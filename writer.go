// Streaming brotli writer and public API.

package brrr

import (
	"errors"
	"io"
	"runtime"
	"strconv"
	"sync"

	"github.com/molecule-man/go-brrr/internal/encoder"
)

// Writer compresses data into brotli format.
//
// Callers must Close the Writer to finalize the brotli stream.
type Writer struct {
	dst      io.Writer
	err      error
	c        encoder.Compressor
	dicts    []*PreparedDictionary // from WriterOptions, preserved across Reset
	quality  int                   // 0 = one-pass, 1 = two-pass, 2+ = streaming
	lgwin    int
	sizeHint uint // from WriterOptions, preserved across Reset
	closed   bool
	reused   bool // true after first Reset; suppresses pool release on Close
}

// NewWriter returns a new Writer compressing data to dst at the given
// quality level. Supported levels are 0 (BestSpeed) through 11
// (BestCompression).
func NewWriter(dst io.Writer, level int) (*Writer, error) {
	return NewWriterOptions(dst, level, WriterOptions{})
}

// NewWriterOptions returns a new Writer compressing data to dst at the given
// quality level with additional tuning options. LGWin range is 10–24; 0
// selects the default (22). Compound dictionaries supplied via opts.Dictionaries
// require level >= 2.
func NewWriterOptions(dst io.Writer, level int, opts WriterOptions) (*Writer, error) {
	if err := checkLevel(level); err != nil {
		return nil, err
	}

	lgwin := opts.LGWin
	if lgwin == 0 {
		lgwin = defaultLGWin
	}
	if lgwin < minLGWin || lgwin > maxLGWin {
		return nil, errors.New("brrr: invalid window size: lgwin=" + strconv.Itoa(lgwin) +
			" (must be " + strconv.Itoa(minLGWin) + "–" + strconv.Itoa(maxLGWin) + ")")
	}

	if len(opts.Dictionaries) > encoder.MaxCompoundDicts {
		return nil, encoder.ErrTooManyDicts
	}
	if len(opts.Dictionaries) > 0 && level < 2 {
		return nil, encoder.ErrQualityTooLow
	}

	w := &Writer{dst: dst, quality: level, lgwin: lgwin, sizeHint: opts.SizeHint, dicts: opts.Dictionaries}
	w.c = encoder.NewCompressor(w.quality, w.lgwin, w.sizeHint)
	for _, pd := range w.dicts {
		_ = w.c.AttachDictionary(pd.impl)
	}
	if level >= minWorkerLevel {
		runtime.SetFinalizer(w, (*Writer).release)
	}
	return w, nil
}

const minWorkerLevel = 10

func (w *Writer) release() {
	if w.c != nil {
		w.c.Release()
		w.c = nil
	}
}

// Compress compresses data at the given quality level and returns the
// brotli-compressed bytes. It is the one-shot counterpart to [Decompress].
// Supported levels are 0 (BestSpeed) through 11 (BestCompression). The exact
// input length is supplied to the encoder as a size hint.
func Compress(data []byte, level int) ([]byte, error) {
	o, err := getOneshotCompressor(level, uint(len(data)))
	if err != nil {
		return nil, err
	}
	err = o.compress(&o.chunks, data)
	if err != nil {
		o.chunks.discard()
		o.drop()
		return nil, err
	}
	out := o.chunks.take()
	oneshotCompressors[level].Put(o)
	return out, nil
}

func AppendCompress(dst, data []byte, level int) ([]byte, error) {
	o, err := getOneshotCompressor(level, uint(len(data)))
	if err != nil {
		return dst, err
	}
	o.appended.buf = dst
	err = o.compress(&o.appended, data)
	out := o.appended.buf
	o.appended.buf = nil
	if err != nil {
		o.drop()
		return dst, err
	}
	oneshotCompressors[level].Put(o)
	return out, nil
}

var oneshotCompressors [maxLevel + 1]sync.Pool

const maxLevel = 11

type oneshotCompressor struct {
	c        encoder.Compressor
	appended appendWriter
	chunks   chunkWriter
}

func getOneshotCompressor(level int, sizeHint uint) (*oneshotCompressor, error) {
	if err := checkLevel(level); err != nil {
		return nil, err
	}
	if v := oneshotCompressors[level].Get(); v != nil {
		o := v.(*oneshotCompressor)
		o.c.ResetSizeHint(sizeHint)
		return o, nil
	}
	o := &oneshotCompressor{
		c:      encoder.NewCompressor(level, defaultLGWin, sizeHint),
		chunks: chunkWriter{pool: &encodeChunkPool, size: encodeChunkSize},
	}
	if level >= minWorkerLevel {
		runtime.SetFinalizer(o, (*oneshotCompressor).drop)
	}
	return o, nil
}

func (o *oneshotCompressor) compress(dst io.Writer, data []byte) error {
	if _, err := o.c.Write(dst, data); err != nil {
		return err
	}
	return o.c.Close(dst)
}

func (o *oneshotCompressor) drop() {
	if o.c != nil {
		o.c.Release()
		o.c = nil
	}
}

func checkLevel(level int) error {
	if level < 0 || level > maxLevel {
		return errors.New("brrr: invalid compression level: " + strconv.Itoa(level))
	}
	return nil
}

type appendWriter struct {
	buf []byte
}

func (w *appendWriter) Write(p []byte) (int, error) {
	w.buf = append(w.buf, p...)
	return len(p), nil
}

const (
	encodeChunkSize = 64 << 10
	decodeChunkSize = 272 << 10
)

var (
	encodeChunkPool = sync.Pool{New: func() any { b := make([]byte, encodeChunkSize); return &b }}
	decodeChunkPool = sync.Pool{New: func() any { b := make([]byte, decodeChunkSize); return &b }}
)

type chunkWriter struct {
	pool   *sync.Pool
	chunks []*[]byte
	size   int
	n      int
}

func (w *chunkWriter) Write(p []byte) (int, error) {
	w.write(p)
	return len(p), nil
}

func (w *chunkWriter) write(p []byte) {
	for len(p) > 0 {
		if w.n == len(w.chunks)*w.size {
			w.chunks = append(w.chunks, w.pool.Get().(*[]byte))
		}
		copied := copy((*w.chunks[len(w.chunks)-1])[w.n%w.size:], p)
		w.n += copied
		p = p[copied:]
	}
}

func (w *chunkWriter) take() []byte {
	if w.n == 0 {
		w.discard()
		return nil
	}
	out := make([]byte, w.n)
	for i, chunk := range w.chunks {
		copy(out[i*w.size:], *chunk)
	}
	w.discard()
	return out
}

func (w *chunkWriter) discard() {
	for i, chunk := range w.chunks {
		w.pool.Put(chunk)
		w.chunks[i] = nil
	}
	w.chunks = w.chunks[:0]
	w.n = 0
}

// Write compresses p and writes it to the underlying writer.
// Data may be buffered internally; call Flush or Close to ensure all
// data is written.
func (w *Writer) Write(p []byte) (int, error) {
	if w.err != nil {
		return 0, w.err
	}
	if w.closed {
		return 0, io.ErrClosedPipe
	}
	n, err := w.c.Write(w.dst, p)
	if err != nil {
		w.err = err
	}
	return n, err
}

// Flush compresses any buffered data and writes it to the underlying
// writer as one or more non-final meta-blocks. Flush does not finalize
// the brotli stream; call Close for that.
func (w *Writer) Flush() error {
	if w.err != nil {
		return w.err
	}
	if w.closed {
		return io.ErrClosedPipe
	}
	w.err = w.c.Flush(w.dst)
	return w.err
}

// Close flushes remaining data, finalizes the brotli stream by writing
// the final empty meta-block, and writes everything to the underlying writer.
// Close does not close the underlying writer.
func (w *Writer) Close() error {
	if w.err != nil {
		return w.err
	}
	if w.closed {
		return nil
	}
	w.closed = true
	w.err = w.c.Close(w.dst)
	if !w.reused {
		w.c.Release()
		w.c = nil
	}
	return w.err
}

// Reset discards internal state and switches to writing to dst.
// This permits reusing a Writer rather than allocating a new one.
// Compound dictionaries supplied via WriterOptions are preserved.
func (w *Writer) Reset(dst io.Writer) {
	w.dst = dst
	w.err = nil
	w.closed = false
	w.reused = true

	if w.c == nil {
		// Compressor was released on a previous Close; re-acquire.
		w.c = encoder.NewCompressor(w.quality, w.lgwin, w.sizeHint)
	} else {
		w.c.Reset()
	}
	for _, pd := range w.dicts {
		_ = w.c.AttachDictionary(pd.impl)
	}
}
