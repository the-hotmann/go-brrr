package brrr

import (
	"bytes"
	"fmt"
	"io"
	"math/rand/v2"
	"os"
	"runtime"
	"runtime/debug"
	"testing"
	"time"
)

func allocTestPayload(tb testing.TB) []byte {
	tb.Helper()
	page, err := os.ReadFile("testdata/gh_172KB.html")
	if err != nil {
		tb.Fatal(err)
	}
	return bytes.Repeat(page, 3)
}

func TestReusedReaderDecodesAStreamWithoutAllocating(t *testing.T) {
	in := allocTestPayload(t)
	for _, level := range []int{1, 5, 11} {
		t.Run(fmt.Sprintf("q%d", level), func(t *testing.T) {
			comp, err := Compress(in, level)
			if err != nil {
				t.Fatal(err)
			}
			src := bytes.NewReader(comp)
			r := NewReader(src)
			out := make([]byte, len(in))
			allocs := testing.AllocsPerRun(3, func() {
				src.Reset(comp)
				r.Reset(src)
				if _, err := io.ReadFull(r, out); err != nil {
					t.Fatal(err)
				}
				if n, err := r.Read(out[:1]); n != 0 || err != io.EOF {
					t.Fatalf("trailing read = %d, %v; want 0, EOF", n, err)
				}
			})
			if allocs != 0 {
				t.Errorf("a reused Reader allocated %.1f times per stream; its ring buffer and tables must be "+
					"kept across Reset so steady-state decoding never touches the heap", allocs)
			}
			if !bytes.Equal(out, in) {
				t.Fatal("decoded bytes differ from the input")
			}
		})
	}
}

func TestReusedWriterCompressesAStreamWithoutAllocating(t *testing.T) {
	in := allocTestPayload(t)
	for level := 0; level <= 11; level++ {
		t.Run(fmt.Sprintf("q%d", level), func(t *testing.T) {
			w, err := NewWriterOptions(io.Discard, level, WriterOptions{SizeHint: uint(len(in))})
			if err != nil {
				t.Fatal(err)
			}
			allocs := testing.AllocsPerRun(2, func() {
				w.Reset(io.Discard)
				if _, err := w.Write(in); err != nil {
					t.Fatal(err)
				}
				if err := w.Close(); err != nil {
					t.Fatal(err)
				}
			})
			if allocs != 0 {
				t.Errorf("a reused q%d Writer allocated %.1f times per stream; buffers and any helper "+
					"goroutine must live as long as the Writer so steady-state compression never touches the heap",
					level, allocs)
			}
		})
	}
}

func waitForGoroutines(tb testing.TB, want int) int {
	tb.Helper()
	got := runtime.NumGoroutine()
	for range 200 {
		if got <= want {
			return got
		}
		runtime.GC()
		time.Sleep(5 * time.Millisecond)
		got = runtime.NumGoroutine()
	}
	return got
}

func TestHighQualityWriterReleasesItsMatchCollectorWhenClosedOrAbandoned(t *testing.T) {
	in := allocTestPayload(t)
	baseline := waitForGoroutines(t, runtime.NumGoroutine())

	w, err := NewWriterOptions(io.Discard, 11, WriterOptions{SizeHint: uint(len(in))})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := w.Write(in); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	if got := waitForGoroutines(t, baseline); got > baseline {
		t.Errorf("closing a q11 Writer left %d goroutines running (baseline %d); the collector must stop "+
			"when the encoder is released or every stream leaks one", got, baseline)
	}

	func() {
		abandoned, err := NewWriterOptions(io.Discard, 10, WriterOptions{SizeHint: uint(len(in))})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := abandoned.Write(in); err != nil {
			t.Fatal(err)
		}
		if err := abandoned.Flush(); err != nil {
			t.Fatal(err)
		}
	}()
	if got := waitForGoroutines(t, baseline); got > baseline {
		t.Errorf("an unclosed q10 Writer that became unreachable left %d goroutines running (baseline %d); "+
			"its finalizer must stop the collector so a forgotten Close cannot leak it", got, baseline)
	}
}

func writerStream(tb testing.TB, in []byte, level int) []byte {
	tb.Helper()
	var buf bytes.Buffer
	w, err := NewWriterOptions(&buf, level, WriterOptions{SizeHint: uint(len(in))})
	if err != nil {
		tb.Fatal(err)
	}
	if _, err := w.Write(in); err != nil {
		tb.Fatal(err)
	}
	if err := w.Close(); err != nil {
		tb.Fatal(err)
	}
	return buf.Bytes()
}

func TestOneshotCompressionMatchesTheStreamingWriterAndOnlyAllocatesTheResult(t *testing.T) {
	in := allocTestPayload(t)
	prefix := []byte("kept-prefix")
	for level := 0; level <= 11; level++ {
		t.Run(fmt.Sprintf("q%d", level), func(t *testing.T) {
			want := writerStream(t, in, level)
			if level < minWorkerLevel {
				overOneMiB := bytes.Repeat(in, 3)
				if _, err := Compress(overOneMiB, level); err != nil {
					t.Fatal(err)
				}

				noisy := make([]byte, 256<<10)
				rng := rand.New(rand.NewPCG(uint64(level), 17))
				for i := range noisy {
					noisy[i] = byte(rng.Uint32())
				}
				gotNoisy, err := Compress(noisy, level)
				if err != nil {
					t.Fatal(err)
				}
				if wantNoisy := writerStream(t, noisy, level); !bytes.Equal(gotNoisy, wantNoisy) ||
					len(gotNoisy) <= encodeChunkSize {
					t.Fatalf("Compress of incompressible input produced %d bytes, the Writer %d; output larger than "+
						"one %d-byte chunk must be stitched from several chunks in order without losing a byte",
						len(gotNoisy), len(wantNoisy), encodeChunkSize)
				}
			}

			got, err := Compress(in, level)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(got, want) {
				t.Fatalf("Compress produced %d bytes that differ from the %d-byte Writer stream; the pooled "+
					"one-shot compressor must reset to exactly the state a fresh Writer starts from, including the "+
					"new input's size hint after a call above 1 MiB", len(got), len(want))
			}
			if cap(got) != len(got) {
				t.Errorf("Compress returned cap %d for len %d; the result must be one exact-size allocation", cap(got), len(got))
			}

			dst := append(make([]byte, 0, len(prefix)+len(want)), prefix...)
			appended, err := AppendCompress(dst, in, level)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(appended[:len(prefix)], prefix) || !bytes.Equal(appended[len(prefix):], want) {
				t.Fatal("AppendCompress must keep dst's existing bytes and append exactly the Writer stream")
			}

			if raceDetectorEnabled {
				return
			}
			if allocs := allocsBetweenCollections(2, func() {
				if _, err := AppendCompress(dst[:len(prefix)], in, level); err != nil {
					t.Fatal(err)
				}
			}); allocs != 0 {
				t.Errorf("AppendCompress into a destination with enough room allocated %.1f times; the pooled "+
					"compressor, its output sink and any helper goroutine must be reused", allocs)
			}
			if allocs := allocsBetweenCollections(2, func() {
				if _, err := Compress(in, level); err != nil {
					t.Fatal(err)
				}
			}); allocs != 1 {
				t.Errorf("Compress allocated %.1f times; it must allocate only the exact-size result and "+
					"collect output in pooled chunks instead of a growing buffer", allocs)
			}
		})
	}
}

func TestOneshotDecompressionMatchesTheInputAndOnlyAllocatesTheResult(t *testing.T) {
	in := allocTestPayload(t)
	prefix := []byte("kept-prefix")
	for _, level := range []int{1, 5, 11} {
		t.Run(fmt.Sprintf("q%d", level), func(t *testing.T) {
			comp, err := Compress(in, level)
			if err != nil {
				t.Fatal(err)
			}

			got, err := Decompress(comp)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(got, in) || cap(got) != len(got) {
				t.Fatalf("Decompress returned %d bytes (cap %d) for a %d-byte input; it must return exactly the "+
					"input in one exact-size slice", len(got), cap(got), len(in))
			}

			dst := append(make([]byte, 0, len(prefix)+len(in)), prefix...)
			appended, err := AppendDecompress(dst, comp)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(appended[:len(prefix)], prefix) || !bytes.Equal(appended[len(prefix):], in) {
				t.Fatal("AppendDecompress must keep dst's existing bytes and append exactly the decoded input")
			}

			if raceDetectorEnabled {
				return
			}
			if allocs := allocsBetweenCollections(3, func() {
				if _, err := AppendDecompress(dst[:len(prefix)], comp); err != nil {
					t.Fatal(err)
				}
			}); allocs != 0 {
				t.Errorf("AppendDecompress into a destination with enough room allocated %.1f times; decoding "+
					"must write straight into dst using the pooled decoder state", allocs)
			}
			if allocs := allocsBetweenCollections(3, func() {
				if _, err := Decompress(comp); err != nil {
					t.Fatal(err)
				}
			}); allocs != 1 {
				t.Errorf("Decompress allocated %.1f times; it must allocate only the exact-size result and "+
					"collect output in pooled chunks instead of doubling a buffer", allocs)
			}
		})
	}
}

func allocsBetweenCollections(runs int, f func()) float64 {
	defer debug.SetGCPercent(debug.SetGCPercent(-1))
	return testing.AllocsPerRun(runs, f)
}

func BenchmarkAppendCompress(b *testing.B) {
	in := allocTestPayload(b)
	for _, level := range []int{1, 5, 9} {
		b.Run(fmt.Sprintf("q=%d", level), func(b *testing.B) {
			dst, err := AppendCompress(nil, in, level)
			if err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.SetBytes(int64(len(in)))
			for range b.N {
				if dst, err = AppendCompress(dst[:0], in, level); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkAppendDecompress(b *testing.B) {
	in := allocTestPayload(b)
	for _, level := range []int{1, 5, 11} {
		b.Run(fmt.Sprintf("q=%d", level), func(b *testing.B) {
			comp, err := Compress(in, level)
			if err != nil {
				b.Fatal(err)
			}
			dst := make([]byte, 0, len(in))
			b.ReportAllocs()
			b.SetBytes(int64(len(in)))
			for range b.N {
				if dst, err = AppendDecompress(dst[:0], comp); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
