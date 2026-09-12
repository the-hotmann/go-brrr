// AVX-512 findBlocksClamp, layered over the SSE2 kernel.
//
// Measured against the SSE2 kernel it replaces, on Zen 4, medians of 6:
//
//	100 histograms   SSE2 26.4 ns   AVX-512 19.1 ns   +38%
//	 50 histograms   SSE2 13.9 ns   AVX-512 10.4 ns   +33%
//
// Eight float64 per iteration is exactly one bitmap byte, so the switch signal
// is one store per group and comes straight out of the comparison mask.
//
// AVX2 is not offered: at 256 bits it measured 29.0 ns and 15.6 ns, i.e. slower
// than the SSE2 kernel, because numHistograms tops out at 100 and the extra
// width does not repay the wider tail.
//
// The fallback is the SSE2 kernel, never the scalar loop, so a CPU without
// AVX-512 is no slower here than it is on a v1 build.

//go:build goexperiment.simd && go1.27 && amd64 && !purego

package encoder

import "simd/archsimd"

// findBlocksHasAVX512 is read once at startup so dispatch is a branch on a
// fixed value rather than a repeated feature query.
var findBlocksHasAVX512 = archsimd.X86.AVX512()

// findBlocksClamp rebases cost against minCost, clamps it at switchCost and
// sets one bit in sig per clamped histogram.
func findBlocksClamp(cost []float64, sig []byte, minCost, switchCost float64) {
	if findBlocksHasAVX512 {
		findBlocksClampAVX512(cost, sig, minCost, switchCost)
		return
	}
	findBlocksClampSSE2(cost, sig, minCost, switchCost)
}

// findBlocksClampAVX512 processes eight float64 per iteration, which is exactly
// one bitmap byte.
//
// The signal byte is OR'd, not assigned: the kernel has to be a drop-in for the
// scalar loop's |=. Assigning happens to work only because findBlocks clears the
// bitmap first, and a caller that stopped doing so would silently lose bits.
func findBlocksClampAVX512(cost []float64, sig []byte, minCost, switchCost float64) {
	mc := archsimd.BroadcastFloat64x8(minCost)
	sc := archsimd.BroadcastFloat64x8(switchCost)
	k := 0
	for ; k+8 <= len(cost); k += 8 {
		v := archsimd.LoadFloat64x8(cost[k:]).Sub(mc)
		sig[k>>3] |= v.GreaterEqual(sc).ToBits()
		v.Min(sc).Store(cost[k:])
	}
	// The tail hands the remaining lanes to the SSE2 kernel rather than a scalar
	// loop, and ORs its bits in, matching what the scalar loop would have done.
	if k < len(cost) {
		findBlocksClampSSE2(cost[k:], sig[k>>3:], minCost, switchCost)
	}
}
