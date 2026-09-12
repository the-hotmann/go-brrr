// findBlocksClamp for amd64 builds without the simd experiment, where the
// AVX-512 kernel in findblocks_clamp_avx512_amd64.go is not compiled.

//go:build amd64 && !purego && !(goexperiment.simd && go1.27)

package encoder

// findBlocksClamp rebases cost against minCost, clamps it at switchCost and
// sets one bit in sig per clamped histogram.
func findBlocksClamp(cost []float64, sig []byte, minCost, switchCost float64) {
	findBlocksClampSSE2(cost, sig, minCost, switchCost)
}
