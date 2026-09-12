//go:build goexperiment.simd && go1.27 && amd64 && !purego

package encoder

import (
	"math"
	"testing"
)

// TestFindBlocksClampAVX512MatchesSSE2 pins the kernel this branch adds against
// the one it layers over, across the shapes that exercise the tail, equality at
// switchCost, and signed zeros.
func TestFindBlocksClampAVX512MatchesSSE2(t *testing.T) {
	for _, shape := range findBlocksKernelShapes {
		for _, n := range findBlocksKernelSizes {
			for _, minCost := range []float64{0.0, 3.5, -1.5} {
				c1, _ := findBlocksKernelFixture(t, n, shape)
				c2 := append([]float64(nil), c1...)
				bl := (n + 7) >> 3
				s1, s2 := make([]byte, bl), make([]byte, bl)
				for i := range s1 {
					s1[i], s2[i] = 0xA5, 0xA5
				}
				findBlocksClampSSE2(c1, s1, minCost, findBlocksTestSwitchCost)
				findBlocksClampAVX512(c2, s2, minCost, findBlocksTestSwitchCost)
				for i := range c1 {
					if math.Float64bits(c2[i]) != math.Float64bits(c1[i]) {
						t.Fatalf("shape=%s n=%d minCost=%v cost[%d]: avx512 %#016x, sse2 %#016x",
							shape, n, minCost, i, math.Float64bits(c2[i]), math.Float64bits(c1[i]))
					}
				}
				for i := range s1 {
					if s2[i] != s1[i] {
						t.Fatalf("shape=%s n=%d minCost=%v sig[%d]: avx512 %#02x, sse2 %#02x; the "+
							"switch bitmap drives the backtrace, so a wrong bit moves block boundaries",
							shape, n, minCost, i, s2[i], s1[i])
					}
				}
			}
		}
	}
}

func benchmarkFindBlocksClampAVX512(b *testing.B, n int) {
	cost, _ := findBlocksKernelFixture(b, n, "mixed")
	sig := make([]byte, (n+7)>>3)
	b.Run("impl=before_sse2_asm", func(b *testing.B) {
		b.ReportAllocs()
		for range b.N {
			findBlocksClampSSE2(cost, sig, 0.5, findBlocksTestSwitchCost)
		}
	})
	b.Run("impl=after_avx512", func(b *testing.B) {
		b.ReportAllocs()
		for range b.N {
			findBlocksClampAVX512(cost, sig, 0.5, findBlocksTestSwitchCost)
		}
	})
}

func BenchmarkFindBlocksClampAVX512_100Histograms(b *testing.B) {
	benchmarkFindBlocksClampAVX512(b, 100)
}

func BenchmarkFindBlocksClampAVX512_50Histograms(b *testing.B) {
	benchmarkFindBlocksClampAVX512(b, 50)
}
