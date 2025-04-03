using StaticArrays, BenchmarkTools

function cmul_acc(mats)
	acc = first(mats)
	for mat in @views mats[2:end]
		acc *= mat
	end
	acc
end

mats = [@SMatrix rand(ComplexF64, 2, 2) for _ in 1:100]

@benchmark cmul_acc(mats)

# BenchmarkTools.Trial: 10000 samples with 156 evaluations per sample.
#  Range (min … max):  658.192 ns …  73.758 μs  ┊ GC (min … max): 0.00% … 98.64%
#  Time  (median):     708.699 ns               ┊ GC (median):    0.00%
#  Time  (mean ± σ):   720.936 ns ± 731.135 ns  ┊ GC (mean ± σ):  1.01% ±  0.99%

#         ▁▃▅▇▃     ▂▄▄██▅▄▄▄▄▄▄▄▃▂▂▂▁   ▃▂▃▃▁ ▁  ▁               ▂
#   ▃▁▁▁▁▃█████▆▃▅▆▇████████████████████████████▇████████▇█▆▇▇▇▇█ █
#   658 ns        Histogram: log(frequency) by time        811 ns <

#  Memory estimate: 80 bytes, allocs estimate: 1.