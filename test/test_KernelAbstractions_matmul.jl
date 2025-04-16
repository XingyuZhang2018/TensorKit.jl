using KernelAbstractions, Test, Random
using BenchmarkTools
include(joinpath(dirname(pathof(KernelAbstractions)), "../examples/utils.jl")) # Load backend

# Simple kernel for matrix multiplication
@kernel function matmul_kernel!(output, a, b)
    i, j = @index(Global, NTuple)

    # creating a temporary sum variable for matrix multiplication
    tmp_sum = zero(eltype(output))
    @inbounds @fastmath for k in 1:size(a)[2]
        tmp_sum += a[i, k] * b[k, j]
    end

    output[i, j] = tmp_sum
end

# Creating a wrapper kernel for launching with error checks
function matmul!(output, a, b)
    if size(a)[2] != size(b)[1]
        println("Matrix size mismatch!")
        return nothing
    end
    backend = KernelAbstractions.get_backend(a)
    kernel! = matmul_kernel!(backend)
    kernel!(output, a, b, ndrange = size(output))
    return
end

N = 1000
a = rand!(allocate(backend, ComplexF64, N, N))
b = rand!(allocate(backend, ComplexF64, N, N))
output = KernelAbstractions.zeros(backend, ComplexF64, N, N)

function foo(output, a, b)
    matmul!(output, a, b)
    KernelAbstractions.synchronize(backend)
end
@btime CUDA.@sync foo($output, $a, $b);
# @btime CUDA.@sync c = $a * $b;
# @test isapprox(output, a * b)