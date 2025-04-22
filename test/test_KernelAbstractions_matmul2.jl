using KernelAbstractions, Test, Random
using BenchmarkTools
include(joinpath(dirname(pathof(KernelAbstractions)), "../examples/utils.jl")) # Load backend

# Simple kernel for matrix multiplication
@kernel function matmul_kernel!(output, @Const(a), @Const(b), N)
    gi, gj = @index(Group, NTuple)
    li, lj = @index(Local, NTuple)
    i = (gi - 1) * @groupsize()[1] + li
    j = (gj - 1) * @groupsize()[2] + lj
    # i, j = @index(Global, NTuple) # Global index for the kernel
    
    # creating a temporary sum variable for matrix multiplication
    N1,N2,_ = N
    tmp_sum = zero(eltype(output))
    @inbounds @fastmath for k in 1:N2
        iA = (k - 1) * N1 + i  # Column-major index for A
        iB = (j - 1) * N2 + k  # Column-major index for B
        tmp_sum += a[iA] * b[iB]
    end

    index = (j - 1) * N1 + i
    @inbounds output[index] = tmp_sum
end

# Creating a wrapper kernel for launching with error checks
function matmul!(output, a, b, N)
    backend = KernelAbstractions.get_backend(a)
    kernel! = matmul_kernel!(backend)
    kernel!(output, a, b, N, ndrange = (N[1],N[3]))
    KernelAbstractions.synchronize(backend)
    return
end

N1 = 1024
N2 = 1024
N3 = 1024
a = rand!(allocate(backend, ComplexF64, N1*N2))
b = rand!(allocate(backend, ComplexF64, N2*N3))
output = KernelAbstractions.zeros(backend, ComplexF64, N1*N3)

# function foo(output, a, b)
matmul!(output, a, b, (N1, N2, N3))

@test reshape(output,N1,N3) ≈ reshape(a,N1,N2) * reshape(b,N2,N3)