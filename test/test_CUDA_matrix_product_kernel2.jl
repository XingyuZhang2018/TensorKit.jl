using CUDA
using LinearAlgebra
using BenchmarkTools

function cuda_matrix_product(A, B, C, matrix_size)
    function kernel(A, B, C, matrix_size)
        i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        
        # Fix row and column calculation (Julia uses 1-based indexing)
        row = (i - 1) % matrix_size[1] + 1
        col = (i - 1) ÷ matrix_size[1] + 1

        sum = zero(ComplexF64)
        
        # Fix matrix element access method
        for k in 1:matrix_size[2]
            iA = (k - 1) * matrix_size[1] + row  # Column-major index for A
            iB = (col - 1) * matrix_size[2] + k  # Column-major index for B
            @inbounds sum += A[iA] * B[iB]
        end
        
        @inbounds C[i] = sum
        return
    end

    # Thread configuration
    threads = 128
    blocks = ceil(Int, length(A)/threads)
            
    # Launch kernel
    @cuda threads=threads blocks=blocks kernel(A, B, C, matrix_size)
    CUDA.synchronize()
    return C
end

# Test case
a = CUDA.rand(ComplexF64, 256*256)
b = CUDA.rand(ComplexF64, 256*256)
c = CUDA.zeros(ComplexF64, 256*256)

# Call matrix multiplication
cuda_matrix_product(a, b, c, (256, 256))

# CPU verification
cpu_result = reshape(reshape(Array(a), 256, 256) * reshape(Array(b), 256, 256), 256*256)

# Compare results (using relative error)
difference = norm(Array(c) - cpu_result) / norm(cpu_result)
println("Relative error: ", difference)

# Performance testing
println("\nPerformance testing:")
@btime CUDA.@sync cuda_matrix_product($a, $b, $c, (256, 256));
@btime CUDA.@sync mul!($reshape(c, 256, 256), $reshape(a, 256, 256), $reshape(b, 256, 256));
