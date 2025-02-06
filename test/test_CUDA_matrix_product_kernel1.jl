using CUDA
using LinearAlgebra
using BenchmarkTools

function cuda_matrix_product(A, B, C)
    function kernel(A, B, C)
        # Calculate global indices
        row = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        col = (blockIdx().y - 1) * blockDim().y + threadIdx().y
        
        # Initialize accumulator
        sum = zero(ComplexF64)
        
        # Compute dot product directly
        for k in 1:size(A, 2)
            @inbounds sum += A[row, k] * B[k, col]
        end
        
        # Write result
        @inbounds C[row, col] = sum
        return
    end

    # Thread configuration
    threads = (16, 16)  # Use smaller thread blocks
    blocks = (ceil(Int, size(A,1)/threads[1]), 
             ceil(Int, size(B,2)/threads[2]))
    
    # Launch kernel
    @cuda threads=threads blocks=blocks kernel(A, B, C)
    CUDA.synchronize()
    return C
end

# Test case
a = CUDA.rand(ComplexF64, 256, 256)
b = CUDA.rand(ComplexF64, 256, 256)
c = CUDA.zeros(ComplexF64, 256, 256)

# Call matrix multiplication
cuda_matrix_product(a, b, c)

# CPU verification
cpu_result = Array(a) * Array(b)

# Compare results (using relative error)
difference = norm(Array(c) - cpu_result) / norm(cpu_result)
println("Relative error: ", difference)

# Performance testing
println("\nPerformance testing:")
@btime CUDA.@sync cuda_matrix_product($a, $b, $c);
@btime CUDA.@sync mul!($c, $a, $b);
