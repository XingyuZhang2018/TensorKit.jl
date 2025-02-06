using CUDA
using LinearAlgebra
using BenchmarkTools

# Binary search device function
function binary_search(prefix_sum, val)
    low = 1
    high = length(prefix_sum) - 1
    while low <= high
        mid = (low + high) ÷ 2
        if prefix_sum[mid] < val
            low = mid + 1
        else
            high = mid - 1
        end
    end

    return high
end

function kernel_matrix_product(A, B, C, matrix_sizes)
    prefix_sumA = CuArray([0; cumsum([prod(d[[1,2]]) for d in matrix_sizes])])
    prefix_sumB = CuArray([0; cumsum([prod(d[[2,3]]) for d in matrix_sizes])])
    prefix_sumC = CuArray([0; cumsum([prod(d[[1,3]]) for d in matrix_sizes])])

    function kernel(A, B, C, matrix_sizes, prefix_sumA, prefix_sumB, prefix_sumC)
        idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        
        # Determine which matrix this thread belongs to
        i_matrix = binary_search(prefix_sumC, idx) 

        # Get dimensions of current matrix
        m1 = matrix_sizes[i_matrix][1]  # number of rows
        m2 = matrix_sizes[i_matrix][2]  # middle dimension

        # Calculate position in result matrix C
        pos_C = idx - prefix_sumC[i_matrix]
        row_C = mod1(pos_C, m1)
        col_C = ((pos_C - 1) ÷ m1) + 1

        sum = zero(ComplexF64)
        
        # Compute matrix multiplication
        for k in 1:m2
            iA = (k - 1) * m1 + row_C + prefix_sumA[i_matrix]
            iB = (col_C - 1) * m2 + k + prefix_sumB[i_matrix]
            @inbounds sum += A[iA] * B[iB]
        end

        @inbounds C[idx] = sum
        return nothing
    end

    # Thread configuration
    threads = 256
    blocks = ceil(Int, length(C)/threads)  # Calculate number of blocks based on C's length
    
    @cuda threads=threads blocks=blocks kernel(A, B, C, matrix_sizes, prefix_sumA, prefix_sumB, prefix_sumC)
    CUDA.synchronize()
    return C
end


# Test cases
matrix_sizes = ((100, 200, 300), (200, 100, 300), (100, 300, 100), (100, 200, 300), (200, 100, 300), (100, 300, 100), (100, 200, 300), (200, 100, 300), (100, 300, 100))
Adim = sum(map(m->prod(m[[1,2]]), matrix_sizes))
Bdim = sum(map(m->prod(m[[2,3]]), matrix_sizes))
Cdim = sum(map(m->prod(m[[1,3]]), matrix_sizes))
a = CUDA.rand(ComplexF64, Adim);
b = CUDA.rand(ComplexF64, Bdim);
c = CUDA.rand(ComplexF64, Cdim);

# 调用矩阵乘法
c = CUDA.zeros(ComplexF64, Cdim);
kernel_matrix_product(a, b, c, matrix_sizes);

# # 串行验证
function serial_matrix_product(A, B, C, matrix_sizes)
    prefix_sumA = [0; cumsum([prod(d[[1,2]]) for d in matrix_sizes])]
    prefix_sumB = [0; cumsum([prod(d[[2,3]]) for d in matrix_sizes])]
    prefix_sumC = [0; cumsum([prod(d[[1,3]]) for d in matrix_sizes])]
    A_matrix = [reshape(view(A, prefix_sumA[i]+1:prefix_sumA[i+1]), matrix_sizes[i][[1,2]]) for i in 1:length(matrix_sizes)]
    B_matrix = [reshape(view(B, prefix_sumB[i]+1:prefix_sumB[i+1]), matrix_sizes[i][[2,3]]) for i in 1:length(matrix_sizes)]

    for i in 1:length(matrix_sizes)
        mul!(reshape(view(C, prefix_sumC[i]+1:prefix_sumC[i+1]), matrix_sizes[i][[1,3]]), A_matrix[i], B_matrix[i])
    end
    return C
end


cs = CUDA.zeros(ComplexF64, Cdim);
serial_result = serial_matrix_product(a, b, cs, matrix_sizes);
Aa = Array(a);
Ab = Array(b);
Acs = Array(cs);

# Compare results (using relative error)
println("Relative error: ", norm(c - cs) / norm(cs))

# # 性能测试
println("kernel_matrix_product (GPU):")
@btime CUDA.@sync kernel_matrix_product($a, $b, $c, $matrix_sizes);
println("serial_matrix_product (GPU):")
@btime CUDA.@sync serial_matrix_product($a, $b, $cs, $matrix_sizes);
println("serial_matrix_product (CPU):")
@btime CUDA.@sync serial_matrix_product($Aa, $Ab, $Acs, $matrix_sizes);