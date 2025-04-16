using CUDA
using LinearAlgebra
using BenchmarkTools

# Binary search device function
function binary_search(prefix_sum, val)
    low = 1
    high = length(prefix_sum) - 1
    while low <= high
        mid = (low + high) >> 1
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
    matrix_sizes1 = CuArray([m[1] for m in matrix_sizes])
    matrix_sizes2 = CuArray([m[2] for m in matrix_sizes])
    function kernel(A, B, C, matrix_sizes1, matrix_sizes2, prefix_sumA, prefix_sumB, prefix_sumC)
        idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        stride = gridDim().x * blockDim().x

        while idx <= length(C)
            # Determine which matrix this thread belongs to
            i_matrix = binary_search(prefix_sumC, idx) 

            # Get dimensions of current matrix
            m1 = matrix_sizes1[i_matrix]  # number of rows
            m2 = matrix_sizes2[i_matrix]  # middle dimension

            # Calculate position in result matrix C
            pos_C = idx - prefix_sumC[i_matrix]
            row_C = mod1(pos_C, m1)
            col_C = ((pos_C - 1) ÷ m1) + 1

            sum = zero(ComplexF64)
            
            # Compute matrix multiplication
            iA_shift = prefix_sumA[i_matrix] + row_C  - m1
            iB_shift = prefix_sumB[i_matrix] + (col_C - 1) * m2
            @inbounds @fastmath for k in 1:m2
                iA = k * m1 + iA_shift
                iB = k + iB_shift
                sum += A[iA] * B[iB]
            end

            @inbounds C[idx] = sum
            idx += stride
        end
        return nothing
    end

    # Thread configuration
    k = @cuda launch=false kernel(A, B, C, matrix_sizes1, matrix_sizes2, prefix_sumA, prefix_sumB, prefix_sumC)
    config = launch_configuration(k.fun)
    # @show config
    threads = min(length(A), 128)
    blocks = min(cld(length(A), threads), config.blocks)
    k(A, B, C, matrix_sizes1, matrix_sizes2, prefix_sumA, prefix_sumB, prefix_sumC; threads, blocks)
    # CUDA.synchronize()
    return C
end


# Test cases
# matrix_sizes = ((100, 200, 300), (200, 100, 300), (100, 300, 100), (100, 200, 300), (200, 100, 300), (100, 300, 100), (100, 200, 300), (200, 100, 300), (100, 300, 100))
Random.seed!(1234)
matrix_sizes = Tuple([Tuple(rand(50:100,3)) for _ in 1:100])
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