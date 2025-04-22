using AMDGPU
using LinearAlgebra
using BenchmarkTools
using KernelAbstractions
using Random

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

@kernel function multi_matmul_kernel!(A, B, C, matrix_sizes, prefix_sumA, prefix_sumB, prefix_sumC)
    idx = @index(Global)

    # Determine which matrix this thread belongs to
    i_matrix = binary_search(prefix_sumC, idx) 

    # Get dimensions of current matrix
    m1 = matrix_sizes[i_matrix,1]  # number of rows
    m2 = matrix_sizes[i_matrix,2]  # middle dimension

    # Calculate position in result matrix C
    pos_C = idx - prefix_sumC[i_matrix]
    row_C = mod1(pos_C, m1)
    col_C = cld(pos_C, m1)

    sum = zero(ComplexF64)
    
    # Compute matrix multiplication
    iA_shift = prefix_sumA[i_matrix] + row_C  - m1
    iB_shift = prefix_sumB[i_matrix] + (col_C - 1) * m2
    @inbounds @fastmath @simd for k in 1:m2
        iA = k * m1 + iA_shift
        iB = k + iB_shift
        sum += A[iA] * B[iB]
    end

    @inbounds C[idx] = sum
end

function kernel_matrix_product(A, B, C, matrix_sizes)
    prefix_sumA = atype([0; cumsum([prod(d[[1,2]]) for d in matrix_sizes])])
    prefix_sumB = atype([0; cumsum([prod(d[[2,3]]) for d in matrix_sizes])])
    prefix_sumC = atype([0; cumsum([prod(d[[1,3]]) for d in matrix_sizes])])
    matrix_sizes = atype(vcat([[m[1] m[2]] for m in matrix_sizes]...))

    backend = KernelAbstractions.get_backend(A)
    kernel! = multi_matmul_kernel!(backend)
    kernel!(A, B, C, matrix_sizes, prefix_sumA, prefix_sumB, prefix_sumC; ndrange = length(C))
    KernelAbstractions.synchronize(backend)
    return
end


# Test cases
# matrix_sizes = ((100, 200, 300), (200, 100, 300), (100, 300, 100), (100, 200, 300), (200, 100, 300), (100, 300, 100), (100, 200, 300), (200, 100, 300), (100, 300, 100))
Random.seed!(1234)
matrix_sizes = Tuple([Tuple(rand(50:100,3)) for _ in 1:100])
Adim = sum(map(m->prod(m[[1,2]]), matrix_sizes))
Bdim = sum(map(m->prod(m[[2,3]]), matrix_sizes))
Cdim = sum(map(m->prod(m[[1,3]]), matrix_sizes))
atype = ROCArray
a = atype(rand(ComplexF64, Adim));
b = atype(rand(ComplexF64, Bdim));
c = atype(zeros(ComplexF64, Cdim));

kernel_matrix_product(a, b, c, matrix_sizes);

# serial verification
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


cs = atype(zeros(ComplexF64, Cdim));
serial_result = serial_matrix_product(a, b, cs, matrix_sizes);
Aa = Array(a);
Ab = Array(b);
Acs = Array(cs);

# Compare results (using relative error)
println("Relative error: ", norm(c - cs) / norm(cs))

# Benchmarking
println("kernel_matrix_product (GPU):")
@btime kernel_matrix_product($a, $b, $c, $matrix_sizes);
println("serial_matrix_product (GPU):")
@btime serial_matrix_product($a, $b, $cs, $matrix_sizes);
println("serial_matrix_product (CPU):")
@btime serial_matrix_product($Aa, $Ab, $Acs, $matrix_sizes);