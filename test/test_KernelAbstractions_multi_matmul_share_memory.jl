using CUDA
using LinearAlgebra
using BenchmarkTools
using KernelAbstractions
using Random

BLOCK_SIZE = 16
# Binary search device function
function binary_search(prefix_sum, val)
    low = 1
    high = length(prefix_sum) - 1
    while low <= high
        mid = (low + high) >> 1 # same to (low + high) ÷ 2
        if prefix_sum[mid] < val
            low = mid + 1
        else
            high = mid - 1
        end
    end

    return high
end

@kernel function multi_matmul_kernel_shared!(A, B, C, matrix_sizes, prefix_sumA, prefix_sumB, prefix_sumC, prefix_sumblock)
    gi = @index(Group)
    li = @index(Local)

    T = @uniform eltype(C)
    # BLOCK_SIZE = @uniform Int(sqrt(@groupsize()[1]))
    sA = @localmem T (BLOCK_SIZE, BLOCK_SIZE)
    sB = @localmem T (BLOCK_SIZE, BLOCK_SIZE)
    i = mod1(li, BLOCK_SIZE)
    j = cld(li, BLOCK_SIZE)

    # which matrix this thread belongs to
    i_matrix = binary_search(prefix_sumblock, gi)

    # dimensions of current matrix
    m1 = matrix_sizes[i_matrix, 1] 
    m2 = matrix_sizes[i_matrix, 2] 
    m3 = matrix_sizes[i_matrix, 3]

    Ni_block = cld(m1, BLOCK_SIZE) # number of blocks in the first dimension
    g_block = gi - prefix_sumblock[i_matrix]  # local block index

    i_global = prefix_sumC[i_matrix] +  # i_matrix shift 
               min((mod1(g_block, Ni_block) - 1) * BLOCK_SIZE + i, m1) + # block i + local i shift
               (min((cld(g_block, Ni_block) - 1) * BLOCK_SIZE + j, m3) - 1) * m1   # block j + local j shift

    pos_C = i_global - prefix_sumC[i_matrix]
    row_C = mod1(pos_C, m1) 
    col_C = cld(pos_C, m1)

    # base shift for A and B
    shift_A = prefix_sumA[i_matrix] + (j - 1) * m1 + row_C
    shift_B = prefix_sumB[i_matrix] + (col_C - 1) * m2 + i

    accumulator = @private T 1
    @inbounds accumulator[1] = zero(T)

    mid_blocks = cld(m2, BLOCK_SIZE)
    for block_idx in 0:(mid_blocks-1)
        k_start = block_idx * BLOCK_SIZE
        
        idx_A = shift_A + k_start * m1 
        if idx_A <= prefix_sumA[i_matrix + 1]
            @inbounds sA[i, j] = A[idx_A]
        else
            @inbounds sA[i, j] = zero(T)
        end
        
        idx_B = shift_B + k_start 
        # @print gi,block_idx,i,j,row_C,col_C,idx_B "\n"
        if idx_B <= prefix_sumB[i_matrix + 1]
            @inbounds sB[i, j] = B[idx_B]
        else
            @inbounds sB[i, j] = zero(T)
        end
        
        @synchronize

        out = zero(T)
        @inbounds @fastmath @simd for k in 1:BLOCK_SIZE
            out += sA[i, k] * sB[k, j]
        end

        accumulator[1] += out
        @synchronize
    end
    

    @inbounds C[i_global] = accumulator[1]
end

function kernel_matrix_product_shared(A, B, C, matrix_sizes)
    block_size = (BLOCK_SIZE,BLOCK_SIZE)

    prefix_sumA = atype([0; cumsum([prod(m[[1,2]]) for m in matrix_sizes])])
    prefix_sumB = atype([0; cumsum([prod(m[[2,3]]) for m in matrix_sizes])])
    prefix_sumC = atype([0; cumsum([prod(m[[1,3]]) for m in matrix_sizes])])
    prefix_sumblock = atype([0; cumsum([cld(m[1], block_size[1]) * cld(m[3], block_size[2]) for m in matrix_sizes])])
    atype_matrix_sizes = atype(vcat([[m[1] m[2] m[3]] for m in matrix_sizes]...))

    backend = KernelAbstractions.get_backend(A)
    grid_size = CUDA.@allowscalar prefix_sumblock[end]
    kernel! = multi_matmul_kernel_shared!(backend)
    kernel!(A, B, C, atype_matrix_sizes, prefix_sumA, prefix_sumB, prefix_sumC, prefix_sumblock; 
            ndrange=grid_size*prod(block_size), workgroupsize=prod(block_size))
    
    KernelAbstractions.synchronize(backend)
    return
end


Random.seed!(1234)
matrix_sizes = Tuple([Tuple(rand(200:500,3)) for _ in 1:10])
Adim = sum(map(m->prod(m[[1,2]]), matrix_sizes))
Bdim = sum(map(m->prod(m[[2,3]]), matrix_sizes))
Cdim = sum(map(m->prod(m[[1,3]]), matrix_sizes))
atype = CuArray
a = atype(rand(ComplexF64, Adim));
b = atype(rand(ComplexF64, Bdim));
c = atype(rand(ComplexF64, Cdim));


c = CUDA.zeros(ComplexF64, Cdim);
kernel_matrix_product_shared(a, b, c, matrix_sizes);

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


cs = CUDA.zeros(ComplexF64, Cdim);
serial_result = serial_matrix_product(a, b, cs, matrix_sizes);
Aa = Array(a);
Ab = Array(b);
Acs = Array(cs);

# Compare results (using relative error)
println("Relative error: ", norm(c - cs) / norm(cs))

# Benchmarking
println("kernel_matrix_product_shared (GPU):")
@btime CUDA.@sync kernel_matrix_product_shared($a, $b, $c, $matrix_sizes);
println("serial_matrix_product (GPU):")
@btime CUDA.@sync serial_matrix_product($a, $b, $cs, $matrix_sizes);
println("serial_matrix_product (CPU):")
@btime CUDA.@sync serial_matrix_product($Aa, $Ab, $Acs, $matrix_sizes);