using BenchmarkTools
using CUDA
using Random
using TensorKit
using Test


import CUDA: CuArray
import Base: Array

CuArray(t::TensorMap) = TensorMap(CuArray(t.data), t.space)
Array(t::TensorMap) = TensorMap(Array(t.data), t.space)

@testset "dense Array" begin
    Random.seed!(42)
    d = Int(10^3/2)
    As = ℂ^d ← ℂ^d
    Bs = ℂ^d ← ℂ^d

    Adata = randn(ComplexF64, d,d)
    Bdata = randn(ComplexF64, d,d)

    ATM = TensorMap(Adata, As)
    BTM = TensorMap(Bdata, Bs)
    CTM = ATM * BTM

    cATM = CuArray(ATM)
    cBTM = CuArray(BTM)
    cCTM = cATM * cBTM
    @test CTM ≈ Array(cCTM)

    # @btime $ATM * $BTM
    @btime CUDA.@sync $cATM * $cBTM

    cAdata = CuArray(Adata)
    cBdata = CuArray(Bdata)
    @btime CUDA.@sync $cAdata * $cAdata
end

@testset "Z2 symmetry" begin
    Random.seed!(42)
    d = Int(10^3/2)
    As = ℤ₂Space(0=>d,1=>d) ← ℤ₂Space(0=>d,1=>d)
    Bs = ℤ₂Space(0=>d,1=>d) ← ℤ₂Space(0=>d,1=>d)

    ATM = randn(ComplexF64, As)
    BTM = randn(ComplexF64, Bs)
    CTM = ATM * BTM

    cATM = CuArray(ATM)
    cBTM = CuArray(BTM)
    cCTM = cATM * cBTM
    @test CTM ≈ Array(cCTM)

    @btime $ATM * $BTM
    @btime CUDA.@sync $cATM * $cBTM
end

@testset "U1 symmetry" begin
    Random.seed!(42)
    d = Int(999/3)
    As = U₁Space(0=>d,1=>d,-1=>d) ← U₁Space(0=>d,1=>d,-1=>d)
    Bs = U₁Space(0=>d,1=>d,-1=>d) ← U₁Space(0=>d,1=>d,-1=>d)

    ATM = randn(ComplexF64, As)
    BTM = randn(ComplexF64, Bs)
    CTM = ATM * BTM

    cATM = CuArray(ATM)
    cBTM = CuArray(BTM)
    cCTM = cATM * cBTM
    @test CTM ≈ Array(cCTM)

    @btime $ATM * $BTM
    @btime CUDA.@sync $cATM * $cBTM
end

@testset "U1 symmetry" begin
    Random.seed!(42)
    d = Int(10^3)
    As = U₁Space(-2=>d,-1=>d,0=>d,1=>d,2=>d) ← U₁Space(-2=>d,-1=>d,0=>d,1=>d,2=>d)
    Bs = U₁Space(-2=>d,-1=>d,0=>d,1=>d,2=>d) ← U₁Space(-2=>d,-1=>d,0=>d,1=>d,2=>d)

    ATM = randn(ComplexF64, As)
    BTM = randn(ComplexF64, Bs)
    CTM = ATM * BTM

    cATM = CuArray(ATM)
    cBTM = CuArray(BTM)
    cCTM = cATM * cBTM
    @test CTM ≈ Array(cCTM)

    @btime $ATM * $BTM
    @btime CUDA.@sync $cATM * $cBTM
end