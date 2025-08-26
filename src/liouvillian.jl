module Liouvillian

using ..Utils: clean!
using LinearAlgebra: I, kron, tr

function liouvillian(H::AbstractMatrix)
    """
    dρ/dt = 1/iħ L ρ
    where L = 1/iħ H
    """
    shape = size(H)
    L = 1 / 1.0im * (linearise(H, I(shape[1])) - linearise(I(shape[2]), H))
    clean!(L)
    return L
end

function vectorise(ρ::AbstractMatrix)::Vector{ComplexF64}
    shape = size(ρ)
    drho = reshape(ρ, shape[1] * shape[2])
    return drho
end

function linearise(A::AbstractMatrix, B::AbstractMatrix)::Matrix{ComplexF64}
    """
    vec (AρB) = (B⊤ ⊗ A) vec(ρ)
    """
    shape = size(A)
    L = kron(B', A)
    return L
end

function normalise(ρ::AbstractMatrix)::Matrix{ComplexF64}
    tr = trace(ρ)
    ρ = ρ / tr
    return ρ
end

function normalise(dρ::Vector{ComplexF64})::Vector{ComplexF64}
    size = size(dρ) # N^2
    shape = isqrt(size) # N
    ρ = reshape(dρ, shape, shape)
    tr = trace(ρ)
    dρ = dρ / tr
    dρ = reshape(dρ, size)
    return dρ
end

function trace(ρ::AbstractMatrix, O::AbstractMatrix)::ComplexF64
    return tr(O * ρ)
end

function trace(dρ::Vector{ComplexF64}, O::AbstractMatrix)::ComplexF64
    shape = isqrt(size(dρ)[1])
    ρ = reshape(dρ, shape, shape)
    return tr(O * ρ)
end

export liouvillian, vectorise, normalise, linearise, trace

end # module
