module Liouvillian

using ..Utils: clean!
using LinearAlgebra: I, kron, tr

function liouvillian(H::AbstractMatrix)::AbstractMatrix
  """
  dρ/dt = 1/iħ L ρ
  where L = 1/iħ H
  """
  shape = size(H)
  L = 1 / 1.0im .* (linearise(H, I(shape[1])) .- linearise(I(shape[2]), H))
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
  In general,
  vec (AρB) = (B⊤ ⊗ A) vec(ρ)

  If B is Hermitian, then B⊤ = B*.
  Thus,
  vec (AρB) = (B* ⊗ A) vec(ρ)

  If B is Hermitian + skew-Hermitian Haberkorn term, B = H - iK/2 where
  H* = H⊤ and K* = K, corresponding Livouvillian is 1/i [H, ρ] - {K/2, ρ},
  which leads to
  vec(L[ρ]) = 1/i (I ⊗ H - H⊤ ⊗ I) vec(ρ) - (I ⊗ K/2 + K/2 ⊗ I) vec(ρ)
            = 1/i(I ⊗ (H - iK/2) - (H⊤ + iK/2) ⊗ I) vec(ρ)
            = 1/i(I ⊗ (H - iK/2) - (H - iK/2)* ⊗ I) vec(ρ).
            = 1/i(I ⊗ B - B* ⊗ I) vec(ρ).

  Thus, it is convenient to use B* instead of B⊤.
  """
  L = kron(conj(B), A)
  return L
end

function normalise(ρ::AbstractMatrix)::Matrix{ComplexF64}
  tr = tr(ρ)
  ρ = ρ ./ tr
  return ρ
end

function normalise(dρ::Vector{ComplexF64})::Vector{ComplexF64}
  size = size(dρ) # N^2
  shape = isqrt(size) # N
  ρ = reshape(dρ, shape, shape)
  tr = tr(ρ)
  dρ = dρ ./ tr
  dρ = reshape(dρ, size)
  return dρ
end

function trace(ρ::AbstractMatrix, O::AbstractMatrix)::Float64
  return real(tr(O * ρ))
end

function trace(dρ::Vector{ComplexF64}, O::AbstractMatrix)::Float64
  shape = isqrt(size(dρ)[1])
  ρ = reshape(dρ, shape, shape)
  return real(tr(O * ρ))
end

export liouvillian, vectorise, normalise, linearise, trace

end # module
