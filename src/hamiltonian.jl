
module Hamiltonian

using ..Utils: clean, sample_from_sphere, sphere_to_cartesian
using ..SpinOps: Sx1, Sy1, Sz1, Sx2, Sy2, Sz2, Ps, Pt, S1S2
using ..SystemModule: System
using ..MoleculeModule: Molecule, is_isotropic, Aiso
using ..Constants: γe

using LinearAlgebra: I, ishermitian, diagm
using StaticArrays: SMatrix
using Distributions: MultivariateNormal

function system_hamiltonian(
    sys::System,
    B::Float64,
    θ::Float64 = 0.0,
    ϕ::Float64 = 0.0,
)::SMatrix{4, 4, ComplexF64}
    return system_hamiltonian(
        B = B,
        θ = θ,
        ϕ = ϕ,
        J = sys.J,
        D = sys.D,
        kS = sys.kS,
        kT = sys.kT,
    )
end

function system_hamiltonian(;
    B::Float64,
    θ::Float64 = 0.0,
    ϕ::Float64 = 0.0,
    J::Float64 = 0.0,
    D::Union{Float64, AbstractMatrix{Float64}} = 0.0,
    kS::Float64 = 0.0,
    kT::Float64 = 0.0,
)::SMatrix{4, 4, ComplexF64}
    Hz = zeeman_hamiltonian(B, θ, ϕ)
    Hj = exchange_hamiltonian(J)
    Hd = dipolar_hamiltonian(D)
    Hk = haberkorn_hamiltonian(kS, kT)
    Hsys = Hz + Hj + Hd + Hk
    Hsys = clean(Hsys)

    return Hsys
end

function zeeman_hamiltonian(
    B::Float64,
    θ::Float64 = 0.0,
    ϕ::Float64 = 0.0,
)::SMatrix{4, 4, ComplexF64}
    """
    Hz = -γe ∑_i (Bx Sx_i + By Sy_i + Bz Sz_i)

    we rescale the Hamiltonian by 1/|γe| and γe < 0.
    """
    @assert B ≥ 0 "B must be non-negative"
    @assert 0 ≤ θ ≤ π "θ must be in [0, π]"
    @assert 0 ≤ ϕ ≤ 2π "ϕ must be in [0, 2π]"

    Bx = B * sin(θ) * cos(ϕ)
    By = B * sin(θ) * sin(ϕ)
    Bz = B * cos(θ)

    Hz = SMatrix{4, 4, ComplexF64}(
        Bx .* (Sx1 .+ Sx2) .+ By .* (Sy1 .+ Sy2) .+ Bz .* (Sz1 .+ Sz2),
    )
    return Hz
end

function exchange_hamiltonian(sys::System)::SMatrix{4, 4, ComplexF64}
    return exchange_hamiltonian(sys.J)
end

function exchange_hamiltonian(J::Float64)::SMatrix{4, 4, ComplexF64}
    Hj = SMatrix{4, 4, ComplexF64}(J .* (2.0 .* S1S2 .- 0.5 .* I(4)))
    return -Hj
end

function dipolar_hamiltonian(sys::System)::SMatrix{4, 4, ComplexF64}
    return dipolar_hamiltonian(sys.D)
end

function dipolar_hamiltonian(D::Float64)::SMatrix{4, 4, ComplexF64}
    @assert D ≤ 0 "D must be non-positive under point-dipole approximation"
    Dtensor = diagm(0 => [2D/3, 2D/3, -4D/3])
    return dipolar_hamiltonian(Dtensor)
end

function dipolar_hamiltonian(D::AbstractMatrix)::SMatrix{4, 4, ComplexF64}
    @assert size(D) == (3, 3) "D must be a 3×3 matrix"
    @assert all(isreal(D)) "D must be real"
    @assert D == D' "D must be symmetric"

    Hd = SMatrix{4, 4, ComplexF64}(
        D[1, 1] .* (Sx1 * Sx2) +
        D[1, 2] .* (Sx1 * Sy2) +
        D[1, 3] .* (Sx1 * Sz2) +
        D[2, 1] .* (Sy1 * Sx2) +
        D[2, 2] .* (Sy1 * Sy2) +
        D[2, 3] .* (Sy1 * Sz2) +
        D[3, 1] .* (Sz1 * Sx2) +
        D[3, 2] .* (Sz1 * Sy2) +
        D[3, 3] .* (Sz1 * Sz2),
    )
    return -Hd
end

function haberkorn_hamiltonian(sys::System)::SMatrix{4, 4, ComplexF64}
    return haberkorn_hamiltonian(sys.kS, sys.kT)
end

function haberkorn_hamiltonian(kS::Float64, kT::Float64)::SMatrix{4, 4, ComplexF64}
    Hk = SMatrix{4, 4, ComplexF64}(
        (-1.0im * kS / 2.0 .* Ps .- 1.0im * kT / 2.0 .* Pt) ./ abs(γe) .* 1e-03,
    )
    return Hk
end

function SchultenWolynes_hamiltonian(
    mol1::Molecule,
    mol2::Molecule,
    N::Integer,
)::Vector{SMatrix{4, 4, ComplexF64}}
    @assert is_isotropic(mol1.A) "mol1 must be isotropic"
    @assert is_isotropic(mol2.A) "mol2 must be isotropic"
    # return SchultenWolynes_hamiltonian(Aiso(mol1.A), Aiso(mol2.A), mol1.I, mol2.I, N)
    return SchultenWolynes_hamiltonian(mol1.A, mol2.A, mol1.I, mol2.I, N)
end

function SchultenWolynes_hamiltonian(
    a1::Union{Vector{Float64}, Array{Float64, 3}},
    a2::Union{Vector{Float64}, Array{Float64, 3}},
    I1::Vector{<:Integer},
    I2::Vector{<:Integer},
    N::Integer,
)::Vector{SMatrix{4, 4, ComplexF64}}
    @assert N ≥ 1 "N must be at least 1"
    Ix1, Iy1, Iz1 = SW_each(a1, I1, N)
    Ix2, Iy2, Iz2 = SW_each(a2, I2, N)
    Hsw = []
    for i in 1:N
        Hsw1 = -(Ix1[i] .* Sx1 .+ Iy1[i] .* Sy1 .+ Iz1[i] .* Sz1)
        Hsw2 = -(Ix2[i] .* Sx2 .+ Iy2[i] .* Sy2 .+ Iz2[i] .* Sz2)
        push!(Hsw, SMatrix{4, 4, ComplexF64}(Hsw1 .+ Hsw2))
    end
    return Hsw
end

function SW_each(
    a::Vector{Float64},
    I::Vector{<:Integer},
    N::Integer,
)::Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}
    @assert N ≥ 1 "N must be at least 1"
    @assert length(a) == length(I) "a and I must have the same length"

    τ = 6.0 / sum((a .^ 2) .* I .* (I .+ 1))

    # Smaple from f(x) = (τ²/4π)^(3/2) exp(-I²τ^2/4)

    μ = [0.0, 0.0, 0.0]
    Σ = diagm(0 => [2/(τ^2), 2/(τ^2), 2/(τ^2)])

    # Sample
    Ixyz_sampled = rand(MultivariateNormal(μ, Σ), Int(N))
    Ix = Ixyz_sampled[1, :]
    Iy = Ixyz_sampled[2, :]
    Iz = Ixyz_sampled[3, :]
    return Ix, Iy, Iz
end

function SW_each(
    A::Array{Float64, 3},
    I::Vector{<:Integer},
    N::Integer,
)::Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}
    N_nuc = size(A, 1)
    @assert size(A) == (N_nuc, 3, 3) "A must be a N_nuc×3×3 matrix but got $(size(A))"
    θ, ϕ = sample_from_sphere((N, N_nuc))
    # Unit vector on the sphere: x = sinθ cosϕ, y = sinθ sinϕ, z = cosθ
    ux, uy, uz = sphere_to_cartesian(θ, ϕ)  # (N, N_nuc)
    @assert size(ux) == (N, N_nuc) "ux must be a N×N_nuc matrix but got $(size(ux))"
    @assert size(uy) == (N, N_nuc) "uy must be a N×N_nuc matrix but got $(size(uy))"
    @assert size(uz) == (N, N_nuc) "uz must be a N×N_nuc matrix but got $(size(uz))"

    vector_length = sqrt.(I .* (I .+ 1))  # (N_nuc,)

    Ix = zeros(Float64, N)
    Iy = zeros(Float64, N)
    Iz = zeros(Float64, N)
    for i in 1:N_nuc
        # Project A_i (3x3) on the random direction u for all N samples
        bx = A[i, 1, 1] .* ux[:, i] .+ A[i, 1, 2] .* uy[:, i] .+ A[i, 1, 3] .* uz[:, i]
        by = A[i, 2, 1] .* ux[:, i] .+ A[i, 2, 2] .* uy[:, i] .+ A[i, 2, 3] .* uz[:, i]
        bz = A[i, 3, 1] .* ux[:, i] .+ A[i, 3, 2] .* uy[:, i] .+ A[i, 3, 3] .* uz[:, i]
        w = vector_length[i]
        Ix .+= w .* bx
        Iy .+= w .* by
        Iz .+= w .* bz
    end

    @assert size(Ix) == (N,) "Ix must be a N-element vector but got $(size(Ix))"
    @assert size(Iy) == (N,) "Iy must be a N-element vector but got $(size(Iy))"
    @assert size(Iz) == (N,) "Iz must be a N-element vector but got $(size(Iz))"
    return Ix, Iy, Iz
end

export system_hamiltonian,
    zeeman_hamiltonian,
    exchange_hamiltonian,
    dipolar_hamiltonian,
    haberkorn_hamiltonian,
    SchultenWolynes_hamiltonian

end # module
