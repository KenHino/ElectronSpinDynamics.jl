
module Hamiltonian

using ..Utils: clean!
using LinearAlgebra: I, ishermitian
using ..SpinOps: Sx1, Sy1, Sz1, Sx2, Sy2, Sz2, Ps, Pt
using ..SystemModule: System
using ..Constants: γe

function system_hamiltonian(sys::System, B::Float64, θ::Float64 = 0.0, ϕ::Float64 = 0.0)
    Hz = zeeman_hamiltonian(B, θ, ϕ)
    Hj = exchange_hamiltonian(sys)
    Hd = dipolar_hamiltonian(sys)
    Hk = haberkorn_hamiltonian(sys)
    Hsys = Hz + Hj + Hd + Hk
    clean!(Hsys)
    # If Hermitian, change type to Hermitian
    if ishermitian(Hsys)
        Hsys = Hermitian(Hsys)
    end
    return Hsys
end

function zeeman_hamiltonian(B::Float64, θ::Float64, ϕ::Float64)
    @assert B ≥ 0 "B must be non-negative"
    @assert 0 ≤ θ ≤ π "θ must be in [0, π]"
    @assert 0 ≤ ϕ ≤ 2π "ϕ must be in [0, 2π]"

    Bx = B * sin(θ) * cos(ϕ)
    By = B * sin(θ) * sin(ϕ)
    Bz = B * cos(θ)

    Hz = zeros(ComplexF64, 4, 4)
    for (Br, Sr1, Sr2) in zip((Bx, By, Bz), (Sx1, Sy1, Sz1), (Sx2, Sy2, Sz2))
        if Br != 0.0
            Hz .+= Br .* (Sr1 + Sr2)
        end
    end
    return Hz
end

function exchange_hamiltonian(sys::System)
    Hj = zeros(ComplexF64, 4, 4)
    if sys.J != 0.0
        Hj += sys.J .* (2.0 .* (Sx1 * Sx2 + Sy1 * Sy2 + Sz1 * Sz2) - 0.5 .* I(4))
    end
    return -Hj
end

function dipolar_hamiltonian(sys::System)
    Hd = zeros(ComplexF64, 4, 4)
    for (i, Sr1) in enumerate((Sx1, Sy1, Sz1))
        for (j, Sr2) in enumerate((Sx2, Sy2, Sz2))
            if sys.D[i, j] != 0.0
                Hd += sys.D[i, j] .* (Sr1 * Sr2)
            end
        end
    end
    return -Hd
end

function haberkorn_hamiltonian(sys::System)
    Hk = zeros(ComplexF64, 4, 4)
    if sys.kS != 0.0
        Hk += -1.0im * sys.kS * 1e-03 / 2.0 * Ps # Since kS in μs
    end
    if sys.kT != 0.0
        Hk += -1.0im * sys.kT * 1e-03 / 2.0 * Pt # Since kT in μs
    end
    Hk ./= abs(γe)
    return Hk
end

export system_hamiltonian,
    zeeman_hamiltonian, exchange_hamiltonian, dipolar_hamiltonian, haberkorn_hamiltonian

end # module
