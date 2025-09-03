using LinearAlgebra
using DifferentialEquations:ODEProblem, solve, EnsembleProblem, EnsembleSerial
using StaticArrays
using Plots
using ElectronSpinDynamics:sample_from_sphere, sphere_to_cartesian, dipolar_hamiltonian, liouvillian, vectorise, zeeman_hamiltonian
using Statistics
using Distributions

A1 = [
    SMatrix{3, 3}([1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]) * 1.0, 
    SMatrix{3, 3}([1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]) * 1.0
]
A2 = [
    SMatrix{3, 3}([1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]) * 1.0,
    SMatrix{3, 3}([1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]) * 1.0
]
C = SMatrix{3, 3}([1.0 1.0 0.1; 1.0 0.0 0.0; 0.1 0.0 0.0]) * 4.0
B = SVector{3}([0.0; 0.0; 1.0]) * 2.0
ω1 = SVector{3}([0.4; 0.0; 0.0]) * 2.0 # asymmetric
ω2 = B
ω1n = [B .* 1e-03, B .* 1e-03]
ω2n = [B .* 1e-03, B .* 1e-03]
kS = 1.0
kT = 1.0
k̄ = (kS+3*kT) / 4.0
Δk = (kS-kT) / 4.0
@assert Δk ≈ 0.0
@assert length(A1) == length(ω1n)
@assert length(A2) == length(ω2n)
Nspin = 2 + length(A1) + length(A2)

θ, ϕ = sample_from_sphere((Nspin,1); seed=123)
ux, uy, uz = sphere_to_cartesian(θ, ϕ)
S1 = [ux[1]; uy[1]; uz[1]] * √3/2
S2 = [ux[2]; uy[2]; uz[2]] * √3/2
I1 = reduce(hcat, [[ux[3]; uy[3]; uz[3]] * √3/2, [ux[3]; uy[3]; uz[3]] * √3/2])
I2 = reduce(hcat, [[ux[4]; uy[4]; uz[4]] * √3/2, [ux[3]; uy[3]; uz[3]] * √3/2])
@assert length(A1) == size(I1)[2]
@assert length(A2) == size(I2)[2]
_u0 = reduce(hcat, [S1, S2, I1, I2])
u0 = reshape(_u0, 3*Nspin)
#u0 = reshape(u0, (3, Nspin))
@show u0[1:3]
@show _u0

function prob_func1(prob, i, repeat)
    θ, ϕ = sample_from_sphere((Nspin,1))
    ux, uy, uz = sphere_to_cartesian(θ, ϕ)
    S1 = [ux[1]; uy[1]; uz[1]] * √3/2
    S2 = [ux[2]; uy[2]; uz[2]] * √3/2
    _I1 = []
    _I2 = []
    for k in 3:length(A1)+2
        push!(_I1, [ux[k]; uy[k]; uz[k]] * √3/2) 
    end
    for k in length(A1)+3:length(A1)+length(A2)+2
        push!(_I2, [ux[k]; uy[k]; uz[k]] * √3/2) 
    end
    I1 = reduce(hcat, _I1)
    I2 = reduce(hcat, _I2)
    newu0 = reduce(hcat, (S1, S2, I1, I2))
    newu0 = reshape(newu0, 3*Nspin)
    prob.u0 .= newu0
    return prob# return a fresh problem
end

function SC1!(du, u, p, t)
    """one dimensional u"""
    S1 = @view u[1:3]
    S2 = @view u[4:6]
    N1 = length(A1)
    N2 = length(A2)
    bgn = 6 
    du[1:3] = cross(ω1 + C*S2 + sum(A1[k]*(u[bgn+3*(k-1)+1:bgn+3*k]) for k in 1:N1), S1)
    bgn = 6 + 3*N1
    du[4:6] = cross(ω2 + C*S1 + sum(A2[k]*(u[bgn+3*(k-1)+1:bgn+3*k]) for k in 1:N2), S2)
    bgn = 6
    for k in 1:N1
        Ik = @view u[bgn+3*(k-1)+1:bgn+3*k]
        du[bgn+3*(k-1)+1:bgn+3*k] = cross(ω1n[k] + A1[k]' * S1, Ik)
    end
    bgn = 6 + 3*N1
    for k in 1:N2
        Ik = @view u[bgn+3*(k-1)+1:bgn+3*k]
        du[bgn+3*(k-1)+1:bgn+3*k] = cross(ω2n[k] +A2[k]' * S2, Ik)
    end
end

dt = 0.05
saveat = 0.0:dt:2.0
tspan = (0.0, last(saveat))
prob = ODEProblem(SC1!, u0, tspan)
sol = solve(prob, dt=dt, saveat=saveat)

function output_func1(sol, i)
    t = sol.t
    N = length(t)
    Ps = ones(Float64, N) * 1/4
    Pt0 = ones(Float64, N) * 1/4
    Ptp = ones(Float64, N) * 1/4
    Ptm = ones(Float64, N) * 1/4

    S0 = sol.u[1]
    S01x, S01y, S01z, S02x, S02y, S02z = S0[1:6]
    
    for k in 1:N
        St = sol.u[k]
        St1x, St1y, St1z, St2x, St2y, St2z = St[1:6]

        Ps[k]  += 4.0*(St1x*St2x+St1y*St2y+St1z*St2z)*(S01x*S02x+S01y*S02y+S01z*S02z)
        Pt0[k] -= 4.0*(St1x*St2x+St1y*St2y-St1z*St2z)*(S01x*S02x+S01y*S02y+S01z*S02z)
        Ptp[k] -= (4.0*St1z*St2z + 2.0*St1z + 2.0*St2z)* (S01x*S02x + S01y*S02y + S01z*S02z) 
        Ptm[k] -= (4.0*St1z*St2z - 2.0*St1z - 2.0*St2z)* (S01x*S02x + S01y*S02y + S01z*S02z) 
    end

     ((t, Ptp .* exp.(-kT*t), Pt0 .* exp.(-kT*t), Ps .* exp.(-kS*t), Ptm .* exp.(-kT*t)), false)
end
eprob1 = EnsembleProblem(prob, output_func=output_func1, prob_func=prob_func1)

data1 = solve(eprob1, dt=dt, saveat=saveat, trajectories=10_000)

# 95% CI for the mean (Normal)
z = 1.959963984540054  # quantile(Normal(), 0.975)
function show(data)
    N  = length(data)
    t  = data[1][1]                 # all replicates share the same x
    # Helper: stack replicates column-wise => size = length(x) × N
    colmat(i) = hcat((d[i] for d in data)...)

    Ptp = colmat(2)
    Pt0 = colmat(3)
    Ps  = colmat(4)
    Ptm = colmat(5)
    Tr = Ptp + Pt0 + Ps + Ptm

    # helper: empirical 95% band + mean (no parametric assumption)
    # mean & sample std (Bessel) per x, then SE = s/sqrt(N)
    mean_std_se(M) = begin
        μ = vec(mean(M, dims=2))
        s = vec(std(M, dims=2))          # corrected=true by default -> sample std
        se = s ./ sqrt(N)
        μ, se
    end

    μ_tp, se_tp = mean_std_se(Ptp)
    μ_t0, se_t0 = mean_std_se(Pt0)
    μ_s,  se_s  = mean_std_se(Ps)
    μ_tm, se_tm = mean_std_se(Ptm)
    μ_tr, se_tr = mean_std_se(Tr)

    plt = plot(t, μ_tp, ribbon=z .* se_tp, label="T+ (95% CI of mean)", lw=3)
    plot!(plt, t, μ_t0, ribbon=z .* se_t0, label="T0 (95% CI of mean)", lw=3)
    plot!(plt, t, μ_s,  ribbon=z .* se_s,  label="S (95% CI of mean)", lw=3)
    plot!(plt, t, μ_tm, ribbon=z .* se_tm, label="T- (95% CI of mean)", lw=3)
    plot!(plt, t, μ_tr, ribbon=z .* se_tr, label="Tr (95% CI of mean)", lw=3)
    plt, μ_s, se_s
end

plt, μ_s_sc1, se_s_sc1 = show(data1)
display(plt)

θ, ϕ = sample_from_sphere((Nspin,1); seed=123)
ux, uy, uz = sphere_to_cartesian(θ, ϕ)
S1 = [ux[1]; uy[1]; uz[1]] * √3/2
S2 = [ux[2]; uy[2]; uz[2]] * √3/2
I1 = reduce(hcat, [[ux[3]; uy[3]; uz[3]] * √3/2, [ux[3]; uy[3]; uz[3]] * √3/2])
I2 = reduce(hcat, [[ux[4]; uy[4]; uz[4]] * √3/2, [ux[3]; uy[3]; uz[3]] * √3/2])
@assert length(A1) == size(I1)[2]
@assert length(A2) == size(I2)[2]
_u0 = reduce(hcat, [S1, S2, I1, I2])
@show _u0
@show transpose(S2)
T12 = S1 * transpose(S2)
@assert reshape(T12, 9)[1:2] ≈ 3/4 * [ux[1]*ux[2], uy[1]*ux[2]]
@show T12
_u0 = reduce(hcat, [_u0, T12])
@show _u0
u0 = reshape(_u0, 3*Nspin+9)

#u0 = reshape(u0, (3, Nspin))
#@show u0[1:3]
#@show _u0
crossmat(v::SVector{3,Float64}) = @SMatrix [ 0.0 -v[3]  v[2];
                                             v[3]  0.0 -v[1];
                                            -v[2]  v[1]  0.0 ]
axial(M::SMatrix{3,3,Float64}) = @SVector [ M[2,3]-M[3,2],
                                            M[3,1]-M[1,3],
                                            M[1,2]-M[2,1] ]
Cxx, Cyx, Czx = C[:, 1]
Cxy, Cyy, Czy = C[:, 2]
Cxz, Cyz, Czz = C[:, 3]

function SC2!(du, u, p, t)
    """one dimensional u"""
    S1 = @view u[1:3]
    S2 = @view u[4:6]
    S1x, S1y, S1z = S1
    S2x, S2y, S2z = S2
    N1 = length(A1)
    N2 = length(A2)
    #T12 = u[3*(N1+N2+2)+1:3*(N1+N2+2)+9]
    Txx, Tyx, Tzx, Txy, Tyy, Tzy, Txz, Tyz, Tzz = @view u[3*(N1+N2+2)+1:3*(N1+N2+2)+9]

    # dS1
    bgn = 6
    env1 = ω1 #+ sum(A1[k]'*u[bgn+3*(k-1)+1:bgn+3*k] for k in 1:N1)
    E1x, E1y, E1z = env1
    
    # ----- S1 |C  = axial(C*T')
    dS1x = (Cyx*Tzx + Cyy*Tzy + Cyz*Tzz) - (Czx*Tyx + Czy*Tyy + Czz*Tyz)
    dS1y = (Czx*Txx + Czy*Txy + Czz*Txz) - (Cxx*Tzx + Cxy*Tzy + Cxz*Tzz)
    dS1z = (Cxx*Tyx + Cxy*Tyy + Cxz*Tyz) - (Cyx*Txx + Cyy*Txy + Cyz*Txz)

    # ----- S2 |C  = - axial(C' * T)
    dS2x = (Cxy*Txz + Cyy*Tyz + Czy*Tzz) - (Cxz*Txy + Cyz*Tyy + Czz*Tzy)
    dS2y = (Cxz*Txx + Cyz*Tyx + Czz*Tzx) - (Cxx*Txz + Cyx*Tyz + Czx*Tzz)
    dS2z = (Cxx*Txy + Cyx*Tyy + Czx*Tzy) - (Cxy*Txx + Cyy*Tyx + Czy*Tzx)
    du[1] = E1y*S1z - E1z*S1y + 0.50*dS1x
    du[2] = E1z*S1x - E1x*S1z + 0.50*dS1y
    du[3] = E1x*S1y - E1y*S1x + 0.50*dS1z

    # keep these: S2 uses ε_{αβγ} C_{βδ} T_{γδ} == axial(C*Tᵀ)
    bgn = 6 + 3*N1
    env2 = ω2 #+ sum(A2[k]'*u[bgn+3*(k-1)+1:bgn+3*k] for k in 1:N2)
    E2x, E2y, E2z = env2
    du[4] = E2y*S2z - E2z*S2y + 0.50*dS2x
    du[5] = E2z*S2x - E2x*S2z + 0.50*dS2y
    du[6] = E2x*S2y - E2y*S2x + 0.50*dS2z


    # dI1
    #bgn = 6
    #for k in 1:N1
    #    du[bgn+3*(k-1)+1:bgn+3*k] .= cross(ω1n[k] + A1[k]' * S1, u[bgn+3*(k-1)+1:bgn+3*k])
    #end
    # dI2
    #bgn = 6 + 3*N1
    #for k in 1:N2
    #    du[bgn+3*(k-1)+1:bgn+3*k] .= cross(ω2n[k] +A2[k]' * S2, u[bgn+3*(k-1)+1:bgn+3*k])
    #end

    bgn = 6 + 3*N1 + 3*N2
    
    # dTxx
    du[bgn+1] = (
        E1y*Tzx - E1z*Tyx  + (E2y*Txz - E2z*Txy) 
        - 0.5*(S1y*Czx - S1z*Cyx) - 0.5*(S2y*Cxz - S2z*Cxy)
    )

    # dTyx
    du[bgn+2] = (
        E1z*Txx - E1x*Tzx + (E2y*Tyz - E2z*Tyy)
        - 0.5*(S1z*Cxx - S1x*Czx) - 0.5*(S2y*Cyz - S2z*Cyy)
    )
    # dTzx
    du[bgn+3] = (
        E1x*Tyx - E1y*Txx + (E2y*Tzz - E2z*Tzy)
        - 0.5*(S1x*Cyx - S1y*Cxx) - 0.5*(S2y*Czz - S2z*Czy)
    )
    # dTxy
    du[bgn+4] =  (
        E1y*Tzy - E1z*Tyy + (E2z*Txx - E2x*Txz)
        - 0.5*(S1y*Czy - S1z*Cyy) - 0.5*(S2z*Cxx - S2x*Cxz)
    )
    # dTyy
    du[bgn+5] = (
        E1z*Txy - E1x*Tzy + (E2z*Tyx - E2x*Tyz)
        - 0.5*(S1z*Cxy - S1x*Czy) - 0.5*(S2z*Cyx - S2x*Cyz)
    )
    # dTzy
    du[bgn+6] = (
        E1x*Tyy - E1y*Txy + (E2z*Tzx - E2x*Tzz)
        - 0.5*(S1x*Cyy - S1y*Cxy) - 0.5*(S2z*Czx - S2x*Czz)
    )
    # dTxz
    du[bgn+7] = (
        E1y*Tzz - E1z*Tyz + (E2x*Txy - E2y*Txx)
        - 0.5*(S1y*Czz - S1z*Cyz) - 0.5*(S2x*Cxy - S2y*Cxx)
    )
    # dTyz
    du[bgn+8] = (
        E1z*Txz - E1x*Tzz + (E2x*Tyy - E2y*Tyx)
        - 0.5*(S1z*Cxz - S1x*Czz) - 0.5*(S2x*Cyy - S2y*Cyx)
    )
    # dTzz
    du[bgn+9] = (
        E1x*Tyz - E1y*Txz + (E2x*Tzy - E2y*Tzx)
        - 0.5*(S1x*Cyz - S1y*Cxz) - 0.5*(S2x*Czy - S2y*Czx)
    )
end
# Cross matrix and axial (vee) map
crossmat(v::SVector{3,Float64}) = @SMatrix [ 0.0 -v[3]  v[2];
                                             v[3]  0.0 -v[1];
                                            -v[2]  v[1]  0.0 ]
axial(M::SMatrix{3,3,Float64}) = @SVector [ M[2,3]-M[3,2],
                                            M[3,1]-M[1,3],
                                            M[1,2]-M[2,1] ]
function SC2!(du,u,p,t)
    S1 = SVector{3,Float64}(u[1:3])
    S2 = SVector{3,Float64}(u[4:6])
    # unpack T (column-major)
    N1 = length(A1)
    N2 = length(A2)
    
    bgn = 6
    ω1′ = ω1 + sum(A1[k]'*(u[bgn+3*(k-1)+1:bgn+3*k]) for k in 1:N1)
    for k in 1:N1
        Ik = @view u[bgn+3*(k-1)+1:bgn+3*k]
        du[bgn+3*(k-1)+1:bgn+3*k] .= cross(ω1n[k] + A1[k]' * S1, Ik)
    end
    
    bgn = 6 + 3*N1
    ω2′ = ω2 + sum(A2[k]'*(u[bgn+3*(k-1)+1:bgn+3*k]) for k in 1:N2)
    for k in 1:N2
        Ik = @view u[bgn+3*(k-1)+1:bgn+3*k]
        du[bgn+3*(k-1)+1:bgn+3*k] .= cross(ω2n[k] +A2[k]' * S2, Ik)
    end
    
    bgn = 6 + 3*N1 + 3*N2
    T  = @SMatrix [u[bgn+1] u[bgn+4] u[bgn+7];
                   u[bgn+2] u[bgn+5] u[bgn+8];
                   u[bgn+3] u[bgn+6] u[bgn+9]]
    # Note: 
    # If Heisenberg EOM is based on pauli matirces σ rather than S 
    # (final results would be identical), 
    # axial(C*T') -> axial(C*T') / 2
    # axial(C' * T) -> axial(C' * T) / 2
    # crossmat(S1)*C/4 -> crossmat(S1)*C / 2
    # C*transpose(crossmat(S2))/4 -> C*transpose(crossmat(S2)) / 2
    dS1 = cross(ω1′, S1) + axial(C*T')
    dS2 = cross(ω2′, S2) + axial(C' * T)
    dT  =  (crossmat(ω1′)*T -
           T*crossmat(ω2′) -
           crossmat(S1)*C/4 -
           C*transpose(crossmat(S2))/4)
    du[1:3]  .= dS1
    du[4:6]  .= dS2
    du[bgn+1:bgn+9] .= vec(dT)
    return nothing
end
prob2 = ODEProblem(SC2!, u0, tspan)

function SC3!(du,u,p,t)
    S1 = SVector{3,Float64}(u[1:3])
    S2 = SVector{3,Float64}(u[4:6])
    p = u[end] # norm
    # unpack T (column-major)
    N1 = length(A1)
    N2 = length(A2)
    
    bgn = 6
    ω1′ = ω1 + sum(A1[k]'*u[bgn+3*(k-1)+1:bgn+3*k] for k in 1:N1)
    factor1 = √3/2/norm(S1)
    for k in 1:N1
        Ik = @view u[bgn+3*(k-1)+1:bgn+3*k]
        du[bgn+3*(k-1)+1:bgn+3*k] .= (
            cross(ω1n[k] + A1[k]' * S1, Ik) * factor1
        )
    end
    
    bgn = 6 + 3*N1
    ω2′ = ω2 + sum(A2[k]'*u[bgn+3*(k-1)+1:bgn+3*k] for k in 1:N2)
    factor2 = √3/2/norm(S2)
    for k in 1:N2
        Ik = @view u[bgn+3*(k-1)+1:bgn+3*k]
        du[bgn+3*(k-1)+1:bgn+3*k] .= (
            cross(ω2n[k] +A2[k]' * S2, Ik) * factor2
        )
    end
    
    bgn = 6 + 3*N1 + 3*N2
    T  = @SMatrix [u[bgn+1] u[bgn+4] u[bgn+7];
                   u[bgn+2] u[bgn+5] u[bgn+8];
                   u[bgn+3] u[bgn+6] u[bgn+9]]
    trT = tr(T)
    dS1 = cross(ω1′, S1) + axial(C*T') - k̄*S1 + Δk*S2
    dS2 = cross(ω2′, S2) + axial(C' * T) - k̄*S2 + Δk*S1
    dT  =  (
        crossmat(ω1′)*T -
        T*crossmat(ω2′) -
        crossmat(S1)*C/4 -
        C*transpose(crossmat(S2))/4 -
        k̄*T +
        Δk*T' +
        Δk*(p/4-trT)*I(3)
    )
    du[1:3]  .= dS1
    du[4:6]  .= dS2
    du[bgn+1:bgn+9] .= vec(dT)
    du[end] = -k̄*p + 4*Δk*trT
    return nothing
end

θ, ϕ = sample_from_sphere((Nspin,1); seed=123)
ux, uy, uz = sphere_to_cartesian(θ, ϕ)
S1 = [ux[1]; uy[1]; uz[1]] * √3/2
S2 = [ux[2]; uy[2]; uz[2]] * √3/2
I1 = reduce(hcat, [[ux[3]; uy[3]; uz[3]] * √3/2, [ux[3]; uy[3]; uz[3]] * √3/2])
I2 = reduce(hcat, [[ux[4]; uy[4]; uz[4]] * √3/2, [ux[3]; uy[3]; uz[3]] * √3/2])
@assert length(A1) == size(I1)[2]
@assert length(A2) == size(I2)[2]
_u0 = reduce(hcat, [S1, S2, I1, I2])
@show _u0
@show transpose(S2)
T12 = S1 * transpose(S2)
@assert reshape(T12, 9)[1:2] ≈ 3/4 * [ux[1]*ux[2], uy[1]*ux[2]]
@show T12
_u0 = reduce(hcat, [_u0, T12])
@show _u0
u0 = reshape(_u0, 3*Nspin+9)
push!(u0, 1.0)
prob3 = ODEProblem(SC3!, u0, tspan)

function prob_func2(prob, i, repeat)
    θ, ϕ = sample_from_sphere((Nspin,1))
    ux, uy, uz = sphere_to_cartesian(θ, ϕ)
    S1 = [ux[1]; uy[1]; uz[1]] * √3/2
    S2 = [ux[2]; uy[2]; uz[2]] * √3/2
    _I1 = []
    _I2 = []
    for k in 3:length(A1)+2
        push!(_I1, [ux[k]; uy[k]; uz[k]] * √3/2) 
    end
    for k in length(A1)+3:length(A1)+length(A2)+2
        push!(_I2, [ux[k]; uy[k]; uz[k]] * √3/2) 
    end
    I1 = reduce(hcat, _I1)
    I2 = reduce(hcat, _I2)
    _u0 = reduce(hcat, (S1, S2, I1, I2))
    T12 = S1 * transpose(S2)
    _u0 = reduce(hcat, [_u0, T12])
    newu0 = reshape(_u0, 3*Nspin+9)
    prob.u0 .= newu0
    return prob# return a fresh problem
end

function prob_func3(prob, i, repeat)
    θ, ϕ = sample_from_sphere((Nspin,1))
    ux, uy, uz = sphere_to_cartesian(θ, ϕ)
    S1 = [ux[1]; uy[1]; uz[1]] * √3/2
    S2 = [ux[2]; uy[2]; uz[2]] * √3/2
    _I1 = []
    _I2 = []
    for k in 3:length(A1)+2
        push!(_I1, [ux[k]; uy[k]; uz[k]] * √3/2) 
    end
    for k in length(A1)+3:length(A1)+length(A2)+2
        push!(_I2, [ux[k]; uy[k]; uz[k]] * √3/2) 
    end
    I1 = reduce(hcat, _I1)
    I2 = reduce(hcat, _I2)
    _u0 = reduce(hcat, (S1, S2, I1, I2))
    T12 = S1 * transpose(S2)
    _u0 = reduce(hcat, [_u0, T12])
    newu0 = reshape(_u0, 3*Nspin+9)
    push!(newu0, 1.0) # trace term
    prob.u0 .= newu0
    return prob# return a fresh problem
end

function output_func2(sol, i)
    t = sol.t
    N = length(t)
    Ps = Array{Float64, 1}(undef, N)
    Pt0 = Array{Float64, 1}(undef, N)
    Ptp = Array{Float64, 1}(undef, N)
    Ptm = Array{Float64, 1}(undef, N)

    u = sol.u[1]
    S01x, S01y, S01z, S02x, S02y, S02z = u[1:6]
    N1 = length(A1)
    N2 = length(A2)
    T12 = u[3*(N1+N2+2)+1:3*(N1+N2+2)+9]
    Txx0, Tyx0, Tzx0, Txy0, Tyy0, Tzy0, Txz0, Tyz0, Tzz0 = T12
    tr0 = Txx0 + Tyy0 + Tzz0
    for k in 1:N
        u = sol.u[k]
        St1x, St1y, St1z, St2x, St2y, St2z = u[1:6]
        T12 = u[3*(N1+N2+2)+1:3*(N1+N2+2)+9]
        Txx, Tyx, Tzx, Txy, Tyy, Tzy, Txz, Tyz, Tzz = T12

        trk = Txx + Tyy + Tzz
        Ps[k] = (1/4 - trk) * (1/4 - tr0) * 4
        Pt0[k] = (1/4 + Txx + Tyy - Tzz) * (1/4 - tr0) * 4
        Ptp[k] = (1/4 + 1/2*St1z + 1/2*St2z + Tzz) * (1/4 - tr0) * 4
        Ptm[k] = (1/4 - 1/2*St1z - 1/2*St2z + Tzz) * (1/4 - tr0) * 4
    end
    ((t, Ptp .* exp.(-kT*t), Pt0 .* exp.(-kT*t), Ps .* exp.(-kS*t), Ptm .* exp.(-kT*t)), false)
end
eprob2 = EnsembleProblem(prob2, output_func=output_func2, prob_func=prob_func2)

function output_func3(sol, i)
    t = sol.t
    N = length(t)
    Ps = Array{Float64, 1}(undef, N)
    Pt0 = Array{Float64, 1}(undef, N)
    Ptp = Array{Float64, 1}(undef, N)
    Ptm = Array{Float64, 1}(undef, N)

    u = sol.u[1]
    S01x, S01y, S01z, S02x, S02y, S02z = u[1:6]
    N1 = length(A1)
    N2 = length(A2)
    T12 = u[3*(N1+N2+2)+1:3*(N1+N2+2)+9]
    Txx0, Tyx0, Tzx0, Txy0, Tyy0, Tzy0, Txz0, Tyz0, Tzz0 = T12
    p0 = u[end]
    tr0 = Txx0 + Tyy0 + Tzz0
    for k in 1:N
        u = sol.u[k]
        p = u[end]
        St1x, St1y, St1z, St2x, St2y, St2z = u[1:6]
        T12 = u[3*(N1+N2+2)+1:3*(N1+N2+2)+9]
        Txx, Tyx, Tzx, Txy, Tyy, Tzy, Txz, Tyz, Tzz = T12

        trk = Txx + Tyy + Tzz
        Ps[k] = (p/4 - trk) * (p0/4 - tr0) * 4
        Pt0[k] = (p/4 + Txx + Tyy - Tzz) * (p0/4 - tr0) * 4
        Ptp[k] = (p/4 + 1/2*St1z + 1/2*St2z + Tzz) * (p0/4 - tr0) * 4
        Ptm[k] = (p/4 - 1/2*St1z - 1/2*St2z + Tzz) * (p0/4 - tr0) * 4
    end
    ((t, Ptp, Pt0, Ps, Ptm), false)
end
eprob3 = EnsembleProblem(prob3, output_func=output_func3, prob_func=prob_func3)

@time data2 = solve(eprob2, dt=dt, saveat=saveat, trajectories=10_000)

@time data3 = solve(eprob3, dt=dt, saveat=saveat, trajectories=10_000)

plt, μ_s_sc2, se_s_sc2 = show(data2)
display(plt)
plt, μ_s_sc3, se_s_sc3 = show(data3)
display(plt)

ham = -dipolar_hamiltonian(C) + zeeman_hamiltonian(ω1, ω2)
@show ishermitian(ham)

L = liouvillian(ham)

U = exp(L*dt)
ρ = vectorise(diagm(0=>[0.0, 0.0, 1.0, 0.0]))
t = sol.t
s = []
tp = []
t0 = []
tm = []
for k in t
    push!(tp, real(ρ[1]))
    push!(t0, real(ρ[6]))
    push!(s, real(ρ[11]))
    push!(tm, real(ρ[16]))
    ρ = U * ρ
end
plt = plot(t, tp, label="T+", lw=3)
plot!(plt, t, t0, label="T0", lw=3)
plot!(plt, t, s,  label="S", lw=3)
plot!(plt, t, tm, label="T-", lw=3)
display(plt)

plt = plot(t, s,  label="QM", lw=3)
plot!(plt, t, μ_s_sc1,  ribbon=z .* se_s_sc1,  label="SC1 (95% CI of mean)", lw=3)
plot!(plt, t, μ_s_sc2,  ribbon=z .* se_s_sc2,  label="SC2 (95% CI of mean)", lw=3)
plot!(plt, t, μ_s_sc3,  ribbon=z .* se_s_sc3,  label="SC3 (95% CI of mean)", lw=3)
display(plt)


