import numpy as np
from numpy import kron
from scipy.linalg import expm
from scipy.integrate import solve_ivp

# ---------- Pauli / tensor (Zeeman basis) ----------
σx = np.array([[0,1],[1,0]], dtype=complex)
σy = np.array([[0,-1j],[1j,0]], dtype=complex)
σz = np.array([[1,0],[0,-1]], dtype=complex)
I2 = np.eye(2, dtype=complex)
σ = [σx, σy, σz]

def σ1(i): return kron(σ[i], I2)
def σ2(j): return kron(I2, σ[j])
def σ1σ2(i,j): return kron(σ[i], σ[j])

Ps = 0.5*np.array([[0,0,0,0],
                   [0,1,-1,0],
                   [0,-1,1,0],
                   [0,0,0,0]], dtype=complex)

def ST_from_rho(r):
    S1 = np.array([np.trace(r@σ1(i)).real for i in range(3)])
    S2 = np.array([np.trace(r@σ2(j)).real for j in range(3)])
    T  = np.array([[np.trace(r@σ1σ2(i,j)).real for j in range(3)] for i in range(3)])
    return S1,S2,T

def crossmat(v):
    vx,vy,vz = v
    return np.array([[0,-vz,vy],[vz,0,-vx],[-vy,vx,0]], dtype=float)

def axial(M):
    return np.array([M[1,2]-M[2,1], M[2,0]-M[0,2], M[0,1]-M[1,0]], dtype=float)

# ---------- Our Hamiltonian ----------
def H_two_electrons(w1,w2,C):
    H = np.zeros((4,4), dtype=complex)
    for i in range(3):
        H += 0.5*w1[i]*σ1(i) + 0.5*w2[i]*σ2(i)
    for i in range(3):
        for j in range(3):
            H += 0.25*C[i,j]*σ1σ2(i,j)
    return H

# ---------- SC2 RHS ----------

def sc2_rhs(S1,S2,T,w1,w2,C):
    dS1 = np.cross(w1,S1) + 0.5*axial(C@T.T)
    dS2 = np.cross(w2,S2) + 0.5*axial(C.T@T)
    dT  =  crossmat(w1)@T        \
         - T@crossmat(w2)        \
         - 0.5*crossmat(S1)@C    \
         - 0.5*C@crossmat(S2).T
    return dS1,dS2,dT

# ---------- Heisenberg check at t=0 ----------
rng = np.random.default_rng(0)
# pick any pure state |ψ> and ρ=|ψ><ψ|
psi = rng.normal(size=(4,))+1j*rng.normal(size=(4,)); psi/=np.linalg.norm(psi)
ρ0 = np.outer(psi, psi.conj())

w1 = np.array([0.4,0.0,0.0])
w2 = np.array([0.0,0.0,2.0])
C  = np.array([[ 3.0,0.1,0.0],
               [ 0.1,0.0,0.0],
               [ 0.0,0.0,0.0]])

H = H_two_electrons(w1,w2,C)
S1,S2,T = ST_from_rho(ρ0)

ρdot = -1j*(H@ρ0 - ρ0@H)
# QM derivatives of S and T:
S1dot_qm = np.array([np.trace(ρdot@σ1(i)).real for i in range(3)])
S2dot_qm = np.array([np.trace(ρdot@σ2(j)).real for j in range(3)])
Tdot_qm  = np.array([[np.trace(ρdot@σ1σ2(i,j)).real for j in range(3)] for i in range(3)])

S1dot_sc2,S2dot_sc2,Tdot_sc2 = sc2_rhs(S1,S2,T,w1,w2,C)

print("‖dS1(QM)-dS1(SC2)‖ =", np.linalg.norm(S1dot_qm-S1dot_sc2))
print("‖dS2(QM)-dS2(SC2)‖ =", np.linalg.norm(S2dot_qm-S2dot_sc2))
print("‖dT (QM)-dT (SC2)‖ =", np.linalg.norm(Tdot_qm -Tdot_sc2))

# ---------- Time evolution comparison ----------
def rhodot_vec(t,ρvec):
    ρ = ρvec.reshape(4,4)
    dρ = -1j*(H@ρ - ρ@H)
    return dρ.reshape(16)

def sc2dot(t,u):
    S1 = u[0:3]; S2 = u[3:6]
    T  = u[6:].reshape(3,3,order='F')  # column-major packing
    dS1,dS2,dT = sc2_rhs(S1,S2,T,w1,w2,C)
    return np.concatenate([dS1,dS2,dT.reshape(9,order='F')])

# initial SC2 state from ρ0 (no scaling ambiguities)
u0 = np.concatenate([S1,S2,T.reshape(9,order='F')])

ts = np.linspace(0,2.0,201)
sol_qm  = solve_ivp(rhodot_vec, [ts[0],ts[-1]], ρ0.reshape(16), t_eval=ts, rtol=1e-12, atol=1e-12)
sol_sc2 = solve_ivp(sc2dot,     [ts[0],ts[-1]], u0,              t_eval=ts, rtol=1e-12, atol=1e-12)

Ps_qm  = np.array([np.trace(Ps@sol_qm.y[:,k].reshape(4,4)).real for k in range(ts.size)])
Ps_sc2 = np.array([0.25-0.25*np.trace(sol_sc2.y[6:,k].reshape(3,3,order='F')).real for k in range(ts.size)])

print("max |Ps_QM - Ps_SC2| =", np.max(np.abs(Ps_qm-Ps_sc2)))
