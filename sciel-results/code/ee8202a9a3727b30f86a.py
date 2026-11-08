"""Robust control pulses for a three-level transmon."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import expm

_DEADLINE = 310.0
_T0 = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _T0)

N_STEPS = 80
DT = 0.5
T_TOTAL = N_STEPS * DT
ANH = -2.0 * np.pi * 0.25
AMP_MAX = 0.4
SLEW_MAX = 0.12
T1 = 25000.0
TPHI = 40000.0

# 3-level ladder operator
_A = np.zeros((3,3), dtype=complex)
_A[0,1] = 1.0
_A[1,2] = np.sqrt(2.0)
_ADAG = _A.conj().T
_N = _ADAG @ _A  # diag(0,1,2)

_L1 = _A / np.sqrt(T1)
_L2 = _N * np.sqrt(2.0/TPHI)
_L1H = _L1.conj().T
_L2H = _L2.conj().T

def _hamiltonian(detuning, scale, I, Q):
    H = detuning * _N + 0.5 * ANH * _N @ (_N - np.eye(3))
    H += 0.5 * scale * (I * (_A + _ADAG) + 1j * Q * (_A - _ADAG))
    return H

def _liouvillian_superop(H):
    """Build the Liouvillian superoperator as a 9x9 matrix."""
    # vec(dρ/dt) = L * vec(ρ)
    # L = -i(H⊗I - I⊗H^T) + sum_k (Lk⊗Lk* - 0.5(Lk†Lk⊗I + I⊗(Lk†Lk)^T))
    I3 = np.eye(3, dtype=complex)
    
    # Coherent part
    L_coh = -1j * (np.kron(H, I3) - np.kron(I3, H.T))
    
    # Dissipative part
    L_diss = np.zeros((9,9), dtype=complex)
    for L, LH in [(_L1, _L1H), (_L2, _L2H)]:
        LLH = LH @ L
        L_diss += np.kron(L, L.conj()) - 0.5 * np.kron(LLH, I3) - 0.5 * np.kron(I3, LLH.T)
    
    return L_coh + L_diss

def _simulate_pulse(pulse, detuning=0.0, scale=1.0):
    I = pulse[:, 0]
    Q = pulse[:, 1]
    
    # Initial states as kets
    kets = [
        np.array([1,0,0], dtype=complex),
        np.array([0,1,0], dtype=complex),
        np.array([1,1,0], dtype=complex)/np.sqrt(2),
        np.array([1,1j,0], dtype=complex)/np.sqrt(2),
        np.array([1,-1j,0], dtype=complex)/np.sqrt(2),
    ]
    
    X = np.array([[0,1,0],[1,0,0],[0,0,0]], dtype=complex)
    
    # Precompute Liouvillian exponentials
    exps = []
    for i in range(N_STEPS):
        H = _hamiltonian(detuning, scale, I[i], Q[i])
        L = _liouvillian_superop(H)
        exps.append(expm(L * DT))
    
    fidelities = []
    for psi in kets:
        rho = np.outer(psi, psi.conj())
        rho_vec = rho.flatten()
        for E in exps:
            rho_vec = E @ rho_vec
        rho_final = rho_vec.reshape(3,3)
        
        # Fidelity with target X|psi>
        psi_target = X @ psi
        fid = np.real(np.vdot(psi_target, rho_final @ psi_target))
        fid = max(0.0, min(1.0, fid))
        fidelities.append(fid)
    
    return np.mean(fidelities)

def _score_pulse(pulse):
    det_grid = [-2*np.pi*0.002, 0.0, 2*np.pi*0.002]
    scale_grid = [0.97, 1.0, 1.03]
    
    infs = []
    for d in det_grid:
        for s in scale_grid:
            fid = _simulate_pulse(pulse, d, s)
            infs.append(1.0 - fid)
    
    infs = np.array(infs)
    return np.mean(infs) + np.max(infs)

def _check_constraints(pulse):
    I = pulse[:, 0]
    Q = pulse[:, 1]
    amps = np.sqrt(I**2 + Q**2)
    if np.max(amps) > AMP_MAX + 1e-9:
        return False
    max_slew = SLEW_MAX * DT
    if np.sqrt(I[0]**2 + Q[0]**2) > max_slew + 1e-9:
        return False
    for i in range(N_STEPS - 1):
        if np.sqrt((I[i+1]-I[i])**2 + (Q[i+1]-Q[i])**2) > max_slew + 1e-9:
            return False
    if np.sqrt(I[-1]**2 + Q[-1]**2) > max_slew + 1e-9:
        return False
    return True

def _project(pulse):
    I = pulse[:, 0].copy()
    Q = pulse[:, 1].copy()
    max_slew = SLEW_MAX * DT
    
    for _ in range(30):
        for i in range(N_STEPS):
            amp = np.sqrt(I[i]**2 + Q[i]**2)
            if amp > AMP_MAX:
                s = AMP_MAX / amp
                I[i] *= s; Q[i] *= s
        
        amp0 = np.sqrt(I[0]**2 + Q[0]**2)
        if amp0 > max_slew:
            s = max_slew / amp0
            I[0] *= s; Q[0] *= s
        
        for i in range(N_STEPS-1):
            dI = I[i+1]-I[i]; dQ = Q[i+1]-Q[i]
            dist = np.sqrt(dI**2+dQ**2)
            if dist > max_slew:
                f = max_slew/dist
                I[i+1] = I[i]+dI*f; Q[i+1] = Q[i]+dQ*f
        
        ampn = np.sqrt(I[-1]**2 + Q[-1]**2)
        if ampn > max_slew:
            s = max_slew/ampn
            I[-1]*=s; Q[-1]*=s
    
    return np.column_stack([I, Q])

def _make_seed():
    t = np.arange(N_STEPS)*DT
    tc = T_TOTAL/2
    sigma = T_TOTAL/6
    g = np.exp(-(t-tc)**2/(2*sigma**2))
    w = np.ones(N_STEPS)
    edge = 4.0
    for i in range(N_STEPS):
        if t[i] < edge: w[i] = t[i]/edge
        elif t[i] > T_TOTAL-edge: w[i] = (T_TOTAL-t[i])/edge
    g *= w
    area = np.trapz(g, dx=DT)
    if area > 1e-10: g *= np.pi/area
    beta = 0.1
    dg = np.gradient(g, DT)
    pulse = np.column_stack([g, beta*dg])
    return _project(pulse)

def _params_to_pulse(p):
    A0, sigma, tc, beta = p
    t = np.arange(N_STEPS)*DT
    g = np.exp(-(t-tc)**2/(2*sigma**2))
    w = np.ones(N_STEPS)
    edge = 3.0
    for i in range(N_STEPS):
        if t[i] < edge: w[i] = t[i]/edge
        elif t[i] > T_TOTAL-edge: w[i] = (T_TOTAL-t[i])/edge
    g *= w
    I = A0*g
    Q = A0*beta*np.gradient(g, DT)
    return _project(np.column_stack([I, Q]))

def _obj(p):
    if _remaining() < 3.0: return 1e10
    pulse = _params_to_pulse(p)
    if not _check_constraints(pulse): return 1e10
    return _score_pulse(pulse)

def pulse(target_gate, hardware_model):
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi")
    
    seed = _make_seed()
    best_pulse = seed.copy()
    best_score = _score_pulse(seed)
    
    sigma0 = T_TOTAL/6
    A00 = np.pi/(sigma0*np.sqrt(2*np.pi))
    x0 = np.array([A00, sigma0, T_TOTAL/2, 0.1])
    bounds = [(0.01, AMP_MAX), (2.0, 15.0), (10.0, 30.0), (-0.5, 0.5)]
    
    try:
        r = minimize(_obj, x0, method='L-BFGS-B', bounds=bounds,
                    options={'maxiter': 150, 'ftol': 1e-10})
        if r.fun < best_score:
            best_pulse = _params_to_pulse(r.x)
            best_score = r.fun
    except: pass
    
    if _remaining() > 20:
        try:
            r2 = minimize(_obj, x0, method='Nelder-Mead',
                        options={'maxiter': 300, 'xatol': 1e-6})
            if r2.fun < best_score:
                best_pulse = _params_to_pulse(r2.x)
                best_score = r2.fun
        except: pass
    
    if _remaining() > 40:
        np.random.seed(42)
        for _ in range(4):
            if _remaining() < 10: break
            xr = np.random.uniform([0.1,3,15,-0.3],[AMP_MAX,12,25,0.3])
            try:
                r = minimize(_obj, xr, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 80})
                if r.fun < best_score:
                    best_pulse = _params_to_pulse(r.x)
                    best_score = r.fun
            except: pass
    
    return _project(best_pulse)

# EVOLVE-BLOCK-END
