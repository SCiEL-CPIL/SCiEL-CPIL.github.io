"""Robust X_pi control pulse for a three-level transmon."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm

_DEADLINE = 325.0
_START_TIME = time.time()

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def _trapz(y, x):
    """Compatible trapz function."""
    if hasattr(np, 'trapezoid'):
        return np.trapezoid(y, x)
    else:
        return np.trapz(y, x)

# Pre-computed constants for the 3-level transmon
_N = 80
_DT = 0.5
_ANH = -2.0 * np.pi * 0.25
_AMP_BOUND = 0.4
_SLEW_BOUND = 0.12
_T1 = 25000.0
_TPHI = 40000.0

# a = diag(sqrt([1, 2]), 1)
_A = np.diag([1.0, np.sqrt(2.0), 1.0])
_ADAG = _A.conj().T
_N_OP = _ADAG @ _A
_I3 = np.eye(3, dtype=complex)

# Lindblad operators
_L1 = _A / np.sqrt(_T1)
_L2 = _N_OP * np.sqrt(2.0 / _TPHI)

# Pre-compute dissipative part of Liouvillian (constant)
# L_diss = sum_k (L_k \otimes L_k^* - 0.5 (I \otimes L_k^dag L_k + L_k^dag L_k \otimes I))
_L_DISS = np.zeros((9, 9), dtype=complex)
for Lk in [_L1, _L2]:
    Lk_dag = Lk.conj().T
    Lk_Lk_dag = Lk @ Lk_dag
    L_DISS += np.kron(Lk, Lk.conj()) - 0.5 * (np.kron(_I3, Lk_Lk_dag) + np.kron(Lk_Lk_dag, _I3))

# Detuning and scale grids for scoring
_DET_GRID = np.array([-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002])
_SCALE_GRID = np.array([0.97, 1.0, 1.03])

# Initial states for fidelity calculation
_STATES = [
    np.array([1, 0, 0], dtype=complex),  # |0>
    np.array([0, 1, 0], dtype=complex),  # |1>
    (np.array([1, 1, 0], dtype=complex)) / np.sqrt(2),
    (np.array([1, -1, 0], dtype=complex)) / np.sqrt(2),
    (np.array([1, 1j, 0], dtype=complex)) / np.sqrt(2),
    (np.array([1, -1j, 0], dtype=complex)) / np.sqrt(2),
]

# Target states for X gate
_TARGETS = [
    np.array([0, 1, 0], dtype=complex),  # X|0> = |1>
    np.array([1, 0, 0], dtype=complex),  # X|1> = |0>
    (np.array([1, 1, 0], dtype=complex)) / np.sqrt(2),  # X(|0>+|1>)/sqrt(2) = (|1>+|0>)/sqrt(2)
    (np.array([-1, 1, 0], dtype=complex)) / np.sqrt(2), # X(|0>-|1>)/sqrt(2) = (|1>-|0>)/sqrt(2)
    (np.array([1j, 1, 0], dtype=complex)) / np.sqrt(2), # X(|0>+i|1>)/sqrt(2) = (i|0>+|1>)/sqrt(2)
    (np.array([-1j, 1, 0], dtype=complex)) / np.sqrt(2),# X(|0>-i|1>)/sqrt(2) = (-i|0>+|1>)/sqrt(2)
]

# Pre-compute initial density matrix vectors
_INIT_RHO_VECTORS = np.array([np.outer(s, s.conj()).flatten() for s in _STATES])
# Shape: (6, 9)

# Pre-compute target density matrices for fidelity
_TARGET_RHOS = np.array([np.outer(t, t.conj()) for t in _TARGETS])
# Shape: (6, 3, 3)

def _liouvillian_step(dt, I, Q, detuning, scale):
    """Compute the Liouvillian superoperator for one step."""
    # H = detuning * n + 0.5 * anharmonicity * n * (n - 1) + 0.5 * scale * (I (a + a^dag) + i Q (a - a^dag))
    H = (detuning * _N_OP + 0.5 * _ANH * (_N_OP @ (_N_OP - _I3)) + 
         0.5 * scale * (I * (_A + _ADAG) + 1j * Q * (_A - _ADAG)))
    
    # Coherent part
    # L_coh = -1j * (H \otimes I - I \otimes H^T)
    L_coh = -1j * (np.kron(H, _I3) - np.kron(_I3, H.T))
    
    L_total = L_coh + _L_DISS
    
    return expm(L_total * dt)

def _simulate_pulse(I_arr, Q_arr, detuning, scale):
    """Simulate the pulse and return final density matrices for all initial states."""
    N = len(I_arr)
    
    # Stack initial states: (6, 9)
    R = _INIT_RHO_VECTORS.copy()
    
    for i in range(N):
        prop = _liouvillian_step(_DT, I_arr[i], Q_arr[i], detuning, scale)
        # R is (6, 9). prop is (9, 9).
        # We need vec(rho_{k+1}) = prop @ vec(rho_k).
        # R is a matrix where each row is a vector.
        # So R_new = R @ prop.T
        R = R @ prop.T
        
    # Convert back to density matrices
    # R is (6, 9). Reshape each row to (3, 3).
    rho_finals = R.reshape(6, 3, 3)
    
    return rho_finals

def _compute_fidelities(rho_finals):
    """Compute fidelities for all 6 states against targets."""
    # rho_finals: (6, 3, 3)
    # _TARGET_RHOS: (6, 3, 3)
    # Fidelity = Tr(target_rho @ rho_final)
    # Using Einstein summation: sum_{i,j} target[i,j] * rho[j,i]
    # Or simply trace for each pair.
    fids = np.zeros(6)
    for i in range(6):
        fids[i] = np.real(np.trace(_TARGET_RHOS[i] @ rho_finals[i]))
    return fids

def _score_pulse(I_arr, Q_arr):
    """Compute the score for a given pulse."""
    all_fidelities = []
    
    for det in _DET_GRID:
        for scale in _SCALE_GRID:
            rho_finals = _simulate_pulse(I_arr, Q_arr, det, scale)
            fids = _compute_fidelities(rho_finals)
            all_fidelities.extend(fids)
    
    all_fidelities = np.array(all_fidelities)
    mean_fidelity = np.mean(all_fidelities)
    max_fidelity = np.max(all_fidelities)
    
    infidelity_mean = 1.0 - mean_fidelity
    infidelity_max = 1.0 - max_fidelity
    
    score = infidelity_mean + infidelity_max
    return score

def _check_constraints(I_arr, Q_arr):
    """Check if the pulse satisfies amplitude and slew rate constraints."""
    N = len(I_arr)
    
    # Check amplitude
    amps = np.sqrt(I_arr**2 + Q_arr**2)
    if np.any(amps > _AMP_BOUND + 1e-10):
        return False
    
    # Check slew rate
    # From zero to first
    if np.sqrt(I_arr[0]**2 + Q_arr[0]**2) / _DT > _SLEW_BOUND + 1e-10:
        return False
    
    # Between steps
    dI = np.diff(I_arr)
    dQ = np.diff(Q_arr)
    slew_amps = np.sqrt(dI**2 + dQ**2) / _DT
    if np.any(slew_amps > _SLEW_BOUND + 1e-10):
        return False
    
    # From last to zero
    if np.sqrt(I_arr[-1]**2 + Q_arr[-1]**2) / _DT > _SLEW_BOUND + 1e-10:
        return False
        
    return True

def _enforce_constraints(I_arr, Q_arr):
    """Enforce amplitude and slew rate constraints by clipping."""
    I_arr = I_arr.copy()
    Q_arr = Q_arr.copy()
    N = len(I_arr)
    
    for _ in range(20):
        # Enforce amplitude
        amps = np.sqrt(I_arr**2 + Q_arr**2)
        mask = amps > _AMP_BOUND
        if np.any(mask):
            scale = _AMP_BOUND / amps[mask]
            I_arr[mask] *= scale
            Q_arr[mask] *= scale
        
        # Enforce slew rate - from zero to first
        amp0 = np.sqrt(I_arr[0]**2 + Q_arr[0]**2)
        if amp0 / _DT > _SLEW_BOUND:
            scale = _SLEW_BOUND * _DT / amp0
            I_arr[0] *= scale
            Q_arr[0] *= scale
        
        # Enforce slew rate - between steps
        for i in range(N - 1):
            dI = I_arr[i+1] - I_arr[i]
            dQ = Q_arr[i+1] - Q_arr[i]
            amp = np.sqrt(dI**2 + dQ**2)
            if amp / _DT > _SLEW_BOUND:
                scale = _SLEW_BOUND * _DT / amp
                I_arr[i+1] = I_arr[i] + dI * scale
                Q_arr[i+1] = Q_arr[i] + dQ * scale
        
        # Enforce slew rate - from last to zero
        amp_last = np.sqrt(I_arr[-1]**2 + Q_arr[-1]**2)
        if amp_last / _DT > _SLEW_BOUND:
            scale = _SLEW_BOUND * _DT / amp_last
            I_arr[-1] *= scale
            Q_arr[-1] *= scale
        
        # Re-check amplitude after slew clipping
        amps = np.sqrt(I_arr**2 + Q_arr**2)
        mask = amps > _AMP_BOUND
        if np.any(mask):
            scale = _AMP_BOUND / amps[mask]
            I_arr[mask] *= scale
            Q_arr[mask] *= scale

    return I_arr, Q_arr

def _make_seed_pulse(N, dt, sigma_ns, area):
    """Create a DRAG-style Gaussian seed pulse."""
    t = np.arange(N) * dt
    center = N * dt / 2.0
    sigma = sigma_ns
    
    # Gaussian envelope
    env = np.exp(-0.5 * ((t - center) / sigma)**2)
    
    # Offset so edges are zero (they are approx zero, but let's ensure)
    # Actually, just scaling is enough for the seed.
    current_area = _trapz(env, t)
    if current_area > 1e-10:
        env = env * (area / current_area)
    
    I_arr = env
    Q_arr = np.zeros(N)
    
    I_arr, Q_arr = _enforce_constraints(I_arr, Q_arr)
    
    return I_arr, Q_arr

def pulse(target_gate, hardware_model):
    """Return (80, 2) array of I and Q amplitudes (rad/ns) for a 40 ns X-pi gate."""
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi supported")
    
    N = 80
    dt = 0.5
    sigma_ns = 40 / 6.0
    area = np.pi
    
    # 1. Generate seed pulse
    I_seed, Q_seed = _make_seed_pulse(N, dt, sigma_ns, area)
    
    if not _check_constraints(I_seed, Q_seed):
        # Fallback to a very simple valid pulse if seed fails (shouldn't happen)
        I_seed = np.zeros(N)
        Q_seed = np.zeros(N)
        
    # 2. Evaluate seed score
    try:
        best_score = _score_pulse(I_seed, Q_seed)
    except Exception:
        best_score = 1e6
        
    I_best = I_seed.copy()
    Q_best = Q_seed.copy()
    
    # 3. Local optimization loop
    # We will try random perturbations and keep the best valid one.
    # This is robust and avoids complex optimizer failures.
    
    rng = np.random.default_rng(42)
    
    while _time_remaining() > 10.0:
        # Generate a perturbation
        # Option 1: Small Gaussian noise on the whole pulse
        # Option 2: Modify a few random samples
        
        # Let's try a mix. Mostly small noise, occasionally larger structural change.
        if rng.random() < 0.8:
            # Small noise
            noise_I = rng.normal(0, 0.01, N)
            noise_Q = rng.normal(0, 0.01, N)
            I_cand = I_best + noise_I
            Q_cand = Q_best + noise_Q
        else:
            # Larger structural change: re-generate seed with slightly different params
            sigma_cand = sigma_ns * rng.uniform(0.8, 1.2)
            area_cand = area * rng.uniform(0.9, 1.1)
            I_cand, Q_cand = _make_seed_pulse(N, dt, sigma_cand, area_cand)
            # Add some noise
            I_cand += rng.normal(0, 0.02, N)
            Q_cand += rng.normal(0, 0.02, N)
            
        # Enforce constraints
        I_cand, Q_cand = _enforce_constraints(I_cand, Q_cand)
        
        # Check validity
        if not _check_constraints(I_cand, Q_cand):
            continue
            
        # Evaluate score
        try:
            score = _score_pulse(I_cand, Q_cand)
        except Exception:
            continue
            
        if score < best_score:
            best_score = score
            I_best = I_cand.copy()
            Q_best = Q_cand.copy()
            
    # Final check
    if not _check_constraints(I_best, Q_best):
        # Return seed if best became invalid (should not happen)
        I_best, Q_best = _make_seed_pulse(N, dt, sigma_ns, area)
        
    result = np.column_stack([I_best, Q_best])
    return result
# EVOLVE-BLOCK-END
