import numpy as np
import time
from scipy.linalg import expm
import scipy.optimize

# ---- Constants from the task contract ----
ANHARMONICITY = -2 * np.pi * 0.25
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DE_TUNINGS = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
SCALES = [0.97, 1.0, 1.03]
T1 = 25000.0
TPHI = 40000.0
N_STEPS = 80
DT = 0.5
TOTAL_TIME = N_STEPS * DT
N_CASES = 9

_DEADLINE = 305.0  # Safety margin before 330s kill
_START_TIME = time.time()

# ---- Precompute static superoperator pieces ----
def _build():
    a = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
    a_dag = a.conj().T
    n = a_dag @ a
    I3 = np.eye(3, dtype=complex)

    def coh(H):
        return -1j * (np.kron(I3, H) - np.kron(H.T, I3))

    H_anh = 0.5 * ANHARMONICITY * n @ (n - I3)
    base_coh = coh(H_anh)
    coh_x = coh(0.5 * (a + a_dag))
    coh_y = coh(0.5j * (a - a_dag))
    det_coh = coh(n)

    def incoh(Lk):
        Lkd = Lk.conj().T
        LkLkd = Lk @ Lkd
        return (np.kron(Lkd.T, Lk) - 0.5 * np.kron(I3, LkLkd) - 0.5 * np.kron(LkLkd.T, I3))
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / TPHI)
    base_inc = incoh(L1) + incoh(L2)

    psi0 = np.array([1, 0, 0], complex)
    psi1 = np.array([0, 1, 0], complex)
    psis = [psi0, psi1,
            (psi0 + psi1) / np.sqrt(2),
            (psi0 - psi1) / np.sqrt(2),
            (psi0 + 1j * psi1) / np.sqrt(2),
            (psi0 - 1j * psi1) / np.sqrt(2)]

    def xswap(p):
        c = np.array([p[1], p[0], p[2]], complex)
        return np.outer(c, c.conj()).reshape(9)

    test = [np.outer(p, p.conj()).reshape(9) for p in psis]
    target = [xswap(p) for p in psis]

    return base_coh, coh_x, coh_y, det_coh, base_inc, test, target

(_base_coh, _coh_x, _coh_y, _det_coh, _base_inc,
 _test_states, _target_states) = _build()

# Precompute the 9 case superoperator bases (without drive terms)
_case_bases = []
for di_val in DE_TUNINGS:
    for sc_val in SCALES:
        base = _base_coh + _base_inc + di_val * _det_coh
        _case_bases.append((base, sc_val))

# ---- Basis for parameterization ----
# Use piecewise constant basis for finer control
_t_grid = np.arange(N_STEPS, dtype=float)
_N_PIECE = 16  # 5 samples per piece
# Build piecewise constant basis: each basis function is 1 over its piece
_piece_indices = np.repeat(np.arange(N_PIECE), N_STEPS // N_PIECE)
_N_PIECE = len(np.unique(_piece_indices))
_piece_basis = np.zeros((N_STEPS, N_PIECE))
for p in range(N_PIECE):
    _piece_basis[_piece_indices == p, p] = 1.0

def _coeffs_to_pulse(coeffs):
    I = _piece_basis @ coeffs[:N_PIECE]
    Q = _piece_basis @ coeffs[N_PIECE:]
    return I, Q

def _pulse_to_coeffs(I, Q):
    I_coeffs = np.linalg.lstsq(_piece_basis, I, rcond=None)[0]
    Q_coeffs = np.linalg.lstsq(_piece_basis, Q, rcond=None)[0]
    return np.concatenate([I_coeffs, Q_coeffs])

def _valid(I, Q):
    if np.max(np.sqrt(I * I + Q * Q)) > AMPLITUDE_BOUND + 1e-9:
        return False
    if np.hypot(I[0], Q[0]) / DT > SLEW_BOUND + 1e-9:
        return False
    if np.hypot(I[-1], Q[-1]) / DT > SLEW_BOUND + 1e-9:
        return False
    for i in range(N_STEPS - 1):
        if np.hypot(I[i + 1] - I[i], Q[i + 1] - Q[i]) / DT > SLEW_BOUND + 1e-9:
            return False
    return True

def _score_fast(I, Q):
    """Compute score using precomputed case bases."""
    infidelities = np.empty(N_CASES)
    k = 0
    for base, sc_val in _case_bases:
        S = np.eye(9, dtype=complex)
        for i in range(N_STEPS):
            L = base + sc_val * (I[i] * _coh_x + Q[i] * _coh_y)
            S = expm(L * DT) @ S
        f = np.empty(6)
        for j in range(6):
            rf = S @ _test_states[j]
            fj = np.real(np.vdot(_target_states[j], rf))
            f[j] = fj if fj > 0 else 0.0
        infidelities[k] = 1.0 - np.mean(f)
        k += 1
    return np.mean(infidelities) + np.max(infidelities)

def _score(I, Q):
    return _score_fast(I, Q)

def _constraint_penalty(I, Q):
    amp = np.sqrt(I**2 + Q**2)
    amp_viol = np.sum(np.maximum(0, amp - AMPLITUDE_BOUND)**2)
    
    slew = np.zeros(N_STEPS + 1)
    slew[0] = np.hypot(I[0], Q[0]) / DT
    slew[-1] = np.hypot(I[-1], Q[-1]) / DT
    if N_STEPS > 1:
        dI = np.diff(I)
        dQ = np.diff(Q)
        slew[1:-1] = np.hypot(dI, dQ) / DT
    slew_viol = np.sum(np.maximum(0, slew - SLEW_BOUND)**2)
    
    return 10.0 * (amp_viol + slew_viol)

def _objective(coeffs):
    I, Q = _coeffs_to_pulse(coeffs)
    score = _score_fast(I, Q)
    pen = _constraint_penalty(I, Q)
    return score + pen

def _generate_seed_pulse():
    """Generate a DRAG-style Gaussian seed pulse."""
    t = (np.arange(N_STEPS) + 0.5) * DT
    center = 20.0
    sigma = 40.0 / 6.0
    
    A = np.pi / (sigma * np.sqrt(2 * np.pi))
    env = A * np.exp(-0.5 * ((t - center) / sigma)**2)
    edge_val = env[0]
    env -= edge_val * 0.5
    
    beta = sigma**2 * A / 2.0
    denv = env * (-(t - center) / sigma**2)
    
    I_seed = env.copy()
    Q_seed = beta * denv
    
    return I_seed, Q_seed

def _clamp_pulse(I, Q):
    """Clamp pulse to satisfy amplitude and slew constraints."""
    I = I.copy()
    Q = Q.copy()
    
    # Clamp amplitude
    amp = np.sqrt(I**2 + Q**2)
    mask = amp > AMPLITUDE_BOUND
    if np.any(mask):
        scale = AMPLITUDE_BOUND / amp[mask]
        I[mask] *= scale
        Q[mask] *= scale
    
    # Clamp slew: process sequentially
    # Start slew
    s0 = np.hypot(I[0], Q[0]) / DT
    if s0 > SLEW_BOUND:
        scale = SLEW_BOUND / s0
        I[0] *= scale
        Q[0] *= scale
    
    # Step-to-step slews
    prev_I, prev_Q = 0.0, 0.0
    for i in range(N_STEPS):
        cur_I, cur_Q = I[i], Q[i]
        dI, dQ = cur_I - prev_I, cur_Q - prev_Q
        s = np.hypot(dI, dQ) / DT
        if s > SLEW_BOUND:
            scale = SLEW_BOUND / s
            I[i] = prev_I + dI * scale
            Q[i] = prev_Q + dQ * scale
        prev_I, prev_Q = I[i], Q[i]
    
    # End slew
    s_end = np.hypot(I[-1], Q[-1]) / DT
    if s_end > SLEW_BOUND:
        scale = SLEW_BOUND / s_end
        I[-1] *= scale
        Q[-1] *= scale
    
    return I, Q

def pulse(target_gate, hardware_model):
    global _START_TIME
    _START_TIME = time.time()
    
    def elapsed():
        return time.time() - _START_TIME

    if target_gate != 'X_pi':
        return np.zeros((80, 2))

    # 1. Generate initial seed pulse
    I_seed, Q_seed = _generate_seed_pulse()
    I_seed, Q_seed = _clamp_pulse(I_seed, Q_seed)
    
    best_I, best_Q = I_seed.copy(), Q_seed.copy()
    best_score = _score_fast(best_I, best_Q)
    best_coeffs = _pulse_to_coeffs(best_I, best_Q)
    
    # 2. L-BFGS-B optimization
    if elapsed() < _DEADLINE - 80:
        try:
            bounds = [(-0.5, 0.5)] * (2 * N_PIECE)
            res = scipy.optimize.minimize(_objective, best_coeffs, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
            I_opt, Q_opt = _coeffs_to_pulse(res.x)
            I_opt, Q_opt = _clamp_pulse(I_opt, Q_opt)
            if _valid(I_opt, Q_opt):
                s_opt = _score_fast(I_opt, Q_opt)
                if s_opt < best_score:
                    best_I, best_Q = I_opt, Q_opt
                    best_score = s_opt
                    best_coeffs = _pulse_to_coeffs(best_I, best_Q)
        except Exception:
            pass
    
    # 3. Multi-start with varied seeds
    if elapsed() < _DEADLINE - 50:
        np.random.seed(42)
        n_starts = 8
        for trial in range(n_starts):
            if elapsed() >= _DEADLINE - 50:
                break
            
            # Generate varied seeds
            t = (np.arange(N_STEPS) + 0.5) * DT
            center = 20.0 + np.random.uniform(-2, 2)
            sigma = (40.0 / 6.0) * np.random.uniform(0.8, 1.2)
            
            A = np.pi / (sigma * np.sqrt(2 * np.pi))
            env = A * np.exp(-0.5 * ((t - center) / sigma)**2)
            edge_val = env[0]
            env -= edge_val * 0.5
            
            beta = sigma**2 * A / 2.0
            denv = env * (-(t - center) / sigma**2)
            
            I_try = env.copy()
            Q_try = beta * denv * np.random.uniform(0.8, 1.2)
            
            I_try, Q_try = _clamp_pulse(I_try, Q_try)
            if not _valid(I_try, Q_try):
                continue
            
            s_try = _score_fast(I_try, Q_try)
            if s_try < best_score:
                best_I, best_Q = I_try.copy(), Q_try.copy()
                best_score = s_try
                best_coeffs = _pulse_to_coeffs(best_I, best_Q)
            
            # Optimize from this start if it's good enough
            if s_try < best_score * 1.5 and elapsed() < _DEADLINE - 50:
                try:
                    bounds = [(-0.5, 0.5)] * (2 * N_PIECE)
                    coeffs_init = _pulse_to_coeffs(I_try, Q_try)
                    res = scipy.optimize.minimize(_objective, coeffs_init, method='L-BFGS-B',
                        bounds=bounds, options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
                    I_opt, Q_opt = _coeffs_to_pulse(res.x)
                    I_opt, Q_opt = _clamp_pulse(I_opt, Q_opt)
                    if _valid(I_opt, Q_opt):
                        s_opt = _score_fast(I_opt, Q_opt)
                        if s_opt < best_score:
                            best_I, best_Q = I_opt, Q_opt
                            best_score = s_opt
                            best_coeffs = _pulse_to_coeffs(best_I, best_Q)
                except Exception:
                    pass
    
    # 4. Fine-tuning
    if elapsed() < _DEADLINE - 20:
        try:
            bounds = [(-0.5, 0.5)] * (2 * N_PIECE)
            res2 = scipy.optimize.minimize(_objective, best_coeffs, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100})
            I_fine, Q_fine = _coeffs_to_pulse(res2.x)
            I_fine, Q_fine = _clamp_pulse(I_fine, Q_fine)
            if _valid(I_fine, Q_fine):
                s_fine = _score_fast(I_fine, Q_fine)
                if s_fine < best_score:
                    best_I, best_Q = I_fine, Q_fine
                    best_score = s_fine
                    best_coeffs = _pulse_to_coeffs(best_I, best_Q)
        except Exception:
            pass
    
    # 5. Direct coordinate search for final refinement
    if elapsed() < _DEADLINE - 5:
        # Try small perturbations in piecewise coefficient space
        np.random.seed(123)
        step_size = 0.01
        for dim in range(2 * N_PIECE):
            if elapsed() >= _DEADLINE - 5:
                break
            for direction in [-1, 1]:
                coeffs_try = best_coeffs.copy()
                coeffs_try[dim] += direction * step_size
                I_try, Q_try = _coeffs_to_pulse(coeffs_try)
                I_try, Q_try = _clamp_pulse(I_try, Q_try)
                if _valid(I_try, Q_try):
                    s_try = _score_fast(I_try, Q_try)
                    if s_try < best_score:
                        best_I, best_Q = I_try, Q_try
                        best_score = s_try
                        best_coeffs = coeffs_try
    
    # Final validation
    if not _valid(best_I, best_Q):
        best_I, best_Q = _clamp_pulse(best_I, best_Q)
        if not _valid(best_I, best_Q):
            best_I = np.ones(N_STEPS) * 0.03
            best_Q = np.zeros(N_STEPS)
    
    return np.column_stack([best_I, best_Q])
