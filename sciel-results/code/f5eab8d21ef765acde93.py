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

_DEADLINE = 310.0
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

# ---- Fast per-pulse simulation (all 9 cases) ----
def _score(I, Q):
    infidelities = np.empty(9)
    k = 0
    det_vals = (0.0, -2 * np.pi * 0.002, 2 * np.pi * 0.002)
    sc_vals = (0.97, 1.0, 1.03)
    for di_val in det_vals:
        for sc_val in sc_vals:
            S = np.eye(9, dtype=complex)
            for i in range(N_STEPS):
                L = _base_coh + _base_inc
                L += di_val * _det_coh
                L += sc_val * (I[i] * _coh_x + Q[i] * _coh_y)
                S = expm(L * DT) @ S
            f = np.empty(6)
            for j in range(6):
                rf = S @ _test_states[j]
                fj = np.real(np.vdot(_target_states[j], rf))
                f[j] = fj if fj > 0 else 0.0
            infidelities[k] = 1.0 - np.mean(f)
            k += 1
    return np.mean(infidelities) + np.max(infidelities)

# ---- Legendre basis pulse representation ----
def _legendre_basis():
    """First 6 Legendre polynomials evaluated at 80 time points, normalized."""
    t = (np.arange(N_STEPS) + 0.5) * DT
    x = 2.0 * t / TOTAL_TIME - 1.0  # map to [-1, 1]
    
    # Compute Legendre polynomials P_0 through P_5
    basis = np.zeros((6, N_STEPS))
    basis[0] = np.ones(N_STEPS)
    basis[1] = x
    for n in range(2, 6):
        basis[n] = ((2*n - 1) * x * basis[n-1] - (n - 1) * basis[n-2]) / n
    
    # Normalize each basis function to have unit L2 norm over [0, TOTAL_TIME]
    for i in range(6):
        norm = np.sqrt(np.sum(basis[i]**2) * DT)
        if norm > 0:
            basis[i] /= norm
    
    return basis

_LEG_BASIS_I = _legendre_basis()
_LEG_BASIS_Q = _legendre_basis()

def _pulse_from_coeffs(coeffs):
    """Convert 12 coefficients (6 for I, 6 for Q) to pulse arrays."""
    I = np.zeros(N_STEPS)
    Q = np.zeros(N_STEPS)
    for i in range(6):
        I += coeffs[i] * _LEG_BASIS_I[i]
        Q += coeffs[i + 6] * _LEG_BASIS_Q[i]
    return I, Q

def _enforce_constraints(I, Q):
    """Enforce amplitude and slew constraints."""
    # Amplitude constraint
    ampv = np.sqrt(I * I + Q * Q)
    ma = np.max(ampv)
    if ma > AMPLITUDE_BOUND:
        s = AMPLITUDE_BOUND / ma
        I = I * s
        Q = Q * s
    
    # Slew rate constraint - iterative clipping
    for _ in range(10):
        changed = False
        # From zero into step 1
        s0 = np.hypot(I[0], Q[0]) / DT
        if s0 > SLEW_BOUND:
            f = SLEW_BOUND * DT / np.hypot(I[0], Q[0])
            I[0] *= f
            Q[0] *= f
            changed = True
        
        # Between steps
        for i in range(N_STEPS - 1):
            dI = I[i + 1] - I[i]
            dQ = Q[i + 1] - Q[i]
            sl = np.hypot(dI, dQ) / DT
            if sl > SLEW_BOUND:
                f = SLEW_BOUND * DT / np.hypot(dI, dQ)
                I[i + 1] = I[i] + dI * f
                Q[i + 1] = Q[i] + dQ * f
                changed = True
        
        # From last step to zero
        sN = np.hypot(I[-1], Q[-1]) / DT
        if sN > SLEW_BOUND:
            f = SLEW_BOUND * DT / np.hypot(I[-1], Q[-1])
            I[-1] *= f
            Q[-1] *= f
            changed = True
        
        if not changed:
            break
    
    return I, Q

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

def _make_seed_pulse():
    """Create a DRAG-style Gaussian seed pulse."""
    sigma = TOTAL_TIME / 6.0
    t = (np.arange(N_STEPS) + 0.5) * DT
    center = TOTAL_TIME / 2.0
    x = (t - center) / sigma
    g = np.exp(-0.5 * x * x)
    area = np.sum(g) * DT
    I = g * (np.pi / area)
    Q = np.zeros(N_STEPS)
    
    # Add DRAG term
    dI = np.empty(N_STEPS)
    dI[1:-1] = (I[2:] - I[:-2]) / (2.0 * DT)
    dI[0] = (I[1] - I[0]) / DT
    dI[-1] = (I[-1] - I[-2]) / DT
    Q = 0.15 * dI
    
    I, Q = _enforce_constraints(I, Q)
    
    # Project onto Legendre basis
    coeffs = np.zeros(12)
    for i in range(6):
        coeffs[i] = np.sum(I * _LEG_BASIS_I[i]) / DT
        coeffs[i + 6] = np.sum(Q * _LEG_BASIS_Q[i]) / DT
    
    return coeffs, I, Q

def pulse(target_gate, hardware_model):
    global _START_TIME
    _START_TIME = time.time()
    if target_gate != 'X_pi':
        return np.zeros((80, 2))

    def elapsed():
        return time.time() - _START_TIME

    # Start with seed pulse
    seed_coeffs, seed_I, seed_Q = _make_seed_pulse()
    best_score = _score(seed_I, seed_Q)
    best_I, best_Q = seed_I.copy(), seed_Q.copy()
    best_coeffs = seed_coeffs.copy()

    # ---- L-BFGS-B optimization with Legendre coefficients ----
    def obj(c):
        if elapsed() > _DEADLINE - 5:
            return best_score
        I, Q = _pulse_from_coeffs(c)
        I, Q = _enforce_constraints(I, Q)
        try:
            s = _score(I, Q)
        except Exception:
            s = 1.0
        return s

    # Multi-start L-BFGS-B
    rng = np.random.RandomState(42)
    n_starts = 8
    for start in range(n_starts):
        if elapsed() > _DEADLINE - 10:
            break
        
        if start == 0:
            c0 = best_coeffs.copy()
        else:
            # Random perturbations around best
            c0 = best_coeffs + 0.1 * rng.randn(12)
        
        try:
            res = scipy.optimize.minimize(obj, c0, method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            c_opt = res.x
            I_opt, Q_opt = _pulse_from_coeffs(c_opt)
            I_opt, Q_opt = _enforce_constraints(I_opt, Q_opt)
            s_opt = obj(c_opt)
            
            if s_opt < best_score:
                best_score = s_opt
                best_coeffs = c_opt.copy()
                best_I, best_Q = I_opt.copy(), Q_opt.copy()
        except Exception:
            pass

    # ---- Second phase: finer optimization from best found ----
    if elapsed() < _DEADLINE - 15:
        try:
            res = scipy.optimize.minimize(obj, best_coeffs, method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
            c_opt = res.x
            I_opt, Q_opt = _pulse_from_coeffs(c_opt)
            I_opt, Q_opt = _enforce_constraints(I_opt, Q_opt)
            s_opt = obj(c_opt)
            
            if s_opt < best_score:
                best_score = s_opt
                best_coeffs = c_opt.copy()
                best_I, best_Q = I_opt.copy(), Q_opt.copy()
        except Exception:
            pass

    # ---- Final safety: keep valid best, else seed ----
    if not _valid(best_I, best_Q):
        best_I, best_Q = seed_I, seed_Q

    return np.column_stack([best_I, best_Q])
