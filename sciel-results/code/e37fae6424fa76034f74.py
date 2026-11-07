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

_DEADLINE = 320.0  # Safety margin before 330s kill
_START_TIME = time.time()

# ---- Precompute static superoperator pieces ----
def _build():
    a = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
    a_dag = a.conj().T
    n = a_dag @ a
    I3 = np.eye(3, dtype=complex)

    # coherent: -i(H rho - rho H)  =>  L = -i( kron(I,H) - kron(H.T, I) )
    def coh(H):
        return -1j * (np.kron(I3, H) - np.kron(H.T, I3))

    H_anh = 0.5 * ANHARMONICITY * n @ (n - I3)
    base_coh = coh(H_anh)
    coh_x = coh(0.5 * (a + a_dag))
    coh_y = coh(0.5j * (a - a_dag))
    det_coh = coh(n)

    # incoherent (constant)
    def incoh(Lk):
        Lkd = Lk.conj().T
        LkLkd = Lk @ Lkd
        return (np.kron(Lkd.T, Lk) - 0.5 * np.kron(I3, LkLkd) - 0.5 * np.kron(LkLkd.T, I3))
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / TPHI)
    base_inc = incoh(L1) + incoh(L2)

    # test states (6) and X-pi target states (6)
    psi0 = np.array([1, 0, 0], complex)
    psi1 = np.array([0, 1, 0], complex)
    psis = [psi0, psi1,
            (psi0 + psi1) / np.sqrt(2),
            (psi0 - psi1) / np.sqrt(2),
            (psi0 + 1j * psi1) / np.sqrt(2),
            (psi0 - 1j * psi1) / np.sqrt(2)]
    
    # X-pi target: |0><->|1>
    def xswap(p):
        c = np.array([p[1], p[0], p[2]], complex)
        return np.outer(c, c.conj()).reshape(9)
        
    test = [np.outer(p, p.conj()).reshape(9) for p in psis]
    target = [xswap(p) for p in psis]

    return base_coh, coh_x, coh_y, det_coh, base_inc, test, target

(_base_coh, _coh_x, _coh_y, _det_coh, _base_inc,
 _test_states, _target_states) = _build()

# ---- Pulse Parameterization ----
# Use Legendre polynomials for smooth pulse shapes
# Map t in [0, 1] to x in [-1, 1]
_t_grid = np.linspace(0, 1, N_STEPS)
_x_grid = 2 * _t_grid - 1
# Use 10 Legendre coefficients for I and 10 for Q
N_COEFFS = 10
_legendre_basis = np.array([
    np.polynomial.legendre.legval(_x_grid, [1 if i == k else 0 for i in range(N_COEFFS)]) 
    for k in range(N_COEFFS)
]).T
# Shape: (N_STEPS, N_COEFFS)

def _coeffs_to_pulse(coeffs):
    """
    coeffs: (2 * N_COEFFS,) array. First N_COEFFS for I, next N_COEFFS for Q.
    Returns I, Q arrays of shape (N_STEPS,).
    """
    I_coeffs = coeffs[:N_COEFFS]
    Q_coeffs = coeffs[N_COEFFS:]
    I = _legendre_basis @ I_coeffs
    Q = _legendre_basis @ Q_coeffs
    return I, Q

def _calc_infidelities(I, Q):
    """
    Calculate infidelities for all 9 cases.
    """
    infidelities = np.empty(9)
    k = 0
    for di_val in DE_TUNINGS:
        for sc_val in SCALES:
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
    return infidelities

def _penalty(I, Q):
    """
    Calculate penalty for constraint violations.
    """
    pen = 0.0
    # Amplitude constraint
    amp = np.sqrt(I**2 + Q**2)
    viol_amp = np.maximum(0, amp - AMPLITUDE_BOUND)
    pen += np.sum(viol_amp**2)
    
    # Slew rate constraint
    # From 0 to step 0
    dist0 = np.hypot(I[0], Q[0])
    if dist0 / DT > SLEW_BOUND:
        pen += ((dist0 / DT) - SLEW_BOUND)**2
        
    # Between steps
    for i in range(N_STEPS - 1):
        dI = I[i+1] - I[i]
        dQ = Q[i+1] - Q[i]
        dist = np.hypot(dI, dQ)
        rate = dist / DT
        if rate > SLEW_BOUND:
            pen += (rate - SLEW_BOUND)**2
            
    # From last step to 0
    distN = np.hypot(I[-1], Q[-1])
    if distN / DT > SLEW_BOUND:
        pen += ((distN / DT) - SLEW_BOUND)**2
        
    return pen

def _objective(coeffs):
    """
    Objective function for optimization.
    Smooth max of infidelities + penalty.
    """
    I, Q = _coeffs_to_pulse(coeffs)
    
    # Soft max approximation for max infidelity
    # f_soft = epsilon * log(sum(exp(f_i / epsilon)))
    # We use a fixed epsilon to maintain smoothness
    eps = 0.01
    infids = _calc_infidelities(I, Q)
    
    # To avoid overflow, subtract max
    max_inf = np.max(infids)
    soft_max = max_inf + eps * np.log(np.sum(np.exp((infids - max_inf) / eps)))
    
    # Mean infidelity
    mean_inf = np.mean(infids)
    
    # Combine: prioritize max, but also mean
    score = 0.5 * mean_inf + 0.5 * soft_max
    
    # Add penalty
    pen = _penalty(I, Q)
    score += 10.0 * pen  # High weight for penalty
    
    return score

def _is_valid(I, Q):
    """
    Strictly check validity.
    """
    amp = np.sqrt(I**2 + Q**2)
    if np.max(amp) > AMPLITUDE_BOUND + 1e-9:
        return False
    if np.hypot(I[0], Q[0]) / DT > SLEW_BOUND + 1e-9:
        return False
    if np.hypot(I[-1], Q[-1]) / DT > SLEW_BOUND + 1e-9:
        return False
    for i in range(N_STEPS - 1):
        if np.hypot(I[i + 1] - I[i], Q[i + 1] - Q[i]) / DT > SLEW_BOUND + 1e-9:
            return False
    return True

def _score_valid(I, Q):
    """
    Calculate the actual score for a valid pulse.
    """
    infids = _calc_infidelities(I, Q)
    return np.mean(infids) + np.max(infids)

def _generate_seed_pulse(type_seed):
    """
    Generate initial seeds for I, Q.
    """
    t = (np.arange(N_STEPS) + 0.5) * DT
    center = TOTAL_TIME / 2.0
    
    if type_seed == 'gaussian':
        sigma = TOTAL_TIME / 6.0
        I = np.exp(-0.5 * ((t - center) / sigma)**2)
        area = np.sum(I) * DT
        I *= (np.pi / area)
        Q = np.zeros(N_STEPS)
    elif type_seed == 'drag':
        # DRAG-like: Gaussian + derivative
        sigma = TOTAL_TIME / 6.0
        I = np.exp(-0.5 * ((t - center) / sigma)**2)
        area = np.sum(I) * DT
        I *= (np.pi / area)
        # Add a derivative term to Q
        dI_dt = np.gradient(I, DT)
        Q = -0.1 * dI_dt  # Heuristic drag coefficient
        # Normalize area of I again
        area = np.sum(I) * DT
        I *= (np.pi / area)
    else: # constant
        I = np.ones(N_STEPS) * (np.pi / TOTAL_TIME)
        Q = np.zeros(N_STEPS)
        
    return I, Q

def pulse(target_gate, hardware_model):
    global _START_TIME
    _START_TIME = time.time()
    if target_gate != 'X_pi':
        return np.zeros((80, 2))

    def elapsed():
        return time.time() - _START_TIME

    best_score = float('inf')
    best_I = np.zeros(N_STEPS)
    best_Q = np.zeros(N_STEPS)
    
    seeds = ['gaussian', 'drag', 'constant']
    
    for seed_type in seeds:
        if elapsed() > _DEADLINE - 50:
            break
            
        I_seed, Q_seed = _generate_seed_pulse(seed_type)
        
        # Convert seed to coeffs
        # Least squares fit to Legendre basis
        I_coeffs = np.linalg.lstsq(_legendre_basis, I_seed, rcond=None)[0]
        Q_coeffs = np.linalg.lstsq(_legendre_basis, Q_seed, rcond=None)[0]
        coeffs0 = np.concatenate([I_coeffs, Q_coeffs])
        
        # Optimize
        try:
            res = scipy.optimize.minimize(
                _objective, 
                coeffs0, 
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if elapsed() > _DEADLINE - 50:
                break
                
            I_opt, Q_opt = _coeffs_to_pulse(res.x)
            
            if _is_valid(I_opt, Q_opt):
                sc = _score_valid(I_opt, Q_opt)
                if sc < best_score:
                    best_score = sc
                    best_I = I_opt.copy()
                    best_Q = Q_opt.copy()
            else:
                # If invalid, try a simple projection and re-score
                # This is a fallback
                I_proj, Q_proj = _enforce_constraints(I_opt, Q_opt)
                if _is_valid(I_proj, Q_proj):
                    sc = _score_valid(I_proj, Q_proj)
                    if sc < best_score:
                        best_score = sc
                        best_I = I_proj.copy()
                        best_Q = Q_proj.copy()
                        
        except Exception:
            pass

    # If no valid pulse found, return a safe constant pulse
    if best_score == float('inf'):
        t = (np.arange(N_STEPS) + 0.5) * DT
        best_I = np.ones(N_STEPS) * (np.pi / TOTAL_TIME)
        best_Q = np.zeros(N_STEPS)
        # Ensure validity for the constant pulse
        if not _is_valid(best_I, best_Q):
             best_I, best_Q = _enforce_constraints(best_I, best_Q)

    return np.column_stack([best_I, best_Q])

def _enforce_constraints(I, Q):
    """
    Simple iterative projection for amplitude and slew rate.
    """
    I = I.copy()
    Q = Q.copy()
    
    for _ in range(10):
        # Amplitude
        amp = np.sqrt(I**2 + Q**2)
        if np.max(amp) > AMPLITUDE_BOUND:
            scale = AMPLITUDE_BOUND / np.max(amp)
            I *= scale
            Q *= scale
            
        # Slew rate
        # From 0 to I[0], Q[0]
        dist0 = np.hypot(I[0], Q[0])
        if dist0 / DT > SLEW_BOUND:
            scale = SLEW_BOUND * DT / dist0
            I[0] *= scale
            Q[0] *= scale
            
        # Between steps
        for i in range(N_STEPS - 1):
            dI = I[i+1] - I[i]
            dQ = Q[i+1] - Q[i]
            dist = np.hypot(dI, dQ)
            if dist / DT > SLEW_BOUND:
                scale = SLEW_BOUND * DT / dist
                I[i+1] = I[i] + dI * scale
                Q[i+1] = Q[i] + dQ * scale
                
        # From last to 0
        distN = np.hypot(I[-1], Q[-1])
        if distN / DT > SLEW_BOUND:
            scale = SLEW_BOUND * DT / distN
            I[-1] *= scale
            Q[-1] *= scale

    return I, Q

# Helper to ensure we don't crash on import
if __name__ == "__main__":
    pass
