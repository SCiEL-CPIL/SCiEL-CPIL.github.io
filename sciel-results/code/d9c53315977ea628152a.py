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

    # Coherent part: -i(H rho - rho H)
    def coh(H):
        return -1j * (np.kron(I3, H) - np.kron(H.T, I3))

    # H = detuning * n + 0.5 * anh * n(n-1) + 0.5 * scale * (I (a + a^dag) + i Q (a - a^dag))
    H_anh = 0.5 * ANHARMONICITY * n @ (n - I3)
    base_coh = coh(H_anh)
    coh_x = coh(0.5 * (a + a_dag))
    coh_y = coh(0.5j * (a - a_dag))
    det_coh = coh(n)

    # Incoherent part
    def incoh(Lk):
        Lkd = Lk.conj().T
        LkLkd = Lk @ Lkd
        return (np.kron(Lkd.T, Lk) - 0.5 * np.kron(I3, LkLkd) - 0.5 * np.kron(LkLkd.T, I3))
    
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / TPHI)
    base_inc = incoh(L1) + incoh(L2)

    # Test states and targets
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

def _score(I, Q):
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
    return np.mean(infidelities) + np.max(infidelities)

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
    
    return 100.0 * (amp_viol + slew_viol)

def pulse(target_gate, hardware_model):
    global _START_TIME
    _START_TIME = time.time()
    
    def elapsed():
        return time.time() - _START_TIME

    if target_gate != 'X_pi':
        return np.zeros((80, 2))

    # Time grid
    t = (np.arange(N_STEPS) + 0.5) * DT
    
    # Physics-informed parametric model: Gaussian DRAG
    # I(t) = A * exp(-0.5 * ((t-tc)/sigma)^2) * cos(phi)
    # Q(t) = -A * exp(-0.5 * ((t-tc)/sigma)^2) * sin(phi) + beta * d(env)/dt
    # Optimal beta for Gaussian DRAG: beta ~ sigma^2 * anh / 4 (approx)
    # We optimize A, sigma, tc, phi. beta is derived.
    
    def param_to_pulse(params):
        A, sigma, tc, phi = params
        if sigma < 0.5: sigma = 0.5
        u = (t - tc) / sigma
        env = A * np.exp(-0.5 * u**2)
        
        # Derivative of envelope
        denv = env * (-u / sigma)
        
        # DRAG coefficient
        # For a Gaussian, the optimal DRAG amplitude to cancel leakage is related to 
        # the anharmonicity and the pulse width.
        # beta_opt = (anh * sigma^2) / 4 is a common approximation in rad/ns units if A is in rad/ns.
        # Let's refine: The drive is H_drive = A/2 (a+ad) cos(wt).
        # The leakage is proportional to A * (anh/4) * <2|a+ad|1> ...
        # Standard result: beta = (sigma^2 * |anh|) / 4
        beta = (sigma**2 * abs(ANHARMONICITY)) / 4.0
        
        I = env * np.cos(phi)
        Q = -env * np.sin(phi) + beta * denv
        return I, Q

    def param_obj(params):
        I, Q = param_to_pulse(params)
        score = _score(I, Q)
        pen = _constraint_penalty(I, Q)
        return score + pen

    # Initial guess: 
    # Area of Gaussian A * sigma * sqrt(2*pi) = pi => A = sqrt(pi/2) / sigma
    # For sigma=3, A ~ 0.7 / 3 ~ 0.23
    # Slew rate check: max slope of Gaussian is A/sigma * exp(-0.5) ~ 0.23/3 * 0.6 ~ 0.04 < 0.12. OK.
    # Detuning correction might require slightly different shape, so we optimize.
    
    x0 = [0.25, 3.0, 20.0, 0.0]
    
    bounds_param = [
        (0.1, 0.4),   # A
        (1.5, 6.0),   # sigma
        (15.0, 25.0), # tc
        (-0.5, 0.5)   # phi
    ]
    
    best_I = np.zeros(N_STEPS)
    best_Q = np.zeros(N_STEPS)
    best_score = np.inf
    
    # 1. Nelder-Mead for global-ish search
    if elapsed() < _DEADLINE - 100:
        try:
            res = scipy.optimize.minimize(param_obj, x0, method='Nelder-Mead', bounds=bounds_param, options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-5})
            if res.fun < best_score:
                best_I, best_Q = param_to_pulse(res.x)
                best_score = _score(best_I, best_Q)
        except Exception:
            pass

    # 2. L-BFGS-B for local refinement
    if elapsed() < _DEADLINE - 20:
        try:
            res2 = scipy.optimize.minimize(param_obj, res.x, method='L-BFGS-B', bounds=bounds_param, options={'maxiter': 200, 'ftol': 1e-10})
            if res2.fun < best_score:
                best_I, best_Q = param_to_pulse(res2.x)
                best_score = _score(best_I, best_Q)
        except Exception:
            pass

    # 3. Fallback if invalid or no improvement
    if not _valid(best_I, best_Q) or best_score == np.inf:
        # Simple valid Gaussian
        t = (np.arange(N_STEPS) + 0.5) * DT
        center = 20.0
        sigma = 4.0
        A = 0.25
        I_seed = A * np.exp(-0.5 * ((t - center) / sigma)**2)
        Q_seed = np.zeros(N_STEPS)
        # Add small DRAG
        denv = I_seed * (-(t - center) / sigma**2)
        beta = (sigma**2 * abs(ANHARMONICITY)) / 4.0
        Q_seed += beta * denv
        best_I, best_Q = I_seed, Q_seed
        best_score = _score(best_I, best_Q)

    # Final check and clamp
    if not _valid(best_I, best_Q):
        # Aggressive clamping
        amp = np.sqrt(best_I**2 + best_Q**2)
        mask = amp > AMPLITUDE_BOUND
        best_I[mask] *= AMPLITUDE_BOUND / amp[mask]
        best_Q[mask] *= AMPLITUDE_BOUND / amp[mask]
        
        # Slew rate clamp is harder, but amplitude clamp usually helps.
        # If still invalid, return zeros (score 0, but better than exception)
        if not _valid(best_I, best_Q):
             return np.zeros((80, 2))

    return np.column_stack([best_I, best_Q])
