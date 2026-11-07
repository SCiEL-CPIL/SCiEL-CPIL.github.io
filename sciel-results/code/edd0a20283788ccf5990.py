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

    # coherent: -i(H rho - rho H)
    def coh(H):
        return -1j * (np.kron(I3, H) - np.kron(H.T, I3))

    H_anh = 0.5 * ANHARMONICITY * n @ (n - I3)
    base_coh = coh(H_anh)
    coh_x = coh(0.5 * (a + a_dag))
    coh_y = coh(0.5j * (a - a_dag))
    det_coh = coh(n)

    # incoherent
    def incoh(Lk):
        Lkd = Lk.conj().T
        LkLkd = Lk @ Lkd
        return (np.kron(Lkd.T, Lk) - 0.5 * np.kron(I3, LkLkd) - 0.5 * np.kron(LkLkd.T, I3))
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / TPHI)
    base_inc = incoh(L1) + incoh(L2)

    # test states
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

# ---- Legendre Basis ----
_t_grid = np.linspace(0, 1, N_STEPS)
_x_grid = 2 * _t_grid - 1
_legendre_basis = np.array([np.polynomial.legendre.legval(_x_grid, [1 if i == k else 0 for i in range(8)]) for k in range(8)]).T

def _coeffs_to_pulse(coeffs):
    I_coeffs = coeffs[:8]
    Q_coeffs = coeffs[8:]
    I = _legendre_basis @ I_coeffs
    Q = _legendre_basis @ Q_coeffs
    return I, Q

def _pulse_to_coeffs(I, Q):
    I_coeffs = np.linalg.lstsq(_legendre_basis, I, rcond=None)[0]
    Q_coeffs = np.linalg.lstsq(_legendre_basis, Q, rcond=None)[0]
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
    
    return 10.0 * (amp_viol + slew_viol)

def pulse(target_gate, hardware_model):
    global _START_TIME
    _START_TIME = time.time()
    
    def elapsed():
        return time.time() - _START_TIME

    if target_gate != 'X_pi':
        return np.zeros((80, 2))

    # 1. Parametric DRAG Optimization with second-order correction
    t = (np.arange(N_STEPS) + 0.5) * DT
    
    def param_to_pulse(params):
        A, sigma, beta, tc, phi = params
        if sigma < 0.1: sigma = 0.1
        u = (t - tc) / sigma
        env = A * np.exp(-0.5 * u**2)
        I = env * np.cos(phi)
        Q = -env * np.sin(phi)
        # DRAG term: beta * d(env)/dt
        # d(env)/dt = A * exp(-0.5 u^2) * (-u / sigma)
        drag = beta * env * (-u / sigma)
        Q += drag
        return I, Q

    def param_obj(params):
        I, Q = param_to_pulse(params)
        score = _score(I, Q)
        pen = _constraint_penalty(I, Q)
        return score + pen

    # Bounds: A, sigma, beta, tc, phi
    bounds_param = [
        (0.1, 0.4),
        (1.0, 10.0),
        (-5.0, 5.0),
        (15.0, 25.0),
        (-0.5, 0.5)
    ]
    
    best_I = np.zeros(N_STEPS)
    best_Q = np.zeros(N_STEPS)
    best_score = np.inf
    
    # Try multiple starting points for Nelder-Mead
    if elapsed() < _DEADLINE - 150:
        # Theoretical DRAG: beta = sigma^2 * A / 2 for optimal leakage cancellation
        # For area pi: A * sigma * sqrt(2*pi) = pi => A = sqrt(pi/2)/sigma
        initial_guesses = [
            [0.3, 4.0, 0.0, 20.0, 0.0],
            [0.35, 3.5, 1.0, 20.0, 0.0],
            [0.3, 4.5, 2.0, 20.0, 0.0],
            [0.25, 5.0, 0.5, 20.0, 0.0],
            [0.35, 4.0, 1.5, 20.0, 0.0],
        ]
        
        for x0 in initial_guesses:
            if elapsed() > _DEADLINE - 150:
                break
            try:
                res_param = scipy.optimize.minimize(param_obj, x0, method='Nelder-Mead', 
                                                    bounds=bounds_param, 
                                                    options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-6})
                I_try, Q_try = param_to_pulse(res_param.x)
                s_try = _score(I_try, Q_try)
                if s_try < best_score:
                    best_I, best_Q = I_try, Q_try
                    best_score = s_try
            except Exception:
                pass

    # 2. Legendre Refinement with multiple restarts
    if elapsed() < _DEADLINE - 20:
        coeffs_init = _pulse_to_coeffs(best_I, best_Q)
        
        def leg_obj(coeffs):
            I, Q = _coeffs_to_pulse(coeffs)
            score = _score(I, Q)
            pen = _constraint_penalty(I, Q)
            return score + pen

        # Try L-BFGS-B from the current best
        try:
            res_leg = scipy.optimize.minimize(leg_obj, coeffs_init, method='L-BFGS-B', 
                                              bounds=[(-0.5, 0.5)]*16, 
                                              options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            I_leg, Q_leg = _coeffs_to_pulse(res_leg.x)
            s_leg = _score(I_leg, Q_leg)
            if s_leg < best_score:
                best_I, best_Q = I_leg, Q_leg
                best_score = s_leg
        except Exception:
            pass
        
        # Try additional restarts with perturbed coefficients
        if elapsed() < _DEADLINE - 20:
            rng = np.random.RandomState(42)
            for _ in range(3):
                if elapsed() > _DEADLINE - 20:
                    break
                try:
                    coeffs_pert = coeffs_init + rng.randn(16) * 0.05
                    res_pert = scipy.optimize.minimize(leg_obj, coeffs_pert, method='L-BFGS-B', 
                                                       bounds=[(-0.5, 0.5)]*16, 
                                                       options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8})
                    I_pert, Q_pert = _coeffs_to_pulse(res_pert.x)
                    s_pert = _score(I_pert, Q_pert)
                    if s_pert < best_score:
                        best_I, best_Q = I_pert, Q_pert
                        best_score = s_pert
                except Exception:
                    pass

    # Final validation
    if not _valid(best_I, best_Q):
        # Fallback to a simple valid Gaussian with DRAG
        t = (np.arange(N_STEPS) + 0.5) * DT
        center = 20.0
        sigma = 4.0
        A = 0.3
        env = A * np.exp(-0.5 * ((t - center) / sigma)**2)
        I_seed = env
        Q_seed = -1.0 * env * ((t - center) / sigma) / sigma  # DRAG term
        best_I, best_Q = I_seed, Q_seed
        best_score = _score(best_I, best_Q)

    # Ensure final pulse is valid
    if not _valid(best_I, best_Q):
        # Last resort: simple valid Gaussian
        t = (np.arange(N_STEPS) + 0.5) * DT
        center = 20.0
        sigma = 4.0
        I_seed = 0.3 * np.exp(-0.5 * ((t - center) / sigma)**2)
        Q_seed = np.zeros(N_STEPS)
        best_I, best_Q = I_seed, Q_seed

    return np.column_stack([best_I, best_Q])
