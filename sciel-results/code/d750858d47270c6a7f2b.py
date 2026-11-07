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

_DEADLINE = 325.0  # Safety margin before 330s kill
_START_TIME = time.time()

# Time grid - midpoint sampling
_t = (np.arange(N_STEPS) + 0.5) * DT

# ---- Precompute static superoperator pieces ----
def _build():
    a = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
    a_dag = a.conj().T
    n = a_dag @ a
    I3 = np.eye(3, dtype=complex)

    # Coherent part of Liouvillian
    def coh(H):
        return -1j * (np.kron(I3, H) - np.kron(H.T, I3))

    # H = delta * n + 0.5 * chi * n * (n - I) + 0.5 * S * (I * (a + a_dagger) + i * Q * (a - a_dagger))
    H_anh = 0.5 * ANHARMONICITY * n @ (n - I3)
    base_coh = coh(H_anh)
    
    H_I = 0.5 * (a + a_dag)
    coh_I = coh(H_I)
    
    H_Q = 0.5j * (a - a_dag)
    coh_Q = coh(H_Q)
    
    H_det = n
    coh_det = coh(H_det)
    
    # Incoherent part
    def incoh(Lk):
        Lkd = Lk.conj().T
        LkLkd = Lk @ Lkd
        return (np.kron(Lkd.T, Lk) - 0.5 * np.kron(I3, LkLkd) - 0.5 * np.kron(LkLkd.T, I3))
        
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / TPHI)
    base_inc = incoh(L1) + incoh(L2)

    # Test states
    psi0 = np.array([1, 0, 0], complex)
    psi1 = np.array([0, 1, 0], complex)
    psis = [psi0, psi1,
            (psi0 + psi1) / np.sqrt(2),
            (psi0 - psi1) / np.sqrt(2),
            (psi0 + 1j * psi1) / np.sqrt(2),
            (psi0 - 1j * psi1) / np.sqrt(2)]
            
    # X-pi target: Swap |0> and |1>
    def xswap(p):
        c = np.array([p[1], p[0], p[2]], complex)
        return np.outer(c, c.conj()).reshape(9)
        
    test = [np.outer(p, p.conj()).reshape(9) for p in psis]
    target = [xswap(p) for p in psis]

    return base_coh, coh_I, coh_Q, coh_det, base_inc, test, target

(_base_coh, _coh_I, _coh_Q, _coh_det, _base_inc,
 _test_states, _target_states) = _build()

def _valid(I, Q):
    """Check if pulse satisfies amplitude and slew constraints."""
    amp = np.sqrt(I * I + Q * Q)
    if np.max(amp) > AMPLITUDE_BOUND + 1e-9:
        return False
    # Slew from zero to first step
    if np.hypot(I[0], Q[0]) / DT > SLEW_BOUND + 1e-9:
        return False
    # Slew between consecutive steps
    dI = np.diff(I)
    dQ = np.diff(Q)
    slew = np.sqrt(dI**2 + dQ**2) / DT
    if np.max(slew) > SLEW_BOUND + 1e-9:
        return False
    # Slew from last step to zero
    if np.hypot(I[-1], Q[-1]) / DT > SLEW_BOUND + 1e-9:
        return False
    return True

def _score(I, Q):
    """Calculate robustness score."""
    infidelities = np.empty(9)
    k = 0
    for di_val in DE_TUNINGS:
        for sc_val in SCALES:
            S = np.eye(9, dtype=complex)
            for i in range(N_STEPS):
                L = _base_coh + _base_inc
                L += di_val * _coh_det
                L += sc_val * (I[i] * _coh_I + Q[i] * _coh_Q)
                S = expm(L * DT) @ S
            f = np.empty(6)
            for j in range(6):
                rf = S @ _test_states[j]
                fj = np.real(np.vdot(_target_states[j], rf))
                f[j] = fj if fj > 0 else 0.0
            infidelities[k] = 1.0 - np.mean(f)
            k += 1
    return np.mean(infidelities) + np.max(infidelities)

def _score_fast(I, Q):
    """Fast approximate score using only nominal case + worst detuning cases."""
    # Use only 3 cases: nominal, max detuning, min detuning
    di_vals = [0.0, DE_TUNINGS[0], DE_TUNINGS[2]]
    sc_val = 1.0
    infidelities = np.empty(3)
    k = 0
    for di_val in di_vals:
        S = np.eye(9, dtype=complex)
        for i in range(N_STEPS):
            L = _base_coh + _base_inc
            L += di_val * _coh_det
            L += sc_val * (I[i] * _coh_I + Q[i] * _coh_Q)
            S = expm(L * DT) @ S
        f = np.empty(6)
        for j in range(6):
            rf = S @ _test_states[j]
            fj = np.real(np.vdot(_target_states[j], rf))
            f[j] = fj if fj > 0 else 0.0
        infidelities[k] = 1.0 - np.mean(f)
        k += 1
    return np.mean(infidelities) + np.max(infidelities)

def pulse(target_gate, hardware_model):
    global _START_TIME
    _START_TIME = time.time()
    if target_gate != 'X_pi':
        return np.zeros((80, 2))

    def elapsed():
        return time.time() - _START_TIME

    # Direct optimization: 80 I-values + 80 Q-values with constraints
    # Use a parametrization that ensures validity
    
    def get_pulse_direct(params):
        """
        params: 160 values, first 80 are I, next 80 are Q
        Apply soft constraints via clipping
        """
        I = params[0:80].copy()
        Q = params[80:160].copy()
        
        # Clip amplitudes
        amp = np.sqrt(I**2 + Q**2)
        mask = amp > AMPLITUDE_BOUND
        if np.any(mask):
            scale = AMPLITUDE_BOUND / amp[mask]
            I[mask] *= scale
            Q[mask] *= scale
        
        # Enforce slew rate constraints by limiting differences
        # From zero to first
        amp0 = np.hypot(I[0], Q[0])
        if amp0 > SLEW_BOUND * DT:
            scale = (SLEW_BOUND * DT) / amp0
            I[0] *= scale
            Q[0] *= scale
        
        # Between steps
        for i in range(1, 80):
            dI = I[i] - I[i-1]
            dQ = Q[i] - Q[i-1]
            slew = np.hypot(dI, dQ)
            if slew > SLEW_BOUND * DT:
                scale = (SLEW_BOUND * DT) / slew
                I[i] = I[i-1] + dI * scale
                Q[i] = Q[i-1] + dQ * scale
        
        # From last to zero
        amplast = np.hypot(I[-1], Q[-1])
        if amplast > SLEW_BOUND * DT:
            scale = (SLEW_BOUND * DT) / amplast
            I[-1] *= scale
            Q[-1] *= scale
        
        return I, Q

    def objective_direct(params):
        if elapsed() > _DEADLINE - 5:
            return 1000.0
        I, Q = get_pulse_direct(params)
        if not _valid(I, Q):
            return 10.0
        return _score_fast(I, Q)

    # Initial guess: DRAG-style
    I_init = np.exp(-0.5 * ((_t - 20.0) / 6.0)**2) * 0.3
    dI = np.gradient(I_init, DT)
    Q_init = 0.08 * dI
    
    # Ensure initial validity
    amp = np.sqrt(I_init**2 + Q_init**2)
    if np.max(amp) > AMPLITUDE_BOUND:
        I_init *= AMPLITUDE_BOUND / np.max(amp)
        Q_init *= AMPLITUDE_BOUND / np.max(amp)
    
    x0 = np.concatenate([I_init, Q_init])
    
    # Bounds for direct optimization
    bounds = [(-0.4, 0.4)] * 80 + [(-0.4, 0.4)] * 80
    
    best_score = 100.0
    best_pulse = None
    best_params = None

    # Phase 1: CMA-ES for global optimization
    if elapsed() < _DEADLINE - 80:
        try:
            import cma
            sigma0 = 0.05
            opts = {'maxiter': 100, 'seed': 42, 'verbose': -100}
            res_cma = cma.fmin(objective_direct, x0, sigma0, opts)
            if res_cma[0] < best_score:
                best_score = res_cma[0]
                best_params = res_cma[1]
                I, Q = get_pulse_direct(best_params)
                if _valid(I, Q):
                    best_pulse = (I, Q)
        except Exception as e:
            pass

    # Phase 2: L-BFGS-B local refinement
    if best_params is not None and elapsed() < _DEADLINE - 20:
        try:
            res_lbf = scipy.optimize.minimize(objective_direct, best_params, method='L-BFGS-B',
                                              bounds=bounds, options={'maxiter': 200, 'ftol': 1e-10})
            if res_lbf.fun < best_score:
                best_score = res_lbf.fun
                best_params = res_lbf.x
                I, Q = get_pulse_direct(best_params)
                if _valid(I, Q):
                    best_pulse = (I, Q)
        except Exception as e:
            pass

    # Phase 3: If we have time, do full score evaluation and refine
    if best_pulse is not None and elapsed() < _DEADLINE - 10:
        try:
            I, Q = best_pulse
            full_score = _score(I, Q)
            
            # Try a few perturbations and keep the best
            for trial in range(3):
                if elapsed() > _DEADLINE - 10:
                    break
                # Small random perturbation
                pert = np.random.normal(0, 0.01, 160)
                new_params = best_params + pert
                new_I, new_Q = get_pulse_direct(new_params)
                if _valid(new_I, new_Q):
                    new_score = _score(new_I, new_Q)
                    if new_score < full_score:
                        full_score = new_score
                        best_pulse = (new_I, new_Q)
                        best_params = new_params
        except Exception as e:
            pass

    # Fallback if no valid pulse found
    if best_pulse is None:
        I = np.exp(-0.5 * ((_t - 20.0) / 6.0)**2) * 0.25
        dI = np.gradient(I, DT)
        Q = 0.05 * dI
        if _valid(I, Q):
            best_pulse = (I, Q)
        else:
            I = np.ones(N_STEPS) * 0.05
            Q = np.zeros(N_STEPS)
            best_pulse = (I, Q)

    I, Q = best_pulse
    return np.column_stack([I, Q])
