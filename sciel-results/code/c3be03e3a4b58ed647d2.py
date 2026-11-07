import numpy as np
import time
from scipy.linalg import expm
import scipy.optimize
from scipy.interpolate import interp1d

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

_DEADLINE = 315.0  # Safety margin before 330s kill
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

    # Coherent part of Liouvillian: L = -i (I⊗H - H^T⊗I)
    def coh(H):
        return -1j * (np.kron(I3, H) - np.kron(H.T, I3))

    # H = delta * n + 0.5 * chi * n * (n - I) + 0.5 * S * (I * (a + a_dag) + i * Q * (a - a_dag))
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

def pulse(target_gate, hardware_model):
    global _START_TIME
    _START_TIME = time.time()
    if target_gate != 'X_pi':
        return np.zeros((80, 2))

    def elapsed():
        return time.time() - _START_TIME

    # PWL Basis: 41 knots (40 segments)
    N_KNOTS = 41
    t_knots = np.linspace(0, TOTAL_TIME, N_KNOTS)
    
    def get_pulse(params):
        """
        params: [I_0..I_40, Q_0..Q_40]
        Returns (I, Q) of length 80 if valid, else None.
        """
        I_knots = params[0:N_KNOTS]
        Q_knots = params[N_KNOTS:2*N_KNOTS]
        
        # Linear interpolation to midpoint grid
        I = np.interp(_t, t_knots, I_knots)
        Q = np.interp(_t, t_knots, Q_knots)
        
        if not _valid(I, Q):
            return None
        return I, Q

    # Create initial guess: DRAG-like pulse projected onto PWL knots
    # Seed: Gaussian for I, derivative for Q
    I_seed_full = np.exp(-0.5 * ((_t - 20.0) / 6.0)**2) * 0.25
    dI_seed_full = np.gradient(I_seed_full, DT)
    Q_seed_full = 0.05 * dI_seed_full
    
    # Interpolate seed to knots to get initial parameters
    I_knots_init = np.interp(t_knots, _t, I_seed_full)
    Q_knots_init = np.interp(t_knots, _t, Q_seed_full)
    
    # Ensure edges are 0 for valid start (slew from 0)
    I_knots_init[0] = 0.0
    I_knots_init[-1] = 0.0
    Q_knots_init[0] = 0.0
    Q_knots_init[-1] = 0.0
    
    x0 = np.concatenate([I_knots_init, Q_knots_init])

    # Verify initial guess is valid; if not, scale down
    test_pulse = get_pulse(x0)
    if test_pulse is None:
        # Scale down until valid
        scale = 1.0
        for _ in range(30):
            x0_scaled = x0 * scale
            test_pulse = get_pulse(x0_scaled)
            if test_pulse is not None:
                x0 = x0_scaled
                break
            scale *= 0.7
        if test_pulse is None:
            # Fallback: constant small pulse (will likely fail slew, so just small constant in middle)
            x0 = np.zeros(2 * N_KNOTS)
            # Set middle knots to small value
            mid = N_KNOTS // 2
            x0[mid-2:mid+2] = 0.05
            x0[N_KNOTS + mid-2 : N_KNOTS + mid+2] = 0.0
            test_pulse = get_pulse(x0)
            if test_pulse is None:
                # Last resort: zero pulse (invalid for gate, but valid constraints? No, gate fails. But better than crash)
                # Actually zero pulse is valid constraints.
                x0 = np.zeros(2 * N_KNOTS)
                test_pulse = get_pulse(x0)

    I_init, Q_init = test_pulse
    best_score = _score(I_init, Q_init)
    best_pulse = (I_init, Q_init)
    best_params = x0.copy()

    def objective(params):
        if elapsed() > _DEADLINE - 5:
            return 1000.0
        pulse_data = get_pulse(params)
        if pulse_data is None:
            # Penalty for invalid pulse
            I_knots = params[0:N_KNOTS]
            Q_knots = params[N_KNOTS:2*N_KNOTS]
            I = np.interp(_t, t_knots, I_knots)
            Q = np.interp(_t, t_knots, Q_knots)
            
            amp = np.sqrt(I*I + Q*Q)
            amp_pen = np.sum(np.maximum(amp - AMPLITUDE_BOUND, 0.0))
            
            dI = np.diff(I)
            dQ = np.diff(Q)
            slew = np.sqrt(dI**2 + dQ**2) / DT
            slew_pen = np.sum(np.maximum(slew - SLEW_BOUND, 0.0))
            slew_pen += np.maximum(np.hypot(I[0], Q[0])/DT - SLEW_BOUND, 0.0)
            slew_pen += np.maximum(np.hypot(I[-1], Q[-1])/DT - SLEW_BOUND, 0.0)
            
            return 10.0 + 100.0 * (amp_pen + slew_pen)
        I, Q = pulse_data
        return _score(I, Q)

    # Phase 1: L-BFGS-B with constraints handled via penalty
    if elapsed() < _DEADLINE - 30:
        try:
            res_l = scipy.optimize.minimize(
                objective, x0, method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 2000}
            )
            if res_l.fun < best_score:
                pulse_data = get_pulse(res_l.x)
                if pulse_data is not None:
                    best_score = res_l.fun
                    best_params = res_l.x
                    best_pulse = pulse_data
        except Exception as e:
            pass

    # Phase 2: Nelder-Mead refinement
    if elapsed() < _DEADLINE - 15:
        try:
            res_nm = scipy.optimize.minimize(
                objective, best_params, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
            )
            if res_nm.fun < best_score:
                pulse_data = get_pulse(res_nm.x)
                if pulse_data is not None:
                    best_score = res_nm.fun
                    best_params = res_nm.x
                    best_pulse = pulse_data
        except Exception as e:
            pass

    # Phase 3: Final high-precision polish with tighter tolerances
    if elapsed() < _DEADLINE - 5:
        try:
            res_final = scipy.optimize.minimize(
                objective, best_params, method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 3000}
            )
            if res_final.fun < best_score:
                pulse_data = get_pulse(res_final.x)
                if pulse_data is not None:
                    best_score = res_final.fun
                    best_params = res_final.x
                    best_pulse = pulse_data
        except Exception as e:
            pass

    # Phase 4: Direct coordinate descent on individual steps for final polish
    if elapsed() < _DEADLINE - 2:
        try:
            I, Q = best_pulse
            I_work = I.copy()
            Q_work = Q.copy()
            improved = True
            while improved and elapsed() < _DEADLINE - 2:
                improved = False
                for i in range(N_STEPS):
                    for delta in [0.001, -0.001, 0.005, -0.005, 0.01, -0.01]:
                        I_test = I_work.copy()
                        Q_test = Q_work.copy()
                        I_test[i] += delta
                        if _valid(I_test, Q_test):
                            s = _score(I_test, Q_test)
                            if s < best_score - 1e-12:
                                I_work = I_test
                                best_score = s
                                best_pulse = (I_work.copy(), Q_work.copy())
                                improved = True
                                break
                    for delta in [0.001, -0.001, 0.005, -0.005, 0.01, -0.01]:
                        I_test = I_work.copy()
                        Q_test = Q_work.copy()
                        Q_test[i] += delta
                        if _valid(I_test, Q_test):
                            s = _score(I_test, Q_test)
                            if s < best_score - 1e-12:
                                Q_work = Q_test
                                best_score = s
                                best_pulse = (I_work.copy(), Q_work.copy())
                                improved = True
                                break
        except Exception as e:
            pass

    I, Q = best_pulse
    return np.column_stack([I, Q])
