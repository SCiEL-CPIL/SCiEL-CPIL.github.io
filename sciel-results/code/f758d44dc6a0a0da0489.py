import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize

def _build_matrices():
    a = np.zeros((3, 3), dtype=complex)
    a[1, 0] = 1.0
    a[2, 1] = np.sqrt(2.0)
    adag = a.conj().T
    n = adag @ a
    return a, adag, n

def _liouvillian_super(I_val, Q_val, detuning, scale, a, adag, n):
    I3 = np.eye(3, dtype=complex)
    H = detuning * n + 0.5 * (-2 * np.pi * 0.25) * (n @ (n - I3))
    H += 0.5 * scale * (I_val * (a + adag) + 1j * Q_val * (a - adag))
    H_super = -1j * (np.kron(H, I3) - np.kron(I3, H.T))
    L1 = a / np.sqrt(25000.0)
    L2 = n * np.sqrt(2.0 / 40000.0)
    L1dag = L1.conj().T
    L2dag = L2.conj().T
    L1_super = np.kron(L1, L1.conj()) - 0.5 * (np.kron(L1dag @ L1, I3) + np.kron(I3, (L1dag @ L1).T))
    L2_super = np.kron(L2, L2.conj()) - 0.5 * (np.kron(L2dag @ L2, I3) + np.kron(I3, (L2dag @ L2).T))
    return H_super + L1_super + L2_super

def _simulate_pulse(IQ, detuning, scale, a, adag, n):
    X_gate = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        (np.array([1, 1, 0], dtype=complex)) / np.sqrt(2),
        (np.array([1, -1, 0], dtype=complex)) / np.sqrt(2),
        (np.array([1, 1j, 0], dtype=complex)) / np.sqrt(2),
        (np.array([1, -1j, 0], dtype=complex)) / np.sqrt(2),
    ]
    supers = [expm(_liouvillian_super(IQ[i, 0], IQ[i, 1], detuning, scale, a, adag, n) * 0.5) for i in range(80)]
    fids = []
    for psi in states:
        rho = np.outer(psi, psi.conj()).flatten()
        for s in supers:
            rho = s @ rho
        rho = rho.reshape(3, 3)
        psi_t = X_gate @ psi
        rho_t = np.outer(psi_t, psi_t.conj())
        fids.append(np.real(np.trace(rho_t @ rho)))
    return np.mean(fids)

def _score(IQ):
    a, adag, n = _build_matrices()
    infs = []
    for det in [-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002]:
        for sc in [0.97, 1.0, 1.03]:
            fid = _simulate_pulse(IQ, det, sc, a, adag, n)
            infs.append(1 - fid)
    return np.mean(infs) + np.max(infs)

def _valid(IQ):
    amps = np.sqrt(IQ[:, 0] ** 2 + IQ[:, 1] ** 2)
    if np.any(amps > 0.4 + 1e-12):
        return False
    for i in range(80):
        prev = IQ[i - 1] if i > 0 else np.array([0.0, 0.0])
        if np.linalg.norm(IQ[i] - prev) / 0.5 > 0.12 + 1e-12:
            return False
    if np.linalg.norm(IQ[-1]) / 0.5 > 0.12 + 1e-12:
        return False
    return True

def _enforce_constraints(IQ):
    """Enforce amplitude and slew-rate constraints by clamping."""
    IQ = IQ.copy()
    # Amplitude bound
    for i in range(80):
        amp = np.sqrt(IQ[i, 0] ** 2 + IQ[i, 1] ** 2)
        if amp > 0.4:
            scale = 0.4 / amp
            IQ[i, 0] *= scale
            IQ[i, 1] *= scale
    # Slew rate: forward pass (from zero into step 0, then step to step)
    max_delta = 0.12 * 0.5
    prev = np.array([0.0, 0.0])
    for i in range(80):
        diff = IQ[i] - prev
        dist = np.linalg.norm(diff)
        if dist > max_delta:
            IQ[i] = prev + (max_delta / dist) * diff
        prev = IQ[i]
    # Slew rate: backward pass (from last step to zero)
    prev = np.array([0.0, 0.0])
    for i in range(79, -1, -1):
        diff = prev - IQ[i]
        dist = np.linalg.norm(diff)
        if dist > max_delta:
            # We need IQ[i] to be within max_delta of prev (which is the next step toward zero)
            # Actually we need ||IQ[i] - prev|| <= max_delta where prev is the value after i
            # Let's re-think: we need ||IQ[i+1] - IQ[i]|| / dt <= 0.12 and ||0 - IQ[79]|| / dt <= 0.12
            # Backward: start from end. prev = 0 (after last step).
            # For i=79: need ||0 - IQ[79]|| <= max_delta
            # For i=78: need ||IQ[79] - IQ[78]|| <= max_delta
            # So we clamp IQ[i] to be within max_delta of prev
            IQ[i] = prev - (max_delta / dist) * diff  # Move IQ[i] toward prev
        prev = IQ[i]
    # Re-check amplitude after slew clamping
    for i in range(80):
        amp = np.sqrt(IQ[i, 0] ** 2 + IQ[i, 1] ** 2)
        if amp > 0.4:
            scale = 0.4 / amp
            IQ[i, 0] *= scale
            IQ[i, 1] *= scale
    return IQ

def _make_seed_pulse(sigma_factor=1.0, drag_coeff=-0.3, amp_factor=1.0):
    t = np.arange(80) * 0.5
    sigma = (40.0 / 6.0) * sigma_factor
    env = np.exp(-(t - 20.0) ** 2 / (2 * sigma ** 2))
    env = np.maximum(env - env[0] - env[-1], 0)
    area = np.sum(env) * 0.5
    if area > 0:
        env *= np.pi / area
    env *= amp_factor
    I = env
    Q = drag_coeff * np.gradient(I, 0.5)
    IQ = np.column_stack([I, Q])
    IQ = _enforce_constraints(IQ)
    return IQ

def pulse(target_gate, hardware_model):
    start_time = time.time()
    deadline = start_time + 320.0  # Safety margin before 330s kill
    
    # Initial seed
    best_IQ = _make_seed_pulse(1.0, -0.3, 1.0)
    best_score = _score(best_IQ)
    
    # Try a few different seeds to pick the best starting point
    seeds = [
        _make_seed_pulse(0.9, -0.4, 1.0),
        _make_seed_pulse(1.0, -0.3, 1.0),
        _make_seed_pulse(1.1, -0.2, 1.0),
        _make_seed_pulse(1.0, -0.5, 1.0),
        _make_seed_pulse(0.85, -0.35, 1.0),
    ]
    for seed in seeds:
        if time.time() > deadline - 30:
            break
        s = _score(seed)
        if s < best_score:
            best_score = s
            best_IQ = seed
    
    # Nelder-Mead optimization over all 160 parameters
    def objective(x):
        IQ = x.reshape(80, 2)
        IQ = _enforce_constraints(IQ)
        if not _valid(IQ):
            return 1e6
        return _score(IQ)
    
    x0 = best_IQ.flatten()
    
    # Nelder-Mead with adaptive step size
    try:
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 500,
                'xatol': 1e-4,
                'fatol': 1e-6,
                'adaptive': True,
                'initial_simplex': None,
            },
            callback=None,
        )
        if result.x is not None:
            candidate_IQ = _enforce_constraints(result.x.reshape(80, 2))
            if _valid(candidate_IQ):
                s = _score(candidate_IQ)
                if s < best_score:
                    best_score = s
                    best_IQ = candidate_IQ
    except Exception:
        pass
    
    # If we have time, try a second round of optimization from the improved point
    if time.time() < deadline - 60:
        try:
            x1 = best_IQ.flatten()
            result2 = minimize(
                objective,
                x1,
                method='Nelder-Mead',
                options={
                    'maxiter': 300,
                    'xatol': 1e-5,
                    'fatol': 1e-7,
                    'adaptive': True,
                },
            )
            if result2.x is not None:
                candidate_IQ = _enforce_constraints(result2.x.reshape(80, 2))
                if _valid(candidate_IQ):
                    s = _score(candidate_IQ)
                    if s < best_score:
                        best_score = s
                        best_IQ = candidate_IQ
        except Exception:
            pass
    
    # Final validity check
    if not _valid(best_IQ):
        best_IQ = _enforce_constraints(best_IQ)
    
    return best_IQ
