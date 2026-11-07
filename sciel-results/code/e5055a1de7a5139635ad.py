import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize

_DEADLINE = time.time() + 310

def _build_hamiltonian(detuning, anharmonicity, scale, I, Q):
    a = np.zeros((3, 3), dtype=complex)
    a[1, 0] = 1.0
    a[2, 1] = np.sqrt(2)
    a_dag = a.conj().T
    n_op = a_dag @ a
    
    H = detuning * n_op
    H += 0.5 * anharmonicity * n_op @ (n_op - np.eye(3))
    H += 0.5 * scale * (I * (a + a_dag) + 1j * Q * (a - a_dag))
    return H

def _build_liouvillian(H, T1, Tphi):
    dim = 3
    a = np.zeros((dim, dim), dtype=complex)
    a[1, 0] = 1.0
    a[2, 1] = np.sqrt(2)
    a_dag = a.conj().T
    n_op = a_dag @ a
    
    L1 = a / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / Tphi)
    
    I3 = np.eye(dim, dtype=complex)
    
    L_vec = -1j * (np.kron(H, I3) - np.kron(I3, H.conj().T))
    for L in [L1, L2]:
        LdL = L.conj().T @ L
        L_vec += np.kron(L, L.conj()) - 0.5 * (np.kron(LdL, I3) + np.kron(I3, LdL))
    
    return L_vec

def _simulate_pulse(IQ, detuning, scale, anharmonicity, T1, Tphi, dt):
    n_steps = len(IQ)
    rho = np.zeros((3, 3), dtype=complex)
    rho[0, 0] = 1.0
    
    for i in range(n_steps):
        I_val = IQ[i, 0]
        Q_val = IQ[i, 1]
        H = _build_hamiltonian(detuning, anharmonicity, scale, I_val, Q_val)
        L_vec = _build_liouvillian(H, T1, Tphi)
        
        rho_vec = rho.ravel()
        rho_vec = expm(L_vec * dt) @ rho_vec
        rho = rho_vec.reshape(3, 3)
    
    return rho

def _compute_fidelity(rho, target_state):
    target_vec = target_state.ravel()
    rho_t = np.outer(target_vec, target_vec.conj())
    fidelity = np.real(np.trace(rho_t @ rho))
    return max(0.0, min(1.0, fidelity))

def _score_pulse(IQ, anharmonicity, T1, Tphi, dt):
    detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    targets = [
        np.array([0, -1j, 0], dtype=complex),
        np.array([1j, 0, 0], dtype=complex),
        np.array([1j, -1j, 0], dtype=complex) / np.sqrt(2),
        np.array([-1j, -1j, 0], dtype=complex) / np.sqrt(2),
        np.array([-1, -1j, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2),
    ]
    
    infidelities = []
    for det in detunings:
        for scl in scales:
            rho = _simulate_pulse(IQ, det, scl, anharmonicity, T1, Tphi, dt)
            case_infs = []
            for target in targets:
                fid = _compute_fidelity(rho, target)
                case_infs.append(1.0 - fid)
            infidelities.append(np.mean(case_infs))
    
    infidelities = np.array(infidelities)
    return np.mean(infidelities) + np.max(infidelities)

def _check_constraints(IQ, dt, amp_bound, slew_bound):
    amplitudes = np.sqrt(IQ[:, 0]**2 + IQ[:, 1]**2)
    if np.any(amplitudes > amp_bound + 1e-9):
        return False
    
    if np.sqrt(IQ[0, 0]**2 + IQ[0, 1]**2) / dt > slew_bound + 1e-9:
        return False
    
    for i in range(len(IQ) - 1):
        diff = IQ[i+1] - IQ[i]
        if np.sqrt(diff[0]**2 + diff[1]**2) / dt > slew_bound + 1e-9:
            return False
    
    if np.sqrt(IQ[-1, 0]**2 + IQ[-1, 1]**2) / dt > slew_bound + 1e-9:
        return False
    
    return True

def _make_pulse(params, n_steps, dt):
    sigma = params[0]
    drag = params[1]
    amp = params[2]
    phase = params[3]
    
    t = np.arange(n_steps) * dt
    t_center = (n_steps - 1) * dt / 2.0
    t_shifted = t - t_center
    
    envelope = np.exp(-0.5 * (t_shifted / sigma)**2)
    area = np.sum(envelope) * dt
    if area < 1e-12:
        return np.zeros((n_steps, 2))
    envelope = envelope * (np.pi / area)
    
    I = amp * np.cos(phase) * envelope
    d_envelope = np.gradient(envelope, dt)
    Q = amp * np.sin(phase) * envelope - drag * amp * np.cos(phase) * d_envelope
    
    IQ = np.column_stack([I, Q])
    
    amplitudes = np.sqrt(IQ[:, 0]**2 + IQ[:, 1]**2)
    max_amp = np.max(amplitudes)
    if max_amp > 0.4:
        IQ *= (0.4 / max_amp)
        
    return IQ

def pulse(target_gate, hardware_model):
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi is supported")
    
    anharmonicity = -2 * np.pi * 0.25
    amp_bound = 0.4
    slew_bound = 0.12
    T1 = 25000.0
    Tphi = 40000.0
    dt = 0.5
    n_steps = 80
    
    # Initial guess:
    # sigma ~ 6.67 ns
    # drag ~ 8 ns (estimated from chi/omega)
    # amp ~ 0.2
    # phase ~ 0
    initial_params = [6.67, 8.0, 0.2, 0.0]
    
    def objective(params):
        if time.time() > _DEADLINE:
            return 10.0
        IQ = _make_pulse(params, n_steps, dt)
        if not _check_constraints(IQ, dt, amp_bound, slew_bound):
            return 10.0
        return _score_pulse(IQ, anharmonicity, T1, Tphi, dt)
    
    best_IQ = _make_pulse(initial_params, n_steps, dt)
    best_score = objective(initial_params)
    
    bounds = [(3.0, 15.0), (0.0, 20.0), (0.1, 0.39), (-np.pi, np.pi)]
    
    try:
        result = minimize(objective, initial_params, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-5})
        if result.fun < best_score:
            best_score = result.fun
            best_IQ = _make_pulse(result.x, n_steps, dt)
    except Exception as e:
        pass
    
    # Random restarts if time permits
    if time.time() < _DEADLINE - 20:
        for _ in range(5):
            if time.time() > _DEADLINE - 10:
                break
            perturbed = np.array([
                np.random.uniform(4.0, 12.0),
                np.random.uniform(2.0, 15.0),
                np.random.uniform(0.15, 0.35),
                np.random.uniform(-np.pi/2, np.pi/2)
            ])
            
            IQ_test = _make_pulse(perturbed, n_steps, dt)
            if not _check_constraints(IQ_test, dt, amp_bound, slew_bound):
                continue
            score_test = _score_pulse(IQ_test, anharmonicity, T1, Tphi, dt)
            if score_test < best_score:
                best_score = score_test
                best_IQ = IQ_test.copy()
                try:
                    res2 = minimize(objective, perturbed, method='L-BFGS-B', bounds=bounds,
                                    options={'maxiter': 50, 'ftol': 1e-6})
                    if res2.fun < best_score:
                        best_score = res2.fun
                        best_IQ = _make_pulse(res2.x, n_steps, dt)
                except:
                    pass
                    
    return best_IQ
