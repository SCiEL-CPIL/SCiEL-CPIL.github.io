"""Robust control pulses for a three-level transmon."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 280  # safety margin before 330s kill

def _liouvillian_step(pulse_step, detuning, scale, T1, Tphi, dt):
    """
    Compute one step of Liouvillian evolution.
    pulse_step: (2,) array [I, Q] in rad/ns
    Returns: (3,3) density matrix update (unitary part only, no decoherence for speed in inner loop)
    """
    # Build 3-level Hamiltonian
    # a = [[0,1,0],[0,0,sqrt(2)],[0,0,0]]
    a = np.zeros((3,3), dtype=complex)
    a[0,1] = 1.0
    a[1,2] = np.sqrt(2.0)
    adag = a.conj().T
    n_op = adag @ a
    
    # H = detuning*n + 0.5*anharmonicity*n*(n-1) + 0.5*scale*(I*(a+adag) + i*Q*(a-adag))
    anharmonicity = -2.0 * np.pi * 0.25
    I_val = pulse_step[0]
    Q_val = pulse_step[1]
    
    H = detuning * n_op + 0.5 * anharmonicity * (n_op @ (n_op - np.eye(3))) 
    H += 0.5 * scale * (I_val * (a + adag) + 1j * Q_val * (a - adag))
    
    # Evolve density matrix: rho' = expm(-iH dt) rho expm(iH dt)
    # For efficiency, use matrix exponential
    U = _expm(-1j * H * dt)
    return U

def _expm(A):
    """Matrix exponential using scipy."""
    from scipy.linalg import expm
    return expm(A)

def _simulate_pulse(pulse, detuning, scale, T1, Tphi, dt=0.5):
    """
    Simulate the full pulse sequence and return final density matrix.
    pulse: (80, 2) array
    Returns: (3,3) final density matrix
    """
    # Initial state: |0><0|
    rho = np.zeros((3,3), dtype=complex)
    rho[0,0] = 1.0
    
    # Precompute Lindblad operators
    a = np.zeros((3,3), dtype=complex)
    a[0,1] = 1.0
    a[1,2] = np.sqrt(2.0)
    adag = a.conj().T
    n_op = adag @ a
    
    sqrt_T1 = np.sqrt(T1)
    sqrt_2_Tphi = np.sqrt(2.0 / Tphi)
    
    # For each step, apply unitary evolution + decoherence
    for i in range(len(pulse)):
        I_val, Q_val = pulse[i]
        
        # Build Hamiltonian
        anharmonicity = -2.0 * np.pi * 0.25
        H = detuning * n_op + 0.5 * anharmonicity * (n_op @ (n_op - np.eye(3)))
        H += 0.5 * scale * (I_val * (a + adag) + 1j * Q_val * (a - adag))
        
        # Unitary evolution
        U = _expm(-1j * H * dt)
        rho = U @ rho @ U.conj().T
        
        # Decoherence (first-order approximation for small dt)
        # L[rho] = (1/T1) (a rho a† - 0.5 a†a rho - 0.5 rho a†a)
        # + (1/Tphi) (n rho n† - 0.5 n†n rho - 0.5 rho n†n)
        a_dag_a = adag @ a
        a_a_dag = a @ adag
        n_dag_n = adag @ a  # n†n = n since n is Hermitian
        
        # Amplitude damping
        rho = rho + (dt / T1) * (a @ rho @ adag - 0.5 * (a_dag_a @ rho + rho @ a_dag_a))
        
        # Dephasing (pure dephasing part: Tphi includes T1 contribution, so pure dephasing rate = 1/Tphi - 1/(2*T1))
        # But the contract says Lindblad operators are a/sqrt(T1) and n*sqrt(2/Tphi)
        # So dephasing: (2/Tphi) * (n rho n† - 0.5 n†n rho - 0.5 rho n†n)
        rho = rho + (dt * 2.0 / Tphi) * (n_op @ rho @ n_op - 0.5 * (n_op @ rho + rho @ n_op))
        
        # Renormalize to keep trace = 1
        tr = np.trace(rho)
        if abs(tr) > 1e-15:
            rho = rho / tr
    
    return rho

def _fidelity_to_X(rho):
    """
    Compute fidelity to X gate.
    X gate maps |0>->|1>, |1>->|0>
    Fidelity = (1/4) * sum over 4 basis states of |<psi|U|phi>|^2 ... 
    Actually, for a unitary U, fidelity to X = (1/4) * Tr(X† U) ... 
    For mixed states: F = (1/4) * sum_{i,j} <i|X† U|j> ... 
    Simpler: F = (1/4) * (Tr(X† U) + Tr(X U†)) ... no.
    
    Standard: For target unitary U_t and actual U, F = (1/4)|Tr(U_t† U)|^2 for pure states.
    For density matrix rho_final, fidelity to X applied to initial state:
    F = <psi_t| rho_final |psi_t> where psi_t = X|psi_0>
    But we need average over 4 input states.
    
    Actually, the contract says: mean fidelity to X over |0>, |1>, (|0>±|1>)/sqrt(2), (|0>±i|1>)/sqrt(2)
    So for each input state |psi>, compute F = <X psi| rho_final |X psi>
    """
    X = np.array([[0,1,0],[1,0,0],[0,0,0]], dtype=complex)
    
    states = [
        np.array([1,0,0], dtype=complex),  # |0>
        np.array([0,1,0], dtype=complex),  # |1>
        np.array([1,1,0], dtype=complex) / np.sqrt(2),  # (|0>+|1>)/sqrt(2)
        np.array([1,-1,0], dtype=complex) / np.sqrt(2),  # (|0>-|1>)/sqrt(2)
        np.array([1,1j,0], dtype=complex) / np.sqrt(2),  # (|0>+i|1>)/sqrt(2)
        np.array([1,-1j,0], dtype=complex) / np.sqrt(2),  # (|0>-i|1>)/sqrt(2)
    ]
    
    fidelities = []
    for psi in states:
        psi_t = X @ psi  # target state
        psi_t = psi_t / np.linalg.norm(psi_t)
        F = np.real(np.vdot(psi_t, rho @ psi_t))
        fidelities.append(np.clip(F, 0, 1))
    
    return np.mean(fidelities)

def _score_pulse(pulse):
    """
    Compute the score: mean + max of infidelity over 9 hardware configs.
    """
    detunings = [-2*np.pi*0.002, 0, 2*np.pi*0.002]
    scales = [0.97, 1.0, 1.03]
    T1 = 25000.0
    Tphi = 40000.0
    dt = 0.5
    
    infidelities = []
    for det in detunings:
        for sc in scales:
            rho = _simulate_pulse(pulse, det, sc, T1, Tphi, dt)
            F = _fidelity_to_X(rho)
            infidelities.append(1.0 - F)
    
    infidelities = np.array(infidelities)
    return np.mean(infidelities) + np.max(infidelities)

def _gaussian_drag_pulse(params):
    """
    Generate pulse from parameters.
    params: [area_scale, drag_coeff, detuning_comp, width_scale, phase_offset]
    """
    N = 80
    dt = 0.5
    t = np.arange(N) * dt
    t_center = 20.0  # 40 ns / 2
    
    # Gaussian envelope
    sigma = (40.0 / 6.0) / params[3] if params[3] != 0 else 40.0/6.0
    env = np.exp(-0.5 * ((t - t_center) / sigma) ** 2)
    
    # Normalize area to pi
    area = np.trapz(env, dx=dt)
    if area > 1e-10:
        env = env * (np.pi / area)
    
    # DRAG: Q channel proportional to derivative of I
    I_env = env
    # Derivative
    dI_dt = np.gradient(I_env, dt)
    Q_env = params[1] * dI_dt
    
    # Add detuning compensation (constant offset)
    I_env = I_env + params[2]
    
    # Apply phase offset
    phase = params[4]
    I = I_env * np.cos(phase) - Q_env * np.sin(phase)
    Q = I_env * np.sin(phase) + Q_env * np.cos(phase)
    
    # Scale overall
    I = I * params[0]
    Q = Q * params[0]
    
    pulse = np.column_stack([I, Q])
    
    # Enforce amplitude bound
    amp = np.sqrt(I**2 + Q**2)
    mask = amp > 0.4
    if np.any(mask):
        scale_down = 0.4 / amp[mask]
        I[mask] *= scale_down
        Q[mask] *= scale_down
    
    # Enforce slew bound
    for i in range(N-1):
        dI = abs(I[i+1] - I[i])
        dQ = abs(Q[i+1] - Q[i])
        slew = np.sqrt(dI**2 + dQ**2) / dt
        if slew > 0.12:
            scale = 0.12 / slew
            I[i+1] = I[i] + (I[i+1] - I[i]) * scale
            Q[i+1] = Q[i] + (Q[i+1] - Q[i]) * scale
    
    # Check slew from 0 to first and last to 0
    slew_start = np.sqrt(I[0]**2 + Q[0]**2) / dt
    if slew_start > 0.12:
        scale = 0.12 / slew_start
        I[0] *= scale
        Q[0] *= scale
    
    slew_end = np.sqrt(I[-1]**2 + Q[-1]**2) / dt
    if slew_end > 0.12:
        scale = 0.12 / slew_end
        I[-1] *= scale
        Q[-1] *= scale
    
    return pulse

def _objective(params):
    """Objective function for optimizer."""
    if time.time() > _DEADLINE:
        return 1e6
    pulse = _gaussian_drag_pulse(params)
    return _score_pulse(pulse)

def pulse(target_gate, hardware_model):
    """
    Return an (80, 2) array of I and Q amplitudes (rad/ns) for a 40 ns X-pi gate.
    """
    if target_gate != 'X_pi':
        return np.zeros((80, 2))
    
    # Initial guess: standard DRAG pulse
    # params: [area_scale, drag_coeff, detuning_comp, width_scale, phase_offset]
    x0 = np.array([1.0, 0.0, 0.0, 1.0, 0.0])
    
    best_pulse = _gaussian_drag_pulse(x0)
    best_score = _score_pulse(best_pulse)
    
    # Try optimization
    try:
        result = minimize(_objective, x0, method='Nelder-Mead', 
                         options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-5,
                                  'adaptive': True})
        if result.fun < best_score:
            best_pulse = _gaussian_drag_pulse(result.x)
            best_score = result.fun
    except Exception:
        pass
    
    # Try a few more random initializations if time permits
    if time.time() < _DEADLINE - 60:
        rng = np.random.RandomState(42)
        for trial in range(5):
            if time.time() > _DEADLINE:
                break
            x0_trial = x0 + rng.randn(5) * np.array([0.1, 0.05, 0.01, 0.1, 0.1])
            try:
                result = minimize(_objective, x0_trial, method='Nelder-Mead',
                                 options={'maxiter': 300, 'xatol': 1e-4, 'fatol': 1e-5,
                                          'adaptive': True})
                if result.fun < best_score:
                    best_pulse = _gaussian_drag_pulse(result.x)
                    best_score = result.fun
            except Exception:
                pass
    
    return best_pulse
