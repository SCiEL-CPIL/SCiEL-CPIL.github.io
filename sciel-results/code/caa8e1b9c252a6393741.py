"""
Robust control pulses for a three-level transmon X-pi gate.
Uses a Chebyshev-modulated Gaussian basis and L-BFGS-B optimization for robustness.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import expm

# Constants from task contract
T1 = 25000.0
Tphi = 40000.0
ANHARMONICITY = -2.0 * np.pi * 0.25
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUND = 2.0 * np.pi * 0.002
DT = 0.5  # ns
N_STEPS = 80
TOTAL_TIME = N_STEPS * DT  # 40 ns

# Precompute ladder operator
_a = np.array([[0, 1, 0],
               [0, 0, np.sqrt(2)],
               [0, 0, 0]], dtype=complex)
_a_dag = _a.conj().T
_n = _a_dag @ _a
_n2 = _n @ _n
_nn1 = _n2 - _n

# Detuning and scale values for the 9 cases
detunings = [-DETUNING_BOUND, 0.0, DETUNING_BOUND]
scales = [0.97, 1.0, 1.03]

# Initial states for fidelity calculation
_states = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [1/np.sqrt(2), 1/np.sqrt(2), 0],
    [1/np.sqrt(2), -1/np.sqrt(2), 0],
    [1/np.sqrt(2), 1j/np.sqrt(2), 0],
    [1/np.sqrt(2), -1j/np.sqrt(2), 0]
], dtype=complex)

# Precompute Chebyshev basis functions at the time grid
# t_mid are the centers of the 80 steps
t_mid = np.linspace(DT/2, TOTAL_TIME - DT/2, N_STEPS)
center = TOTAL_TIME / 2.0
# Normalized time in [-1, 1] for Chebyshev
t_norm = (t_mid - center) / (TOTAL_TIME / 2.0)

# Number of Chebyshev coefficients
N_CHEB = 10
# Precompute T_k(x) for k=0..N_CHEB-1
_cheb_basis = np.zeros((N_STEPS, N_CHEB))
for k in range(N_CHEB):
    _cheb_basis[:, k] = np.cos(k * np.arccos(t_norm))

def _build_hamiltonian(I, Q, detuning, scale):
    """Build 3x3 Hamiltonian."""
    H = (detuning * _n 
         + 0.5 * ANHARMONICITY * _nn1
         + 0.5 * scale * (I * (_a + _a_dag) + 1j * Q * (_a - _a_dag)))
    return H

def _evolve_pulse(I_pulse, Q_pulse, detuning, scale):
    """Evolve the system with a given pulse."""
    fidelities = []
    for state in _states:
        psi = state.copy()
        for i in range(N_STEPS):
            H = _build_hamiltonian(I_pulse[i], Q_pulse[i], detuning, scale)
            U = expm(-1j * H * DT)
            psi = U @ psi
        target = np.array([state[1], state[0], 0], dtype=complex)
        fid = np.abs(np.vdot(target, psi))**2
        fidelities.append(fid)
    mean_fid = np.mean(fidelities)
    return 1.0 - mean_fid

def _score_params(params):
    """Score a set of parameters across all 9 cases."""
    # params: [sigma, amp_scale, phase, c0, c1, ..., c9]
    sigma = params[0]
    amp_scale = params[1]
    phase = params[2]
    c = params[3:]
    
    if sigma <= 0.5 or sigma > 20.0:
        return 1.0
    if abs(amp_scale) > 2.0:
        return 1.0
    if np.any(np.abs(c) > 1.0):
        return 1.0
        
    # Generate envelope
    A = np.pi * amp_scale / (sigma * np.sqrt(2.0 * np.pi))
    envelope = A * np.exp(-(t_mid - center)**2 / (2.0 * sigma**2))
    
    # Generate modulation
    modulation = _cheb_basis @ c
    
    # Pulse components
    I_pulse = envelope * modulation * np.cos(phase)
    Q_pulse = envelope * modulation * np.sin(phase)
    
    # Apply amplitude bound
    amp = np.sqrt(I_pulse**2 + Q_pulse**2)
    mask = amp > AMPLITUDE_BOUND
    if np.any(mask):
        scale_factor = AMPLITUDE_BOUND / amp[mask]
        I_pulse[mask] *= scale_factor
        Q_pulse[mask] *= scale_factor
    
    # Fix slew rate violations
    for i in range(N_STEPS - 1):
        dI = I_pulse[i+1] - I_pulse[i]
        dQ = Q_pulse[i+1] - Q_pulse[i]
        slew = np.sqrt(dI**2 + dQ**2) / DT
        if slew > SLEW_BOUND:
            factor = SLEW_BOUND * DT / np.sqrt(dI**2 + dQ**2)
            I_pulse[i+1] = I_pulse[i] + dI * factor
            Q_pulse[i+1] = Q_pulse[i] + dQ * factor
    
    # From zero to first
    slew0 = np.sqrt(I_pulse[0]**2 + Q_pulse[0]**2) / DT
    if slew0 > SLEW_BOUND:
        factor = SLEW_BOUND * DT / np.sqrt(I_pulse[0]**2 + Q_pulse[0]**2)
        I_pulse[0] *= factor
        Q_pulse[0] *= factor
    
    # From last to zero
    slewn = np.sqrt(I_pulse[-1]**2 + Q_pulse[-1]**2) / DT
    if slewn > SLEW_BOUND:
        factor = SLEW_BOUND * DT / np.sqrt(I_pulse[-1]**2 + Q_pulse[-1]**2)
        I_pulse[-1] *= factor
        Q_pulse[-1] *= factor
    
    infidelities = []
    for det in detunings:
        for sc in scales:
            inf = _evolve_pulse(I_pulse, Q_pulse, det, sc)
            infidelities.append(inf)
    
    inf_arr = np.array(infidelities)
    score = np.mean(inf_arr) + np.max(inf_arr)
    return score

def _generate_pulse(params):
    """Generate the final (80, 2) pulse array from parameters."""
    sigma = params[0]
    amp_scale = params[1]
    phase = params[2]
    c = params[3:]
    
    A = np.pi * amp_scale / (sigma * np.sqrt(2.0 * np.pi))
    envelope = A * np.exp(-(t_mid - center)**2 / (2.0 * sigma**2))
    modulation = _cheb_basis @ c
    
    I_pulse = envelope * modulation * np.cos(phase)
    Q_pulse = envelope * modulation * np.sin(phase)
    
    # Apply amplitude bound
    amp = np.sqrt(I_pulse**2 + Q_pulse**2)
    mask = amp > AMPLITUDE_BOUND
    if np.any(mask):
        scale_factor = AMPLITUDE_BOUND / amp[mask]
        I_pulse[mask] *= scale_factor
        Q_pulse[mask] *= scale_factor
    
    # Fix slew rate violations
    for i in range(N_STEPS - 1):
        dI = I_pulse[i+1] - I_pulse[i]
        dQ = Q_pulse[i+1] - Q_pulse[i]
        slew = np.sqrt(dI**2 + dQ**2) / DT
        if slew > SLEW_BOUND:
            factor = SLEW_BOUND * DT / np.sqrt(dI**2 + dQ**2)
            I_pulse[i+1] = I_pulse[i] + dI * factor
            Q_pulse[i+1] = Q_pulse[i] + dQ * factor
    
    # From zero to first
    slew0 = np.sqrt(I_pulse[0]**2 + Q_pulse[0]**2) / DT
    if slew0 > SLEW_BOUND:
        factor = SLEW_BOUND * DT / np.sqrt(I_pulse[0]**2 + Q_pulse[0]**2)
        I_pulse[0] *= factor
        Q_pulse[0] *= factor
    
    # From last to zero
    slewn = np.sqrt(I_pulse[-1]**2 + Q_pulse[-1]**2) / DT
    if slewn > SLEW_BOUND:
        factor = SLEW_BOUND * DT / np.sqrt(I_pulse[-1]**2 + Q_pulse[-1]**2)
        I_pulse[-1] *= factor
        Q_pulse[-1] *= factor
    
    return np.column_stack([I_pulse, Q_pulse])

def pulse(target_gate, hardware_model):
    """Main entry point. Returns (80, 2) array of I,Q amplitudes."""
    if target_gate != 'X_pi':
        return np.zeros((80, 2))
    
    start_time = time.time()
    deadline = start_time + 300.0
    
    # Initial guess: Gaussian with sigma=6.67, amp_scale=1, phase=0, c0=1, others=0
    x0 = np.zeros(N_CHEB + 3)
    x0[0] = 40.0 / 6.0  # sigma
    x0[1] = 1.0         # amp_scale
    x0[2] = 0.0         # phase
    x0[3] = 1.0         # c0
    
    bounds = [
        (3.0, 15.0),    # sigma
        (0.5, 1.5),     # amp_scale
        (-0.5, 0.5),    # phase
    ] + [(-1.0, 1.0)] * N_CHEB
    
    best_params = x0.copy()
    best_score = _score_params(x0)
    
    # Use L-BFGS-B for fast gradient-based optimization
    try:
        res = minimize(
            _score_params,
            best_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-5, 'maxfun': 2000}
        )
        if res.fun < best_score:
            best_params = res.x
            best_score = res.fun
    except Exception:
        pass
    
    # Generate final pulse
    pulse_array = _generate_pulse(best_params)
    
    # Final validity check
    amp = np.sqrt(pulse_array[:, 0]**2 + pulse_array[:, 1]**2)
    if np.any(amp > AMPLITUDE_BOUND + 1e-9):
        mask = amp > AMPLITUDE_BOUND
        factor = AMPLITUDE_BOUND / amp[mask]
        pulse_array[mask, 0] *= factor
        pulse_array[mask, 1] *= factor
    
    # Slew rate check
    for i in range(len(pulse_array) - 1):
        dI = pulse_array[i+1, 0] - pulse_array[i, 0]
        dQ = pulse_array[i+1, 1] - pulse_array[i, 1]
        slew = np.sqrt(dI**2 + dQ**2) / DT
        if slew > SLEW_BOUND + 1e-9:
            factor = SLEW_BOUND * DT / np.sqrt(dI**2 + dQ**2)
            pulse_array[i+1, 0] = pulse_array[i, 0] + dI * factor
            pulse_array[i+1, 1] = pulse_array[i, 1] + dQ * factor
    
    # From zero to first
    slew0 = np.sqrt(pulse_array[0, 0]**2 + pulse_array[0, 1]**2) / DT
    if slew0 > SLEW_BOUND + 1e-9:
        factor = SLEW_BOUND * DT / np.sqrt(pulse_array[0, 0]**2 + pulse_array[0, 1]**2)
        pulse_array[0, 0] *= factor
        pulse_array[0, 1] *= factor
    
    # From last to zero
    slewn = np.sqrt(pulse_array[-1, 0]**2 + pulse_array[-1, 1]**2) / DT
    if slewn > SLEW_BOUND + 1e-9:
        factor = SLEW_BOUND * DT / np.sqrt(pulse_array[-1, 0]**2 + pulse_array[-1, 1]**2)
        pulse_array[-1, 0] *= factor
        pulse_array[-1, 1] *= factor
    
    return pulse_array
# EVOLVE-BLOCK-END
