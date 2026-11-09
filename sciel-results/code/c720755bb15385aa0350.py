"""Robust control pulses for a three-level transmon."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.linalg as sla
from scipy.integrate import solve_ivp

# --- Constants ---
DT = 0.5
N_STEPS = 80
GATE_TIME = 40.0
T1 = 25000.0
Tphi = 40000.0
ANHARMONICITY = -2 * np.pi * 0.25
AMP_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_VALUES = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
SCALE_VALUES = [0.97, 1.0, 1.03]

# Precompute matrices
_a = np.array([[0, 1, 0],
               [0, 0, np.sqrt(2)],
               [0, 0, 0]], dtype=complex)
_adag = _a.conj().T
_n = _adag @ _a

# Lindblad operators
_L1 = _a / np.sqrt(T1)
_L2 = _n * np.sqrt(2.0 / Tphi)

def _build_hamiltonian(detuning, scale, I, Q):
    """Build the 3-level Hamiltonian."""
    H = (detuning * _n + 0.5 * ANHARMONICITY * _n @ (_n - np.eye(3))
         + 0.5 * scale * (I * (_a + _adag) + 1j * Q * (_a - _adag)))
    return H

def _build_liouvillian(detuning, scale, I, Q):
    """Build the Liouvillian superoperator for density matrix evolution."""
    H = _build_hamiltonian(detuning, scale, I, Q)
    # Liouvillian: L(rho) = -i[H, rho] + sum_k (L_k rho L_k^dag - 0.5 {L_k^dag L_k, rho})
    # We'll construct it as a matrix acting on vec(rho)
    # vec(A rho B) = (B^T ⊗ A) vec(rho)
    
    # Coherent part: -i(H ⊗ I - I ⊗ H^T)
    I3 = np.eye(3, dtype=complex)
    L_coh = -1j * (np.kron(H, I3) - np.kron(I3, H.T))
    
    # Dissipative part
    L_diss = np.zeros((9, 9), dtype=complex)
    for Lk in [_L1, _L2]:
        Lkd = Lk.conj().T
        LdL = Lkd @ Lk
        # Lk rho Lk^dag
        term1 = np.kron(Lkd.T, Lk)
        # -0.5 {Lk^dag Lk, rho} = -0.5 (LdL rho + rho LdL)
        term2 = -0.5 * (np.kron(I3, LdL.T) + np.kron(LdL, I3))
        L_diss += term1 + term2
    
    return L_coh + L_diss

def _evolve_state(pulse_I, pulse_Q, detuning, scale):
    """Evolve the density matrix through the pulse sequence."""
    rho = np.zeros((3, 3), dtype=complex)
    rho[0, 0] = 1.0  # Start in |0>
    
    for i in range(N_STEPS):
        I_val = pulse_I[i]
        Q_val = pulse_Q[i]
        L = _build_liouvillian(detuning, scale, I_val, Q_val)
        # Evolve: rho(t+dt) = exp(L * dt) rho(t)
        # Use matrix exponential
        expL = sla.expm(L * DT)
        rho_vec = rho.flatten()
        rho_vec = expL @ rho_vec
        rho = rho_vec.reshape(3, 3)
    
    return rho

def _fidelity_X(rho):
    """Compute fidelity to X gate: F = |<0|rho|1>|^2 + |<1|rho|0>|^2 ... 
    Actually, for a unitary X gate, the ideal output from |psi> is X|psi>.
    Fidelity = <psi| X^dag rho X |psi>
    For the 4 states, we average.
    
    Actually, the task says: "mean fidelity to X over |0>, |1>, (|0> +- |1>)/sqrt(2), (|0> +- i|1>)/sqrt(2)"
    Fidelity for state |psi> is |<psi| X^dag rho X |psi>|? No.
    
    Standard fidelity: F = <psi_out| rho |psi_out> where |psi_out> = X|psi_in>
    So F = (X|psi>)^dag rho (X|psi>) = <psi| X^dag rho X |psi>
    """
    X = np.array([[0, 1, 0],
                  [1, 0, 0],
                  [0, 0, 0]], dtype=complex)
    
    states = []
    for psi in [np.array([1, 0, 0], dtype=complex),
                np.array([0, 1, 0], dtype=complex),
                (np.array([1, 0, 0], dtype=complex) + np.array([0, 1, 0], dtype=complex)) / np.sqrt(2),
                (np.array([1, 0, 0], dtype=complex) - np.array([0, 1, 0], dtype=complex)) / np.sqrt(2),
                (np.array([1, 0, 0], dtype=complex) + 1j * np.array([0, 1, 0], dtype=complex)) / np.sqrt(2),
                (np.array([1, 0, 0], dtype=complex) - 1j * np.array([0, 1, 0], dtype=complex)) / np.sqrt(2)]:
        psi_out = X @ psi
        F = np.real(np.vdot(psi_out, rho @ psi_out))
        F = max(0, min(1, F))  # Clip to [0,1]
        states.append(F)
    
    return np.mean(states)

def _compute_infidelity(pulse_I, pulse_Q):
    """Compute infidelity = 1 - mean fidelity over 9 cases."""
    fidelities = []
    for det in DETUNING_VALUES:
        for scale in SCALE_VALUES:
            rho = _evolve_state(pulse_I, pulse_Q, det, scale)
            F = _fidelity_X(rho)
            fidelities.append(F)
    
    mean_F = np.mean(fidelities)
    return 1.0 - mean_F

def _is_valid(pulse_I, pulse_Q):
    """Check amplitude and slew constraints."""
    # Amplitude constraint
    amp = np.sqrt(pulse_I**2 + pulse_Q**2)
    if np.any(amp > AMP_BOUND + 1e-12):
        return False
    
    # Slew constraint: ||p[i+1] - p[i]|| / dt <= 0.12
    # Including from zero to step 0 and from last step to zero
    for i in range(N_STEPS - 1):
        dI = pulse_I[i+1] - pulse_I[i]
        dQ = pulse_Q[i+1] - pulse_Q[i]
        slew = np.sqrt(dI**2 + dQ**2) / DT
        if slew > SLEW_BOUND + 1e-12:
            return False
    
    # From zero to step 0
    dI = pulse_I[0] - 0.0
    dQ = pulse_Q[0] - 0.0
    slew = np.sqrt(dI**2 + dQ**2) / DT
    if slew > SLEW_BOUND + 1e-12:
        return False
    
    # From last step to zero
    dI = 0.0 - pulse_I[-1]
    dQ = 0.0 - pulse_Q[-1]
    slew = np.sqrt(dI**2 + dQ**2) / DT
    if slew > SLEW_BOUND + 1e-12:
        return False
    
    return True

def _seed_pulse():
    """Generate the DRAG-style Gaussian seed pulse."""
    t = np.arange(N_STEPS) * DT
    sigma = 40.0 / 6.0
    # Gaussian envelope
    g = np.exp(-0.5 * (t - GATE_TIME/2.0)**2 / sigma**2)
    # Offset to 0 at edges
    g[0] = 0.0
    g[-1] = 0.0
    # Scale to have area pi
    area = np.sum(g) * DT
    if area > 0:
        g *= np.pi / area
    
    # DRAG: add derivative term (beta * dg/dt)
    # For a simple seed, just use the Gaussian in I, Q=0
    pulse_I = g
    pulse_Q = np.zeros(N_STEPS)
    
    # Ensure constraints
    # Check amplitude
    amp = np.sqrt(pulse_I**2 + pulse_Q**2)
    if np.max(amp) > AMP_BOUND:
        pulse_I *= AMP_BOUND / np.max(amp)
        pulse_Q *= AMP_BOUND / np.max(amp)
    
    # Check slew
    for i in range(N_STEPS - 1):
        dI = pulse_I[i+1] - pulse_I[i]
        dQ = pulse_Q[i+1] - pulse_Q[i]
        slew = np.sqrt(dI**2 + dQ**2) / DT
        if slew > SLEW_BOUND:
            # Reduce the pulse
            scale = SLEW_BOUND * DT / slew
            pulse_I[i+1] = pulse_I[i] + dI * scale
            pulse_Q[i+1] = pulse_Q[i] + dQ * scale
    
    return pulse_I, pulse_Q

def _optimize_pulse(pulse_I, pulse_Q, max_iters=200, time_limit=300):
    """Simple gradient-free optimization using random perturbations."""
    best_I = pulse_I.copy()
    best_Q = pulse_Q.copy()
    best_infid = _compute_infidelity(best_I, best_Q)
    
    start_time = time.time()
    np.random.seed(42)
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Random perturbation
        sigma_I = 0.05
        sigma_Q = 0.05
        dI = np.random.normal(0, sigma_I, N_STEPS)
        dQ = np.random.normal(0, sigma_Q, N_STEPS)
        
        new_I = best_I + dI
        new_Q = best_Q + dQ
        
        # Enforce amplitude constraint
        amp = np.sqrt(new_I**2 + new_Q**2)
        mask = amp > AMP_BOUND
        if np.any(mask):
            new_I[mask] *= AMP_BOUND / amp[mask]
            new_Q[mask] *= AMP_BOUND / amp[mask]
        
        # Enforce slew constraint (simple projection)
        for i in range(N_STEPS - 1):
            dI_step = new_I[i+1] - new_I[i]
            dQ_step = new_Q[i+1] - new_Q[i]
            slew = np.sqrt(dI_step**2 + dQ_step**2) / DT
            if slew > SLEW_BOUND:
                scale = SLEW_BOUND * DT / slew
                new_I[i+1] = new_I[i] + dI_step * scale
                new_Q[i+1] = new_Q[i] + dQ_step * scale
        
        # Check edge slews
        dI0 = new_I[0]
        dQ0 = new_Q[0]
        slew0 = np.sqrt(dI0**2 + dQ0**2) / DT
        if slew0 > SLEW_BOUND:
            scale = SLEW_BOUND * DT / slew0
            new_I[0] = dI0 * scale
            new_Q[0] = dQ0 * scale
        
        dIl = new_I[-1]
        dQl = new_Q[-1]
        slewl = np.sqrt(dIl**2 + dQl**2) / DT
        if slewl > SLEW_BOUND:
            scale = SLEW_BOUND * DT / slewl
            new_I[-1] = dIl * scale
            new_Q[-1] = dQl * scale
        
        if not _is_valid(new_I, new_Q):
            continue
        
        new_infid = _compute_infidelity(new_I, new_Q)
        
        if new_infid < best_infid:
            best_I = new_I.copy()
            best_Q = new_Q.copy()
            best_infid = new_infid
    
    return best_I, best_Q, best_infid

def pulse(target_gate, hardware_model):
    """Return an (80, 2) array of I and Q amplitudes for an X-pi gate."""
    if target_gate != 'X_pi':
        # Return zero pulse for unknown gates
        return np.zeros((80, 2))
    
    start_time = time.time()
    
    # Generate seed pulse
    pulse_I, pulse_Q = _seed_pulse()
    
    # Optimize
    pulse_I, pulse_Q, infid = _optimize_pulse(pulse_I, pulse_Q, max_iters=100, time_limit=280)
    
    # Return as (80, 2) array
    result = np.column_stack((pulse_I, pulse_Q))
    
    return result
# EVOLVE-BLOCK-END
