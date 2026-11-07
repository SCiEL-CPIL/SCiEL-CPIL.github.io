"""Robust control pulses for a three-level transmon."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Hardware parameters
ANHARMONICITY = -2 * np.pi * 0.25  # rad/ns
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUNDS = [-2 * np.pi * 0.002, 2 * np.pi * 0.002]
DRIVE_SCALE_BOUNDS = [0.97, 1.03]
T1 = 25000.0
T2 = 40000.0  # Tphi (dephasing time)

N_STEPS = 80
DT = 0.5
TOTAL_TIME = N_STEPS * DT  # 40 ns

# Deadline guard
DEADLINE_SECONDS = 300.0
START_TIME = time.time()

def build_operators():
    """Build a, a_dagger, n for 3-level system."""
    a = np.zeros((3, 3), dtype=complex)
    a[1, 0] = 1.0
    a[2, 1] = np.sqrt(2.0)
    a_dag = a.conj().T
    n = a_dag @ a
    return a, a_dag, n

a_op, a_dag_op, n_op = build_operators()

def hamiltonian(I, Q, detuning, scale):
    """Build Hamiltonian for given pulse amplitudes and hardware parameters."""
    # H = detuning * n + 0.5 * anharmonicity * n*(n-1) + 0.5 * scale * (I*(a+a^dag) + i*Q*(a-a^dag))
    H = detuning * n_op + 0.5 * ANHARMONICITY * (n_op @ (n_op - np.eye(3)))
    drive = 0.5 * scale * (I * (a_op + a_dag_op) + 1j * Q * (a_op - a_dag_op))
    H = H + drive
    return H

def liouvillian_step(pulse_I, pulse_Q, detuning, scale, step_idx):
    """
    Compute the Liouvillian superoperator for one time step.
    Returns the unitary evolution matrix for the density matrix in vectorized form.
    """
    I_val = pulse_I[step_idx]
    Q_val = pulse_Q[step_idx]
    
    H = hamiltonian(I_val, Q_val, detuning, scale)
    
    # Lindblad operators
    L1 = a_op / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / T2)
    
    # Liouvillian: dρ/dt = -i[H,ρ] + Σ(LρL† - 0.5(L†Lρ + ρL†L))
    # In vectorized form: dvec(ρ)/dt = L_super * vec(ρ)
    # L_super = -i(kron(I,H) - kron(H.T,I)) + Σ(kron(L†,L) - 0.5(kron(L†L,I) + kron(I,L†L)))
    
    n_dim = 3
    eye_n = np.eye(n_dim, dtype=complex)
    
    L_super = -1j * (np.eye(n_dim**2) @ H - H.T @ np.eye(n_dim**2))  # This is wrong, need kron
    
    # Proper Liouvillian in vectorized form:
    # For dρ/dt = -i[H,ρ], the superop is -i*(I⊗H - H^T⊗I) acting on vec(ρ)
    # Wait, let me use the standard: vec(AXB) = (B^T ⊗ A) vec(X)
    # -i[H,ρ] = -i(Hρ - ρH) = -i(HρI - IρH)
    # vec(HρI) = (I^T ⊗ H) vec(ρ) = (I ⊗ H) vec(ρ)
    # vec(IρH) = (H^T ⊗ I) vec(ρ)
    # So -i[H,ρ] -> -i((I⊗H) - (H^T⊗I)) vec(ρ)
    
    L_super = -1j * (np.kron(eye_n, H) - np.kron(H.T, eye_n))
    
    for L in [L1, L2]:
        Ldag = L.conj().T
        LdagL = Ldag @ L
        # LρL†: vec(LρL†) = ((L†)^T ⊗ L) vec(ρ) = (L.conj() ⊗ L) vec(ρ)
        # 0.5(L†Lρ + ρL†L): vec(L†Lρ) = (I⊗L†L)vec(ρ), vec(ρL†L) = ((L†L)^T ⊗ I)vec(ρ)
        L_super += np.kron(L.conj(), L) - 0.5 * (np.kron(eye_n, LdagL) + np.kron(LdagL.T, eye_n))
    
    # Exponentiate: U_step = expm(L_super * dt)
    # Use scipy's expm
    from scipy.linalg import expm
    U_step = expm(L_super * DT)
    return U_step

def evolve_pulse(pulse_I, pulse_Q, detuning, scale):
    """
    Evolve the density matrix through the entire pulse.
    Returns final density matrix for each initial state.
    """
    n_dim = 3
    eye_n = np.eye(n_dim, dtype=complex)
    
    # Initial states: |0>, |1>, (|0>+|1>)/sqrt(2), (|0>-|1>)/sqrt(2), (|0>+i|1>)/sqrt(2), (|0>-i|1>)/sqrt(2)
    psi_0 = np.array([1.0, 0.0, 0.0], dtype=complex)
    psi_1 = np.array([0.0, 1.0, 0.0], dtype=complex)
    psi_p = (psi_0 + psi_1) / np.sqrt(2.0)
    psi_m = (psi_0 - psi_1) / np.sqrt(2.0)
    psi_ip = (psi_0 + 1j * psi_1) / np.sqrt(2.0)
    psi_im = (psi_0 - 1j * psi_1) / np.sqrt(2.0)
    
    targets = [psi_0, psi_1, psi_p, psi_m, psi_ip, psi_im]
    
    # For each initial state, compute final state
    final_states = []
    for psi in targets:
        rho = np.outer(psi, psi.conj())  # Pure state density matrix
        rho_vec = rho.flatten()
        
        # Apply each step
        for step_idx in range(N_STEPS):
            U_step = liouvillian_step(pulse_I, pulse_Q, detuning, scale, step_idx)
            rho_vec = U_step @ rho_vec
        final_states.append(rho_vec.reshape((n_dim, n_dim)))
    
    return final_states

def fidelity_with_X(final_rho, target_state):
    """
    Compute fidelity between final density matrix and target state after X gate.
    The X gate maps |0> -> |1>, |1> -> |0>.
    Target state after X gate from initial state psi is X|psi>.
    Fidelity = <psi_target|rho_final|psi_target>
    """
    # X gate in 2-level subspace (ignore |2> for target)
    X_gate = np.zeros((3, 3), dtype=complex)
    X_gate[0, 1] = 1.0
    X_gate[1, 0] = 1.0
    
    target_psi = X_gate @ target_state
    # Normalize
    norm = np.sqrt(np.vdot(target_psi, target_psi).real)
    if norm > 0:
        target_psi = target_psi / norm
    
    # Fidelity for pure target: <psi|rho|psi>
    fidelity = np.vdot(target_psi, final_rho @ target_psi).real
    return np.clip(fidelity, 0.0, 1.0)

def compute_score(pulse_I, pulse_Q):
    """
    Compute the score: mean + max of infidelity over 9 cases.
    Detuning in (-2*pi*0.002, 0, 2*pi*0.002), scale in (0.97, 1, 1.03)
    """
    detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    # Initial states
    psi_0 = np.array([1.0, 0.0, 0.0], dtype=complex)
    psi_1 = np.array([0.0, 1.0, 0.0], dtype=complex)
    psi_p = (psi_0 + psi_1) / np.sqrt(2.0)
    psi_m = (psi_0 - psi_1) / np.sqrt(2.0)
    psi_ip = (psi_0 + 1j * psi_1) / np.sqrt(2.0)
    psi_im = (psi_0 - 1j * psi_1) / np.sqrt(2.0)
    
    targets = [psi_0, psi_1, psi_p, psi_m, psi_ip, psi_im]
    
    infidelities = []
    for detuning in detunings:
        for scale in scales:
            final_states = evolve_pulse(pulse_I, pulse_Q, detuning, scale)
            case_infs = []
            for idx, target_state in enumerate(targets):
                fid = fidelity_with_X(final_states[idx], target_state)
                case_infs.append(1.0 - fid)
            infidelities.append(np.mean(case_infs))
    
    infidelities = np.array(infidelities)
    score = np.mean(infidelities) + np.max(infidelities)
    return score

def is_valid_pulse(pulse_I, pulse_Q):
    """Check amplitude and slew constraints."""
    # Amplitude: sqrt(I^2 + Q^2) <= 0.4
    for i in range(N_STEPS):
        amp = np.sqrt(pulse_I[i]**2 + pulse_Q[i]**2)
        if amp > AMPLITUDE_BOUND + 1e-9:
            return False
    
    # Slew: ||p[i+1] - p[i]|| / dt <= 0.12
    # From zero into step 1
    for i in range(N_STEPS):
        if i == 0:
            p_prev = np.array([0.0, 0.0])
        else:
            p_prev = np.array([pulse_I[i-1], pulse_Q[i-1]])
        p_curr = np.array([pulse_I[i], pulse_Q[i]])
        slew = np.linalg.norm(p_curr - p_prev) / DT
        if slew > SLEW_BOUND + 1e-9:
            return False
    
    # From last step to zero
    p_last = np.array([pulse_I[N_STEPS-1], pulse_Q[N_STEPS-1]])
    slew_end = np.linalg.norm(p_last) / DT
    if slew_end > SLEW_BOUND + 1e-9:
        return False
    
    return True

def make_seed_pulse():
    """Create DRAG-style Gaussian seed pulse with area pi."""
    t = np.linspace(0, TOTAL_TIME, N_STEPS, endpoint=False) + DT/2  # Centers of steps
    sigma = 40.0 / 6.0  # ~6.67 ns
    
    # Gaussian with offset to 0 at edges
    # DRAG: I = A * exp(-t^2/(2*sigma^2)), Q = alpha * dI/dt
    # Area should be pi
    
    # Start with simple Gaussian for I
    I_raw = np.exp(-t**2 / (2 * sigma**2))
    
    # Offset to make edges ~0 (they already are small, but ensure)
    # Normalize so that sum(I * dt) = pi
    area = np.sum(I_raw) * DT
    I_scaled = I_raw * (np.pi / area)
    
    # Q component: derivative-based (DRAG)
    dI_dt = np.gradient(I_scaled, DT)
    alpha = 0.5  # DRAG alpha parameter
    Q_scaled = alpha * dI_dt
    
    # Ensure constraints
    pulse_I = I_scaled.copy()
    pulse_Q = Q_scaled.copy()
    
    # Clamp amplitudes
    for i in range(N_STEPS):
        amp = np.sqrt(pulse_I[i]**2 + pulse_Q[i]**2)
        if amp > AMPLITUDE_BOUND:
            scale = AMPLITUDE_BOUND / amp
            pulse_I[i] *= scale
            pulse_Q[i] *= scale
    
    return pulse_I, pulse_Q

def optimize_pulse():
    """Optimize the pulse using gradient-free method since computing gradients is expensive."""
    # Start with seed
    pulse_I, pulse_Q = make_seed_pulse()
    x0 = np.concatenate([pulse_I, pulse_Q])
    
    def objective(x):
        elapsed = time.time() - START_TIME
        if elapsed > DEADLINE_SECONDS - 10:
            return np.inf  # Stop searching
        
        pI = x[:N_STEPS]
        pQ = x[N_STEPS:]
        
        if not is_valid_pulse(pI, pQ):
            # Penalize violations
            penalty = 0.0
            for i in range(N_STEPS):
                amp = np.sqrt(pI[i]**2 + pQ[i]**2)
                if amp > AMPLITUDE_BOUND:
                    penalty += (amp - AMPLITUDE_BOUND) * 100
            for i in range(N_STEPS):
                if i == 0:
                    prev = np.array([0.0, 0.0])
                else:
                    prev = np.array([pI[i-1], pQ[i-1]])
                curr = np.array([pI[i], pQ[i]])
                slew = np.linalg.norm(curr - prev) / DT
                if slew > SLEW_BOUND:
                    penalty += (slew - SLEW_BOUND) * 100
            p_last = np.array([pI[N_STEPS-1], pQ[N_STEPS-1]])
            slew_end = np.linalg.norm(p_last) / DT
            if slew_end > SLEW_BOUND:
                penalty += (slew_end - SLEW_BOUND) * 100
            return penalty
        
        return compute_score(pI, pQ)
    
    # Use Nelder-Mead for optimization
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
    
    best_x = result.x
    best_I = best_x[:N_STEPS]
    best_Q = best_x[N_STEPS:]
    
    # Ensure validity
    if not is_valid_pulse(best_I, best_Q):
        # Fall back to seed
        best_I, best_Q = make_seed_pulse()
    
    return best_I, best_Q

# Precompute for speed: cache liouvillian steps
_liouvillian_cache = {}

def liouvillian_step_cached(pulse_I, pulse_Q, detuning, scale, step_idx):
    """Cached version of liouvillian_step."""
    key = (step_idx, detuning, scale, round(pulse_I[step_idx], 6), round(pulse_Q[step_idx], 6))
    if key in _liouvillian_cache:
        return _liouvillian_cache[key]
    
    U_step = liouvillian_step(pulse_I, pulse_Q, detuning, scale, step_idx)
    _liouvillian_cache[key] = U_step
    return U_step

# Override evolve_pulse to use cached version
def evolve_pulse_cached(pulse_I, pulse_Q, detuning, scale):
    """Evolve with caching."""
    n_dim = 3
    eye_n = np.eye(n_dim, dtype=complex)
    
    psi_0 = np.array([1.0, 0.0, 0.0], dtype=complex)
    psi_1 = np.array([0.0, 1.0, 0.0], dtype=complex)
    psi_p = (psi_0 + psi_1) / np.sqrt(2.0)
    psi_m = (psi_0 - psi_1) / np.sqrt(2.0)
    psi_ip = (psi_0 + 1j * psi_1) / np.sqrt(2.0)
    psi_im = (psi_0 - 1j * psi_1) / np.sqrt(2.0)
    
    targets = [psi_0, psi_1, psi_p, psi_m, psi_ip, psi_im]
    
    final_states = []
    for psi in targets:
        rho = np.outer(psi, psi.conj())
        rho_vec = rho.flatten()
        
        for step_idx in range(N_STEPS):
            U_step = liouvillian_step_cached(pulse_I, pulse_Q, detuning, scale, step_idx)
            rho_vec = U_step @ rho_vec
        final_states.append(rho_vec.reshape((n_dim, n_dim)))
    
    return final_states

# Use cached version in compute_score
def compute_score_cached(pulse_I, pulse_Q):
    detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    psi_0 = np.array([1.0, 0.0, 0.0], dtype=complex)
    psi_1 = np.array([0.0, 1.0, 0.0], dtype=complex)
    psi_p = (psi_0 + psi_1) / np.sqrt(2.0)
    psi_m = (psi_0 - psi_1) / np.sqrt(2.0)
    psi_ip = (psi_0 + 1j * psi_1) / np.sqrt(2.0)
    psi_im = (psi_0 - 1j * psi_1) / np.sqrt(2.0)
    
    targets = [psi_0, psi_1, psi_p, psi_m, psi_ip, psi_im]
    
    infidelities = []
    for detuning in detunings:
        for scale in scales:
            final_states = evolve_pulse_cached(pulse_I, pulse_Q, detuning, scale)
            case_infs = []
            for idx, target_state in enumerate(targets):
                fid = fidelity_with_X(final_states[idx], target_state)
                case_infs.append(1.0 - fid)
            infidelities.append(np.mean(case_infs))
    
    infidelities = np.array(infidelities)
    score = np.mean(infidelities) + np.max(infidelities)
    return score

def pulse(target_gate, hardware_model):
    """Main entry point."""
    global START_TIME
    START_TIME = time.time()
    
    if target_gate != 'X_pi':
        # Return zero pulse for invalid target
        return np.zeros((80, 2))
    
    # Try optimization with time limit
    try:
        # First, get a valid seed
        pulse_I, pulse_Q = make_seed_pulse()
        
        # Quick check if we have time for optimization
        elapsed = time.time() - START_TIME
        if elapsed > 5.0:
            # No time for optimization
            result = np.column_stack([pulse_I, pulse_Q])
            return result
        
        # Optimize
        _liouvillian_cache.clear()
        best_I, best_Q = optimize_pulse()
        
        # Final validation
        if not is_valid_pulse(best_I, best_Q):
            best_I, best_Q = make_seed_pulse()
        
        result = np.column_stack([best_I, best_Q])
        return result
        
    except Exception as e:
        # Fallback to seed
        pulse_I, pulse_Q = make_seed_pulse()
        result = np.column_stack([pulse_I, pulse_Q])
        return result

# EVOLVE-BLOCK-END
