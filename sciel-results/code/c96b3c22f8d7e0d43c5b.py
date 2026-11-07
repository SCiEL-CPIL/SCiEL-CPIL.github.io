"""Optimized pulse for three-level transmon X_pi gate with robustness."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize, minimize_scalar
import sys

_DEADLINE = time.time() + 320  # safety margin before 330s kill

# Hardware parameters
T1 = 25000.0
Tphi = 40000.0
ANHARMONICITY = -2 * np.pi * 0.25
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUNDS = (-2 * np.pi * 0.002, 2 * np.pi * 0.002)
SCALE_BOUNDS = (0.97, 1.03)
N_STEPS = 80
DT = 0.5

# Precompute ladder operators
_a = np.zeros((3, 3), dtype=complex)
_a[1, 0] = np.sqrt(1.0)
_a[2, 1] = np.sqrt(2.0)
_a_dag = _a.conj().T
_n_op = _a_dag @ _a

# Lindblad operators
_L1 = _a / np.sqrt(T1)
_L2 = _n_op * np.sqrt(2.0 / Tphi)
_L1_dag = _L1.conj().T
_L2_dag = _L2.conj().T

# Precompute L^dag L terms
_L1_dag_L1 = _L1_dag @ _L1
_L2_dag_L2 = _L2_dag @ _L2


def _build_liouvillian(detuning, scale, I, Q):
    """Build the Liouvillian superoperator for one step."""
    H = detuning * _n_op
    H += 0.5 * ANHARMONICITY * (_n_op @ (_n_op - np.eye(3)))
    H += 0.5 * scale * (I * (_a + _a_dag) + 1j * Q * (_a - _a_dag))
    
    # Liouvillian: L(rho) = -i[H, rho] + sum_k (L_k rho L_k^dag - 0.5{L_k^dag L_k, rho})
    # In vectorized form, we can use the matrix representation
    # For efficiency, we'll build it as a 9x9 matrix acting on vec(rho)
    
    # -i[H, rho] term: -i(H ⊗ I - I ⊗ H^T) in vec form
    # Actually: vec(-i[H, rho]) = -i(I ⊗ H^T - H^T ⊗ I) vec(rho) ... let me be careful
    # vec(AXB) = (B^T ⊗ A) vec(X)
    # vec([H, rho]) = vec(H rho - rho H) = (I ⊗ H^T - H^T ⊗ I) vec(rho)
    # So vec(-i[H, rho]) = -i(I ⊗ H^T - H^T ⊗ I) vec(rho)
    
    I3 = np.eye(3)
    H_t = H.T
    
    L_ham = -1j * (np.kron(I3, H_t) - np.kron(H_t, I3))
    
    # Dissipative terms: L_k rho L_k^dag - 0.5{L_k^dag L_k, rho}
    # vec(L_k rho L_k^dag) = (L_k^dag^T ⊗ L_k) vec(rho)
    # vec(L_k^dag L_k rho) = (I ⊗ (L_k^dag L_k)^T) vec(rho)
    # vec(rho L_k^dag L_k) = ((L_k^dag L_k) ⊗ I) vec(rho)
    
    L_diss = np.zeros((9, 9), dtype=complex)
    for Lk in [_L1, _L2]:
        Lk_dag = Lk.conj().T
        Lk_dag_Lk = Lk_dag @ Lk
        L_diss += np.kron(Lk_dag.T, Lk) - 0.5 * (np.kron(I3, Lk_dag_Lk.T) + np.kron(Lk_dag_Lk, I3))
    
    return L_ham + L_diss


def _simulate_pulse(pulse, detuning, scale):
    """Simulate the pulse and return the final density matrix."""
    rho = np.zeros((3, 3), dtype=complex)
    rho[0, 0] = 1.0  # Start in |0>
    
    for i in range(N_STEPS):
        I_val = pulse[i, 0]
        Q_val = pulse[i, 1]
        L = _build_liouvillian(detuning, scale, I_val, Q_val)
        # Exponentiate: rho_next = expm(L * dt) @ rho_vec
        L_mat = expm(L * DT)
        rho_vec = rho.flatten()
        rho_vec = L_mat @ rho_vec
        rho = rho_vec.reshape(3, 3)
    
    return rho


def _fidelity(rho, target_state):
    """Compute fidelity between density matrix and pure target state."""
    # Fidelity = <psi|rho|psi> for pure state
    val = np.vdot(target_state, rho @ target_state).real
    return np.clip(val, 0.0, 1.0)


def _compute_score(pulse):
    """Compute the full score for a pulse."""
    # Check validity
    for i in range(N_STEPS):
        amp = np.sqrt(pulse[i, 0]**2 + pulse[i, 1]**2)
        if amp > AMPLITUDE_BOUND + 1e-9:
            return None
    
    # Slew rate check
    prev = np.array([0.0, 0.0])
    for i in range(N_STEPS):
        curr = pulse[i]
        if np.linalg.norm(curr - prev) / DT > SLEW_BOUND + 1e-9:
            return None
        prev = curr
    # Check last step to zero
    if np.linalg.norm(prev) / DT > SLEW_BOUND + 1e-9:
        return None
    
    # Target states
    target_states = [
        np.array([1, 0, 0], dtype=complex),       # |0>
        np.array([0, 1, 0], dtype=complex),       # |1>
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),   # (|0>+|1>)/sqrt(2)
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),  # (|0>-|1>)/sqrt(2)
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),  # (|0>+i|1>)/sqrt(2)
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2), # (|0>-i|1>)/sqrt(2)
    ]
    
    # X gate: |0> -> |1>, |1> -> |0>
    # So target for input |0> is |1>, for |1> is |0>, etc.
    # X = [[0,1],[1,0]] in 2-level, extended to 3-level as |0><1| + |1><0|
    # Target states after X:
    # |0> -> |1>
    # |1> -> |0>
    # (|0>+|1>)/sqrt(2) -> (|1>+|0>)/sqrt(2) = same
    # (|0>-|1>)/sqrt(2) -> (|1>-|0>)/sqrt(2) = -(|0>-|1>)/sqrt(2)
    # (|0>+i|1>)/sqrt(2) -> (|1>+i|0>)/sqrt(2) = i(|0>-i|1>)/sqrt(2) ... let's just use X matrix
    
    X_gate = np.zeros((3, 3), dtype=complex)
    X_gate[0, 1] = 1.0
    X_gate[1, 0] = 1.0
    
    detunings = [-2*np.pi*0.002, 0.0, 2*np.pi*0.002]
    scales = [0.97, 1.0, 1.03]
    
    fidelities = []
    for det in detunings:
        for scl in scales:
            for init_state in target_states:
                rho = _simulate_pulse(pulse, det, scl)
                target = X_gate @ init_state
                f = _fidelity(rho, target)
                fidelities.append(f)
    
    fidelities = np.array(fidelities)
    infidelities = 1.0 - fidelities
    mean_inf = np.mean(infidelities)
    max_inf = np.max(infidelities)
    score = mean_inf + max_inf
    return score


def _seed_pulse():
    """Generate DRAG-style Gaussian seed pulse."""
    t = np.arange(N_STEPS) * DT
    t_center = (N_STEPS - 1) * DT / 2.0
    sigma = 40.0 / 6.0
    
    # Gaussian envelope
    envelope = np.exp(-0.5 * ((t - t_center) / sigma)**2)
    
    # Offset to 0 at edges
    # Scale so area = pi
    raw_area = np.sum(envelope) * DT
    envelope = envelope * (np.pi / raw_area)
    
    # DRAG: add derivative term
    dt_drive = 1.0  # For derivative
    drag_coeff = -0.1  # Typical DRAG coefficient for transmon
    envelope_deriv = np.gradient(envelope, DT)
    
    I_pulse = envelope
    Q_pulse = drag_coeff * envelope_deriv
    
    pulse = np.column_stack([I_pulse, Q_pulse])
    
    # Enforce constraints
    # Amplitude bound
    for i in range(N_STEPS):
        amp = np.sqrt(pulse[i, 0]**2 + pulse[i, 1]**2)
        if amp > AMPLITUDE_BOUND:
            scale = AMPLITUDE_BOUND / amp
            pulse[i, 0] *= scale
            pulse[i, 1] *= scale
    
    # Slew rate bound - simple projection
    prev = np.array([0.0, 0.0])
    for i in range(N_STEPS):
        delta = pulse[i] - prev
        norm = np.linalg.norm(delta) / DT
        if norm > SLEW_BOUND:
            pulse[i] = prev + SLEW_BOUND * DT * delta / (norm * DT)
        prev = pulse[i]
    # Final step to zero
    norm_final = np.linalg.norm(prev) / DT
    if norm_final > SLEW_BOUND:
        # Scale down the last step
        pulse[-1] = prev * (SLEW_BOUND * DT / np.linalg.norm(prev))
    
    return pulse


def pulse(target_gate, hardware_model):
    """Main entry point."""
    start_time = time.time()
    
    if target_gate != 'X_pi':
        # Return zero pulse for invalid gate
        return np.zeros((N_STEPS, 2))
    
    # Start with seed
    best_pulse = _seed_pulse()
    best_score = _compute_score(best_pulse)
    if best_score is None:
        # Seed is invalid, try to fix
        best_pulse = np.zeros((N_STEPS, 2))
        best_score = _compute_score(best_pulse)
    
    # Optimization: parameterize the pulse with a few parameters
    # Use a basis of smooth functions to keep slew rate low
    # Parameters: amplitude, phase offset, DRAG strength, detuning compensation
    
    def _param_pulse(params):
        """Generate pulse from parameters."""
        amp, phase, drag, det_comp = params
        t = np.arange(N_STEPS) * DT
        t_center = (N_STEPS - 1) * DT / 2.0
        sigma = 40.0 / 6.0
        
        envelope = amp * np.exp(-0.5 * ((t - t_center) / sigma)**2)
        envelope_deriv = np.gradient(envelope, DT)
        
        I_pulse = envelope * np.cos(phase)
        Q_pulse = envelope * np.sin(phase) + drag * envelope_deriv
        
        # Detuning compensation: add a small linear ramp to Q
        Q_pulse += det_comp * (t - t_center) / t_center
        
        pulse = np.column_stack([I_pulse, Q_pulse])
        
        # Enforce amplitude bound
        for i in range(N_STEPS):
            a = np.sqrt(pulse[i, 0]**2 + pulse[i, 1]**2)
            if a > AMPLITUDE_BOUND:
                s = AMPLITUDE_BOUND / a
                pulse[i, 0] *= s
                pulse[i, 1] *= s
        
        # Enforce slew rate
        prev = np.array([0.0, 0.0])
        for i in range(N_STEPS):
            d = pulse[i] - prev
            n = np.linalg.norm(d) / DT
            if n > SLEW_BOUND:
                pulse[i] = prev + SLEW_BOUND * DT * d / (n * DT)
            prev = pulse[i]
        # Final to zero
        nf = np.linalg.norm(prev) / DT
        if nf > SLEW_BOUND:
            pulse[-1] = prev * (SLEW_BOUND * DT / np.linalg.norm(prev))
        
        return pulse
    
    def _objective(params):
        if time.time() > _DEADLINE - 10:
            return best_score if best_score is not None else 1.0
        p = _param_pulse(params)
        s = _compute_score(p)
        if s is None:
            return 1.0  # Invalid
        return s
    
    # Optimize parameters
    initial_params = [np.pi / 2, 0.0, -0.1, 0.0]  # amp, phase, drag, det_comp
    # Actually, let's think about this more carefully.
    # The X-pi gate needs a pi rotation. The Rabi frequency is related to the drive amplitude.
    # For a 2-level system, pi pulse area = pi.
    # The seed already has area pi.
    
    # Let's use a simpler parameterization:
    # amp: peak amplitude (0 to 0.4)
    # drag: DRAG coefficient
    # det_comp: detuning compensation strength
    
    def _objective2(params):
        if time.time() > _DEADLINE - 10:
            return best_score if best_score is not None else 1.0
        amp, drag, det_comp = params
        t = np.arange(N_STEPS) * DT
        t_center = (N_STEPS - 1) * DT / 2.0
        sigma = 40.0 / 6.0
        
        envelope = amp * np.exp(-0.5 * ((t - t_center) / sigma)**2)
        # Normalize area to pi
        raw_area = np.sum(envelope) * DT
        if raw_area > 0:
            envelope = envelope * (np.pi / raw_area)
        
        envelope_deriv = np.gradient(envelope, DT)
        
        I_pulse = envelope
        Q_pulse = drag * envelope_deriv + det_comp * (t - t_center) / t_center
        
        p = np.column_stack([I_pulse, Q_pulse])
        
        # Enforce amplitude bound
        for i in range(N_STEPS):
            a = np.sqrt(p[i, 0]**2 + p[i, 1]**2)
            if a > AMPLITUDE_BOUND:
                s = AMPLITUDE_BOUND / a
                p[i, 0] *= s
                p[i, 1] *= s
        
        # Enforce slew rate
        prev = np.array([0.0, 0.0])
        for i in range(N_STEPS):
            d = p[i] - prev
            n = np.linalg.norm(d) / DT
            if n > SLEW_BOUND:
                p[i] = prev + SLEW_BOUND * DT * d / (n * DT)
            prev = p[i]
        nf = np.linalg.norm(prev) / DT
        if nf > SLEW_BOUND:
            p[-1] = prev * (SLEW_BOUND * DT / np.linalg.norm(prev))
        
        s = _compute_score(p)
        if s is None:
            return 1.0
        return s
    
    # Try optimization
    try:
        result = minimize(_objective2, [0.3, -0.1, 0.0], method='Nelder-Mead',
                         options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6})
        if result.fun < (best_score if best_score is not None else 1.0):
            best_score = result.fun
            best_pulse = _param_pulse([0.0, 0.0, 0.0, 0.0])  # Placeholder, need to regenerate
            # Regenerate properly
            amp, drag, det_comp = result.x
            t = np.arange(N_STEPS) * DT
            t_center = (N_STEPS - 1) * DT / 2.0
            sigma = 40.0 / 6.0
            envelope = amp * np.exp(-0.5 * ((t - t_center) / sigma)**2)
            raw_area = np.sum(envelope) * DT
            if raw_area > 0:
                envelope = envelope * (np.pi / raw_area)
            envelope_deriv = np.gradient(envelope, DT)
            I_pulse = envelope
            Q_pulse = drag * envelope_deriv + det_comp * (t - t_center) / t_center
            best_pulse = np.column_stack([I_pulse, Q_pulse])
            # Re-enforce constraints
            for i in range(N_STEPS):
                a = np.sqrt(best_pulse[i, 0]**2 + best_pulse[i, 1]**2)
                if a > AMPLITUDE_BOUND:
                    s = AMPLITUDE_BOUND / a
                    best_pulse[i, 0] *= s
                    best_pulse[i, 1] *= s
            prev = np.array([0.0, 0.0])
            for i in range(N_STEPS):
                d = best_pulse[i] - prev
                n = np.linalg.norm(d) / DT
                if n > SLEW_BOUND:
                    best_pulse[i] = prev + SLEW_BOUND * DT * d / (n * DT)
                prev = best_pulse[i]
            nf = np.linalg.norm(prev) / DT
            if nf > SLEW_BOUND:
                best_pulse[-1] = prev * (SLEW_BOUND * DT / np.linalg.norm(prev))
    except Exception:
        pass
    
    # Check deadline
    if time.time() > _DEADLINE - 5:
        return best_pulse
    
    # Try a few more local optimizations with different initial guesses
    for init in [[0.25, -0.05, 0.0], [0.35, -0.15, 0.001], [0.3, -0.08, -0.001]]:
        if time.time() > _DEADLINE - 10:
            break
        try:
            result = minimize(_objective2, init, method='Nelder-Mead',
                             options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-6})
            if result.fun < (best_score if best_score is not None else 1.0):
                best_score = result.fun
                amp, drag, det_comp = result.x
                t = np.arange(N_STEPS) * DT
                t_center = (N_STEPS - 1) * DT / 2.0
                sigma = 40.0 / 6.0
                envelope = amp * np.exp(-0.5 * ((t - t_center) / sigma)**2)
                raw_area = np.sum(envelope) * DT
                if raw_area > 0:
                    envelope = envelope * (np.pi / raw_area)
                envelope_deriv = np.gradient(envelope, DT)
                I_pulse = envelope
                Q_pulse = drag * envelope_deriv + det_comp * (t - t_center) / t_center
                best_pulse = np.column_stack([I_pulse, Q_pulse])
                for i in range(N_STEPS):
                    a = np.sqrt(best_pulse[i, 0]**2 + best_pulse[i, 1]**2)
                    if a > AMPLITUDE_BOUND:
                        s = AMPLITUDE_BOUND / a
                        best_pulse[i, 0] *= s
                        best_pulse[i, 1] *= s
                prev = np.array([0.0, 0.0])
                for i in range(N_STEPS):
                    d = best_pulse[i] - prev
                    n = np.linalg.norm(d) / DT
                    if n > SLEW_BOUND:
                        best_pulse[i] = prev + SLEW_BOUND * DT * d / (n * DT)
                    prev = best_pulse[i]
                nf = np.linalg.norm(prev) / DT
                if nf > SLEW_BOUND:
                    best_pulse[-1] = prev * (SLEW_BOUND * DT / np.linalg.norm(prev))
        except Exception:
            pass
    
    return best_pulse
# EVOLVE-BLOCK-END
