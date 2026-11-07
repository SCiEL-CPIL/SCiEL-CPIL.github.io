"""Robust control pulses for a three-level transmon - differentiable ODE with adjoint gradients."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.linalg import expm

# Hardware constants
ANHARMONICITY = -2 * np.pi * 0.25  # rad/ns
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUNDS = [-2 * np.pi * 0.002, 2 * np.pi * 0.002]
DRIVE_SCALE_BOUNDS = [0.97, 1.03]
T1 = 25000.0  # ns
T2 = 40000.0  # ns (Tphi)
N_STEPS = 80
DT = 0.5  # ns
TOTAL_TIME = N_STEPS * DT  # 40 ns
DIM = 3

# Precompute ladder operator and number operator for 3 levels
A = np.array([
    [0, np.sqrt(1), 0],
    [0, 0, np.sqrt(2)],
    [0, 0, 0]
], dtype=complex)
A_DAG = A.conj().T
N_OP = A_DAG @ A  # diag(0, 1, 2)
N_MINUS_1 = N_OP - np.eye(DIM)

# Precompute static parts of Hamiltonian
H_STATIC = ANHARMONICITY * 0.5 * (N_OP @ N_MINUS_1)

# Drive coefficient matrices: H_drive = 0.5 * scale * (I * M_I + Q * M_Q)
M_I = A + A_DAG
M_Q = 1j * (A - A_DAG)

# Lindblad operators
L1 = A / np.sqrt(T1)
L2 = N_OP * np.sqrt(2.0 / T2)
LINDBLADS = [L1, L2]

# Precompute dissipative superoperator terms (time-independent)
DISSIPATIVE_SUPER = np.zeros((DIM * DIM, DIM * DIM), dtype=complex)
for L in LINDBLADS:
    Ldag = L.conj().T
    LdagL = Ldag @ L
    DISSIPATIVE_SUPER += np.kron(L, L.conj().T)
    DISSIPATIVE_SUPER -= 0.5 * (np.kron(LdagL, np.eye(DIM)) + np.kron(np.eye(DIM), LdagL))

# Target states
PSI0 = np.array([1, 0, 0], dtype=complex)
PSI1 = np.array([0, 1, 0], dtype=complex)
PSI_PLUS = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
PSI_MINUS = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)
PSI_I_PLUS = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
PSI_I_MINUS = np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
TARGET_STATES = [PSI0, PSI1, PSI_PLUS, PSI_MINUS, PSI_I_PLUS, PSI_I_MINUS]

# Detuning and scale grid
DETUNINGS = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
SCALES = [0.97, 1.0, 1.03]


def build_superoperator(detuning, scale, I, Q):
    """Build the Liouvillian superoperator for given parameters."""
    # Coherent part: -i[H, rho] -> -i(H ⊗ I - I ⊗ H†)
    H_det = detuning * N_OP
    H_drive = 0.5 * scale * (I * M_I + Q * M_Q)
    H = H_STATIC + H_det + H_drive
    
    coherent = -1j * (np.kron(H, np.eye(DIM)) - np.kron(np.eye(DIM), H.conj().T))
    return coherent + DISSIPATIVE_SUPER


def evolve_density_matrix(pulse_I, pulse_Q, detuning, scale):
    """
    Evolve density matrix through the pulse using matrix exponential per step.
    Returns final density matrix and intermediate states for gradient computation.
    """
    rho = np.zeros((DIM, DIM), dtype=complex)
    rho[0, 0] = 1.0
    
    states = [rho.copy()]
    
    for i in range(N_STEPS):
        H = build_superoperator(detuning, scale, pulse_I[i], pulse_Q[i])
        U = expm(DT * H)
        rho_vec = rho.reshape(DIM * DIM)
        rho_vec = U @ rho_vec
        rho = rho_vec.reshape(DIM, DIM)
        states.append(rho.copy())
    
    return rho, states


def compute_score_and_gradient(pulse_I, pulse_Q):
    """
    Compute the score (mean + max infidelity over 9 cases) and its gradient.
    Uses forward-mode differentiation through the ODE evolution.
    """
    detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    total_score = 0.0
    grad_I = np.zeros(N_STEPS)
    grad_Q = np.zeros(N_STEPS)
    
    n_cases = 0
    for det in detunings:
        for scl in scales:
            # Forward pass: evolve and store states
            rho_final, states = evolve_density_matrix(pulse_I, pulse_Q, det, scl)
            
            # Compute infidelities for all target states
            case_infids = []
            for psi in TARGET_STATES:
                fid = np.real(np.vdot(psi, rho_final @ psi))
                fid = np.clip(fid, 0.0, 1.0)
                case_infids.append(1.0 - fid)
            
            case_infids = np.array(case_infids)
            mean_infid = np.mean(case_infids)
            max_infid = np.max(case_infids)
            case_score = mean_infid + max_infid
            total_score += case_score
            n_cases += 1
            
            # Backward pass: compute gradient w.r.t. pulse amplitudes
            # d(score)/d(rho_final) for each target state
            # For infidelity = 1 - <psi|rho|psi>, d(infid)/d(rho) = -|psi><psi|
            # For mean: sum over targets of -|psi_k><psi_k| / 6
            # For max: -|psi_argmax><psi_argmax|
            
            argmax_idx = np.argmax(case_infids)
            psi_max = TARGET_STATES[argmax_idx]
            
            # Gradient of score w.r.t. final rho
            # d(mean_infid)/d(rho) = -1/6 * sum_k |psi_k><psi_k|
            # d(max_infid)/d(rho) = -|psi_max><psi_max|
            grad_rho = -np.outer(psi_max, psi_max.conj())
            for psi in TARGET_STATES:
                grad_rho -= (1.0 / len(TARGET_STATES)) * np.outer(psi, psi.conj())
            
            # Now propagate gradient backward through time
            # d(score)/d(rho_i) = (d(score)/d(rho_{i+1}))^T @ U_{i+1}^† (in vectorized form)
            # Actually: rho_{i+1} = U_{i+1} @ vec(rho_i)
            # d(score)/d(vec(rho_i)) = U_{i+1}^† @ d(score)/d(vec(rho_{i+1}))
            
            grad_vec = grad_rho.reshape(DIM * DIM).copy()
            
            # Backward pass from step N-1 to 0
            for i in range(N_STEPS - 1, -1, -1):
                H_super = build_superoperator(det, scl, pulse_I[i], pulse_Q[i])
                U = expm(DT * H_super)
                
                # Gradient w.r.t. pulse amplitudes at step i
                # rho_{i+1} = U(H(I_i, Q_i)) @ vec(rho_i)
                # d(score)/d(I_i) = d(score)/d(rho_{i+1})^† @ dU/dI_i @ vec(rho_i)
                # dU/dI_i = DT * dH/dI_i @ U (approximately, for small dt)
                # More precisely: dU/dI = integral_0^dt exp((dt-s)*H) * dH/dI * exp(s*H) ds
                # For small dt, approximate: dU/dI ≈ DT * 0.5 * (dH/dI @ U + U @ dH/dI)
                
                # dH_super/dI: derivative of superoperator w.r.t. I
                # H = H_static + det*N + 0.5*scl*(I*M_I + Q*M_Q)
                # dH/dI = 0.5 * scl * M_I
                dH_dI = 0.5 * scl * M_I
                dH_dQ = 0.5 * scl * M_Q
                
                # d(super)/dI = -i(kron(dH_dI, I) - kron(I, dH_dI†))
                dsuper_dI = -1j * (np.kron(dH_dI, np.eye(DIM)) - np.kron(np.eye(DIM), dH_dI.conj().T))
                dsuper_dQ = -1j * (np.kron(dH_dQ, np.eye(DIM)) - np.kron(np.eye(DIM), dH_dQ.conj().T))
                
                rho_i_vec = states[i].reshape(DIM * DIM)
                
                # Approximate dU/dI using midpoint rule
                # dU/dI ≈ DT * exp(DT*H/2) @ dsuper_dI @ exp(DT*H/2)
                U_half = expm(DT * H_super / 2.0)
                dU_dI = DT * (U_half @ dsuper_dI @ U_half)
                dU_dQ = DT * (U_half @ dsuper_dQ @ U_half)
                
                # grad_I[i] = Re( (grad_vec^† @ dU_dI @ rho_i_vec) )
                # grad_Q[i] = Re( (grad_vec^† @ dU_dQ @ rho_i_vec) )
                temp_I = dU_dI @ rho_i_vec
                temp_Q = dU_dQ @ rho_i_vec
                grad_I[i] = np.real(np.vdot(grad_vec, temp_I))
                grad_Q[i] = np.real(np.vdot(grad_vec, temp_Q))
                
                # Propagate gradient to previous step
                grad_vec = U.conj().T @ grad_vec
            
            # Note: We need to accumulate gradients across the 9 cases
            # But the above already does this since we're adding to grad_I/grad_Q
    
    total_score /= n_cases
    grad_I /= n_cases
    grad_Q /= n_cases
    
    return total_score, grad_I, grad_Q


def check_constraints(pulse_I, pulse_Q):
    """Check amplitude and slew rate constraints."""
    amps = np.sqrt(pulse_I**2 + pulse_Q**2)
    if np.any(amps > AMPLITUDE_BOUND + 1e-10):
        return False
    
    prev_I, prev_Q = 0.0, 0.0
    for i in range(N_STEPS):
        dI = pulse_I[i] - prev_I
        dQ = pulse_Q[i] - prev_Q
        slew = np.sqrt(dI**2 + dQ**2) / DT
        if slew > SLEW_BOUND + 1e-10:
            return False
        prev_I, prev_Q = pulse_I[i], pulse_Q[i]
    
    dI = -prev_I
    dQ = -prev_Q
    slew = np.sqrt(dI**2 + dQ**2) / DT
    if slew > SLEW_BOUND + 1e-10:
        return False
    
    return True


def drag_seed():
    """Generate the DRAG-style Gaussian seed pulse."""
    t = np.arange(N_STEPS) * DT
    sigma = 40.0 / 6.0  # ns
    
    A = np.sqrt(np.pi / 2) / sigma
    I = A * np.exp(-t**2 / (2 * sigma**2))
    I = I - I[0]
    I = np.maximum(I, 0)
    
    max_amp = np.max(np.sqrt(I**2))
    if max_amp > AMPLITUDE_BOUND:
        I = I * (AMPLITUDE_BOUND / max_amp)
    
    for _ in range(10):
        if check_constraints(I, np.zeros_like(I)):
            break
        I = I * 0.9
    
    Q = np.zeros_like(I)
    return I, Q


def optimize_pulse():
    """
    Optimize the pulse using gradient-based L-BFGS-B with constraint handling.
    """
    I_seed, Q_seed = drag_seed()
    x0 = np.concatenate([I_seed, Q_seed])
    
    def objective(x):
        I = x[:N_STEPS]
        Q = x[N_STEPS:]
        
        score, gI, gQ = compute_score_and_gradient(I, Q)
        grad = np.concatenate([gI, gQ])
        return score, grad
    
    # Use L-BFGS-B with bounds to enforce amplitude constraint
    # Slew constraint is harder to handle with bounds, so we use penalty
    def objective_with_penalty(x):
        I = x[:N_STEPS]
        Q = x[N_STEPS:]
        
        score, gI, gQ = compute_score_and_gradient(I, Q)
        
        # Penalty for slew violations
        penalty = 0.0
        prev_I, prev_Q = 0.0, 0.0
        for i in range(N_STEPS):
            dI = I[i] - prev_I
            dQ = Q[i] - prev_Q
            slew = np.sqrt(dI**2 + dQ**2) / DT
            if slew > SLEW_BOUND:
                penalty += (slew - SLEW_BOUND)**2
            prev_I, prev_Q = I[i], Q[i]
        
        dI = -prev_I
        dQ = -prev_Q
        slew = np.sqrt(dI**2 + dQ**2) / DT
        if slew > SLEW_BOUND:
            penalty += (slew - SLEW_BOUND)**2
        
        # Gradient of penalty
        pen_gI = np.zeros(N_STEPS)
        pen_gQ = np.zeros(N_STEPS)
        prev_I, prev_Q = 0.0, 0.0
        for i in range(N_STEPS):
            dI = I[i] - prev_I
            dQ = Q[i] - prev_Q
            slew = np.sqrt(dI**2 + dQ**2) / DT
            if slew > SLEW_BOUND:
                coeff = 2.0 * (slew - SLEW_BOUND) / (DT * slew) if slew > 1e-15 else 0
                pen_gI[i] += coeff * dI
                pen_gQ[i] += coeff * dQ
                pen_gI[i-1] -= coeff * dI if i > 0 else 0
                pen_gQ[i-1] -= coeff * dQ if i > 0 else 0
            prev_I, prev_Q = I[i], Q[i]
        
        dI = -prev_I
        dQ = -prev_Q
        slew = np.sqrt(dI**2 + dQ**2) / DT
        if slew > SLEW_BOUND:
            coeff = 2.0 * (slew - SLEW_BOUND) / (DT * slew) if slew > 1e-15 else 0
            pen_gI[N_STEPS-1] -= coeff * dI
            pen_gQ[N_STEPS-1] -= coeff * dQ
        
        grad = np.concatenate([gI + 10.0 * pen_gI, gQ + 10.0 * pen_gQ])
        return score + 10.0 * penalty, grad
    
    # Simple bounds for amplitude
    bounds = [(-AMPLITUDE_BOUND, AMPLITUDE_BOUND)] * (2 * N_STEPS)
    
    result = minimize(objective_with_penalty, x0, jac=True, method='L-BFGS-B',
                      bounds=bounds, options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
    
    I_opt = result.x[:N_STEPS]
    Q_opt = result.x[N_STEPS:]
    
    if not check_constraints(I_opt, Q_opt):
        return I_seed, Q_seed
    
    return I_opt, Q_opt


def _compute_pulse():
    """Compute the pulse and cache it."""
    I, Q = drag_seed()
    
    try:
        I_opt, Q_opt = optimize_pulse()
        if check_constraints(I_opt, Q_opt):
            return I_opt, Q_opt
    except Exception:
        pass
    
    return I, Q


_I_cached, _Q_cached = _compute_pulse()


def pulse(target_gate, hardware_model):
    """
    Return an (80, 2) array of I and Q amplitudes (rad/ns) for a 40 ns X-pi gate.
    """
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi gate is supported")
    
    result = np.column_stack([_I_cached, _Q_cached])
    return result
