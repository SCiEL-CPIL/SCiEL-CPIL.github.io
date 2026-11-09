"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution

# Hardware parameters
DT = 0.5  # ns
N_STEPS = 80
T_GATE = 40.0  # ns
AMP_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUNDS = [-2 * np.pi * 0.002, 2 * np.pi * 0.002]
SCALE_BOUNDS = [0.97, 1.03]
T1 = 25000.0  # ns
Tphi = 40000.0  # ns
ANHARMONICITY = -2 * np.pi * 0.25  # rad/ns

# Three-level system matrices
# a = [[0, 1, 0], [0, 0, sqrt(2)], [0, 0, 0]]
a_op = np.array([
    [0, 1, 0],
    [0, 0, np.sqrt(2)],
    [0, 0, 0]
], dtype=complex)
ad_op = a_op.conj().T
n_op = ad_op @ a_op

def liouvillian(H, L_ops):
    """Construct Liouvillian superoperator."""
    dim = H.shape[0]
    L = np.zeros((dim*dim, dim*dim), dtype=complex)
    # Coherent part: -i (H rho - rho H)
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                for l in range(dim):
                    # (H rho)_{ik} = sum_j H_{ij} rho_{jk}
                    # (rho H)_{ik} = sum_j rho_{ij} H_{jk}
                    pass
    # More efficient: use Kronecker products
    # L = -i (H ⊗ I - I ⊗ H^T) + sum_k L_k ⊗ L_k^* - 0.5 {L_k^† L_k} ⊗ I - 0.5 I ⊗ {L_k^† L_k}^T
    I = np.eye(dim, dtype=complex)
    L = -1j * (np.kron(H, I) - np.kron(I, H.T))
    for Lk in L_ops:
        Lkd = Lk.conj().T
        LkLkd = Lkd @ Lk
        L += np.kron(Lk, Lk.conj().T)
        L -= 0.5 * np.kron(LkLkd, I)
        L -= 0.5 * np.kron(I, LkLkd.T)
    return L

def hamiltonian(detuning, scale, I, Q):
    """Construct Hamiltonian for given detuning, scale, I, Q."""
    H = detuning * n_op + 0.5 * ANHARMONICITY * n_op @ (n_op - np.eye(3))
    drive = 0.5 * scale * (I * (a_op + ad_op) + 1j * Q * (a_op - ad_op))
    return H + drive

def simulate_pulse(pulse_I, pulse_Q, detuning, scale):
    """Simulate the pulse and return final density matrix."""
    # Lindblad operators
    L1 = a_op / np.sqrt(T1)
    L2 = n_op * np.sqrt(2.0 / Tphi)
    
    rho = np.zeros((3, 3), dtype=complex)
    rho[0, 0] = 1.0  # Start in |0>
    
    # We'll evolve the density matrix step by step
    for step in range(N_STEPS):
        I_val = pulse_I[step]
        Q_val = pulse_Q[step]
        
        H = hamiltonian(detuning, scale, I_val, Q_val)
        L_ops = [L1, L2]
        
        L = liouvillian(H, L_ops)
        
        # Vectorize rho
        rho_vec = rho.flatten()
        
        # Evolve for DT
        # Use matrix exponential for small steps
        # For efficiency, we can use expm
        from scipy.linalg import expm
        L_mat = L @ np.eye(9, dtype=complex)  # Ensure it's a proper matrix
        # Actually L is already 9x9
        rho_vec_new = expm(L * DT) @ rho_vec
        rho = rho_vec_new.reshape((3, 3))
    
    return rho

def fidelity(rho, target_state):
    """Calculate fidelity between density matrix and pure state."""
    target_vec = target_state.flatten()
    # Fidelity = <psi|rho|psi>
    fid = np.real(np.conj(target_vec) @ rho @ target_vec)
    return max(0.0, min(1.0, fid))

def infidelity_score(pulse_I, pulse_Q):
    """Calculate the infidelity score for a given pulse."""
    # Target states
    psi_0 = np.array([1, 0, 0], dtype=complex)
    psi_1 = np.array([0, 1, 0], dtype=complex)
    psi_plus = (psi_0 + psi_1) / np.sqrt(2)
    psi_minus = (psi_0 - psi_1) / np.sqrt(2)
    psi_plus_i = (psi_0 + 1j * psi_1) / np.sqrt(2)
    psi_minus_i = (psi_0 - 1j * psi_1) / np.sqrt(2)
    
    target_states = [psi_0, psi_1, psi_plus, psi_minus, psi_plus_i, psi_minus_i]
    
    # 9 cases: detuning in (-2pi*0.002, 0, 2pi*0.002), scale in (0.97, 1, 1.03)
    detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    fidelities = []
    
    for det in detunings:
        for sc in scales:
            for psi in target_states:
                # Evolve from |0>
                rho_final = simulate_pulse(pulse_I, pulse_Q, det, sc)
                fid = fidelity(rho_final, psi)
                fidelities.append(fid)
    
    mean_fid = np.mean(fidelities)
    max_fid = np.max(fidelities)
    # Score = mean infidelity + max infidelity = (1 - mean_fid) + (1 - max_fid)
    score = (1.0 - mean_fid) + (1.0 - max_fid)
    return score

def check_constraints(pulse_I, pulse_Q):
    """Check if pulse satisfies amplitude and slew constraints."""
    # Check amplitude: sqrt(I^2 + Q^2) <= 0.4
    for i in range(N_STEPS):
        amp = np.sqrt(pulse_I[i]**2 + pulse_Q[i]**2)
        if amp > AMP_BOUND + 1e-9:
            return False
    
    # Check slew rate
    # From zero to step 1
    slew0 = np.sqrt(pulse_I[0]**2 + pulse_Q[0]**2) / DT
    if slew0 > SLEW_BOUND + 1e-9:
        return False
    
    # Between consecutive steps
    for i in range(N_STEPS - 1):
        dI = pulse_I[i+1] - pulse_I[i]
        dQ = pulse_Q[i+1] - pulse_Q[i]
        slew = np.sqrt(dI**2 + dQ**2) / DT
        if slew > SLEW_BOUND + 1e-9:
            return False
    
    # From last step to zero
    slew_end = np.sqrt(pulse_I[-1]**2 + pulse_Q[-1]**2) / DT
    if slew_end > SLEW_BOUND + 1e-9:
        return False
    
    return True

def pulse_from_params(params):
    """Convert low-dimensional parameters to full pulse.
    
    params: [amp_scale, phase, drag_amp, drag_center, drag_width, extra_amp, extra_center, extra_width]
    We use a Gaussian for the main pulse and a derivative-like term for DRAG.
    """
    amp_scale = params[0]
    phase = params[1]
    drag_amp = params[2]
    drag_center = params[3]
    drag_width = params[4]
    
    # Time array
    t = np.arange(N_STEPS) * DT
    
    # Main Gaussian pulse
    # Area should be pi for X-pi gate
    # Gaussian: A * exp(-t^2 / (2*sigma^2))
    # Area = A * sigma * sqrt(2*pi)
    # We want area = pi, so A = pi / (sigma * sqrt(2*pi)) = sqrt(pi/2) / sigma
    
    sigma = 40.0 / 6.0  # From seed description
    # But we can vary sigma
    sigma = 3.0 + 4.0 * (params[5] if len(params) > 5 else 0.0)  # sigma in 3-7 ns
    
    A_main = np.sqrt(np.pi / 2.0) / sigma
    A_main *= amp_scale
    
    # Center the pulse
    center = T_GATE / 2.0
    t_centered = t - center
    
    main_pulse = A_main * np.exp(-t_centered**2 / (2 * sigma**2))
    
    # DRAG term: proportional to derivative of Gaussian
    # d/dt exp(-t^2/(2sigma^2)) = -t/sigma^2 * exp(-t^2/(2sigma^2))
    drag_term = drag_amp * (-t_centered / sigma**2) * np.exp(-t_centered**2 / (2 * sigma**2))
    
    # Apply phase
    pulse_I = main_pulse * np.cos(phase) + drag_term * np.cos(phase + np.pi/2)
    pulse_Q = main_pulse * np.sin(phase) + drag_term * np.sin(phase + np.pi/2)
    
    # Offset to ensure zero at edges (Gaussian already does this approximately)
    # Subtract edge values
    edge_val_I = (pulse_I[0] + pulse_I[-1]) / 2
    edge_val_Q = (pulse_Q[0] + pulse_Q[-1]) / 2
    pulse_I -= edge_val_I
    pulse_Q -= edge_val_Q
    
    # Enforce constraints by clipping
    for i in range(N_STEPS):
        amp = np.sqrt(pulse_I[i]**2 + pulse_Q[i]**2)
        if amp > AMP_BOUND:
            scale_factor = AMP_BOUND / amp
            pulse_I[i] *= scale_factor
            pulse_Q[i] *= scale_factor
    
    return pulse_I, pulse_Q

def objective(params):
    """Objective function for optimization."""
    pulse_I, pulse_Q = pulse_from_params(params)
    
    if not check_constraints(pulse_I, pulse_Q):
        return 10.0  # Penalty for constraint violation
    
    score = infidelity_score(pulse_I, pulse_Q)
    return score

def pulse(target_gate, hardware_model):
    """Generate pulse for the given target gate and hardware model."""
    # Parameter bounds
    bounds = [
        (0.5, 1.5),      # amp_scale
        (-np.pi/4, np.pi/4),  # phase
        (-0.1, 0.1),     # drag_amp
        (0, 1),          # drag_center (unused for now)
        (0, 1),          # drag_width (unused for now)
        (0, 1),          # sigma scale
    ]
    
    # Run differential evolution
    result = differential_evolution(
        objective,
        bounds,
        seed=42,
        maxiter=50,
        popsize=10,
        tol=1e-4,
        init='sobol',
        polish=False
    )
    
    best_params = result.x
    pulse_I, pulse_Q = pulse_from_params(best_params)
    
    # Final constraint check and enforcement
    for i in range(N_STEPS):
        amp = np.sqrt(pulse_I[i]**2 + pulse_Q[i]**2)
        if amp > AMP_BOUND:
            scale_factor = AMP_BOUND / amp
            pulse_I[i] *= scale_factor
            pulse_Q[i] *= scale_factor
    
    return np.column_stack([pulse_I, pulse_Q])
# EVOLVE-BLOCK-END
