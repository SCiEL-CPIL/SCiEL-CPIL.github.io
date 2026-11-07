"""
Robust control pulses for a three-level transmon.
Uses Legendre polynomial basis expansion with L-BFGS-B optimization.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import expm
from scipy.special import lpmv

# Constants
DT = 0.5  # ns
N_STEPS = 80
T_GATE = 40.0  # ns
N_BASIS = 8  # Number of Legendre polynomials

# Hardware parameters
ANHARMONICITY = -2 * np.pi * 0.25
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUNDS = (-2 * np.pi * 0.002, 2 * np.pi * 0.002)
DRIVE_SCALE_BOUNDS = (0.97, 1.03)
T1 = 25000.0  # ns
Tphi = 40000.0  # ns

# Precompute Legendre basis matrix (80 x 8)
def compute_legendre_basis():
    """Compute the Legendre polynomial basis evaluated at 80 time points."""
    # Time points in [0, 1]
    t = np.linspace(0, 1, N_STEPS)
    basis = np.zeros((N_STEPS, N_BASIS))
    for k in range(N_BASIS):
        basis[:, k] = lpmv(0, k, 2 * t - 1)  # Map [0,1] to [-1,1] for Legendre
    return basis

LEGENDRE_BASIS = compute_legendre_basis()

def coefficients_to_pulse(coeffs):
    """Convert Legendre coefficients to I and Q pulse arrays."""
    # coeffs is a flat array of length 2*N_BASIS: first N_BASIS for I, next N_BASIS for Q
    I_coeffs = coeffs[:N_BASIS]
    Q_coeffs = coeffs[N_BASIS:]
    
    I = LEGENDRE_BASIS @ I_coeffs
    Q = LEGENDRE_BASIS @ Q_coeffs
    
    return I, Q

def pulse_to_coefficients(I, Q):
    """Convert I and Q pulse arrays to Legendre coefficients (least squares)."""
    I_coeffs, _, _, _ = np.linalg.lstsq(LEGENDRE_BASIS, I, rcond=None)
    Q_coeffs, _, _, _ = np.linalg.lstsq(LEGENDRE_BASIS, Q, rcond=None)
    return np.concatenate([I_coeffs, Q_coeffs])

def check_validity(I, Q):
    """Check if the pulse satisfies amplitude and slew constraints."""
    # Amplitude constraint: sqrt(I^2 + Q^2) <= AMPLITUDE_BOUND
    amp = np.sqrt(I**2 + Q**2)
    if np.any(amp > AMPLITUDE_BOUND + 1e-10):
        return False
    
    # Slew constraint: ||p[i+1] - p[i]|| / dt <= SLEW_BOUND
    # From zero to step 1
    slew_start = np.sqrt(I[0]**2 + Q[0]**2) / DT
    if slew_start > SLEW_BOUND + 1e-10:
        return False
    
    # Between consecutive steps
    dI = np.diff(I)
    dQ = np.diff(Q)
    slews = np.sqrt(dI**2 + dQ**2) / DT
    if np.any(slews > SLEW_BOUND + 1e-10):
        return False
    
    # From last step to zero
    slew_end = np.sqrt(I[-1]**2 + Q[-1]**2) / DT
    if slew_end > SLEW_BOUND + 1e-10:
        return False
    
    return True

def project_to_valid(I, Q):
    """Project pulse to satisfy constraints by scaling down if needed."""
    # Check amplitude
    amp = np.sqrt(I**2 + Q**2)
    max_amp = np.max(amp)
    if max_amp > AMPLITUDE_BOUND:
        scale = AMPLITUDE_BOUND / max_amp
        I = I * scale
        Q = Q * scale
        amp = np.sqrt(I**2 + Q**2)
        max_amp = np.max(amp)
    
    # Check slew from zero to step 1
    slew_start = np.sqrt(I[0]**2 + Q[0]**2) / DT
    if slew_start > SLEW_BOUND:
        scale = SLEW_BOUND * DT / np.sqrt(I[0]**2 + Q[0]**2) if (I[0]**2 + Q[0]**2) > 0 else 1.0
        # Only scale the first point
        I[0] *= scale
        Q[0] *= scale
    
    # Check slew between steps
    dI = np.diff(I)
    dQ = np.diff(Q)
    slews = np.sqrt(dI**2 + dQ**2) / DT
    max_slew = np.max(slews)
    if max_slew > SLEW_BOUND:
        scale = SLEW_BOUND / max_slew
        # This is a rough projection; in practice we rely on the optimizer
        # For now, just return as is and let the optimizer handle it
    
    # Check slew from last step to zero
    slew_end = np.sqrt(I[-1]**2 + Q[-1]**2) / DT
    if slew_end > SLEW_BOUND:
        scale = SLEW_BOUND * DT / np.sqrt(I[-1]**2 + Q[-1]**2) if (I[-1]**2 + Q[-1]**2) > 0 else 1.0
        I[-1] *= scale
        Q[-1] *= scale
    
    return I, Q

def build_hamiltonian(detuning, scale, I_val, Q_val):
    """Build the 3x3 Hamiltonian matrix."""
    # a = diag(sqrt([1, 2]), 1) -> creation operator
    # a = [[0, 1, 0], [0, 0, sqrt(2)], [0, 0, 0]]
    a = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
    adag = a.conj().T
    n = adag @ a
    
    H = detuning * n + 0.5 * ANHARMONICITY * n @ (n - np.eye(3))
    H += 0.5 * scale * (I_val * (a + adag) + 1j * Q_val * (a - adag))
    
    return H

def build_liouvillian(H, T1_val, Tphi_val):
    """Build the Liouvillian superoperator for the Lindblad evolution."""
    dim = 3
    L = np.zeros((dim*dim, dim*dim), dtype=complex)
    
    # Coherent part: -i * (H ⊗ I - I ⊗ H^T)
    H_vec = H.reshape(dim*dim)
    for i in range(dim):
        for j in range(dim):
            L[i*dim+j, i*dim+j] = -1j * (H[i,j] - np.conj(H[i,j]))
    
    # More carefully: L[ρ] = -i[H, ρ] + Σ_k γ_k (L_k ρ L_k† - 1/2 {L_k† L_k, ρ})
    # Vectorize: vec(L[ρ]) = L_super * vec(ρ)
    # Coherent: -i (H ⊗ I - I ⊗ H^T) vec(ρ)
    H_kron_I = np.kron(H, np.eye(dim))
    I_kron_HT = np.kron(np.eye(dim), H.T)
    L = -1j * (H_kron_I - I_kron_HT)
    
    # Dissipative: a / sqrt(T1)
    a = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
    adag = a.conj().T
    L1 = a / np.sqrt(T1_val)
    L1_adag_L1 = L1.conj().T @ L1
    
    # n * sqrt(2/Tphi)
    n = adag @ a
    L2 = n * np.sqrt(2.0 / Tphi_val)
    L2_adag_L2 = L2.conj().T @ L2
    
    for Lk in [L1, L2]:
        Lk_adag_Lk = Lk.conj().T @ Lk
        # L_k ρ L_k† -> vec(L_k ρ L_k†) = (L_k† ⊗ L_k) vec(ρ)
        L += np.kron(Lk.conj().T, Lk)
        # -1/2 {L_k† L_k, ρ} -> -1/2 (L_k† L_k ρ + ρ L_k† L_k)
        # vec(A ρ) = (I ⊗ A) vec(ρ)
        # vec(ρ A) = (A^T ⊗ I) vec(ρ)
        L += -0.5 * np.kron(np.eye(dim), Lk_adag_Lk)
        L += -0.5 * np.kron(Lk_adag_Lk.T, np.eye(dim))
    
    return L

def simulate_pulse(I, Q, detuning, scale, n_substeps=1):
    """Simulate the pulse and return the final density matrix."""
    dim = 3
    rho = np.zeros((dim, dim), dtype=complex)
    rho[0, 0] = 1.0  # Start in |0>
    
    rho_vec = rho.reshape(dim*dim)
    
    for step in range(N_STEPS):
        I_val = I[step]
        Q_val = Q[step]
        
        H = build_hamiltonian(detuning, scale, I_val, Q_val)
        L = build_liouvillian(H, T1, Tphi)
        
        # Exponentiate the Liouvillian for dt
        L_exp = expm(L * DT)
        rho_vec = L_exp @ rho_vec
        rho = rho_vec.reshape(dim, dim)
    
    return rho

def compute_fidelity(rho, target_state):
    """Compute fidelity F = |<target|rho|target>| for pure target state."""
    # F = Tr(|target><target| rho) = <target|rho|target>
    return np.real(np.vdot(target_state, rho @ target_state))

def compute_infidelity(I, Q, detuning, scale):
    """Compute mean infidelity over the 4 test states."""
    dim = 3
    # Target: X gate swaps |0> and |1>
    # X|0> = |1>, X|1> = |0>
    # X(|0>+|1>)/sqrt(2) = (|1>+|0>)/sqrt(2)
    # X(|0>+i|1>)/sqrt(2) = (|1>-i|0>)/sqrt(2) = -i(|0>+i|1>)/sqrt(2)
    
    states = [
        np.array([1, 0, 0], dtype=complex),  # |0>
        np.array([0, 1, 0], dtype=complex),  # |1>
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),  # (|0>+|1>)/sqrt(2)
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2),  # (|0>-i|1>)/sqrt(2)
    ]
    
    # Target states after X gate
    target_states = [
        np.array([0, 1, 0], dtype=complex),  # X|0> = |1>
        np.array([1, 0, 0], dtype=complex),  # X|1> = |0>
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),  # X(|0>+|1>)/sqrt(2)
        np.array([-1j, 1, 0], dtype=complex) / np.sqrt(2),  # X(|0>-i|1>)/sqrt(2)
    ]
    
    fidelities = []
    for state, target in zip(states, target_states):
        rho = simulate_pulse(I, Q, detuning, scale)
        f = compute_fidelity(rho, target)
        f = np.clip(f, 0.0, 1.0)
        fidelities.append(f)
    
    mean_fid = np.mean(fidelities)
    infidelity = 1.0 - mean_fid
    return infidelity

def objective(coeffs):
    """Objective function for optimization: mean + max infidelity over 9 cases."""
    I, Q = coefficients_to_pulse(coeffs)
    I, Q = project_to_valid(I, Q)
    
    if not check_validity(I, Q):
        return 100.0  # Large penalty for invalid pulses
    
    detunings = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]
    
    infidelities = []
    for det in detunings:
        for sc in scales:
            inf = compute_infidelity(I, Q, det, sc)
            infidelities.append(inf)
    
    mean_inf = np.mean(infidelities)
    max_inf = np.max(infidelities)
    return mean_inf + max_inf

def initial_guess():
    """Generate an initial guess using a DRAG-style Gaussian."""
    t = np.linspace(0, 1, N_STEPS)
    # Gaussian with sigma = 40/6 ns, but we're in normalized time [0,1]
    # 40 ns gate, so sigma in normalized time = (40/6)/40 = 1/6
    sigma = 1.0 / 6.0
    center = 0.5
    I = np.exp(-0.5 * ((t - center) / sigma)**2)
    
    # Scale to have area ~ pi (integral of I dt in ns units)
    # Area in normalized time = sum(I) * dt_normalized
    # dt_normalized = 1/80
    # We want integral in ns = sum(I) * DT = pi
    # So sum(I) = pi / DT = pi / 0.5 = 2*pi
    current_area = np.sum(I) * DT
    if current_area > 0:
        I = I * (np.pi / current_area)
    
    Q = np.zeros(N_STEPS)
    
    # Project to valid
    I, Q = project_to_valid(I, Q)
    
    # Convert to coefficients
    coeffs = pulse_to_coefficients(I, Q)
    return coeffs

def optimize_pulse(deadline):
    """Optimize the pulse using L-BFGS-B."""
    start_time = time.time()
    x0 = initial_guess()
    
    best_coeffs = x0.copy()
    best_obj = objective(x0)
    
    # Bounds for coefficients: loose bounds to allow exploration
    bounds = [(-1.0, 1.0)] * (2 * N_BASIS)
    
    try:
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
        )
        
        if result.fun < best_obj:
            best_coeffs = result.x
            best_obj = result.fun
        
        # Check deadline
        if time.time() - start_time > deadline - 60:
            I, Q = coefficients_to_pulse(best_coeffs)
            I, Q = project_to_valid(I, Q)
            return I, Q
        
    except Exception as e:
        print(f"Optimization error: {e}")
    
    I, Q = coefficients_to_pulse(best_coeffs)
    I, Q = project_to_valid(I, Q)
    return I, Q

def pulse(target_gate, hardware_model):
    """
    Main entry point.
    Returns an (80, 2) array of I and Q amplitudes.
    """
    start_time = time.time()
    deadline = start_time + 300.0  # 300 second deadline
    
    if target_gate != 'X_pi':
        # Return zero pulse for invalid gates
        return np.zeros((N_STEPS, 2))
    
    I, Q = optimize_pulse(deadline)
    
    # Final validity check
    if not check_validity(I, Q):
        # Fallback to a simple valid pulse
        t = np.linspace(0, 1, N_STEPS)
        I = 0.3 * np.exp(-0.5 * ((t - 0.5) / 0.15)**2)
        Q = np.zeros(N_STEPS)
        I, Q = project_to_valid(I, Q)
    
    return np.column_stack([I, Q])
