"""Structural change: DRAG-informed Legendre-basis pulse with constrained local optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, Bounds, LinearConstraint
from scipy.linalg import expm
from scipy.integrate import solve_ivp

# --- Constants from the task ---
DT = 0.5          # ns
N_STEPS = 80
T_TOTAL = N_STEPS * DT  # 40 ns
AMPL_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUND = 2 * np.pi * 0.002
SCALE_LO, SCALE_HI = 0.97, 1.03
T1 = 25000.0
T2 = 40000.0  # Tphi
ANHARMONICITY = -2 * np.pi * 0.25

# 3-level system matrices
# a = [[0, sqrt(1), 0], [0, 0, sqrt(2)], [0, 0, 0]]
sqrt1 = np.sqrt(1.0)
sqrt2 = np.sqrt(2.0)
a_mat = np.array([[0, sqrt1, 0],
                  [0, 0, sqrt2],
                  [0, 0, 0]], dtype=complex)
ad_mat = a_mat.conj().T
n_mat = ad_mat @ a_mat  # diag(0, 1, 2)

# Lindblad operators
L1 = a_mat / np.sqrt(T1)
L2 = n_mat * np.sqrt(2.0 / T2)

def build_liouvillian(detuning, scale, I_val, Q_val):
    """Build the Liouvillian superoperator for one step."""
    # Hamiltonian
    H = (detuning * n_mat 
         + 0.5 * ANHARMONICITY * n_mat @ (n_mat - 3 * np.eye(3))  # n(n-1) with n=diag(0,1,2): n(n-1)=diag(0,0,2)
         + 0.5 * scale * (I_val * (a_mat + ad_mat) + 1j * Q_val * (a_mat - ad_mat)))
    
    # Wait, let me re-read: H = detuning n + 0.5 anharmonicity n (n-1) + 0.5 scale (I (a+ad) + iQ(a-ad))
    # n(n-1) for n=diag(0,1,2) gives diag(0, 0, 2)
    H = (detuning * n_mat 
         + 0.5 * ANHARMONICITY * (n_mat @ (n_mat - np.eye(3)))
         + 0.5 * scale * (I_val * (a_mat + ad_mat) + 1j * Q_val * (a_mat - ad_mat)))
    
    # Liouvillian: d|psi><psi|/dt = -i[H, rho] + sum_k (L_k rho L_k^dag - 0.5{L_k^dag L_k, rho})
    # In vectorized form: L_vec = -i(Kron(I,H) - Kron(H,I)) + ...
    # But for state vector evolution: d|psi>/dt = -i H |psi> (coherent part)
    # For Lindblad on density matrix, we need the full superoperator.
    # However, for fidelity we can track the state vector if we ignore decoherence,
    # but the task says "each step exponentiates the Liouvillian" so we need the full dynamics.
    
    # Let's use the density matrix approach with the Liouvillian superoperator.
    # dim = 3, so rho is 3x3, vectorized to 9x1.
    
    # Coherent part: -i[H, rho] = -i(H rho - rho H)
    # In vec form: -i(Kron(I, H^T) - Kron(H, I)) vec(rho)  ... actually it's -i(Kron(I,H) - Kron(H,I))^T depending on convention
    # Standard: vec(A X B) = (B^T ⊗ A) vec(X)
    # So vec(H rho) = (I ⊗ H) vec(rho), vec(rho H) = (H^T ⊗ I) vec(rho)
    # -i[H, rho] -> -i((I ⊗ H) - (H^T ⊗ I)) vec(rho)
    
    I3 = np.eye(3, dtype=complex)
    coherent = -1j * (np.kron(I3, H) - np.kron(H.T, I3))
    
    # Dissipative part
    dissipative = np.zeros((9, 9), dtype=complex)
    for L in [L1, L2]:
        Ldag = L.conj().T
        # L rho Ldag: vec(L rho Ldag) = (Ldag^T ⊗ L) vec(rho)
        # -0.5{Ldag L, rho}: vec(Ldag L rho) = (I ⊗ Ldag L) vec(rho), vec(rho Ldag L) = ((Ldag L)^T ⊗ I) vec(rho)
        dissipative += (np.kron(Ldag.T, L) 
                       - 0.5 * (np.kron(I3, Ldag @ L) + np.kron((Ldag @ L).T, I3)))
    
    L_total = coherent + dissipative
    return L_total

def evolve_state(pulse_IQ, detuning, scale):
    """Evolve the density matrix through all steps. Returns final rho (3x3)."""
    rho = np.zeros((3, 3), dtype=complex)
    rho[0, 0] = 1.0  # Start in |0>
    
    for i in range(N_STEPS):
        I_val = pulse_IQ[i, 0]
        Q_val = pulse_IQ[i, 1]
        L = build_liouvillian(detuning, scale, I_val, Q_val)
        U = expm(L * DT)
        rho_vec = U @ rho.flatten()
        rho = rho_vec.reshape(3, 3)
    
    return rho

def fidelity_with_target(rho, target_state):
    """Fidelity F = <psi_target| rho |psi_target>."""
    psi = target_state.flatten()
    return np.real(np.conj(psi) @ rho @ psi)

def compute_infidelity(pulse_IQ):
    """Compute mean infidelity over 9 cases and 4 initial states."""
    detunings = [-DETUNING_BOUND, 0.0, DETUNING_BOUND]
    scales = [SCALE_LO, 1.0, SCALE_HI]
    
    # Target: X gate maps |0>->|1>, |1>->|0>, so X|0> = |1>, X|1> = |0>
    # For initial state |psi>, target is X|psi>
    # |0> = [1,0,0], |1> = [0,1,0]
    # X|0> = |1> = [0,1,0]
    # X|1> = |0> = [1,0,0]
    # X(|0>+|1>)/sqrt(2) = (|1>+|0>)/sqrt(2) = same
    # X(|0>-|1>)/sqrt(2) = (|1>-|0>)/sqrt(2) = -(|0>-|1>)/sqrt(2)
    # X(|0>+i|1>)/sqrt(2) = (|1>+i|0>)/sqrt(2)
    # X(|0>-i|1>)/sqrt(2) = (|1>-i|0>)/sqrt(2)
    
    psi0 = np.array([1, 0, 0], dtype=complex)
    psi1 = np.array([0, 1, 0], dtype=complex)
    psi_plus = (psi0 + psi1) / np.sqrt(2)
    psi_minus = (psi0 - psi1) / np.sqrt(2)
    psi_i_plus = (psi0 + 1j * psi1) / np.sqrt(2)
    psi_i_minus = (psi0 - 1j * psi1) / np.sqrt(2)
    
    initial_states = [psi0, psi1, psi_plus, psi_minus, psi_i_plus, psi_i_minus]
    # Actually the task says 4 states: |0>, |1>, (|0>+/-|1>)/sqrt(2), (|0>+/-i|1>)/sqrt(2)
    # That's 6 states? No, it lists: |0>, |1>, (|0> +- |1>)/sqrt(2), (|0> +- i|1>)/sqrt(2)
    # The +- means both + and -, so 2 + 2 + 2 = 6 states? Or is it 4?
    # Re-reading: "fidelity to X over |0>, |1>, (|0> +- |1>)/sqrt(2), (|0> +- i|1>)/sqrt(2)"
    # I think this means 4 categories but with +- giving 6 states. Let me assume 6 states.
    # Actually wait, maybe it's 4: |0>, |1>, one superposition, one phase superposition?
    # The notation (|0> +- |1>) typically means both signs. Let me use all 6.
    
    total_infidelity = 0.0
    count = 0
    for det in detunings:
        for sc in scales:
            for psi_init in initial_states:
                # Target state is X applied to psi_init
                # X in 3-level: X|0>=|1>, X|1>=|0>, X|2>=0 (or undefined, but we only care about 0,1)
                if psi_init[0] != 0 or psi_init[1] != 0:
                    # X swaps 0 and 1
                    psi_target = np.array([psi_init[1], psi_init[0], psi_init[2]], dtype=complex)
                else:
                    psi_target = psi_init
                
                rho_final = evolve_state(pulse_IQ, det, sc)
                F = fidelity_with_target(rho_final, psi_target)
                inf = 1.0 - F
                total_infidelity += inf
                count += 1
    
    mean_inf = total_infility / count if count > 0 else 0
    return mean_inf

def compute_score(pulse_IQ):
    """Compute the score: mean + max of the 9 cases' infidelities."""
    detunings = [-DETUNING_BOUND, 0.0, DETUNING_BOUND]
    scales = [SCALE_LO, 1.0, SCALE_HI]
    
    psi0 = np.array([1, 0, 0], dtype=complex)
    psi1 = np.array([0, 1, 0], dtype=complex)
    psi_plus = (psi0 + psi1) / np.sqrt(2)
    psi_minus = (psi0 - psi1) / np.sqrt(2)
    psi_i_plus = (psi0 + 1j * psi1) / np.sqrt(2)
    psi_i_minus = (psi0 - 1j * psi1) / np.sqrt(2)
    initial_states = [psi0, psi1, psi_plus, psi_minus, psi_i_plus, psi_i_minus]
    
    case_infidelities = []
    for det in detunings:
        for sc in scales:
            case_inf = 0.0
            for psi_init in initial_states:
                psi_target = np.array([psi_init[1], psi_init[0], psi_init[2]], dtype=complex)
                rho_final = evolve_state(pulse_IQ, det, sc)
                F = fidelity_with_target(rho_final, psi_target)
                case_inf += (1.0 - F)
            case_inf /= len(initial_states)
            case_infidelities.append(case_inf)
    
    mean_inf = np.mean(case_infidelities)
    max_inf = np.max(case_infidelities)
    return mean_inf + max_inf

def is_valid(pulse_IQ):
    """Check amplitude and slew constraints."""
    # Amplitude
    amps = np.sqrt(pulse_IQ[:, 0]**2 + pulse_IQ[:, 1]**2)
    if np.any(amps > AMPL_BOUND + 1e-12):
        return False
    
    # Slew: from zero to step 1, between steps, and last step to zero
    # ||p[i+1] - p[i]|| / dt <= 0.12
    prev = np.array([0.0, 0.0])
    for i in range(N_STEPS):
        curr = pulse_IQ[i]
        if np.linalg.norm(curr - prev) / DT > SLEW_BOUND + 1e-12:
            return False
        prev = curr
    # From last step to zero
    if np.linalg.norm(prev - np.array([0.0, 0.0])) / DT > SLEW_BOUND + 1e-12:
        return False
    
    return True

def gaussian_drag_pulse(beta=0.0, sigma_factor=40.0/6.0):
    """Generate a DRAG-style Gaussian pulse."""
    t = np.arange(N_STEPS) * DT
    t_center = (N_STEPS - 1) * DT / 2.0
    sigma = sigma_factor
    # Gaussian envelope with area pi
    # Integral of A * exp(-(t-tc)^2/(2*sigma^2)) dt = A * sigma * sqrt(2*pi) = pi
    # So A = pi / (sigma * sqrt(2*pi)) = sqrt(pi/2) / sigma
    A = np.sqrt(np.pi / 2.0) / sigma
    envelope = A * np.exp(-(t - t_center)**2 / (2 * sigma**2))
    
    # Offset to 0 at edges
    # The Gaussian is already close to 0 at edges, but let's ensure
    # Actually, "offset to 0 at the edges" might mean subtract the edge value
    edge_val = A * np.exp(-(t_center)**2 / (2 * sigma**2))
    envelope = envelope - edge_val
    # Re-normalize area to pi? Or just use as is?
    # Let's keep it simple and just use the Gaussian
    
    # DRAG: I = envelope, Q = beta * d(envelope)/dt
    d_envelope = np.gradient(envelope, DT)
    
    I = envelope
    Q = beta * d_envelope
    
    return np.column_stack([I, Q])

def legendre_basis_pulse(coeffs):
    """Generate pulse from Legendre polynomial coefficients.
    coeffs is a flat array of 2 * n_terms values: [I_0, I_1, ..., Q_0, Q_1, ...]
    Each is expanded in Legendre polynomials on [-1, 1]."""
    n_terms = len(coeffs) // 2
    I_coeffs = coeffs[:n_terms]
    Q_coeffs = coeffs[n_terms:]
    
    t = np.linspace(-1, 1, N_STEPS)
    
    I = np.zeros(N_STEPS)
    Q = np.zeros(N_STEPS)
    
    for k in range(n_terms):
        # Evaluate Legendre polynomial P_k(t)
        Pk = np.polynomial.legendre.legval(t, [0]*k + [1])
        I += I_coeffs[k] * Pk
        Q += Q_coeffs[k] * Pk
    
    # Ensure zero at edges for slew constraint
    # Actually, let's just return and let the constraint handler deal with it
    return np.column_stack([I, Q])

def pulse_to_params(pulse_IQ):
    """Convert pulse to Legendre coefficients."""
    n_terms = 10  # Use 10 terms for each of I and Q
    t = np.linspace(-1, 1, N_STEPS)
    
    I_coeffs = np.polynomial.legendre.legfit(t, pulse_IQ[:, 0], n_terms - 1)
    Q_coeffs = np.polynomial.legendre.legfit(t, pulse_IQ[:, 1], n_terms - 1)
    
    return np.concatenate([I_coeffs, Q_coeffs])

def params_to_pulse(params):
    """Convert Legendre coefficients to pulse."""
    n_terms = len(params) // 2
    I_coeffs = params[:n_terms]
    Q_coeffs = params[n_terms:]
    
    t = np.linspace(-1, 1, N_STEPS)
    
    I = np.polynomial.legendre.legval(t, I_coeffs)
    Q = np.polynomial.legendre.legval(t, Q_coeffs)
    
    return np.column_stack([I, Q])

def objective(params):
    """Objective function for optimization."""
    pulse_IQ = params_to_pulse(params)
    
    if not is_valid(pulse_IQ):
        return 1e6
    
    return compute_score(pulse_IQ)

def pulse(target_gate, hardware_model):
    """Main entry point."""
    start_time = time.time()
    DEADLINE = 300.0  # seconds
    
    # Generate initial DRAG pulse
    # Analytical beta for DRAG: beta = -g * anharmonicity / (2 * Omega^2) * something
    # For a pi pulse, a common DRAG beta is around -0.1 to -0.3
    # Let's start with beta = -0.2
    pulse_IQ = gaussian_drag_pulse(beta=-0.2)
    
    # Check validity, if not valid, reduce amplitude
    while not is_valid(pulse_IQ):
        pulse_IQ *= 0.95
        if np.all(np.abs(pulse_IQ) < 1e-10):
            break
    
    best_score = compute_score(pulse_IQ)
    best_pulse = pulse_IQ.copy()
    
    # Convert to Legendre parameters for optimization
    params = pulse_to_params(best_pulse)
    n_terms = len(params) // 2
    
    # Define bounds for Legendre coefficients
    # Each coefficient should be small enough to keep amplitude < 0.4
    bound_per_coeff = AMPL_BOUND / n_terms * 2  # Rough bound
    bounds = Bounds(-bound_per_coeff, bound_per_coeff)
    
    # Try Nelder-Mead optimization
    try:
        if time.time() - start_time < DEADLINE - 60:
            result = minimize(objective, params, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8,
                                      'adaptive': True})
            if result.fun < best_score:
                best_pulse = params_to_pulse(result.x)
                best_score = result.fun
    except Exception:
        pass
    
    # Ensure final pulse is valid
    if not is_valid(best_pulse):
        # Fall back to a simple valid pulse
        best_pulse = gaussian_drag_pulse(beta=-0.1)
        while not is_valid(best_pulse):
            best_pulse *= 0.95
    
    return best_pulse
# EVOLVE-BLOCK-END
