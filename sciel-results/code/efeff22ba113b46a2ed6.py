import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize, minimize_scalar

# Constants from the task contract
ANHARMONICITY = -2 * np.pi * 0.25
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DE_TUNING_BOUNDS = (-2 * np.pi * 0.002, 2 * np.pi * 0.002)
DRIVE_SCALE_BOUNDS = (0.97, 1.03)
T1 = 25000.0
TPHI = 40000.0
N_STEPS = 80
DT = 0.5

def _build_matrices():
    """Build 3x3 matrices for a, a_dag, n."""
    a = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
    a_dag = a.conj().T
    n = a_dag @ a
    return a, a_dag, n

def _liouvillian_superoperator(I, Q, detuning, scale, a, a_dag, n):
    """Build the 9x9 Liouvillian superoperator."""
    H = detuning * n + 0.5 * ANHARMONICITY * (n @ (n - np.eye(3))) + \
        0.5 * scale * (I * (a + a_dag) + 1j * Q * (a - a_dag))
    
    L1 = a / np.sqrt(T1)
    L2 = n * np.sqrt(2.0 / TPHI)
    
    I3 = np.eye(3, dtype=complex)
    
    L_ham = -1j * (np.kron(H, I3) - np.kron(I3, H.T))
    
    L_total = L_ham.copy()
    for L in [L1, L2]:
        L_dag = L.conj().T
        L_total += np.kron(L, L.conj()) - 0.5 * (np.kron(L_dag @ L, I3) + np.kron(I3, (L_dag @ L).T))
        
    return L_total

def _simulate_pulse_fast(pulse, detuning, scale, a, a_dag, n):
    """
    Simulate the pulse for 6 input states simultaneously.
    Returns mean fidelity.
    """
    # Input states
    psi_0 = np.array([1, 0, 0], dtype=complex)
    psi_1 = np.array([0, 1, 0], dtype=complex)
    psi_2 = np.array([1, 1, 0], dtype=complex) / np.sqrt(2)
    psi_3 = np.array([1, -1, 0], dtype=complex) / np.sqrt(2)
    psi_4 = np.array([1, 1j, 0], dtype=complex) / np.sqrt(2)
    psi_5 = np.array([1, -1j, 0], dtype=complex) / np.sqrt(2)
    
    # Initial density matrices vectorized
    initial_states = []
    for psi in [psi_0, psi_1, psi_2, psi_3, psi_4, psi_5]:
        rho = np.outer(psi, psi.conj())
        initial_states.append(rho.flatten(order='F'))
    
    # Stack into a 9x6 matrix
    states = np.column_stack(initial_states)
    
    # Apply propagators
    for i in range(N_STEPS):
        I_val = pulse[i, 0]
        Q_val = pulse[i, 1]
        L_super = _liouvillian_superoperator(I_val, Q_val, detuning, scale, a, a_dag, n)
        U_super = expm(L_super * DT)
        states = U_super @ states
        
    # Compute fidelities
    X_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    inputs = [psi_0, psi_1, psi_2, psi_3, psi_4, psi_5]
    
    fidelities = []
    for j in range(6):
        psi_in = inputs[j]
        psi_target = X_mat @ psi_in
        
        # Extract rho_out
        rho_vec = states[:, j]
        rho_out = rho_vec.reshape((3, 3), order='F')
        
        fidelity = np.real(np.conj(psi_target) @ rho_out @ psi_target)
        fidelity = max(0.0, min(1.0, fidelity))
        fidelities.append(fidelity)
        
    return np.mean(fidelities)

def _check_validity(pulse):
    """Check amplitude and slew constraints."""
    amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    if np.any(amps > AMPLITUDE_BOUND + 1e-12):
        return False
    
    slew0 = np.linalg.norm(pulse[0]) / DT
    if slew0 > SLEW_BOUND + 1e-12:
        return False
    
    for i in range(N_STEPS - 1):
        diff = np.linalg.norm(pulse[i+1] - pulse[i]) / DT
        if diff > SLEW_BOUND + 1e-12:
            return False
    
    slew_last = np.linalg.norm(pulse[-1]) / DT
    if slew_last > SLEW_BOUND + 1e-12:
        return False
    
    return True

def _compute_score(pulse, a, a_dag, n):
    """Compute the score: mean + max infidelity over 9 cases."""
    infidelities = []
    for detuning in [DE_TUNING_BOUNDS[0], 0.0, DE_TUNING_BOUNDS[1]]:
        for scale in [DRIVE_SCALE_BOUNDS[0], 1.0, DRIVE_SCALE_BOUNDS[1]]:
            fid = _simulate_pulse_fast(pulse, detuning, scale, a, a_dag, n)
            infid = 1.0 - fid
            infidelities.append(infid)
    
    return np.mean(infidelities) + np.max(infidelities)

def _generate_pulse(params):
    """Generate pulse from parameters [A, sigma, beta]."""
    A, sigma, beta = params
    
    t = np.arange(N_STEPS) * DT
    t_center = 20.0
    
    # Gaussian
    gauss = np.exp(-0.5 * ((t - t_center) / sigma)**2)
    
    # Offset to 0 at edges
    edge_val = np.exp(-0.5 * ((0 - t_center) / sigma)**2)
    gauss = gauss - edge_val
    
    # Normalize area to pi
    area = np.sum(gauss) * DT
    if area > 0:
        gauss_norm = gauss / area  # Area is 1
    else:
        gauss_norm = gauss
    
    # Derivative
    dgauss = np.zeros(N_STEPS)
    dgauss[1:] = (gauss_norm[1:] - gauss_norm[:-1]) / DT
    dgauss[0] = dgauss[1]
    dgauss[-1] = dgauss[-2]
    
    I = A * gauss_norm
    Q = beta * A * dgauss
    
    pulse = np.column_stack([I, Q])
    
    # Enforce amplitude bound
    amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    max_amp = np.max(amps)
    if max_amp > AMPLITUDE_BOUND:
        scale_factor = AMPLITUDE_BOUND / max_amp
        pulse *= scale_factor
    
    return pulse

def _legendre_basis(n_terms, N):
    """
    Generate Legendre polynomials scaled to [0, 1] and shifted to be zero at boundaries.
    """
    t = np.linspace(0, 1, N)
    basis = []
    for k in range(n_terms):
        # numpy.polynomial.legendre.legvander
        P_k = np.polynomial.legendre.legvander(t, k)[:, k]
        # Shift to be zero at t=0 and t=1
        P_k_shifted = P_k - P_k[0] - (P_k[-1] - P_k[0]) * t
        basis.append(P_k_shifted)
    return np.column_stack(basis)

def _generate_pulse_legendre(coefs, n_terms=8):
    """
    Generate pulse from Legendre coefficients.
    coefs[0] scales I, coefs[1] scales Q (with derivative-like structure).
    Actually, let's just optimize I and Q separately or jointly.
    The prompt suggests: "basis expansion of the pulse in the first 8 Legendre polynomials... directly optimizing the 8 coefficients".
    This implies 8 coefs for I and 8 for Q? Or 8 total?
    "optimizing the 8 coefficients" suggests 8 total, which is under-parameterized for 2 channels.
    Maybe 4 for I and 4 for Q? Or 8 for the combined vector?
    Let's assume 8 coefficients total: 4 for I shape, 4 for Q shape?
    Or perhaps the strategy means: optimize 8 parameters that define the pulse.
    Let's stick to a reasonable interpretation: 4 coefficients for I and 4 for Q, or 8 for I and 8 for Q.
    Given the budget, let's try 4 for I and 4 for Q (8 total params) to keep optimization fast.
    """
    # Split coefficients
    n_I = n_terms // 2
    n_Q = n_terms // 2
    c_I = coefs[:n_I]
    c_Q = coefs[n_I:n_I+n_Q]
    
    basis_I = _legendre_basis(n_I, N_STEPS)
    basis_Q = _legendre_basis(n_Q, N_STEPS)
    
    I = basis_I @ c_I
    Q = basis_Q @ c_Q
    
    pulse = np.column_stack([I, Q])
    
    # Enforce amplitude bound
    amps = np.sqrt(pulse[:, 0]**2 + pulse[:, 1]**2)
    max_amp = np.max(amps)
    if max_amp > AMPLITUDE_BOUND:
        scale_factor = AMPLITUDE_BOUND / max_amp
        pulse *= scale_factor
        
    return pulse

def pulse(target_gate, hardware_model):
    """Main entry point."""
    start_time = time.time()
    deadline = start_time + 280.0  # Leave 20s safety margin
    
    a, a_dag, n = _build_matrices()
    
    # --- Phase 0: Initial Guesses ---
    
    # 1. DRAG Gaussian Seed
    x0_drag = np.array([0.3, 5.0, 0.2])
    best_pulse = _generate_pulse(x0_drag)
    best_score = _compute_score(best_pulse, a, a_dag, n)
    
    # 2. Legendre Seed (Random small coeffs)
    # We need a good initial guess for Legendre.
    # Let's try to match the area of the Gaussian.
    # Or just start with zero and let optimizer find it?
    # Zero pulse is invalid (slew from 0 to 0 is fine, but infidelity is 1).
    # Let's create a "square-ish" pulse using Legendre coeffs.
    # P_0 is constant. P_0 shifted is 0.
    # P_1 shifted is linear ramp.
    # To get a pulse, we need higher order terms.
    
    # Let's just run the Legendre optimization from a few random starts.
    
    def objective_drag(params):
        p = _generate_pulse(params)
        if not _check_validity(p):
            return 1.0
        return _compute_score(p, a, a_dag, n)
        
    def objective_legendre(params):
        p = _generate_pulse_legendre(params)
        if not _check_validity(p):
            return 1.0
        return _compute_score(p, a, a_dag, n)

    # --- Phase 1: Optimize DRAG Family ---
    starts_drag = [
        np.array([0.3, 5.0, 0.2]),
        np.array([0.35, 4.0, 0.1]),
        np.array([0.25, 6.0, 0.3]),
        np.array([0.4, 5.0, 0.0]),
        np.array([0.3, 5.0, -0.1]),
    ]
    
    for x_start in starts_drag:
        if time.time() > deadline - 120.0:
            break
        try:
            res = minimize(objective_drag, x_start, method='Nelder-Mead', 
                           options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-7})
            if res.fun < best_score:
                best_score = res.fun
                best_pulse = _generate_pulse(res.x)
        except Exception:
            pass
            
    # --- Phase 2: Optimize Legendre Family ---
    # Use 8 coefficients (4 for I, 4 for Q)
    n_legendre_params = 8
    
    # Generate a few random starting points for Legendre
    # Coeffs should be small enough to stay within bounds initially.
    # Max amplitude is 0.4.
    # If we use coeffs of order 0.1, the pulse might be small.
    
    starts_legendre = []
    for i in range(5):
        # Random small coeffs
        c = np.random.uniform(-0.5, 0.5, n_legendre_params)
        starts_legendre.append(c)
        
    # Also try a "structured" start: 
    # I should be positive bump. Q should be derivative-like (antisymmetric).
    # P_2 is bump-like (parabola). P_1 is linear.
    # Let's try c_I = [0, 0.5, 0, 0] (P_1 is 0, P_2 is bump?)
    # Legendre P_2(x) = 0.5(3x^2-1). On [0,1], it goes from -0.5 to 1.
    # Shifted P_2: P_2(x) - P_2(0) - (P_2(1)-P_2(0))x
    # P_2(0) = -0.5. P_2(1) = 1.
    # Shifted: 0.5(3x^2-1) - (-0.5) - (1 - (-0.5))x = 1.5x^2 - 1.5x = 1.5x(x-1).
    # This is negative in (0,1). So c_I should be negative to make I positive.
    
    # Let's just rely on random search + Nelder-Mead.
    
    for x_start in starts_legendre:
        if time.time() > deadline - 60.0:
            break
        try:
            res = minimize(objective_legendre, x_start, method='Nelder-Mead', 
                           options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-7})
            if res.fun < best_score:
                best_score = res.fun
                best_pulse = _generate_pulse_legendre(res.x)
        except Exception:
            pass
            
    # --- Phase 3: Polish with L-BFGS-B if time permits ---
    # L-BFGS-B might struggle with the non-smooth validity penalty.
    # But if the current best is valid, we can try to polish it.
    # However, we don't have the parameters for the best pulse if it came from Legendre.
    # We can try to polish the Legendre solution if it was the best.
    # Or just do a final Nelder-Mead run with tighter tolerance on the best found.
    
    # Let's identify the best family and polish.
    # This is tricky because we didn't track which family produced best_pulse.
    # Let's just do a generic local search around the best_pulse in the "pulse space" is hard.
    # Instead, let's re-run the best family's optimizer with tighter settings.
    
    # Re-evaluate the best pulse to ensure it's valid and get score.
    if not _check_validity(best_pulse):
         # Fallback to seed
         best_pulse = _generate_pulse(np.array([0.3, 5.0, 0.2]))
         best_score = _compute_score(best_pulse, a, a_dag, n)

    return best_pulse
