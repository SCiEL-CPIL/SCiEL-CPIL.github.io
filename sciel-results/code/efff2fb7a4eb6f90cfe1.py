"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import expm

# --- Constants ---
N_STEPS = 80
DT = 0.5
T_TOTAL = N_STEPS * DT  # 40 ns

# Hardware parameters
ANHARMONICITY = -2 * np.pi * 0.25
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNING_BOUND = 2 * np.pi * 0.002
SCALE_MIN = 0.97
SCALE_MAX = 1.03
T1 = 25000.0
Tphi = 40000.0

# --- Precomputed matrices ---
# Ladder operator a in |0>, |1>, |2> basis
_a = np.zeros((3, 3), dtype=complex)
_a[0, 1] = 1.0
_a[1, 2] = np.sqrt(2.0)
_adag = _a.conj().T
_n = _adag @ _a
_nn_minus_1 = _n @ (_n - np.eye(3))

# Lindblad operators
_L1 = _a / np.sqrt(T1)
_L2 = _n * np.sqrt(2.0 / Tphi)

# Precompute Liouvillian components
_I3 = np.eye(3)

# Coherent part structure: -i[H, rho] = -i(H⊗I - I⊗H^T)
# We'll build this on the fly since H depends on I, Q

# Dissipative part is constant
_L1diss = np.kron(_L1.conj().T, _L1) - 0.5 * (np.kron(_I3, _L1.conj().T @ _L1) + np.kron((_L1.conj().T @ _L1).T, _I3))
_L2diss = np.kron(_L2.conj().T, _L2) - 0.5 * (np.kron(_I3, _L2.conj().T @ _L2) + np.kron((_L2.conj().T @ _L2).T, _I3))
_L_dissipative = _L1diss + _L2diss

# Drive part: 0.5 * scale * (I*(a+a†) + i*Q*(a-a†))
_a_plus_adag = _a + _adag
_a_minus_adag = _a - _adag

def _build_liouvillian(detuning, scale, I_val, Q_val):
    """Build the 9x9 Liouvillian for given parameters."""
    # Drive term
    drive = 0.5 * scale * (I_val * _a_plus_adag + 1j * Q_val * _a_minus_adag)
    # Full Hamiltonian
    H = detuning * _n + 0.5 * ANHARMONICITY * _nn_minus_1 + drive
    
    # Coherent part
    L_coherent = -1j * (np.kron(_I3, H) - np.kron(H.T, _I3))
    
    return L_coherent + _L_dissipative

def _simulate_pulse(I, Q, detuning, scale):
    """Simulate pulse and return final density matrix for |0> initial state."""
    rho_vec = np.zeros(9, dtype=complex)
    rho_vec[0] = 1.0  # |0><0|
    
    for i in range(N_STEPS):
        L = _build_liouvillian(detuning, scale, I[i], Q[i])
        U = expm(L * DT)
        rho_vec = U @ rho_vec
    
    rho_final = rho_vec.reshape((3, 3))
    rho_final = 0.5 * (rho_final + rho_final.conj().T)
    trace = np.trace(rho_final)
    if abs(trace) > 1e-15:
        rho_final /= trace
    return rho_final

def _compute_score(I, Q):
    """Compute score across all 9 cases."""
    detunings = [-DETUNING_BOUND, 0.0, DETUNING_BOUND]
    scales = [SCALE_MIN, 1.0, SCALE_MAX]
    
    # Target states for X gate
    X = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
    
    initial_states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2),
    ]
    
    infidelities = []
    
    for det in detunings:
        for scl in scales:
            case_infids = []
            for psi in initial_states:
                # Simulate
                rho = np.outer(psi, psi.conj())
                rho_vec = rho.flatten()
                
                for i in range(N_STEPS):
                    L = _build_liouvillian(det, scl, I[i], Q[i])
                    U = expm(L * DT)
                    rho_vec = U @ rho_vec
                
                rho_final = rho_vec.reshape((3, 3))
                rho_final = 0.5 * (rho_final + rho_final.conj().T)
                trace = np.trace(rho_final)
                if abs(trace) > 1e-15:
                    rho_final /= trace
                
                # Fidelity to X|psi>
                psi_target = X @ psi
                fid = np.abs(np.vdot(psi_target, rho_final @ psi_target)) ** 2
                case_infids.append(1.0 - fid)
            
            infidelities.append(np.mean(case_infids))
    
    return np.mean(infidelities) + np.max(infidelities)

def _check_validity(I, Q):
    """Check amplitude and slew constraints."""
    amplitudes = np.sqrt(I**2 + Q**2)
    if np.any(amplitudes > AMPLITUDE_BOUND + 1e-10):
        return False
    
    # Slew: zero to first
    if np.sqrt(I[0]**2 + Q[0]**2) / DT > SLEW_BOUND + 1e-10:
        return False
    
    # Between steps
    for i in range(N_STEPS - 1):
        dI = I[i+1] - I[i]
        dQ = Q[i+1] - Q[i]
        if np.sqrt(dI**2 + dQ**2) / DT > SLEW_BOUND + 1e-10:
            return False
    
    # Last to zero
    if np.sqrt(I[-1]**2 + Q[-1]**2) / DT > SLEW_BOUND + 1e-10:
        return False
    
    return True

def _generate_drag_pulse(alpha=1.0, beta=0.0, sigma_factor=1.0):
    """Generate DRAG-style pulse with adjustable parameters."""
    t = np.arange(N_STEPS) * DT
    t_center = T_TOTAL / 2.0
    sigma = T_TOTAL / 6.0 * sigma_factor
    
    # Main Gaussian
    gauss = np.exp(-0.5 * ((t - t_center) / sigma) ** 2)
    
    # Offset to 0 at edges
    gauss_offset = gauss - gauss[0]
    gauss_offset = np.maximum(gauss_offset, 0)
    
    # Normalize area to pi
    area = np.sum(gauss_offset) * DT
    if area > 0:
        gauss_offset = gauss_offset * (np.pi / area)
    
    # Derivative for DRAG
    deriv = -((t - t_center) / sigma**2) * gauss_offset
    
    I = alpha * gauss_offset
    Q = beta * deriv
    
    # Ensure validity
    for _ in range(20):
        if _check_validity(I, Q):
            break
        I *= 0.95
        Q *= 0.95
    
    return I, Q

def _legendre_basis(n_steps, n_terms):
    """Generate Legendre polynomial basis for pulse representation."""
    t = np.linspace(-1, 1, n_steps)
    basis = np.zeros((n_steps, n_terms))
    for k in range(n_terms):
        # Use numpy's Legendre polynomials
        from numpy.polynomial import legendre
        p = legendre.legvander(t, k)
        basis[:, k] = p[:, k]
    return basis

def _pulse_from_legendre(coeffs, n_steps, alpha_scale=1.0, beta_scale=0.0):
    """Generate pulse from Legendre coefficients."""
    t = np.linspace(-1, 1, n_steps)
    from numpy.polynomial import legendre
    
    I = np.zeros(n_steps)
    Q = np.zeros(n_steps)
    
    for k, c in enumerate(coeffs[:n_terms]):
        p = legendre.legval(t, [0]*k + [c])
        I += p
    
    # Add DRAG component if beta_scale != 0
    if abs(beta_scale) > 1e-10:
        # Derivative of the I pulse
        dI_dt = np.gradient(I, t)
        Q = beta_scale * dI_dt
    
    return I * alpha_scale, Q * alpha_scale

def _check_legendre_validity(I, Q):
    return _check_validity(I, Q)

def _objective_legendre(params):
    """Objective function for Legendre optimization."""
    n_terms = len(params) // 2
    alpha = params[0]
    beta = params[1]
    coeffs = params[2:]
    
    I, Q = _pulse_from_legendre(coeffs, N_STEPS, alpha, beta)
    
    if not _check_validity(I, Q):
        return 1e6
    
    return _compute_score(I, Q)

def _generate_initial_guess_legendre(n_terms=6):
    """Generate initial guess for Legendre coefficients."""
    # Start with a simple Gaussian-like profile
    t = np.linspace(-1, 1, N_STEPS)
    gauss = np.exp(-3 * t**2)
    
    # Fit to Legendre polynomials
    from numpy.polynomial import legendre
    coeffs = np.zeros(n_terms)
    for k in range(n_terms):
        p = legendre.legvander(t, k)
        coeffs[k] = np.dot(p[:, k], gauss) / np.dot(p[:, k], p[:, k])
    
    return np.concatenate([[1.0, 0.0], coeffs])

def pulse(target_gate, hardware_model):
    """
    Main entry point.
    Returns an (80, 2) array of I and Q amplitudes.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # 280s safety margin before 300s limit
    
    if target_gate != 'X_pi':
        return np.zeros((N_STEPS, 2))
    
    # Generate initial DRAG pulse
    I, Q = _generate_drag_pulse(alpha=1.0, beta=0.0)
    best_I = I.copy()
    best_Q = Q.copy()
    best_score = _compute_score(best_I, best_Q)
    
    # Try different DRAG parameters
    for alpha in [0.9, 0.95, 1.0, 1.05]:
        for beta in [0.0, 0.05, 0.1, 0.15, 0.2]:
            if time.time() > deadline - 30.0:
                break
            I, Q = _generate_drag_pulse(alpha=alpha, beta=beta)
            if _check_validity(I, Q):
                score = _compute_score(I, Q)
                if score < best_score:
                    best_score = score
                    best_I = I.copy()
                    best_Q = Q.copy()
    
    # Try Legendre optimization
    if time.time() < deadline - 60.0:
        try:
            from scipy.optimize import minimize
            
            n_terms = 5
            x0 = _generate_initial_guess_legendre(n_terms)
            
            result = minimize(
                _objective_legendre,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4}
            )
            
            if result.success and result.fun < best_score:
                params = result.x
                alpha = params[0]
                beta = params[1]
                coeffs = params[2:]
                I, Q = _pulse_from_legendre(coeffs, N_STEPS, alpha, beta)
                if _check_validity(I, Q):
                    best_I = I.copy()
                    best_Q = Q.copy()
                    best_score = result.fun
        except Exception:
            pass
    
    return np.column_stack([best_I, best_Q])
# EVOLVE-BLOCK-END
