"""
Fourier basis expansion pulse optimization for 3-level transmon X-pi gate.
Uses a 16-term Fourier series (8 cos, 8 sin) to represent the pulse,
allowing for more complex shapes than a simple Gaussian.
Differential Evolution is used for global optimization over the 16 coefficients.
"""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.linalg as sla
import time
from scipy.optimize import differential_evolution, minimize

# --- Constants from task ---
DT = 0.5
N_STEPS = 80
GATE_TIME = 40.0
T1 = 25000.0
Tphi = 40000.0
ANHARMONICITY = -2 * np.pi * 0.25
DETUNING_VALUES = np.array([-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002])
SCALE_VALUES = np.array([0.97, 1.0, 1.03])
AMP_BOUND = 0.4
SLEW_BOUND = 0.12

# Precompute constant matrices
A_MAT = np.array([[0, 1, 0],
                  [0, 0, np.sqrt(2)],
                  [0, 0, 0]], dtype=complex)
AD_MAT = A_MAT.conj().T
N_MAT = AD_MAT @ A_MAT
I3 = np.eye(3, dtype=complex)

# Lindblad operators
G1 = A_MAT / np.sqrt(T1)
G2 = N_MAT * np.sqrt(2.0 / Tphi)
G1DAG = G1.conj().T
G2DAG = G2.conj().T
G1DAG_G1 = G1DAG @ G1
G2DAG_G2 = G2DAG @ G2

# Precompute Liouvillian terms that don't depend on H
DISS_L = (np.kron(G1, G1.conj()) - 0.5 * (np.kron(G1DAG_G1, I3) + np.kron(I3, G1DAG_G1.T)) +
          np.kron(G2, G2.conj()) - 0.5 * (np.kron(G2DAG_G2, I3) + np.kron(I3, G2DAG_G2.T)))

# Target operator X
X3 = np.array([[0, 1, 0],
               [1, 0, 0],
               [0, 0, 1]], dtype=complex)

# Test states
STATES = [
    np.array([1, 0, 0], dtype=complex),
    np.array([0, 1, 0], dtype=complex),
    np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
    np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
    np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
    np.array([1, -1j, 0], dtype=complex) / np.sqrt(2),
]

def build_hamiltonian(detuning, scale, I, Q):
    NN1 = N_MAT @ (N_MAT - I3)
    H = (detuning * N_MAT + 0.5 * ANHARMONICITY * NN1 +
         0.5 * scale * (I * (A_MAT + AD_MAT) + 1j * Q * (A_MAT - AD_MAT)))
    return H

def build_liouvillian_full(H):
    coh_L = -1j * (np.kron(H, I3) - np.kron(I3, H.T))
    return coh_L + DISS_L

def check_validity(pulse):
    for i in range(len(pulse)):
        amp = np.sqrt(pulse[i, 0]**2 + pulse[i, 1]**2)
        if amp > AMP_BOUND + 1e-9:
            return False
    prev = np.array([0.0, 0.0])
    for i in range(len(pulse)):
        curr = pulse[i]
        diff = np.linalg.norm(curr - prev) / DT
        if diff > SLEW_BOUND + 1e-9:
            return False
        prev = curr
    diff = np.linalg.norm(prev - np.array([0.0, 0.0])) / DT
    if diff > SLEW_BOUND + 1e-9:
        return False
    return True

def enforce_constraints(pulse):
    pulse = pulse.copy()
    # Amplitude constraint
    for i in range(N_STEPS):
        amp = np.sqrt(pulse[i, 0]**2 + pulse[i, 1]**2)
        if amp > AMP_BOUND:
            scale = AMP_BOUND / amp
            pulse[i] *= scale
    
    # Slew rate constraint (forward pass)
    prev = np.array([0.0, 0.0])
    for i in range(N_STEPS):
        curr = pulse[i].copy()
        diff = curr - prev
        diff_norm = np.linalg.norm(diff)
        max_diff = SLEW_BOUND * DT
        if diff_norm > max_diff:
            curr = prev + diff * (max_diff / diff_norm)
        pulse[i] = curr
        prev = curr
    
    # Slew rate constraint (backward pass)
    prev = np.array([0.0, 0.0])
    for i in range(N_STEPS - 1, -1, -1):
        curr = pulse[i].copy()
        diff = curr - prev
        diff_norm = np.linalg.norm(diff)
        max_diff = SLEW_BOUND * DT
        if diff_norm > max_diff:
            curr = prev + diff * (max_diff / diff_norm)
        pulse[i] = curr
        prev = curr
    
    # Re-apply amplitude constraint after slew adjustments
    for i in range(N_STEPS):
        amp = np.sqrt(pulse[i, 0]**2 + pulse[i, 1]**2)
        if amp > AMP_BOUND:
            scale = AMP_BOUND / amp
            pulse[i] *= scale
            
    return pulse

def generate_pulse_from_params(params):
    """
    params: 16 parameters representing Fourier coefficients.
    First 8 are cosine coefficients, next 8 are sine coefficients.
    The pulse is constructed as I(t) = sum c_k cos(k * pi * t / T) + sum s_k sin(k * pi * t / T)
    We use a window function to ensure the pulse starts and ends at 0.
    """
    t = np.arange(N_STEPS) * DT
    # Use a Hann window to ensure edges are zero
    window = 0.5 * (1 - np.cos(2 * np.pi * t / GATE_TIME))
    
    I_vals = np.zeros(N_STEPS)
    Q_vals = np.zeros(N_STEPS)
    
    # Cosine terms
    for k in range(8):
        freq = (k + 1) * np.pi / GATE_TIME
        I_vals += params[k] * np.cos(freq * t)
        Q_vals += params[k + 8] * np.cos(freq * t) # Using same coeffs for Q for simplicity? 
        # Actually, let's separate I and Q.
        # Let's use params[0:8] for I cos, params[8:16] for I sin?
        # No, let's do:
        # params[0:4] I cos, params[4:8] I sin
        # params[8:12] Q cos, params[12:16] Q sin
        pass
        
    # Let's re-define the mapping clearly in the function body below
    # I = sum_{k=1}^4 c_ik cos(k pi t / T) + sum_{k=1}^4 s_ik sin(k pi t / T)
    # Q = sum_{k=1}^4 c_qk cos(k pi t / T) + sum_{k=1}^4 s_qk sin(k pi t / T)
    
    I_vals = np.zeros(N_STEPS)
    Q_vals = np.zeros(N_STEPS)
    
    for k in range(1, 5):
        freq = k * np.pi / GATE_TIME
        I_vals += params[k-1] * np.cos(freq * t)
        I_vals += params[4 + k - 1] * np.sin(freq * t)
        
        Q_vals += params[8 + k - 1] * np.cos(freq * t)
        Q_vals += params[12 + k - 1] * np.sin(freq * t)
        
    I_vals *= window
    Q_vals *= window
    
    pulse = np.column_stack((I_vals, Q_vals))
    return enforce_constraints(pulse)

def compute_score(pulse):
    total_score = 0.0
    max_infidelity = -np.inf
    case_infideltas = []
    
    for ci, (detuning, scale) in enumerate(zip(np.repeat(DETUNING_VALUES, 3), np.tile(SCALE_VALUES, 3))):
        exp_mats = []
        
        for i in range(N_STEPS):
            I_val, Q_val = pulse[i]
            H = build_hamiltonian(detuning, scale, I_val, Q_val)
            L = build_liouvillian_full(H)
            exp_mats.append(sla.expm(L * DT))
            
        infidelities = []
        for psi0 in STATES:
            rho0 = np.outer(psi0, psi0.conj())
            rho_vec = rho0.reshape(-1).copy()
            
            for i in range(N_STEPS):
                rho_vec = exp_mats[i] @ rho_vec
            
            rho = rho_vec.reshape(3, 3)
            psi_target = X3 @ psi0
            f = np.vdot(psi_target, rho @ psi_target).real
            f = np.clip(f, 0, 1)
            infidelities.append(1.0 - f)
            
        mean_inf = np.mean(infidelities)
        case_infideltas.append(mean_inf)
        if mean_inf > max_infidelity:
            max_infidelity = mean_inf
            
    mean_inf = np.mean(case_infideltas)
    score = mean_inf + max_infidelity
    return score

def pulse(target_gate, hardware_model):
    if target_gate != 'X_pi':
        return np.zeros((80, 2))
    
    start_time = time.time()
    deadline = start_time + 290
    
    # Initial guess: Gaussian-like Fourier coefficients
    # A Gaussian centered at T/2 with sigma ~ 5ns
    # We can approximate this with a few Fourier terms.
    # Or just start with small random values.
    np.random.seed(42)
    x0 = np.random.uniform(-0.1, 0.1, 16)
    
    # Bounds for Fourier coefficients
    # The pulse amplitude must be <= 0.4.
    # Sum of |coeffs| * window <= 0.4
    # If we have 8 terms, each coeff should be roughly < 0.4/8 = 0.05 to be safe?
    # But window is <= 1. So sum of coeffs <= 0.4.
    # Let's use bounds [-0.3, 0.3] for each coeff. The enforce_constraints will handle the rest.
    bounds = [(-0.3, 0.3)] * 16
    
    best_params = x0.copy()
    best_pulse = generate_pulse_from_params(best_params)
    best_score = compute_score(best_pulse)
    
    def objective(params):
        if time.time() > deadline:
            return best_score
        p = generate_pulse_from_params(params)
        s = compute_score(p)
        return s
        
    # Run Differential Evolution
    try:
        result_de = differential_evolution(
            objective, bounds, 
            seed=42, 
            maxiter=50, 
            popsize=15,
            tol=1e-4,
            polish=False
        )
        if result_de.fun < best_score:
            best_params = result_de.x
            best_pulse = generate_pulse_from_params(best_params)
            best_score = compute_score(best_pulse)
    except Exception:
        pass
    
    # Local refinement with Nelder-Mead (since gradient is not easily available for FD in DE context without extra cost)
    # Or just do a few more DE iterations if time permits?
    # Let's try a quick local polish with Nelder-Mead on the 16 params
    if time.time() < deadline - 30:
        try:
            result_nm = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if result_nm.fun < best_score:
                best_params = result_nm.x
                best_pulse = generate_pulse_from_params(best_params)
                best_score = compute_score(best_pulse)
        except Exception:
            pass
            
    return best_pulse
# EVOLVE-BLOCK-END
