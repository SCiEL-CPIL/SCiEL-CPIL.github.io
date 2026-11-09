"""
Low-dimensional basis expansion pulse optimization for 3-level transmon X-pi gate.
Uses Differential Evolution for global search over 8 parameters,
then L-BFGS-B for local refinement.
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
    for i in range(N_STEPS):
        amp = np.sqrt(pulse[i, 0]**2 + pulse[i, 1]**2)
        if amp > AMP_BOUND:
            scale = AMP_BOUND / amp
            pulse[i] *= scale
    
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
    
    for i in range(N_STEPS):
        amp = np.sqrt(pulse[i, 0]**2 + pulse[i, 1]**2)
        if amp > AMP_BOUND:
            scale = AMP_BOUND / amp
            pulse[i] *= scale
            
    return pulse

def generate_pulse_from_params(params):
    """
    params: [amp, sigma, center, drag_coeff, phase, taper_left, taper_right, offset]
    Generates pulse using a Gaussian-DRAG basis.
    """
    t = np.arange(N_STEPS) * DT
    amp = params[0]
    sigma = params[1]
    center = params[2]
    drag_coeff = params[3]
    phase = params[4]
    taper_left = params[5]
    taper_right = params[6]
    offset = params[7]
    
    g = np.exp(-0.5 * ((t - center) / sigma)**2)
    dg = (t - center) / (sigma**2) * g
    
    # Tapering to ensure edges are close to 0
    # Linear taper at edges
    edge_width = 5.0
    taper_mask_left = (t < edge_width)
    taper_mask_right = (t > GATE_TIME - edge_width)
    
    g[taper_mask_left] *= (t[taper_mask_left] / edge_width) ** taper_left
    g[taper_mask_right] *= ((GATE_TIME - t[taper_mask_right]) / edge_width) ** taper_right
    
    # Apply phase rotation
    I = amp * g * np.cos(phase)
    Q = amp * (g * np.sin(phase) + drag_coeff * dg)
    
    pulse = np.column_stack((I, Q))
    
    # Normalize area roughly to pi if needed? 
    # The task seed normalizes area to pi. 
    # Let's calculate area and scale if it's far from pi?
    # No, let the optimizer find the right amplitude.
    
    return enforce_constraints(pulse)

def compute_score_and_gradient(pulse):
    p_flat = pulse.flatten()
    total_score = 0.0
    total_grad = np.zeros_like(p_flat)
    max_infidelity = -np.inf
    max_case_idx = -1
    case_infideltas = []
    
    for ci, (detuning, scale) in enumerate(zip(np.repeat(DETUNING_VALUES, 3), np.tile(SCALE_VALUES, 3))):
        exp_mats = []
        exp_mats_dagger = []
        
        for i in range(N_STEPS):
            I_val, Q_val = pulse[i]
            H = build_hamiltonian(detuning, scale, I_val, Q_val)
            L = build_liouvillian_full(H)
            exp_mats.append(sla.expm(L * DT))
            exp_mats_dagger.append(exp_mats[-1].conj().T)
            
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
            max_case_idx = ci
            
    mean_inf = np.mean(case_infideltas)
    score = mean_inf + max_infidelity
    
    for ci, (detuning, scale) in enumerate(zip(np.repeat(DETUNING_VALUES, 3), np.tile(SCALE_VALUES, 3))):
        weight = 1.0 / 9.0
        if ci == max_case_idx:
            weight += 1.0
            
        if weight == 0: continue
        
        E_list = []
        for i in range(N_STEPS):
            I_val, Q_val = pulse[i]
            H = build_hamiltonian(detuning, scale, I_val, Q_val)
            L = build_liouvillian_full(H)
            E_list.append(sla.expm(L * DT))
            
        grad_case = np.zeros_like(p_flat)
        
        for psi0 in STATES:
            rho0 = np.outer(psi0, psi0.conj()).reshape(-1)
            psi_target = X3 @ psi0
            psi_target_vec = psi_target.reshape(-1)
            
            rho_hist = [rho0]
            for i in range(N_STEPS):
                rho_hist.append(E_list[i] @ rho_hist[-1])
                
            lambda_i = np.zeros(9, dtype=complex)
            P = np.outer(psi_target, psi_target.conj())
            lambda_N_minus_1 = -P.reshape(-1)
            
            for i in range(N_STEPS - 1, -1, -1):
                I_val, Q_val = pulse[i]
                H = build_hamiltonian(detuning, scale, I_val, Q_val)
                
                rho_i = E_list[i] @ rho_hist[i]
                
                X_drive = A_MAT + AD_MAT
                Y_drive = A_MAT - AD_MAT
                
                term_I = np.vdot(lambda_i, (np.kron(X_drive, I3) - np.kron(I3, X_drive.T)) @ rho_i)
                term_Q = np.vdot(lambda_i, (np.kron(Y_drive, I3) - np.kron(I3, Y_drive.T)) @ rho_i)
                
                dJ_dI = 0.5 * scale * term_I
                dJ_dQ = 0.5 * scale * 1j * term_Q
                
                grad_case[i*2] += weight * np.real(dJ_dI)
                grad_case[i*2+1] += weight * np.real(dJ_dQ)
                
                if i > 0:
                    lambda_i = E_list[i].conj().T @ lambda_i
                else:
                    lambda_i = np.zeros(9, dtype=complex)
            
            grad_case /= 6.0
            total_grad += grad_case
            
    return score, total_grad

def pulse(target_gate, hardware_model):
    if target_gate != 'X_pi':
        return np.zeros((80, 2))
    
    start_time = time.time()
    deadline = start_time + 280
    
    # Initial guess
    # amp ~ 0.2, sigma ~ 5, center ~ 20, drag ~ -0.1, phase ~ 0, taper ~ 1, offset ~ 0
    x0 = np.array([0.2, 5.0, 20.0, -0.1, 0.0, 1.0, 1.0, 0.0])
    
    # Differential Evolution
    # Bounds:
    # amp: [0.05, 0.35]
    # sigma: [3.0, 8.0]
    # center: [15.0, 25.0]
    # drag: [-0.5, 0.5]
    # phase: [-0.5, 0.5]
    # taper_left: [0.5, 3.0]
    # taper_right: [0.5, 3.0]
    # offset: [-0.1, 0.1]
    
    bounds = [
        (0.05, 0.35),
        (3.0, 8.0),
        (15.0, 25.0),
        (-0.5, 0.5),
        (-0.5, 0.5),
        (0.5, 3.0),
        (0.5, 3.0),
        (-0.1, 0.1)
    ]
    
    best_params = x0.copy()
    best_pulse = generate_pulse_from_params(best_params)
    best_score, _ = compute_score_and_gradient(best_pulse)
    
    def objective(params):
        if time.time() > deadline:
            return best_score
        p = generate_pulse_from_params(params)
        s, _ = compute_score_and_gradient(p)
        return s
        
    # Run DE
    try:
        result_de = differential_evolution(
            objective, bounds, 
            seed=42, 
            maxiter=20, 
            popsize=10,
            tol=1e-4,
            polish=False
        )
        if result_de.fun < best_score:
            best_params = result_de.x
            best_pulse = generate_pulse_from_params(best_params)
            best_score, _ = compute_score_and_gradient(best_pulse)
    except Exception:
        pass
    
    # Local refinement with L-BFGS-B on the 8 parameters
    def objective_grad(params):
        if time.time() > deadline:
            return best_score, np.zeros_like(params)
        p = generate_pulse_from_params(params)
        s, g_pulse = compute_score_and_gradient(p)
        # We need gradient with respect to params.
        # Since generate_pulse_from_params is differentiable (mostly), 
        # but we don't have analytic gradient for the mapping.
        # We can use finite difference for the 8 params, or just use the pulse gradient 
        # if we had a direct mapping. 
        # Actually, L-BFGS-B on params requires dS/dparams.
        # dS/dparams = sum_i (dS/dpulse_i) * (dpulse_i/dparams)
        # This is expensive to compute via FD (8 * 2 evals).
        # Let's just do a simple Nelder-Mead or Powell on the 8 params using score_only.
        # Or use L-BFGS-B with FD gradient.
        
        # FD gradient for 8 params
        eps = 1e-4
        grad = np.zeros(8)
        f0 = s
        for i in range(8):
            params_plus = params.copy()
            params_plus[i] += eps
            p_plus = generate_pulse_from_params(params_plus)
            s_plus, _ = compute_score_and_gradient(p_plus)
            grad[i] = (s_plus - f0) / eps
        return s, grad
        
    try:
        result_lbfgs = minimize(
            objective_grad,
            best_params,
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 50, 'gtol': 1e-6}
        )
        if result_lbfgs.fun < best_score:
            best_params = result_lbfgs.x
            best_pulse = generate_pulse_from_params(best_params)
            best_score, _ = compute_score_and_gradient(best_pulse)
    except Exception:
        pass
        
    return best_pulse
# EVOLVE-BLOCK-END
