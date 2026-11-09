"""Gradient-based pulse optimization for 3-level transmon X-pi gate."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.linalg as sla
import time
from scipy.optimize import minimize

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
# L(rho) = -i[H,rho] + D1(rho) + D2(rho)
# In vec form: L = -i*(H⊗I - I⊗H^T) + sum_k (L_k⊗L_k^* - 0.5*(L_k^†L_k⊗I + I⊗(L_k^†L_k)^T))
# The dissipative part is constant
DISS_L = (np.kron(G1, G1.conj()) - 0.5 * (np.kron(G1DAG_G1, I3) + np.kron(I3, G1DAG_G1.T)) +
          np.kron(G2, G2.conj()) - 0.5 * (np.kron(G2DAG_G2, I3) + np.kron(I3, G2DAG_G2.T)))

def build_liouvillian_full(H):
    """Build full Liouvillian for given Hamiltonian."""
    # Coherent part: -i*(H⊗I - I⊗H^T)
    coh_L = -1j * (np.kron(H, I3) - np.kron(I3, H.T))
    return coh_L + DISS_L

def build_hamiltonian(detuning, scale, I, Q):
    """Build the 3-level Hamiltonian matrix."""
    # H = detuning * n + 0.5 * anharmonicity * n(n-1) + 0.5 * scale * (I*(a+ad) + i*Q*(a-ad))
    # Precompute n(n-1) term
    NN1 = N_MAT @ (N_MAT - I3)
    H = (detuning * N_MAT + 0.5 * ANHARMONICITY * NN1 +
         0.5 * scale * (I * (A_MAT + AD_MAT) + 1j * Q * (A_MAT - AD_MAT)))
    return H

def evolve_all_states(pulse, detuning, scale):
    """Evolve all 6 test states and return fidelities."""
    # pulse is (80, 2)
    # Precompute Liouvillian exponentials for each step
    # For efficiency, we'll evolve each state separately but reuse L matrices
    
    # The 6 initial states
    states = [
        np.array([1, 0, 0], dtype=complex),
        np.array([0, 1, 0], dtype=complex),
        np.array([1, 1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1, 0], dtype=complex) / np.sqrt(2),
        np.array([1, 1j, 0], dtype=complex) / np.sqrt(2),
        np.array([1, -1j, 0], dtype=complex) / np.sqrt(2),
    ]
    
    # X gate in 3-level space
    X3 = np.array([[0, 1, 0],
                   [1, 0, 0],
                   [0, 0, 1]], dtype=complex)
    
    # Precompute exp(L*dt) for each step
    exp_mats = []
    for i in range(N_STEPS):
        I_val, Q_val = pulse[i]
        H = build_hamiltonian(detuning, scale, I_val, Q_val)
        L = build_liouvillian_full(H)
        exp_mats.append(sla.expm(L * DT))
    
    fidelities = []
    for psi0 in states:
        rho0 = np.outer(psi0, psi0.conj())
        rho_vec = rho0.reshape(-1).copy()
        
        for i in range(N_STEPS):
            rho_vec = exp_mats[i] @ rho_vec
        
        rho = rho_vec.reshape(3, 3)
        psi_target = X3 @ psi0
        f = np.vdot(psi_target, rho @ psi_target).real
        f = np.clip(f, 0, 1)
        fidelities.append(f)
    
    return np.mean(fidelities)

def compute_score_and_gradient(pulse):
    """Compute score and gradient using finite differences."""
    # Score = mean + max of infidelity over 9 cases
    # Infidelity = 1 - mean fidelity
    # So score = mean(1-f_i) + max(1-f_i) where f_i is mean fidelity for case i
    
    # First compute all 9 fidelities
    case_fidelities = []
    for detuning in DETUNING_VALUES:
        for scale in SCALE_VALUES:
            f = evolve_all_states(pulse, detuning, scale)
            case_fidelities.append(f)
    
    case_fidelities = np.array(case_fidelities)
    case_infidelities = 1.0 - case_fidelities
    
    mean_inf = np.mean(case_infidelities)
    max_idx = np.argmax(case_infidelities)
    max_inf = case_infidelities[max_idx]
    score = mean_inf + max_inf
    
    # Compute gradient via central differences
    # For 160 parameters, this is 320 evaluations... too slow.
    # Use forward differences with small step
    # Actually, let's use the adjoint method or just do a few random directions
    
    # For now, use forward differences with a small step
    # But 160 * 9 * 6 evolutions per gradient step is too expensive.
    
    # Alternative: use the fact that the score is smooth and use a smaller step
    # Or use a hybrid: compute gradient for a subset of parameters
    
    # Let's use a practical approach: compute gradient via finite differences
    # but only for the most important parameters, or use a smaller step size
    
    # Actually, for L-BFGS-B, we can provide an analytic gradient if we do adjoint.
    # But for simplicity and robustness, let's use a smaller number of evaluations.
    
    # Compromise: use forward differences with step=1e-3, but this is 160 evaluations
    # Each evaluation is 9*6=54 state evolutions, each with 80 matrix exps.
    # That's 160 * 54 * 80 = 691,200 matrix exps. Too slow.
    
    # Better approach: use the adjoint method for exact gradient.
    # But that's complex. Let's use a different strategy:
    # Use Nelder-Mead or Powell which don't need gradients.
    
    # Actually, let's just compute the score and use a gradient-free method.
    # Or use a very small number of gradient evaluations.
    
    # For now, return score only and let the optimizer handle it.
    # We'll use Powell's method which doesn't require gradients.
    
    return score

def score_only(pulse):
    """Compute score only (no gradient)."""
    case_fidelities = []
    for detuning in DETUNING_VALUES:
        for scale in SCALE_VALUES:
            f = evolve_all_states(pulse, detuning, scale)
            case_fidelities.append(f)
    
    case_fidelities = np.array(case_fidelities)
    case_infidelities = 1.0 - case_fidelities
    
    mean_inf = np.mean(case_infidelities)
    max_inf = np.max(case_infidelities)
    score = mean_inf + max_inf
    
    return score

def check_validity(pulse):
    """Check amplitude and slew constraints."""
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
    """Project pulse onto feasible set."""
    pulse = pulse.copy()
    
    # Enforce amplitude bound
    for i in range(N_STEPS):
        amp = np.sqrt(pulse[i, 0]**2 + pulse[i, 1]**2)
        if amp > AMP_BOUND:
            scale = AMP_BOUND / amp
            pulse[i] *= scale
    
    # Enforce slew bound forward
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
    
    # Enforce slew bound backward (from last to zero)
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
    
    # Re-enforce amplitude after slew projection
    for i in range(N_STEPS):
        amp = np.sqrt(pulse[i, 0]**2 + pulse[i, 1]**2)
        if amp > AMP_BOUND:
            scale = AMP_BOUND / amp
            pulse[i] *= scale
    
    return pulse

def seed_pulse():
    """Generate the seed DRAG-style Gaussian pulse."""
    t = np.arange(N_STEPS) * DT
    sigma = 40.0 / 6.0
    center = 20.0
    g = np.exp(-0.5 * ((t - center) / sigma)**2)
    edge_val = np.exp(-0.5 * ((t[0] - center) / sigma)**2)
    g = g - edge_val
    area = np.sum(g) * DT
    if area > 0:
        g = g * (np.pi / area)
    pulse = np.zeros((N_STEPS, 2))
    pulse[:, 0] = g
    return pulse

def pulse(target_gate, hardware_model):
    """Main entry point."""
    if target_gate != 'X_pi':
        return np.zeros((80, 2))
    
    start_time = time.time()
    deadline = start_time + 280  # Leave 20s margin
    
    # Start with seed
    current_pulse = seed_pulse()
    current_pulse = enforce_constraints(current_pulse)
    best_pulse = current_pulse.copy()
    best_score = score_only(best_pulse)
    
    if time.time() > deadline:
        return best_pulse
    
    # Use Powell's method (derivative-free) for optimization
    # Flatten pulse to 1D
    x0 = current_pulse.flatten()
    
    def objective(x):
        p = x.reshape(N_STEPS, 2)
        p = enforce_constraints(p)
        return score_only(p)
    
    # Powell's method with bounds
    # Bounds: each component in [-0.4, 0.4]
    bounds = [(-AMP_BOUND, AMP_BOUND)] * (2 * N_STEPS)
    
    try:
        result = minimize(
            objective,
            x0,
            method='Powell',
            bounds=bounds,
            options={
                'maxiter': 50,
                'maxfev': 200,
                'xatol': 1e-4,
                'fatol': 1e-6,
                'disp': False
            }
        )
        
        if result.success or result.fun < best_score:
            opt_pulse = enforce_constraints(result.x.reshape(N_STEPS, 2))
            opt_score = score_only(opt_pulse)
            if opt_score < best_score:
                best_pulse = opt_pulse
                best_score = opt_score
        
    except Exception as e:
        pass
    
    if time.time() > deadline:
        return best_pulse
    
    # Try a few more local refinements with Nelder-Mead
    for _ in range(3):
        if time.time() > deadline:
            break
        
        x0 = best_pulse.flatten()
        try:
            result = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 30,
                    'xatol': 1e-5,
                    'fatol': 1e-7,
                    'adaptive': True,
                    'disp': False
                }
            )
            
            if result.fun < best_score:
                opt_pulse = enforce_constraints(result.x.reshape(N_STEPS, 2))
                opt_score = score_only(opt_pulse)
                if opt_score < best_score:
                    best_pulse = opt_pulse
                    best_score = opt_score
                    
        except Exception:
            pass
    
    return best_pulse
# EVOLVE-BLOCK-END
