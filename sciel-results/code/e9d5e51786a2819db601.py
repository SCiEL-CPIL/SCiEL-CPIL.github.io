"""
Rewritten program: parametric DRAG pulse optimization for three-level transmon X-pi gate.

Strategy:
- Represent the pulse with 4 parameters: Gaussian width (sigma), peak amplitude (A),
  DRAG coefficient (k), and a small Q-phase offset (phi).
- The pulse is I(t) = A * exp(-t^2/(2*sigma^2)) and Q(t) = k * dI/dt + phi * I(t).
- Optimize these 4 parameters to minimize the worst-case infidelity over 9 detuning/scale cases.
- Use scipy.optimize.minimize with a robust method (L-BFGS-B or Powell).
- Validate constraints: amplitude <= 0.4, slew rate <= 0.12.
- Deadline guard: return best result if time runs out.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import time

# Constants
ANHARMONICITY = -2 * np.pi * 0.25  # rad/ns
AMPLITUDE_BOUND = 0.4
SLEW_BOUND = 0.12
DETUNINGS = [-2 * np.pi * 0.002, 0.0, 2 * np.pi * 0.002]
SCALES = [0.97, 1.0, 1.03]
T1 = 25000.0  # ns
Tphi = 40000.0  # ns
DT = 0.5  # ns
N_STEPS = 80
DURATION = 40.0  # ns

# Time grid for simulation
t_grid = np.arange(N_STEPS) * DT

# Basis operators for 3-level system
# a = [[0, 1, 0], [0, 0, sqrt(2)], [0, 0, 0]]
a_op = np.array([[0, 1, 0],
                 [0, 0, np.sqrt(2)],
                 [0, 0, 0]], dtype=complex)
a_dag = a_op.conj().T

# n = a^dag a
n_op = a_dag @ a_op

# Precompute Lindblad operators
# L1 = a / sqrt(T1)
L1 = a_op / np.sqrt(T1)
# L2 = n * sqrt(2 / Tphi)
L2 = n_op * np.sqrt(2.0 / Tphi)

# Precompute L^dag L and L L^dag terms for the Liouvillian
# For Lindblad: sum_k (L_k rho L_k^dag - 0.5 {L_k^dag L_k, rho})
# We'll build the superoperator on-the-fly for each detuning/scale

def build_liouvillian(detuning, scale, I, Q):
    """Build the Liouvillian superoperator for given parameters."""
    # H = detuning * n + 0.5 * anharm * n*(n-1) + 0.5 * scale * (I*(a+a_dag) + 1j*Q*(a-a_dag))
    
    # n*(n-1)
    n_minus_1 = n_op - np.eye(3)
    n_n_minus_1 = n_op @ n_minus_1
    
    H = (detuning * n_op + 
         0.5 * ANHARMONICITY * n_n_minus_1 +
         0.5 * scale * (I * (a_op + a_dag) + 1j * Q * (a_op - a_dag)))
    
    # Liouvillian: L(rho) = -i [H, rho] + sum_k (L_k rho L_k^dag - 0.5 {L_k^dag L_k, rho})
    # We'll represent this as a 9x9 matrix acting on vec(rho)
    
    dim = 3
    L_mat = np.zeros((dim*dim, dim*dim), dtype=complex)
    
    # Coherent part: -i [H, rho] -> -i (H rho - rho H)
    # In vectorized form: -i (I ⊗ H^T - H^* ⊗ I)
    I_mat = np.eye(dim)
    H_vec = -1j * (np.kron(I_mat, H.conj().T) - np.kron(H, I_mat))
    L_mat += H_vec
    
    # Dissipative part
    for Lk in [L1, L2]:
        Lk_dag = Lk.conj().T
        # L_k rho L_k^dag -> (L_k^dag^T ⊗ L_k) vec(rho)
        term1 = np.kron(Lk_dag.T, Lk)
        # 0.5 {L_k^dag L_k, rho} -> 0.5 (L_k^dag L_k rho + rho L_k^dag L_k)
        Lk_dag_Lk = Lk_dag @ Lk
        term2 = 0.5 * (np.kron(I_mat, Lk_dag_Lk.conj().T) + np.kron(Lk_dag_Lk, I_mat))
        L_mat += term1 - term2
    
    return L_mat

def simulate_pulse(I, Q, detuning, scale, method='euler'):
    """Simulate the pulse and return the final state vector for |0> initial state."""
    # Start with |0><0|
    rho = np.zeros((3, 3), dtype=complex)
    rho[0, 0] = 1.0
    
    for i in range(N_STEPS):
        I_i = I[i]
        Q_i = Q[i]
        
        L_mat = build_liouvillian(detuning, scale, I_i, Q_i)
        
        # Evolve rho for DT: rho_new = expm(L_mat * DT) @ rho
        # Use Euler step for speed: rho_new = (I + L_mat * DT) @ rho
        # This is first-order, but for small DT it's acceptable
        rho = (np.eye(3, dtype=complex) + L_mat * DT) @ rho
        
        # Renormalize trace to combat numerical drift
        tr = np.trace(rho)
        if abs(tr) > 1e-12:
            rho = rho / tr
    
    return rho

def compute_fidelity(rho_final, target_state):
    """Compute fidelity between rho_final and target pure state."""
    # Fidelity = <psi| rho |psi> for pure target state
    return np.real(np.vdot(target_state, rho_final @ target_state))

def compute_score(I, Q):
    """Compute the score: mean + max infidelity over 9 cases and 5 initial states."""
    # Target states: |0>, |1>, (|0>±|1>)/sqrt(2), (|0>±i|1>)/sqrt(2)
    e0 = np.array([1, 0, 0], dtype=complex)
    e1 = np.array([0, 1, 0], dtype=complex)
    p_plus = (e0 + e1) / np.sqrt(2)
    p_minus = (e0 - e1) / np.sqrt(2)
    q_plus = (e0 + 1j*e1) / np.sqrt(2)
    q_minus = (e0 - 1j*e1) / np.sqrt(2)
    
    target_states = [e0, e1, p_plus, p_minus, q_plus, q_minus]
    
    # Target gate is X_pi: swaps |0> and |1>
    # So target for |0> is |1>, for |1> is |0>, for superpositions accordingly
    # X|0> = |1>, X|1> = |0>
    # X(|0>+|1>)/sqrt(2) = (|1>+|0>)/sqrt(2) = same
    # X(|0>-|1>)/sqrt(2) = (|1>-|0>)/sqrt(2) = -(|0>-|1>)/sqrt(2)
    # X(|0>+i|1>)/sqrt(2) = (|1>+i|0>)/sqrt(2)
    # X(|0>-i|1>)/sqrt(2) = (|1>-i|0>)/sqrt(2)
    
    target_map = {
        e0: e1,
        e1: e0,
        p_plus: p_plus,  # X|+> = |+>
        p_minus: -p_minus,  # X|-> = -|->
        q_plus: (e1 + 1j*e0)/np.sqrt(2),  # X|+i> 
        q_minus: (e1 - 1j*e0)/np.sqrt(2),  # X|-i>
    }
    
    # Actually, let's just apply X to each target state
    X_op = np.array([[0, 1, 0],
                     [1, 0, 0],
                     [0, 0, 0]], dtype=complex)
    
    infidelities = []
    
    for detuning in DETUNINGS:
        for scale in SCALES:
            for psi_init in target_states:
                # Simulate
                rho_final = simulate_pulse(I, Q, detuning, scale)
                
                # Target state is X|psi_init>
                psi_target = X_op @ psi_init
                
                # Fidelity
                fid = compute_fidelity(rho_final, psi_target)
                infid = 1.0 - fid
                infidelities.append(infid)
    
    infidelities = np.array(infid)
    return np.mean(infid) + np.max(infid)

def make_pulse(params):
    """Generate I and Q from parameters: [sigma, A, k, phi]."""
    sigma = params[0]
    A = params[1]
    k = params[2]
    phi = params[3]
    
    I = np.zeros(N_STEPS)
    Q = np.zeros(N_STEPS)
    
    for i in range(N_STEPS):
        t = i * DT
        I[i] = A * np.exp(-t**2 / (2 * sigma**2))
        # dI/dt = -t/sigma^2 * I
        dI_dt = -t / sigma**2 * I[i]
        Q[i] = k * dI_dt + phi * I[i]
    
    return I, Q

def check_constraints(I, Q):
    """Check amplitude and slew rate constraints."""
    # Amplitude
    for i in range(N_STEPS):
        if np.sqrt(I[i]**2 + Q[i]**2) > AMPLITUDE_BOUND + 1e-9:
            return False
    
    # Slew rate: ||p[i+1] - p[i]|| / dt <= SLEW_BOUND
    # Including from zero to step 0 and from last step to zero
    for i in range(N_STEPS - 1):
        dI = I[i+1] - I[i]
        dQ = Q[i+1] - Q[i]
        if np.sqrt(dI**2 + dQ**2) / DT > SLEW_BOUND + 1e-9:
            return False
    
    # From zero to step 0
    if np.sqrt(I[0]**2 + Q[0]**2) / DT > SLEW_BOUND + 1e-9:
        return False
    
    # From last step to zero
    if np.sqrt(I[-1]**2 + Q[-1]**2) / DT > SLEW_BOUND + 1e-9:
        return False
    
    return True

def objective(params):
    """Objective function for optimization."""
    sigma, A, k, phi = params
    
    # Basic bounds
    if sigma <= 0 or sigma > DURATION:
        return 1e10
    if A < 0 or A > AMPLITUDE_BOUND:
        return 1e10
    
    I, Q = make_pulse(params)
    
    if not check_constraints(I, Q):
        return 1e10
    
    return compute_score(I, Q)

def pulse(target_gate, hardware_model):
    """Main entry point. Returns (80, 2) array of I and Q amplitudes."""
    if target_gate != 'X_pi':
        raise ValueError("Only X_pi gate is supported")
    
    start_time = time.time()
    deadline = start_time + 320.0  # 10s safety margin
    
    # Initial guess: DRAG-style Gaussian
    # Area = pi, so A * sigma * sqrt(2*pi) = pi -> A = 1/(sigma*sqrt(2))
    # For sigma = 40/6 ≈ 6.67, A ≈ 1/(6.67*1.414) ≈ 0.106
    # But we need to hit amplitude bound, so let's start with reasonable values
    
    best_I = None
    best_Q = None
    best_score = float('inf')
    
    # Try multiple initial points
    initial_guesses = [
        [6.67, 0.35, 0.05, 0.0],  # DRAG-like
        [5.0, 0.4, 0.03, 0.0],
        [8.0, 0.3, 0.07, 0.0],
        [6.0, 0.38, 0.04, 0.01],
    ]
    
    for guess in initial_guesses:
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                objective,
                guess,
                method='L-BFGS-B',
                bounds=[(1.0, 20.0), (0.05, AMPLITUDE_BOUND), (-0.5, 0.5), (-0.5, 0.5)],
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            if result.success or result.fun < best_score:
                I, Q = make_pulse(result.x)
                if check_constraints(I, Q):
                    score = result.fun
                    if score < best_score:
                        best_score = score
                        best_I = I.copy()
                        best_Q = Q.copy()
        except Exception:
            continue
    
    # If no valid solution found, use a simple DRAG pulse
    if best_I is None:
        sigma = 6.67
        A = 0.35
        k = 0.05
        phi = 0.0
        best_I, best_Q = make_pulse([sigma, A, k, phi])
        
        # If still invalid, scale down
        if not check_constraints(best_I, best_Q):
            scale_factor = 0.5
            best_I = best_I * scale_factor
            best_Q = best_Q * scale_factor
    
    return np.column_stack([best_I, best_Q])
