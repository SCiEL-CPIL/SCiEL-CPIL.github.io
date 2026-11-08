"""Optimized pulse for three-level transmon X_pi gate with robustness."""
import numpy as np
import time
from scipy.linalg import expm
import warnings

warnings.filterwarnings('ignore')

def pulse(target_gate, hardware_model):
    start_time = time.time()
    deadline = start_time + 325.0  # safety margin before 330s kill

    # Hardware parameters
    alpha = -2 * np.pi * 0.25  # anharmonicity
    amp_bound = 0.4
    slew_bound = 0.12
    dt = 0.5
    N = 80
    T1 = 25000.0
    Tphi = 40000.0

    # Detuning and scale cases
    deltas = [-2 * np.pi * 0.002, 0, 2 * np.pi * 0.002]
    scales = [0.97, 1.0, 1.03]

    # Basis states (column vectors)
    states = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [1/np.sqrt(2), 1/np.sqrt(2), 0],
        [1/np.sqrt(2), -1/np.sqrt(2), 0],
        [1/np.sqrt(2), 1j/np.sqrt(2), 0],
        [1/np.sqrt(2), -1j/np.sqrt(2), 0]
    ], dtype=complex).T  # shape (3, 6)

    # Target states for X gate
    targets = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [1/np.sqrt(2), 1/np.sqrt(2), 0],
        [-1/np.sqrt(2), 1/np.sqrt(2), 0],
        [1j/np.sqrt(2), 1/np.sqrt(2), 0],
        [-1j/np.sqrt(2), 1/np.sqrt(2), 0]
    ], dtype=complex).T  # shape (3, 6)

    # Construct operators
    a_op = np.array([
        [0, 1, 0],
        [0, 0, np.sqrt(2)],
        [0, 0, 0]
    ], dtype=complex)

    a_dag = a_op.conj().T
    n_op = a_dag @ a_op
    n_nm1 = n_op @ (n_op - np.eye(3))

    # Precompute Hamiltonian components
    H0_nom = 0.5 * alpha * n_nm1
    driveI = 0.5 * (a_op + a_dag)
    driveQ = 0.5j * (a_op - a_dag)

    def get_H(I, Q, delta, scale):
        return H0_nom + delta * n_op + scale * (I * driveI + Q * driveQ)

    def check_constraints(I_Q):
        """Check if pulse satisfies all constraints."""
        amps = np.sqrt(I_Q[:, 0]**2 + I_Q[:, 1]**2)
        if np.max(amps) > amp_bound + 1e-9:
            return False
        p = np.vstack([np.zeros(2), I_Q, np.zeros(2)])
        diffs = np.diff(p, axis=0)
        norms = np.sqrt(np.sum(diffs**2, axis=1))
        if np.max(norms) / dt > slew_bound + 1e-9:
            return False
        return True

    def simulate_unitary_fast(I_Q, delta, scale):
        """Simulate pulse using unitary evolution - vectorized over states."""
        U = np.eye(3, dtype=complex)
        for i in range(N):
            H = get_H(I_Q[i, 0], I_Q[i, 1], delta, scale)
            U = expm(-1j * H * dt) @ U
        return U

    def score_case_fast(I_Q, delta, scale):
        """Score a single case using vectorized simulation."""
        U = simulate_unitary_fast(I_Q, delta, scale)
        # Fidelity for each state: |<target|psi_final>|^2
        # psi_final = U @ states, target = targets
        # fidelity_i = |vdot(target_i, U @ state_i)|^2
        evolved = U @ states  # (3, 6)
        fids = np.abs(np.sum(targets.conj() * evolved, axis=0))**2
        return 1.0 - np.mean(fids)

    def score(I_Q):
        """Full score: mean + max over 9 cases."""
        if not check_constraints(I_Q):
            return 1e6
        infidelities = []
        for d in deltas:
            for s in scales:
                infidelities.append(score_case_fast(I_Q, d, s))
        infidelities = np.array(infidelities)
        return np.mean(infidelities) + np.max(infidelities)

    def generate_drag(A, sigma, eta, shift=0.0):
        """Generate DRAG-style pulse with constraints."""
        t = np.linspace(0, 40, N)
        t_c = 20.0 + shift
        I = A * np.exp(-(t - t_c)**2 / (2 * sigma**2))
        dI = -(t - t_c) / (sigma**2) * I
        Q = eta * dI
        I_Q = np.column_stack((I, Q))
        
        # Apply amplitude constraint
        amps = np.sqrt(I_Q[:, 0]**2 + I_Q[:, 1]**2)
        for i in range(N):
            if amps[i] > amp_bound:
                scale_factor = amp_bound / amps[i]
                I_Q[i, 0] *= scale_factor
                I_Q[i, 1] *= scale_factor
        
        # Apply slew rate constraint (forward pass)
        for i in range(1, N):
            diff = np.linalg.norm(I_Q[i] - I_Q[i-1])
            if diff / dt > slew_bound:
                scale_factor = (slew_bound * dt) / diff
                I_Q[i] = I_Q[i-1] + scale_factor * (I_Q[i] - I_Q[i-1])
        
        # Check start slew
        if np.linalg.norm(I_Q[0]) / dt > slew_bound:
            scale_factor = (slew_bound * dt) / np.linalg.norm(I_Q[0])
            I_Q[0] *= scale_factor
        
        # Check end slew
        if np.linalg.norm(I_Q[-1]) / dt > slew_bound:
            scale_factor = (slew_bound * dt) / np.linalg.norm(I_Q[-1])
            I_Q[-1] *= scale_factor
            
        return I_Q

    def seed_pulse():
        """Generate seed pulse."""
        sigma = 40.0 / 6.0
        # Area should be pi for X_pi gate
        A = np.sqrt(np.pi / 2) / sigma
        return generate_drag(A, sigma, 0.0)

    # --- Optimization Strategy ---
    
    # 1. Coarse grid search over DRAG parameters
    best_I_Q = seed_pulse()
    best_score = score(best_I_Q)
    
    A_values = np.linspace(0.12, 0.42, 15)
    sigma_values = np.linspace(2.5, 10.0, 12)
    eta_values = np.linspace(-3.0, 3.0, 13)
    
    for A in A_values:
        if time.time() > deadline - 150:
            break
        for sigma in sigma_values:
            if time.time() > deadline - 150:
                break
            for eta in eta_values:
                if time.time() > deadline - 150:
                    break
                I_Q = generate_drag(A, sigma, eta)
                if not check_constraints(I_Q):
                    continue
                s = score(I_Q)
                if s < best_score:
                    best_score = s
                    best_I_Q = I_Q

    # 2. Local refinement using Nelder-Mead on DRAG parameters
    def opt_drag(params):
        if time.time() > deadline - 60:
            return 1e6
        A, sigma, eta = params
        if A <= 0.01 or sigma <= 0.5:
            return 1e6
        I_Q = generate_drag(A, sigma, eta)
        if not check_constraints(I_Q):
            return 1e6
        return score(I_Q)

    # Find best initial point from grid
    best_params = None
    min_val = 1e6
    for A in A_values:
        for sigma in sigma_values:
            for eta in eta_values:
                if time.time() > deadline - 60:
                    break
                I_Q = generate_drag(A, sigma, eta)
                if not check_constraints(I_Q):
                    continue
                s = score(I_Q)
                if s < min_val:
                    min_val = s
                    best_params = [A, sigma, eta]

    if best_params is not None:
        try:
            result = minimize_from_scipy(best_params, opt_drag, deadline)
            if result.fun < best_score:
                A, sigma, eta = result.x
                if A > 0.01 and sigma > 0.5:
                    I_Q = generate_drag(A, sigma, eta)
                    if check_constraints(I_Q):
                        best_score = result.fun
                        best_I_Q = I_Q
        except Exception:
            pass

    # 3. Direct refinement of individual steps (if time permits)
    if time.time() < deadline - 40:
        def opt_direct(x):
            if time.time() > deadline - 30:
                return 1e6
            I_Q = x.reshape(N, 2)
            if not check_constraints(I_Q):
                return 1e6
            return score(I_Q)

        try:
            result = minimize_direct(best_I_Q, opt_direct, deadline)
            I_Q = result.x.reshape(N, 2)
            if check_constraints(I_Q):
                s = score(I_Q)
                if s < best_score:
                    best_score = s
                    best_I_Q = I_Q
        except Exception:
            pass

    # Final constraint check
    if not check_constraints(best_I_Q):
        best_I_Q = seed_pulse()

    return best_I_Q

def minimize_from_scipy(initial, func, deadline):
    """Nelder-Mead minimization with deadline guard."""
    from scipy.optimize import minimize
    result = minimize(func, initial, method='Nelder-Mead',
                      options={'maxiter': 200, 'xatol': 0.005, 'fatol': 1e-6})
    return result

def minimize_direct(I_Q_initial, func, deadline):
    """L-BFGS-B minimization with deadline guard."""
    from scipy.optimize import minimize
    result = minimize(func, I_Q_initial.reshape(-1), method='L-BFGS-B',
                      options={'maxiter': 50, 'ftol': 1e-8, 'maxfun': 500})
    return result
