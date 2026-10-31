import numpy as np
import time
from scipy.optimize import minimize, root
from scipy.special import factorial

def construct():
    start_time = time.time()
    deadline = start_time + 180.0

    # Indices of Hermite polynomials
    # We need H0, H4, H8, H12
    # Coefficients c0, c4, c8 are free. c12 is fixed by g(0)=0.

    # Precompute H_n(0) values
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0

    # Constraint: c0*H0_0 + c4*H4_0 + c8*H8_0 + c12*H12_0 = 0
    K = H12_0

    # Indices in the full space [0, 4, 8, 12]
    indices = [0, 4, 8, 12]
    N = len(indices)

    sqrt_pi = np.sqrt(np.pi)

    # Precompute factorials for indices up to max(indices)+2
    max_idx = max(indices) + 2
    fact = [1] * (max_idx + 1)
    for i in range(2, max_idx + 1):
        fact[i] = fact[i - 1] * i

    M_norm = np.zeros((N, N))
    M_pos = np.zeros((N, N))
    M_mom = np.zeros((N, N))

    def int_h(a, b):
        if a == b and a >= 0:
            return sqrt_pi * (2 ** a) * fact[a]
        return 0.0

    for i, m in enumerate(indices):
        for j, n in enumerate(indices):
            # Norm
            if m == n:
                M_norm[i, j] = sqrt_pi * (2 ** n) * fact[n]

            # Position: int x^2 H_m H_n e^{-x^2}
            # x^2 H_n = 0.5 H_{n+2} + n H_n + 0.5 n(n-1) H_{n-2}
            if m == n + 2:
                M_pos[i, j] += 0.5 * sqrt_pi * (2 ** m) * fact[m]
            if m == n:
                M_pos[i, j] += n * sqrt_pi * (2 ** n) * fact[n]
            if m == n - 2:
                M_pos[i, j] += 0.5 * n * (n - 1) * sqrt_pi * (2 ** m) * fact[m]

            # Momentum: int Q_m Q_n e^{-x^2}
            # Q_k = 1.5 k H_{k-1} - 0.5 H_{k+1}
            # Q_m Q_n = 2.25 mn H_{m-1}H_{n-1} - 0.75 m H_{m-1}H_{n+1} - 0.75 n H_{m+1}H_{n-1} + 0.25 H_{m+1}H_{n+1}
            M_mom[i, j] = (2.25 * m * n * int_h(m-1, n-1)
                           - 0.75 * m * int_h(m-1, n+1)
                           - 0.75 * n * int_h(m+1, n-1)
                           + 0.25 * int_h(m+1, n+1))

    # Construct A matrix to map v=[c0,c4,c8] to C=[c0,c4,c8,c12]
    # c12 = -(1*c0 + 12*c4 + 1680*c8) / 665280
    A = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0/K, -12.0/K, -1680.0/K]
    ])

    M_norm_p = A.T @ M_norm @ A
    M_pos_p = A.T @ M_pos @ A
    M_mom_p = A.T @ M_mom @ A

    # Objective function: minimize (pos * mom) / norm^2
    # This is equivalent to minimizing the generalized eigenvalue problem:
    # (M_pos * M_mom) v = lambda * (M_norm^2) v  -- wait, this is not quite right.
    # Let J(v) = (v^T P v)(v^T M v) / (v^T N v)^2
    # We want to minimize J.
    # Let's use the substitution u = v / sqrt(v^T N v) so that u^T N u = 1.
    # Then J = (u^T P u)(u^T M u).
    # This is still not a standard eigenvalue problem.
    #
    # Alternative approach:
    # Let's just use multi-start optimization with more robust methods.
    
    def objective(v):
        norm_val = v @ M_norm_p @ v
        if norm_val <= 1e-12:
            return 1e20
        pos_val = v @ M_pos_p @ v
        mom_val = v @ M_mom_p @ v
        return (pos_val * mom_val) / (norm_val * norm_val)
        
    def gradient(v):
        norm_val = v @ M_norm_p @ v
        pos_val = v @ M_pos_p @ v
        mom_val = v @ M_mom_p @ v
        
        dP = 2 * M_pos_p @ v
        dM = 2 * M_mom_p @ v
        dN = 2 * M_norm_p @ v
        
        num = (dP * mom_val + pos_val * dM) * norm_val - 2 * pos_val * mom_val * dN
        den = norm_val ** 3
        
        return num / den

    # Multi-start optimization
    best_v = None
    best_val = np.inf

    # Initial guess from seed
    v0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    v0_norm = np.linalg.norm(v0)
    if v0_norm > 1e-9:
        v0 = v0 / v0_norm
    else:
        v0 = np.array([1.0, 0.0, 0.0])
        
    # Generate starts
    starts = [v0]
    
    # Random starts on sphere
    rng = np.random.default_rng(42)
    for _ in range(100):
        v_rand = rng.normal(size=3)
        norm = np.linalg.norm(v_rand)
        if norm > 1e-9:
            starts.append(v_rand / norm)
            
    # Axis-aligned starts
    starts.append(np.array([1.0, 0.0, 0.0]))
    starts.append(np.array([-1.0, 0.0, 0.0]))
    starts.append(np.array([0.0, 1.0, 0.0]))
    starts.append(np.array([0.0, -1.0, 0.0]))
    starts.append(np.array([0.0, 0.0, 1.0]))
    starts.append(np.array([0.0, 0.0, -1.0]))

    # Also try some structured starts
    for alpha in np.linspace(-1, 1, 11):
        for beta in np.linspace(-1, 1, 11):
            v_start = np.array([alpha, beta, 0.0])
            norm = np.linalg.norm(v_start)
            if norm > 1e-9:
                starts.append(v_start / norm)

    for v_start in starts:
        if time.time() > deadline - 10.0:
            break
            
        try:
            res = minimize(objective, v_start, jac=gradient, method='BFGS', 
                          options={'maxiter': 1000, 'gtol': 1e-10})
            if res.success:
                v_sol = res.x
                val = objective(v_sol)
                if val < best_val:
                    best_val = val
                    best_v = v_sol
        except Exception:
            pass

    if best_v is None:
        best_v = v0
        best_val = objective(v0)
        
    return np.array(best_v)
