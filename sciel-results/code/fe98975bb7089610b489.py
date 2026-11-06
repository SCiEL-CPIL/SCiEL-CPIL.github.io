"""Semi-infinite LP quadrature design with fixed Gauss-Legendre nodes."""
import numpy as np
import time
import scipy.optimize as so
from scipy.optimize import linprog

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    Omega_m = 0.3
    Omega_L = 0.7

    # Fixed nodes: Gauss-Legendre on [0,1]
    gl8_nodes, gl8_weights = np.polynomial.legendre.leggauss(8)
    gl8_nodes = (gl8_nodes + 1.0) / 2.0
    gl8_weights = gl8_weights / 2.0
    nodes = gl8_nodes  # Fixed, not optimized

    # High-accuracy reference using 256-point Gauss-Legendre
    n_ref = 256
    ref_nodes, ref_weights = np.polynomial.legendre.leggauss(n_ref)
    ref_nodes = (ref_nodes + 1.0) / 2.0
    ref_weights = ref_weights / 2.0

    # Diverse test set for the LP constraints
    z_vals = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0])
    w0_vals = np.array([-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    wa_vals = np.array([-3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0])

    test_cases = []
    for z in z_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                test_cases.append((z, w0, wa))

    n_tests = len(test_cases)
    z_arr = np.array([tc[0] for tc in test_cases])
    w0_arr = np.array([tc[1] for tc in test_cases])
    wa_arr = np.array([tc[2] for tc in test_cases])

    def _compute_inv_E(a_vals, w0, wa):
        """Compute 1/E(a) for given a values and CPL parameters."""
        a_vals = np.clip(a_vals, 1e-15, 1.0)
        log_term = 3.0 * (1.0 + w0 + wa) * np.log(a_vals) - 3.0 * wa * (a_vals - 1.0)
        log_term = np.clip(log_term, -700, 700)
        term_L = Omega_L * np.exp(log_term)
        term_m = Omega_m * (a_vals ** -3)
        E2 = np.maximum(term_m + term_L, 1e-300)
        E = np.sqrt(E2)
        return 1.0 / E

    # Precompute reference integrals and integrand values at our 8 nodes
    ref_vals = np.zeros(n_tests)
    A_matrix = np.zeros((n_tests, 8))  # integrand values at our 8 nodes

    for i, (z, w0, wa) in enumerate(test_cases):
        if time.time() > deadline - 40.0:
            # Fill remaining with zeros and break
            ref_vals[i:] = 0
            A_matrix[i:] = 0
            break
        # Reference integral
        a_ref = 1.0 / (1.0 + z * ref_nodes)
        inv_E_ref = _compute_inv_E(a_ref, w0, wa)
        ref_vals[i] = np.dot(ref_weights, inv_E_ref)

        # Integrand at our 8 nodes
        a_our = 1.0 / (1.0 + z * nodes)
        A_matrix[i] = _compute_inv_E(a_our, w0, wa)

    # Filter out invalid test cases (zero reference)
    valid_mask = ref_vals > 1e-10
    ref_vals = ref_vals[valid_mask]
    A_matrix = A_matrix[valid_mask]
    n_valid = len(ref_vals)

    if n_valid == 0:
        # Fallback: return Gauss-Legendre weights
        return {"nodes": nodes, "weights": gl8_weights}

    # Semi-infinite LP:
    # Minimize t
    # Subject to:
    #   A @ w - t * ref <= 0   (for each test case: A_i @ w <= t * ref_i)
    #   -A @ w - t * ref <= 0  (for each test case: A_i @ w >= -t * ref_i)
    #   sum(w) = 1
    #   w >= 0

    # Variables: [w_0, ..., w_7, t]  (9 variables)
    n_vars = 9

    # Objective: minimize t
    c = np.zeros(n_vars)
    c[8] = 1.0  # minimize t

    # Inequality constraints: A @ w - t * ref <= 0  and  -A @ w - t * ref <= 0
    # For each test case i:
    #   A_i @ w - t * ref_i <= 0  =>  [A_i, -ref_i] @ [w, t] <= 0
    #   -A_i @ w - t * ref_i <= 0  =>  [-A_i, -ref_i] @ [w, t] <= 0

    A_ineq = np.zeros((2 * n_valid, n_vars))
    b_ineq = np.zeros(2 * n_valid)

    for i in range(n_valid):
        A_ineq[2*i, :8] = A_matrix[i]
        A_ineq[2*i, 8] = -ref_vals[i]
        A_ineq[2*i+1, :8] = -A_matrix[i]
        A_ineq[2*i+1, 8] = -ref_vals[i]

    # Equality constraint: sum(w) = 1
    A_eq = np.zeros((1, n_vars))
    A_eq[0, :8] = 1.0
    b_eq = np.array([1.0])

    # Bounds: w_i >= 0, t >= 0
    bounds = [(0, None)] * 8 + [(0, None)]

    # Solve LP
    try:
        if time.time() < deadline - 5.0:
            res = linprog(c, A_ub=A_ineq, b_ub=b_ineq, A_eq=A_eq, b_eq=b_eq,
                          bounds=bounds, method='highs')
            if res.success:
                best_weights = res.x[:8]
                best_t = res.x[8]
            else:
                best_weights = gl8_weights.copy()
                best_t = np.inf
        else:
            best_weights = gl8_weights.copy()
            best_t = np.inf
    except Exception:
        best_weights = gl8_weights.copy()
        best_t = np.inf

    # Ensure weights are positive and sum to 1
    best_weights = np.clip(best_weights, 1e-12, None)
    best_weights = best_weights / np.sum(best_weights)

    # Verify and potentially refine with a second LP pass on a subset
    if time.time() < deadline - 5.0:
        # Evaluate current solution
        qvals = A_matrix @ best_weights
        rel_errs = np.abs(qvals - ref_vals) / ref_vals * 100.0
        current_max_err = np.max(rel_errs)

        # Try a refined LP with tighter tolerance on the worst cases
        # Identify the top 50 worst cases and re-solve
        if n_valid > 50 and current_max_err > 0.1:
            worst_indices = np.argsort(-rel_errs)[:50]
            A_sub = A_matrix[worst_indices]
            ref_sub = ref_vals[worst_indices]

            A_ineq_sub = np.zeros((2 * len(worst_indices), n_vars))
            b_ineq_sub = np.zeros(2 * len(worst_indices))
            for i, idx in enumerate(worst_indices):
                A_ineq_sub[2*i, :8] = A_sub[i]
                A_ineq_sub[2*i, 8] = -ref_sub[i]
                A_ineq_sub[2*i+1, :8] = -A_sub[i]
                A_ineq_sub[2*i+1, 8] = -ref_sub[i]

            try:
                res2 = linprog(c, A_ub=A_ineq_sub, b_ub=b_ineq_sub, A_eq=A_eq, b_eq=b_eq,
                               bounds=bounds, method='highs')
                if res2.success:
                    w2 = res2.x[:8]
                    w2 = np.clip(w2, 1e-12, None)
                    w2 = w2 / np.sum(w2)
                    qvals2 = A_matrix @ w2
                    rel_errs2 = np.abs(qvals2 - ref_vals) / ref_vals * 100.0
                    if np.max(rel_errs2) < current_max_err:
                        best_weights = w2
            except Exception:
                pass

    # Final cleanup
    best_nodes = nodes.copy()
    best_nodes = np.clip(best_nodes, 1e-10, 1.0 - 1e-10)
    for i in range(1, 8):
        if best_nodes[i] <= best_nodes[i-1]:
            best_nodes[i] = best_nodes[i-1] + 1e-10
    best_weights = np.clip(best_weights, 1e-12, None)
    best_weights = best_weights / np.sum(best_weights)

    return {"nodes": best_nodes, "weights": best_weights}
