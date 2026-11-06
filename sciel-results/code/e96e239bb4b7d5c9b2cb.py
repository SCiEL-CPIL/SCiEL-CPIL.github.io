"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog
from scipy.integrate import quad

def construct():
    """
    Discover an eight-node positive quadrature rule for dimensionless luminosity distance
    in spatially flat, time-varying dark-energy models.
    """
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin before 90s kill

    n_nodes = 8

    # Define E(z) for CPL models (Chevallier-Polarski parameterization)
    def E_cpl(z, w0, wa, Omega_m=0.3):
        """Compute E(z) = H(z)/H0 for CPL model using power-law approximation."""
        Omega_DE = 1.0 - Omega_m
        term_m = Omega_m * (1.0 + z)**3
        term_de = Omega_DE * (1.0 + z)**(3.0 * (1.0 + w0 + wa))
        return np.sqrt(term_m + term_de)

    # Build a comprehensive set of test cosmologies
    test_models = []
    for Om in [0.20, 0.25, 0.30, 0.35, 0.40]:
        for w0, wa in [
            (-1.0, 0.0),
            (-1.2, 0.2),
            (-1.0, 0.4),
            (-0.8, 0.3),
            (-0.5, 0.5),
            (-0.3, 0.2),
            (-1.5, 0.1),
            (-1.0, -0.3),
            (-0.8, -0.5),
            (-0.5, -0.2),
            (-1.2, -0.4),
            (0.0, 0.0),
            (-2.0, 0.0),
            (-1.0, 0.8),
            (-0.5, 1.0),
            (-1.5, 0.5),
            (-1.0, 1.0),
            (-0.5, 0.0),
        ]:
            test_models.append((Om, w0, wa))

    # Test redshifts covering the relevant range
    z_values = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0])

    # Build test cases: each is (E_func, z)
    test_cases = []
    for Om, w0, wa in test_models:
        for z in z_values:
            test_cases.append((Om, w0, wa, z))

    n_cases = len(test_cases)

    # Precompute true integrals using adaptive quadrature
    # The integral is: ∫_0^1 dt / E(z*t)
    true_integrals = np.zeros(n_cases)
    for k, (Om, w0, wa, z) in enumerate(test_cases):
        def integrand(t, Om=Om, w0=w0, wa=wa, z=z):
            return 1.0 / E_cpl(z * t, w0, wa, Om)
        val, _ = quad(integrand, 0, 1, limit=200, epsabs=1e-12, epsrel=1e-12)
        true_integrals[k] = val

    def solve_weights_lp(nodes):
        """
        Solve LP to find weights w that minimize max relative error.
        The quadrature approximates: ∫_0^1 f(t) dt ≈ sum_i w_i f(x_i)
        where f(t) = 1/E(z*t)
        Relative error for case k: |sum_i w_i f_k(x_i) - true_k| / true_k
        We want to minimize max_k relative error subject to sum w_i = 1, w_i >= 0
        """
        # A[k, i] = f_k(x_i) / true_k = 1/(E_k(z_k * x_i) * true_k)
        A = np.zeros((n_cases, n_nodes))
        for k, (Om, w0, wa, z) in enumerate(test_cases):
            for i in range(n_nodes):
                A[k, i] = 1.0 / (E_cpl(z * nodes[i], w0, wa, Om) * true_integrals[k])

        # Minimize t subject to:
        # A[k,:] @ w - t <= 1 for all k
        # -A[k,:] @ w - t <= -1 for all k
        # sum w = 1
        # w >= 0, t >= 0

        c = np.zeros(n_nodes + 1)
        c[-1] = 1.0

        A_ub = np.zeros((2 * n_cases, n_nodes + 1))
        b_ub = np.zeros(2 * n_cases)

        for k in range(n_cases):
            A_ub[2*k, :n_nodes] = A[k, :]
            A_ub[2*k, -1] = -1.0
            b_ub[2*k] = 1.0

            A_ub[2*k+1, :n_nodes] = -A[k, :]
            A_ub[2*k+1, -1] = -1.0
            b_ub[2*k+1] = -1.0

        A_eq = np.zeros((1, n_nodes + 1))
        A_eq[0, :n_nodes] = 1.0
        b_eq = np.array([1.0])

        bounds = [(0, None)] * n_nodes + [(0, None)]

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method='highs', options={'presolve': True})

        if res.success:
            return res.x[:n_nodes], res.x[-1]
        else:
            return np.ones(n_nodes) / n_nodes, 1.0

    def compute_worst_error(nodes, weights):
        """Compute the worst-case relative error across all test cases."""
        max_err = 0.0
        for k, (Om, w0, wa, z) in enumerate(test_cases):
            quad_val = 0.0
            for i in range(n_nodes):
                quad_val += weights[i] / E_cpl(z * nodes[i], w0, wa, Om)
            rel_err = abs(quad_val - true_integrals[k]) / true_integrals[k]
            if rel_err > max_err:
                max_err = rel_err
        return max_err

    # Strategy: Use Chebyshev alternation / iterative minimax approach
    # 1. Start with Gauss-Legendre nodes
    # 2. Solve LP for weights
    # 3. Identify worst-case cases
    # 4. Adjust nodes using a subgradient or direct search
    # 5. Repeat until convergence or time limit

    # Initial nodes: Gauss-Legendre
    gl_nodes, _ = np.polynomial.legendre.leggauss(n_nodes)
    init_nodes = (gl_nodes + 1.0) / 2.0
    init_nodes = np.clip(init_nodes, 1e-10, 1 - 1e-10)
    # Ensure strictly increasing
    for i in range(1, n_nodes):
        if init_nodes[i] <= init_nodes[i-1]:
            init_nodes[i] = init_nodes[i-1] + 1e-12

    best_nodes = init_nodes.copy()
    best_weights, best_err = solve_weights_lp(init_nodes)
    best_weights = np.maximum(best_weights, 1e-15)
    best_weights = best_weights / np.sum(best_weights)
    best_err = compute_worst_error(best_nodes, best_weights)

    # Iterative minimax: adjust nodes to reduce worst-case error
    # Use a simple coordinate ascent / hill climbing on node positions
    # with LP re-solve at each step

    def perturb_nodes(nodes, step_size):
        """Generate a perturbation of nodes while maintaining strict increase in [0,1]."""
        new_nodes = nodes.copy()
        # Randomly perturb interior nodes
        for i in range(1, n_nodes - 1):
            delta = np.random.uniform(-step_size, step_size)
            new_nodes[i] += delta
        # Also allow boundary nodes to move slightly
        new_nodes[0] += np.random.uniform(-step_size * 0.5, step_size * 0.5)
        new_nodes[-1] += np.random.uniform(-step_size * 0.5, step_size * 0.5)
        # Clip to [0,1]
        new_nodes = np.clip(new_nodes, 1e-10, 1 - 1e-10)
        # Enforce strict increase
        for i in range(1, n_nodes):
            if new_nodes[i] <= new_nodes[i-1]:
                new_nodes[i] = new_nodes[i-1] + 1e-12
        # Re-clip
        new_nodes = np.clip(new_nodes, 1e-10, 1 - 1e-10)
        return new_nodes

    # Adaptive step size
    step_size = 0.05
    no_improve_count = 0
    max_no_improve = 50

    while time.time() < deadline - 10.0:
        # Try multiple perturbations
        improved = False
        for trial in range(5):
            new_nodes = perturb_nodes(best_nodes, step_size)

            # Quick check: compute worst error with LP weights
            new_weights, lp_err = solve_weights_lp(new_nodes)
            if not np.all(new_weights >= 0) or abs(np.sum(new_weights) - 1.0) > 1e-6:
                continue

            new_weights = np.maximum(new_weights, 1e-15)
            new_weights = new_weights / np.sum(new_weights)
            actual_err = compute_worst_error(new_nodes, new_weights)

            if actual_err < best_err - 1e-10:
                best_nodes = new_nodes.copy()
                best_weights = new_weights.copy()
                best_err = actual_err
                improved = True
                no_improve_count = 0
                break
            else:
                no_improve_count += 1

        if not improved:
            no_improve_count += 5
            if no_improve_count > max_no_improve:
                # Reduce step size and try again
                step_size *= 0.7
                no_improve_count = 0
                if step_size < 1e-5:
                    break
        else:
            # Keep step size or slightly increase if improving fast
            step_size = min(step_size * 1.1, 0.1)

    # Final: try a few more local refinements with very small steps
    step_size = 1e-4
    for _ in range(20):
        if time.time() > deadline - 5.0:
            break
        improved = False
        for trial in range(3):
            new_nodes = perturb_nodes(best_nodes, step_size)
            new_weights, _ = solve_weights_lp(new_nodes)
            if not np.all(new_weights >= 0):
                continue
            new_weights = np.maximum(new_weights, 1e-15)
            new_weights = new_weights / np.sum(new_weights)
            actual_err = compute_worst_error(new_nodes, new_weights)
            if actual_err < best_err - 1e-12:
                best_nodes = new_nodes.copy()
                best_weights = new_weights.copy()
                best_err = actual_err
                improved = True
                break

    # Final validation and cleanup
    best_nodes = np.clip(best_nodes, 1e-15, 1 - 1e-15)
    for i in range(1, n_nodes):
        if best_nodes[i] <= best_nodes[i-1]:
            best_nodes[i] = best_nodes[i-1] + 1e-12
    best_nodes = np.clip(best_nodes, 1e-15, 1 - 1e-15)

    best_weights = np.maximum(best_weights, 1e-15)
    best_weights = best_weights / np.sum(best_weights)

    # Ensure weights sum to 1 within 1e-12
    best_weights = best_weights / np.sum(best_weights)

    if not np.all(np.diff(best_nodes) > 0) or not np.all(best_nodes > 0) or not np.all(best_nodes < 1) or not np.all(best_weights > 0):
        best_nodes = (np.arange(n_nodes) + 0.5) / n_nodes
        best_weights = np.full(n_nodes, 1.0 / n_nodes)

    return {"nodes": best_nodes, "weights": best_weights}
# EVOLVE-BLOCK-END
