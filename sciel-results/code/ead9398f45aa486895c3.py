"""Chebyshev equioscillation approach for luminosity distance quadrature."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad

_START_TIME = time.time()
_DEADLINE = _START_TIME + 88.0

def _time_left():
    return _DEADLINE - time.time()

def _E_cpl(a, w0, wa, Om_m=0.3, Om_r=5e-5):
    Om_de = 1.0 - Om_m - Om_r
    term_m = Om_m * a**(-3)
    term_r = Om_r * a**(-4)
    cpl_factor = 1.0 + wa * (1.0 - a)
    term_de = Om_de * a**(-3.0 * (1.0 + w0 + wa)) * (cpl_factor**3)
    val = term_m + term_r + term_de
    val = np.maximum(val, 1e-30)
    return np.sqrt(val)

# Precompute high-accuracy Gauss-Legendre nodes/weights for reference integration
_REF_NODES, _REF_WEIGHTS = np.polynomial.legendre.leggauss(64)
_REF_NODES = 0.5 * (_REF_NODES + 1.0)
_REF_WEIGHTS = 0.5 * _REF_WEIGHTS

def _true_dL(z, w0, wa):
    if z == 0:
        return 0.0
    z_nodes = _REF_NODES * z
    a_vals = 1.0 / (1.0 + z_nodes)
    E_vals = _E_cpl(a_vals, w0, wa)
    integral = np.sum(_REF_WEIGHTS / E_vals) * z
    return (1.0 + z) * integral

def _quad_dL(z, nodes, weights, w0, wa):
    a_vals = 1.0 / (1.0 + z * nodes)
    E_vals = _E_cpl(a_vals, w0, wa)
    integral_approx = np.sum(weights / E_vals)
    return (1.0 + z) * z * integral_approx

def _rel_error_vector(nodes, weights, test_cosmos, test_zs):
    """Compute vector of relative errors across all test cases."""
    errors = []
    for w0, wa in test_cosmos:
        for z in test_zs:
            true_val = _true_dL(z, w0, wa)
            quad_val = _quad_dL(z, nodes, weights, w0, wa)
            if true_val > 1e-10:
                rel_err = abs(quad_val - true_val) / true_val
                errors.append(rel_err)
    return np.array(errors)

def _solve_weights_remez(nodes, test_cosmos, test_zs, max_iter=200):
    """
    Solve for optimal weights using a Remez-like equioscillation approach.
    For fixed nodes, we want to minimize max |sum_i w_i / E(a_i) - I_true| / I_true
    where I_true = true integral / z.
    
    This is equivalent to: minimize max |sum_i w_i * f_i - c| where f_i = 1/E(a_i)
    and c = true_integral / z.
    
    We use a linear programming / bisection approach on the L-infinity norm.
    """
    n = len(nodes)
    
    # Precompute all (cosmo, z) pairs
    cases = []
    for w0, wa in test_cosmos:
        for z in test_zs:
            if z <= 0:
                continue
            a_vals = 1.0 / (1.0 + z * nodes)
            E_vals = _E_cpl(a_vals, w0, wa)
            f_vals = 1.0 / E_vals  # integrand values at nodes
            true_val = _true_dL(z, w0, wa)
            target = true_val / ((1.0 + z) * z)  # target sum
            cases.append((f_vals, target))
    
    if not cases:
        return np.ones(n) / n
    
    # We want to find weights w (positive, sum=1) that minimize
    # max over all cases of |sum_i w_i * f_i - target| / target
    
    # Use a bisection + feasibility approach
    # For a given epsilon, check if there exist w >= 0, sum(w)=1,
    # such that for all cases: |sum_i w_i * f_i - target| <= epsilon * target
    
    def feasible(eps):
        """Check if there exist weights w >= 0, sum(w)=1 with max error <= eps."""
        # This is a linear programming feasibility problem
        # Variables: w_0, ..., w_{n-1} (n variables)
        # Constraints: 
        #   w_i >= 0 for all i
        #   sum(w) = 1
        #   For each case j: target_j * (1-eps) <= sum_i w_i * f_{j,i} <= target_j * (1+eps)
        
        # Use scipy's linprog or manual bisection
        # Since n=8 is small, we can use a simpler approach:
        # Start with uniform weights and do projected gradient or simplex
        
        # Actually, let's use a different approach: 
        # For fixed nodes, the optimal weights in L-infinity sense can be found
        # by solving a sequence of LPs. But for speed, let's use an iterative
        # equioscillation method.
        
        # Simpler: use scipy.optimize.linprog
        try:
            from scipy.optimize import linprog
        except ImportError:
            return False
        
        n_vars = n
        # Minimize 0 subject to constraints
        c = np.zeros(n_vars)
        
        # Inequality constraints: A_ub @ w <= b_ub
        A_ub_list = []
        b_ub_list = []
        
        # w_i >= 0 handled by bounds
        # sum(w) = 1: A_eq @ w = b_eq
        A_eq = np.array([[1.0] * n_vars])
        b_eq = np.array([1.0])
        
        # For each case: |sum_i w_i * f_i - target| <= eps * target
        # => target - eps*target <= sum_i w_i * f_i <= target + eps*target
        # => -sum_i w_i * f_i <= -target + eps*target
        # => sum_i w_i * f_i <= target + eps*target
        
        for f_vals, target in cases:
            A_ub_list.append(f_vals.copy())
            b_ub_list.append(target * (1.0 + eps))
            A_ub_list.append(-f_vals.copy())
            b_ub_list.append(-target * (1.0 - eps))
        
        A_ub = np.array(A_ub_list)
        b_ub = np.array(b_ub_list)
        
        bounds = [(0.0, 1.0)] * n_vars
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                         bounds=bounds, method='highs', options={'presolve': True})
            return res.status == 0
        except Exception:
            return False
    
    # Bisection on epsilon
    lo, hi = 0.0, 1.0
    best_eps = hi
    best_w = None
    
    for _ in range(40):
        if _time_left() < 2.0:
            break
        mid = (lo + hi) / 2.0
        if feasible(mid):
            best_eps = mid
            # Find the actual weights by solving the LP
            try:
                from scipy.optimize import linprog
                c = np.zeros(n)
                A_ub_list = []
                b_ub_list = []
                for f_vals, target in cases:
                    A_ub_list.append(f_vals.copy())
                    b_ub_list.append(target * (1.0 + mid))
                    A_ub_list.append(-f_vals.copy())
                    b_ub_list.append(-target * (1.0 - mid))
                A_ub = np.array(A_ub_list)
                b_ub = np.array(b_ub_list)
                A_eq = np.array([[1.0] * n])
                b_eq = np.array([1.0])
                bounds = [(0.0, 1.0)] * n
                res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                             bounds=bounds, method='highs')
                if res.status == 0:
                    best_w = res.x
            except Exception:
                pass
            hi = mid
        else:
            lo = mid
    
    if best_w is not None and np.all(best_w >= -1e-12):
        best_w = np.maximum(best_w, 0)
        if np.sum(best_w) > 1e-15:
            best_w = best_w / np.sum(best_w)
        return best_w
    else:
        return np.ones(n) / n

def _chebyshev_nodes_8():
    """Generate 8 Chebyshev nodes on [0,1]."""
    k = np.arange(8)
    # Chebyshev roots on [-1,1]: cos((2k+1)*pi/(2n))
    x = np.cos((2 * k + 1) * np.pi / 16)
    # Map to [0,1]
    nodes = 0.5 * (x + 1.0)
    nodes = np.sort(nodes)
    return nodes

def _gauss_legendre_init():
    x_gl, w_gl = np.polynomial.legendre.leggauss(8)
    nodes = 0.5 * (x_gl + 1.0)
    weights = 0.5 * w_gl
    idx = np.argsort(nodes)
    nodes = nodes[idx]
    weights = weights[idx]
    weights = weights / np.sum(weights)
    return nodes, weights

def _evaluate_worst(nodes, weights, test_cosmos, test_zs):
    """Compute worst-case relative error."""
    errs = _rel_error_vector(nodes, weights, test_cosmos, test_zs)
    if len(errs) == 0:
        return 0.0
    return np.max(errs)

def _coordinate_descent_nodes(weights, test_cosmos, test_zs, max_iter=100):
    """
    Coordinate descent on node positions, with weights re-optimized at each step.
    """
    nodes = np.copy(weights)  # placeholder, will be set
    # Actually, let's pass nodes as argument
    pass

def _optimize_nodes_with_remez_weights(initial_nodes, test_cosmos, test_zs):
    """
    Optimize node positions while re-solving for optimal weights at each step.
    Uses coordinate descent on nodes.
    """
    nodes = np.array(initial_nodes, dtype=float)
    n = len(nodes)
    
    # Initial weights via Remez
    weights = _solve_weights_remez(nodes, test_cosmos, test_zs)
    current_err = _evaluate_worst(nodes, weights, test_cosmos, test_zs)
    
    for it in range(max_iter):
        if _time_left() < 3.0:
            break
        
        improved = False
        for i in range(n):
            # Try perturbing node i in both directions
            for delta in [0.01, -0.01, 0.005, -0.005, 0.002, -0.002]:
                new_nodes = np.copy(nodes)
                new_nodes[i] += delta
                # Ensure ordering and bounds
                new_nodes = np.sort(new_nodes)
                new_nodes = np.clip(new_nodes, 1e-8, 1.0 - 1e-8)
                # Ensure strictly increasing
                for j in range(1, 8):
                    if new_nodes[j] <= new_nodes[j-1] + 1e-10:
                        new_nodes[j] = new_nodes[j-1] + 1e-10
                
                # Solve for optimal weights
                new_weights = _solve_weights_remez(new_nodes, test_cosmos, test_zs)
                new_err = _evaluate_worst(new_nodes, new_weights, test_cosmos, test_zs)
                
                if new_err < current_err - 1e-12:
                    nodes = new_nodes
                    weights = new_weights
                    current_err = new_err
                    improved = True
                    break
        
        if not improved:
            # Try smaller steps
            for i in range(n):
                for delta in [0.001, -0.001, 0.0005, -0.0005]:
                    new_nodes = np.copy(nodes)
                    new_nodes[i] += delta
                    new_nodes = np.sort(new_nodes)
                    new_nodes = np.clip(new_nodes, 1e-8, 1.0 - 1e-8)
                    for j in range(1, 8):
                        if new_nodes[j] <= new_nodes[j-1] + 1e-10:
                            new_nodes[j] = new_nodes[j-1] + 1e-10
                    
                    new_weights = _solve_weights_remez(new_nodes, test_cosmos, test_zs)
                    new_err = _evaluate_worst(new_nodes, new_weights, test_cosmos, test_zs)
                    
                    if new_err < current_err - 1e-12:
                        nodes = new_nodes
                        weights = new_weights
                        current_err = new_err
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return nodes, weights, current_err

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 88.0
    
    test_cosmos = [
        (0.0, 0.0),
        (-1.0, 0.0),
        (0.5, -1.0),
        (-1.5, 0.5),
        (-0.5, 0.5),
        (0.3, -0.3),
        (-2.0, 1.0),
        (1.0, -2.0),
        (-0.8, 0.8),
        (0.7, -0.7),
    ]
    test_zs = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    
    best_nodes = None
    best_weights = None
    best_err = np.inf
    
    # Multiple initializations
    initializations = []
    
    # 1. Chebyshev nodes
    cheb_nodes = _chebyshev_nodes_8()
    initializations.append(cheb_nodes)
    
    # 2. Gauss-Legendre nodes
    gl_nodes, gl_weights = _gauss_legendre_init()
    initializations.append(gl_nodes)
    
    # 3. Clenshaw-Curtis nodes
    k = np.arange(8)
    cc_nodes = 0.5 * (1.0 + np.cos(np.pi * k / 7.0))
    cc_nodes = np.sort(cc_nodes)
    initializations.append(cc_nodes)
    
    # 4. Equally spaced
    eq_nodes = np.linspace(0.05, 0.95, 8)
    initializations.append(eq_nodes)
    
    # 5. Slightly perturbed Chebyshev
    for _ in range(5):
        pert = cheb_nodes + np.random.randn(8) * 0.05
        pert = np.sort(pert)
        pert = np.clip(pert, 1e-8, 1.0 - 1e-8)
        for j in range(1, 8):
            if pert[j] <= pert[j-1] + 1e-10:
                pert[j] = pert[j-1] + 1e-10
        initializations.append(pert)
    
    # Optimize from each initialization
    for init_nodes in initializations:
        if _time_left() < 8.0:
            break
        
        try:
            nodes, weights, err = _optimize_nodes_with_remez_weights(
                init_nodes, test_cosmos, test_zs, max_iter=50)
            
            if err < best_err:
                best_err = err
                best_nodes = nodes.copy()
                best_weights = weights.copy()
        except Exception:
            pass
    
    # If we didn't find anything good, fall back to GL
    if best_nodes is None:
        best_nodes, best_weights = _gauss_legendre_init()
    else:
        # Final refinement with more iterations
        if _time_left() > 10.0:
            try:
                nodes, weights, err = _optimize_nodes_with_remez_weights(
                    best_nodes, test_cosmos, test_zs, max_iter=100)
                if err < best_err:
                    best_err = err
                    best_nodes = nodes.copy()
                    best_weights = weights.copy()
            except Exception:
                pass
    
    # Ensure validity
    best_nodes = np.clip(best_nodes, 1e-12, 1.0 - 1e-12)
    for i in range(1, 8):
        if best_nodes[i] <= best_nodes[i-1]:
            best_nodes[i] = best_nodes[i-1] + 1e-12
    best_weights = np.abs(best_weights)
    if np.sum(best_weights) < 1e-15:
        best_weights = np.ones(8) / 8.0
    best_weights = best_weights / np.sum(best_weights)
    
    return {"nodes": best_nodes, "weights": best_weights}
