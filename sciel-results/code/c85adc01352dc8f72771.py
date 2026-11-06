"""
Strategy: Decoupled Deterministic Node Search + Constrained Weight Optimization.

1. Instead of optimizing 16 variables simultaneously with a generic optimizer,
   we parameterize the nodes to maintain their structural properties (monotonicity,
   spacing) while allowing them to adapt to the specific non-polynomial shape of
   1/E(z).
   
   We use a parameterization:
   t_i = (1 + \alpha * x_i) / 2 + \beta * (x_i - 0.5)
   where x_i are standard Gauss-Legendre nodes on [-1, 1].
   \alpha controls the "spread" (scaling) and \beta controls the "skew" (shift of the center of mass relative to the midpoint).
   
   Actually, a simpler and more robust parameterization for nodes in [0,1] that are strictly increasing:
   Let x_i be GL nodes on [-1,1].
   Map to [0,1] via affine transform: t_i = (1 + s * x_i + shift) / 2.
   This might not capture all shapes.
   
   Better: Use a polynomial perturbation of the GL nodes.
   t_i = (1 + x_i)/2 + \epsilon * P(x_i)
   where P(x) is a low-degree polynomial with coefficients \epsilon_k.
   However, ensuring strict increase and bounds is tricky.
   
   Let's stick to a robust 2D search:
   The integrand 1/E(z*t) is monotonic in t (decreasing if E is increasing, which it is for z>0).
   The "mass" of the integral is concentrated where the integrand is large (small t).
   Standard GL nodes are symmetric. We might need to shift nodes towards t=0.
   
   We will search over two parameters:
   1. A "shift" parameter that moves the entire node distribution towards 0 or 1.
   2. A "stretch" parameter that changes the effective width.
   
   Specifically, we map the standard GL nodes x_i \in [-1, 1] to t_i \in [0, 1] using:
   t_i = (1 + x_i * scale + shift) / 2
   Constraints: t_i must be in (0, 1).
   This affine map preserves the relative spacing ratios of the GL nodes but allows global translation and scaling.
   
   Wait, affine maps only have 2 degrees of freedom. The error might depend on the specific non-linear shape.
   Let's use a cubic mapping:
   t_i = (1 + x_i)/2 + c1 * x_i^2 + c2 * x_i^3 ... this gets complex.
   
   Alternative:
   Use a grid search on the "center" of the mass distribution.
   Let's try searching over a 2D grid of (shift, scale) for the affine transform of GL nodes.
   For each candidate node set, we compute the optimal weights via Non-Negative Least Squares (NNLS) or Constrained Least Squares (sum w=1, w>0) against a large set of reference integrals.
   The reference integrals are computed once using high-precision `quad`.
   The weights are then found by solving a linear system or optimization problem.
   We evaluate the worst-case error for the resulting (nodes, weights) pair.
   We pick the best pair.
   
   This is much faster because:
   - The node search is a coarse grid (e.g., 20x20 = 400 points).
   - The weight optimization is a linear/convex problem (fast).
   - The reference data is pre-computed and reused.
"""

import time
import numpy as np
from scipy.optimize import nnls, minimize
from scipy.integrate import quad
from scipy.special import roots_legendre

_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0

def _E_of_z(z, w0, wa, Omega_m=0.3):
    """Compute E(z) = H(z)/H0."""
    Omega_de = 1.0 - Omega_m
    # Handle z=0 carefully
    if np.any(z == 0):
        return np.where(z == 0, 1.0, np.sqrt(Omega_m * (1 + z)**3 + Omega_de * (1 + z)**(3 * (1 + w0 + wa * z))))
    return np.sqrt(Omega_m * (1 + z)**3 + Omega_de * (1 + z)**(3 * (1 + w0 + wa * z)))

def _true_integral(z, w0, wa, Omega_m=0.3):
    """Compute \int_0^1 dt / E(z*t) using quad."""
    if z == 0:
        return 1.0
    def integrand(t):
        return 1.0 / _E_of_z(z * t, w0, wa, Omega_m)
    val, _ = quad(integrand, 0, 1, limit=100, epsabs=1e-12, epsrel=1e-12)
    return val

def _approx_integral(z, w0, wa, nodes, weights, Omega_m=0.3):
    """Compute \sum w_i / E(z*t_i)."""
    if z == 0:
        return np.sum(weights)
    z_nodes = z * nodes
    E_vals = _E_of_z(z_nodes, w0, wa, Omega_m)
    return np.dot(weights, 1.0 / E_vals)

def construct():
    if time.time() > _DEADLINE - 15:
        # Fallback: Standard Gauss-Legendre
        n = 8
        nodes, weights = roots_legendre(n)
        nodes = (nodes + 1) / 2
        weights = weights / 2
        return {"nodes": nodes, "weights": weights}

    # 1. Generate a large, diverse test set for training/validation
    # We need to cover the parameter space likely to be tested.
    # w0 in [-1.5, -0.5], wa in [-0.5, 0.5]
    # z in [0.01, 5.0]
    
    w0_vals = np.linspace(-1.5, -0.5, 15)
    wa_vals = np.linspace(-0.5, 0.5, 15)
    z_vals = np.logspace(-2, np.log10(5.0), 30)
    
    test_set = []
    for w0 in w0_vals:
        for wa in wa_vals:
            for z in z_vals:
                test_set.append((z, w0, wa))
    
    # Pre-compute true integrals
    num_tests = len(test_set)
    true_vals = np.zeros(num_tests)
    for i, (z, w0, wa) in enumerate(test_set):
        true_vals[i] = _true_integral(z, w0, wa)

    # 2. Define node parameterization
    # Base: Gauss-Legendre nodes on [-1, 1]
    n = 8
    gl_nodes_std, _ = roots_legendre(n)
    
    # We will search over affine transformations: t = (1 + scale * x + shift) / 2
    # Constraints: 0 < t_i < 1 for all i.
    # Since x_i \in [-1, 1], t_min = (1 - scale + shift)/2, t_max = (1 + scale + shift)/2.
    # We need 1 - scale + shift > 0 and 1 + scale + shift < 2.
    # i.e., shift > scale - 1 and shift < 1 - scale.
    # Also scale > 0.
    
    # Let's define a grid for scale and shift.
    # Scale is effectively the "width" of the node distribution in [0,1].
    # GL nodes span [-1, 1], so mapped to [0,1] with scale=1, shift=0.
    # If we want to shift mass to the left (smaller t), we decrease shift.
    # If we want to compress/expand, we change scale.
    
    # Let's try a 2D grid search.
    # Scale range: [0.5, 1.5] (don't go too far out of bounds)
    # Shift range: depends on scale.
    
    best_score = np.inf
    best_nodes = None
    best_weights = None
    
    # Grid sizes
    scale_grid = np.linspace(0.6, 1.4, 15)
    # For each scale, valid shift range:
    # (1 - scale + shift) > 0 => shift > scale - 1
    # (1 + scale + shift) < 2 => shift < 1 - scale
    # Let's use 10 points for shift in the valid range.
    
    for scale in scale_grid:
        min_shift = scale - 1 + 0.01  # small margin
        max_shift = 1 - scale - 0.01
        if min_shift >= max_shift:
            continue
        
        # We can also consider that the "center" of the GL nodes is 0.
        # The affine map t = (1 + s*x + c)/2 has center (1+c)/2.
        # Standard GL has center 0.5 (c=0).
        # Shifting c moves the center.
        
        shift_grid = np.linspace(min_shift, max_shift, 10)
        
        for shift in shift_grid:
            # Compute candidate nodes
            cand_nodes = (1 + scale * gl_nodes_std + shift) / 2.0
            
            # Check strict increase and bounds
            if not np.all(np.diff(cand_nodes) > 1e-6):
                continue
            if not (np.all(cand_nodes > 1e-6) and np.all(cand_nodes < 1 - 1e-6)):
                continue
            
            # Build matrix A for this node set
            # A[i, j] = 1/E(z_i * t_j)
            # We want to find w such that A @ w \approx true_vals
            # Constraints: sum(w) = 1, w > 0
            
            # Pre-compute 1/E for all test points and these nodes
            # This is expensive if done naively in the loop, but vectorizable.
            # z_vals is (30, 1), w0/wa are fixed per row of test_set? No, test_set is flattened.
            # Let's vectorize the E computation for all tests and these specific nodes.
            
            # test_set is a list of tuples. Let's convert to arrays for speed.
            # We already have true_vals. We need A.
            
            # To avoid rebuilding A from scratch for every (scale, shift), we can't easily do it
            # because z*t changes.
            # However, the number of tests is 15*15*30 = 6750.
            # The number of grid points is 15*10 = 150.
            # Total E computations: 6750 * 150 * 8 = 8.1 million.
            # This might be too slow in Python loops, but vectorized NumPy should handle it.
            
            # Let's optimize the inner loop.
            # Instead of computing A for every grid point, we can just compute the error
            # for a *fixed* set of weights? No, weights depend on nodes.
            
            # We must solve for weights for each node set.
            
            # Construct A
            # z_array: (num_tests, 1)
            # t_array: (1, n)
            # z_nodes: (num_tests, n)
            
            z_arr = np.array([t[0] for t in test_set])
            w0_arr = np.array([t[1] for t in test_set])
            wa_arr = np.array([t[2] for t in test_set])
            
            # z_nodes = z_arr[:, None] * cand_nodes[None, :]
            z_nodes = z_arr[:, None] * cand_nodes[None, :]
            
            # Compute E(z_nodes)
            # E = sqrt(Omega_m (1+z)^3 + Omega_de (1+z)^(3(1+w0+wa z)))
            # This is vectorized over the 2D array.
            
            # To save time, we can subsample the test set for the grid search phase.
            # Use a subset of 500 points for the grid search, then refine with full set.
            
            if best_nodes is None:
                # First iteration, maybe use full set? No, use subsample for speed.
                pass
            
            # Subsample indices for speed
            # We'll use a fixed random subsample of size 1000
            # But we need deterministic results.
            # Let's just use every 5th point.
            step = max(1, num_tests // 1000)
            indices = np.arange(0, num_tests, step)
            
            z_sub = z_arr[indices]
            w0_sub = w0_arr[indices]
            wa_sub = wa_arr[indices]
            true_sub = true_vals[indices]
            
            z_nodes_sub = z_sub[:, None] * cand_nodes[None, :]
            
            # Calculate E
            Om_m = 0.3
            Om_de = 0.7
            one_plus_z = 1 + z_nodes_sub
            term_m = Om_m * one_plus_z**3
            # w0, wa are vectors of length len(indices)
            # We need to broadcast w0, wa to match z_nodes_sub shape (len(indices), n)
            # w0_sub[:, None] * ones(1, n)
            
            # Exponent: 3 * (1 + w0 + wa * z)
            # z here is the actual redshift z_i * t_j
            # So exponent varies with j as well.
            
            # exp_arg = 3 * (1 + w0_sub[:, None] + wa_sub[:, None] * z_nodes_sub)
            exp_arg = 3 * (1 + w0_sub[:, None] + wa_sub[:, None] * z_nodes_sub)
            term_de = Om_de * one_plus_z**exp_arg
            
            E_sub = np.sqrt(term_m + term_de)
            A_sub = 1.0 / E_sub
            
            # Solve for weights
            # Minimize || A_sub w - true_sub ||^2
            # Subject to sum(w)=1, w>=0
            
            # We can use scipy.optimize.nnls, but it doesn't handle sum(w)=1 directly.
            # We can add a row of 1s to A and 1 to b?
            # No, that changes the objective.
            
            # Better: Use scipy.optimize.lsq_linear with bounds and equality constraint?
            # Or just use least_squares with a penalty for sum(w)-1.
            
            # Given the small size (8 vars), we can use a simple projected gradient or just
            # solve the unconstrained LS and project?
            # Unconstrained LS might yield negative weights.
            
            # Let's use scipy.optimize.nnls on an augmented system?
            # Actually, for 8 variables, we can just use `scipy.optimize.minimize` with SLSQP
            # or `scipy.optimize.lsq_linear` (which supports bounds but not equality).
            
            # Let's use `scipy.optimize.lsq_linear` with bounds [0, 1] and then normalize?
            # Normalization might break the optimality.
            
            # Alternative: Use the fact that we want to minimize error.
            # Let's try `scipy.optimize.nnls` first. If the sum is not 1, we can scale?
            # No, scaling changes the value.
            
            # Let's use a simple grid of "total weight" normalization? No.
            
            # Let's use `scipy.optimize.least_squares` with bounds and a large penalty for sum(w)-1.
            # Or better: parameterize w = softmax(v) / sum(softmax(v))?
            # Or just use `scipy.optimize.minimize` with method='SLSQP' for each node set.
            # 150 node sets * SLSQP overhead might be too slow.
            
            # Let's try a faster approach:
            # Solve unconstrained LS: w_ls = pinv(A) b
            # If w_ls has negatives, project to nearest point in the simplex (w>=0, sum=1).
            # Projection onto simplex is O(n log n).
            
            def project_simplex(v):
                """Project v onto the probability simplex."""
                n = len(v)
                u = np.sort(v)[::-1]
                cssv = np.cumsum(u)
                rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
                theta = (cssv[rho] - 1) / (rho + 1.0)
                w = np.maximum(v - theta, 0)
                return w

            # Compute LS solution
            try:
                w_ls, _, _, _ = np.linalg.lstsq(A_sub, true_sub, rcond=None)
                w_proj = project_simplex(w_ls)
                # Normalize just in case (projection should sum to 1, but floating point)
                w_proj = w_proj / np.sum(w_proj)
            except:
                continue
            
            # Evaluate worst-case error on the FULL test set (or a larger validation subset)
            # To save time, we evaluate on the full set only for the top candidates?
            # Or just use the subsample for the grid search, and refine later.
            
            # For now, compute error on subsample
            # A_sub is already computed.
            approx_sub = A_sub @ w_proj
            rel_errs = np.abs(approx_sub - true_sub) / true_sub
            worst_err = np.max(rel_errs)
            
            if worst_err < best_score:
                best_score = worst_err
                best_nodes = cand_nodes.copy()
                best_weights = w_proj.copy()
                
                # Check time
                if time.time() > _DEADLINE - 20:
                    break
        if time.time() > _DEADLINE - 20:
            break
    
    if best_nodes is None:
        # Fallback
        nodes, weights = roots_legendre(8)
        nodes = (nodes + 1) / 2
        weights = weights / 2
        return {"nodes": nodes, "weights": weights}

    # 3. Refine with full test set and local optimization of weights
    # Use the best nodes found.
    # Solve for weights using the FULL test set with constrained optimization.
    
    z_arr = np.array([t[0] for t in test_set])
    w0_arr = np.array([t[1] for t in test_set])
    wa_arr = np.array([t[2] for t in test_set])
    
    z_nodes_full = z_arr[:, None] * best_nodes[None, :]
    
    one_plus_z = 1 + z_nodes_full
    Om_m = 0.3
    Om_de = 0.7
    term_m = Om_m * one_plus_z**3
    exp_arg = 3 * (1 + w0_arr[:, None] + wa_arr[:, None] * z_nodes_full)
    term_de = Om_de * one_plus_z**exp_arg
    E_full = np.sqrt(term_m + term_de)
    A_full = 1.0 / E_full
    
    # Optimize weights to minimize worst-case error on full set
    # Since the problem is convex in weights (linear approx, linear constraints),
    # we can use a more robust solver.
    
    # Let's use scipy.optimize.lsq_linear with bounds, then check sum.
    # Actually, lsq_linear minimizes ||Aw-b||. It doesn't enforce sum(w)=1.
    # We can add a large weight to the sum constraint?
    
    # Let's stick to the projection method but on the full set.
    try:
        w_ls_full, _, _, _ = np.linalg.lstsq(A_full, true_vals, rcond=None)
        w_opt = project_simplex(w_ls_full)
        w_opt = w_opt / np.sum(w_opt)
    except:
        w_opt = best_weights

    # 4. Final local refinement of nodes?
    # Given the time budget, we might not have time for a full node refinement.
    # The affine search covered a good range.
    
    # Let's do a quick local refinement of weights using least_squares on the full set
    # to ensure they are as optimal as possible for the fixed nodes.
    
    def weight_obj(w):
        # w is 8x1
        approx = A_full @ w
        rel_errs = (approx - true_vals) / true_vals
        # Minimize sum of squared relative errors
        return np.sum(rel_errs**2)
    
    # Initial guess
    w0_init = w_opt
    
    # Bounds
    bounds = [(0, 1)] * 8
    
    # We need to enforce sum(w)=1.
    # Let's parameterize w = softmax(x) / sum(softmax(x))?
    # Or just use SLSQP with equality constraint.
    
    def constraint_sum(w):
        return np.sum(w) - 1.0
    
    cons = [{'type': 'eq', 'fun': constraint_sum}]
    
    try:
        res = minimize(weight_obj, w0_init, method='SLSQP', bounds=bounds, 
                       constraints=cons, options={'maxiter': 100, 'ftol': 1e-10})
        if res.success:
            w_final = res.x
        else:
            w_final = w_opt
    except:
        w_final = w_opt
    
    # Ensure strict positivity and sum
    w_final = np.maximum(w_final, 1e-10)
    w_final = w_final / np.sum(w_final)
    
    # Final check: ensure nodes are strictly increasing
    if not np.all(np.diff(best_nodes) > 1e-6):
        # Fallback to GL
        nodes, weights = roots_legendre(8)
        nodes = (nodes + 1) / 2
        weights = weights / 2
        return {"nodes": nodes, "weights": weights}
        
    return {"nodes": best_nodes, "weights": w_final}

# EVOLVE-BLOCK-END
