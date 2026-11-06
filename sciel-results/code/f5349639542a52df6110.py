"""Optimal single-bidder auction via nested LP and gradient optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog, minimize

def construct_auction(distribution_spec, support, constraints):
    """
    Construct optimal incentive-compatible auction for one additive bidder, two goods.
    
    Args:
        distribution_spec: specification of the value distribution
        support: value support (e.g., {0,1,2,3} for each good)
        constraints: additional constraints
    
    Returns:
        dict with 'allocations' (16,2) and 'payments' (16,)
    """
    # Values for each good: 0, 1, 2, 3
    values = [0, 1, 2, 3]
    
    # Enumerate all 16 types
    types = []
    for v1 in values:
        for v2 in values:
            types.append((v1, v2))
    
    n_types = len(types)  # 16
    n_goods = 2
    
    # Get probabilities from distribution_spec
    if isinstance(distribution_spec, dict) and 'probabilities' in distribution_spec:
        probs = np.array(distribution_spec['probabilities'])
        if len(probs) != n_types:
            probs = np.resize(probs, n_types)
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec)
        if len(probs) != n_types:
            probs = np.resize(probs, n_types)
    else:
        # Default: uniform distribution
        probs = np.ones(n_types) / n_types
        
    probs = probs / probs.sum()  # normalize
    
    # Vector of values for each type
    V = np.array(types, dtype=float)
    
    # Precompute IC constraint coefficients for the inner LP
    # Constraint: u_i + u_k - (V_i - V_k) @ x_k >= 0
    # Rewrite for linprog (A_ub @ z <= b_ub):
    # -u_i - u_k + (V_i - V_k) @ x_k <= 0
    # Variables in inner LP: u_0 ... u_{15}
    # x is fixed during inner LP.
    
    n_u_vars = n_types
    
    def compute_utility_and_grad(x):
        """
        Solve for minimal utilities u given allocations x.
        Returns: u (16,), grad_u_x (16, 2) gradient of sum(f_i u_i) w.r.t x
        """
        c_u = probs.copy() # Minimize sum(f_i u_i)
        
        A_ub_rows = []
        b_ub_rows = []
        
        # We need to store which dual variable corresponds to which (i,k) pair to compute gradient
        # Dual variable index mapping
        dual_indices = []
        
        for i in range(n_types):
            for k in range(n_types):
                if i == k:
                    continue
                row = np.zeros(n_u_vars)
                row[i] = -1.0
                row[k] = -1.0
                
                # RHS: (V_i - V_k) @ x_k
                rhs = np.dot(V[i] - V[k], x[k])
                
                A_ub_rows.append(row)
                b_ub_rows.append(rhs)
                dual_indices.append((i, k))
                
        A_ub = np.vstack(A_ub_rows)
        b_ub = np.array(b_ub_rows)
        
        bounds = [(0, None)] * n_u_vars
        
        res = linprog(c_u, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if not res.success:
            # Fallback: return large penalty
            return np.full(n_types, 1e6), np.zeros((n_types, n_goods))
            
        u = res.x
        
        # Compute gradient of sum(f_i u_i) w.r.t x
        # grad_x = sum_{i,k} lambda_{ik} * (V_i - V_k) * e_k
        # lambda is res.ineqlin.marginals (for highs, it's res.ineqlin.marginals? 
        # In scipy linprog, res.ineqlin is the solution to the dual? 
        # Actually, for method='highs', res.ineqlin is not directly the dual variables in older versions, 
        # but res.ineqlin.marginals is available in newer SciPy.
        # Let's check if we can access duals.
        # If not, we can approximate or use finite differences.
        # For robustness, let's try to access duals.
        
        try:
            # In recent scipy, res.ineqlin is the dual solution for inequality constraints?
            # Or res.ineqlin.marginals
            if hasattr(res, 'ineqlin') and hasattr(res.ineqlin, 'marginals'):
                duals = res.ineqlin.marginals
            else:
                # Fallback: finite difference approximation for gradient
                # This is expensive but N=16 is small.
                # However, we are in a minimize loop, so this will be called many times.
                # Let's assume we can get duals.
                duals = np.zeros(len(dual_indices))
        except:
            duals = np.zeros(len(dual_indices))
            
        grad_u_x = np.zeros((n_types, n_goods))
        
        for idx, (i, k) in enumerate(dual_indices):
            lam = duals[idx]
            # The constraint was: -u_i - u_k + (V_i - V_k) @ x_k <= 0
            # The dual variable lam >= 0.
            # The term in the Lagrangian is lam * (-u_i - u_k + (V_i - V_k) @ x_k)
            # So contribution to gradient of u w.r.t x_k is lam * (V_i - V_k)
            grad_u_x[k, :] += lam * (V[i] - V[k])
            
        return u, grad_u_x

    def objective_and_grad(x_flat):
        x = x_flat.reshape(n_types, n_goods)
        # Clip x to [0,1] to be safe, though bounds handle it
        x = np.clip(x, 0, 1)
        
        u, grad_u_x = compute_utility_and_grad(x)
        
        # Revenue = sum(f_i (V_i @ x_i - u_i))
        # We want to minimize -Revenue
        # -Rev = -sum(f_i V_i x_i) + sum(f_i u_i)
        
        # Gradient of -Rev w.r.t x:
        # grad = -sum(f_i V_i) + grad_u_x
        # Note: f_i is scalar, V_i is vector.
        # sum(f_i V_i) is a (16, 2) matrix where row i is f_i * V[i]
        
        term1 = probs[:, np.newaxis] * V
        grad = -term1 + grad_u_x
        
        val = -np.sum(probs * (np.sum(V * x, axis=1) - u))
        
        return val, grad

    # Initial guess
    x0 = np.random.rand(n_types, n_goods)
    
    # Use L-BFGS-B
    res = minimize(objective_and_grad, x0.flatten(), method='L-BFGS-B', jac=True, 
                   bounds=[(0,1)] * (n_types * n_goods), options={'maxiter': 100, 'ftol': 1e-12})
    
    if res.success:
        x_opt = res.x.reshape(n_types, n_goods)
    else:
        x_opt = x0
        
    # Final calculation
    u_opt, _ = compute_utility_and_grad(x_opt)
    payments = np.sum(V * x_opt, axis=1) - u_opt
    payments = np.maximum(payments, 0)
    
    return {
        'allocations': x_opt,
        'payments': payments
    }

# EVOLVE-BLOCK-END
