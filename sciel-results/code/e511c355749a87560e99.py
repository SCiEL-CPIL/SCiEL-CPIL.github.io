import numpy as np
import time
from scipy.optimize import minimize
from scipy.integrate import quad

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin for 90s limit

    # Basis functions to capture the behavior of 1/E(a)
    # The integrand in x-space (where x = z/z_max or similar, here x is the variable in [0,1])
    # The formula is: (1+z) * z * sum w_i / E(a_i)
    # Wait, the task says: dL/(c/H0) = (1+z)*z*sum_i weights[i]/E(a_i)
    # where a_i = 1/(1+z*nodes[i]).
    # This implies the integral is approximating: int_0^1 1/E(1/(1+z*x)) dx
    # Let f_z(x) = 1/E(1/(1+z*x)).
    # We want a quadrature rule (nodes, weights) that approximates int_0^1 f_z(x) dx well for all z and all models.
    
    # Since z is a parameter, f_z(x) changes with z.
    # However, the structure is similar.
    # Let's define a set of test functions that mimic the behavior of f_z(x) for different z and models.
    # For z=1, a = 1/(1+x).
    # E(a) ~ sqrt(Om a^-3 + ...).
    # 1/E(a) ~ a^(3/2) / sqrt(Om + ... a^3) ~ 1/sqrt(Om) for small a (high z).
    # For large a (low z), 1/E(a) varies.
    
    # Instead of specific models, we use a basis of functions that span the likely curvature.
    # Common basis: Polynomials, Powers of (1-x), Powers of x, Exponentials.
    
    # Let's define the basis functions in x-space directly.
    # The integrand is g(x) = 1/E(1/(1+z*x)).
    # For a fixed z, this function is smooth.
    # We want a rule that works for all z.
    # This is hard to do with a single basis unless we sample z.
    
    # Alternative Interpretation:
    # The "universal rule" must work for all z.
    # This implies the rule should be accurate for a wide class of functions.
    # Let's use a basis that includes:
    # 1. Polynomials up to degree 7 (standard quadrature requirement).
    # 2. Functions that look like 1/sqrt(1+z*x) or similar, which arise from matter-dominated E(a) ~ a^-1.5 -> 1/E ~ a^1.5 ~ (1+z*x)^-1.5.
    # 3. Functions that look like constant (Lambda dominated).
    
    # Let's construct a set of test functions f_k(x) and their exact integrals I_k.
    # We will minimize the error for these functions.
    
    # Test Functions:
    # 1. x^k for k=0..7
    # 2. (1+x)^p for p in [-3, -2, -1, 0, 1, 2, 3] (approximates matter/radiation/curvature behavior)
    # 3. exp(-gamma * x) for gamma in [1, 2, 5, 10] (approximates dark energy evolution)
    
    test_funcs = []
    true_integrals = []
    
    # Polynomials
    for k in range(8):
        def f(x, k=k):
            return x**k
        test_funcs.append(f)
        # Integral of x^k from 0 to 1 is 1/(k+1)
        true_integrals.append(1.0 / (k + 1))
        
    # Powers of (1+x)
    # Note: a = 1/(1+z*x). If z=1, a=1/(1+x).
    # 1/E(a) for matter dominated (w=0, wa=0) is a^1.5 = (1+x)^-1.5.
    # For Lambda (w=-1), 1/E(a) is constant (if Om=0) or mixed.
    # Let's include (1+x)^p for various p.
    p_vals = [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    for p in p_vals:
        def f(x, p=p):
            return (1.0 + x)**p
        test_funcs.append(f)
        # Integral of (1+x)^p from 0 to 1
        # If p != -1: [ (1+x)^(p+1) / (p+1) ]_0^1 = (2^(p+1) - 1) / (p+1)
        # If p == -1: ln(2)
        if abs(p + 1.0) < 1e-9:
            val = np.log(2.0)
        else:
            val = (2.0**(p + 1.0) - 1.0) / (p + 1.0)
        true_integrals.append(val)
        
    # Exponentials
    # Dark energy often introduces exponential-like behavior in x.
    gamma_vals = [0.5, 1.0, 2.0, 5.0, 10.0]
    for gamma in gamma_vals:
        def f(x, gamma=gamma):
            return np.exp(-gamma * x)
        test_funcs.append(f)
        # Integral of exp(-gamma x) from 0 to 1 = (1 - exp(-gamma)) / gamma
        val = (1.0 - np.exp(-gamma)) / gamma
        true_integrals.append(val)
        
    # Also include 1/(1+x)^p which is more directly related to a^p
    # a = 1/(1+x) (for z=1). So a^p = (1+x)^-p.
    # We already have (1+x)^p. Let's ensure we cover negative powers well.
    
    # We have a good set of test functions.
    # Now, define the objective function.
    
    def decode_params(params):
        # Nodes: 8 params. Use sigmoid to map to (0, 1).
        # To ensure strict ordering and spread, we can use the gap method but allow interior nodes.
        # Or simply optimize nodes directly with bounds and sort?
        # Sorting breaks the derivative flow for optimization.
        # Let's use the gap method:
        # y_0..y_7 > 0.
        # d_i = exp(y_i).
        # Total = sum(d).
        # Nodes are cumulative sums / Total.
        # This forces last node to 1.0.
        # To allow last node < 1.0, we can add an extra gap y_8 for the tail?
        # No, the integral is 0 to 1.
        # If we use Gauss-Legendre, nodes are interior.
        # Let's try a different parameterization:
        # nodes_i = sigmoid(u_i).
        # Then sort nodes.
        # Weights: softmax(v_i).
        # But sorting nodes and keeping weights associated is tricky if nodes cross.
        # However, if we start near GL nodes, they are well separated.
        # Let's use the "sorted sigmoid" approach but penalize crossings?
        # Or better: Optimize the nodes directly in (0,1) with a penalty for non-increasing order.
        
        # Let's stick to the gap method but shift the interval?
        # No, let's use the "cumulative sum of exponentials" but normalize to sum to 1?
        # That puts last node at 1.
        
        # Let's try: nodes = 0.5 * (1 + tanh(u_i))? No, same issue.
        
        # Actually, the previous program's gap method (last node = 1) might be okay if the integrand is well-behaved at 1.
        # But GL nodes are better.
        # Let's use a parameterization that allows interior nodes.
        # n_i = n_{i-1} + delta_i.
        # delta_i = exp(y_i) / (1 + sum(exp(y_j))).
        # This makes sum(delta) = 1.
        # n_0 = delta_0.
        # n_7 = sum(delta_0..delta_7) = 1.
        
        # To get interior nodes, we can define the nodes as the midpoints of the intervals defined by the gaps?
        # Or simply use 9 gaps?
        # Let's use 9 gaps: y_0..y_8.
        # d_i = exp(y_i).
        # Total = sum(d).
        # Breakpoints b_i = sum(d_0..d_i) / Total. b_0=0, b_8=1.
        # Nodes are midpoints: n_i = (b_i + b_{i+1}) / 2.
        # This ensures n_i in (0,1) and strictly increasing.
        
        y = params[:9]
        v = params[9:]
        
        d = np.exp(y)
        total_d = np.sum(d)
        cumsum_d = np.concatenate([[0.0], np.cumsum(d)])
        breakpoints = cumsum_d / total_d
        # breakpoints: 0, ..., 1 (9 points)
        nodes = (breakpoints[:-1] + breakpoints[1:]) / 2.0
        
        # Weights
        v_max = np.max(v)
        exp_v = np.exp(v - v_max)
        weights = exp_v / np.sum(exp_v)
        
        return nodes, weights

    def objective(params):
        if time.time() > deadline:
            return 1e10
            
        nodes, weights = decode_params(params)
        
        # Check validity
        if not np.all(np.diff(nodes) > 1e-6):
            return 1e10
        if not np.all(weights > 0):
            return 1e10
        if abs(np.sum(weights) - 1.0) > 1e-10:
            return 1e10
        if nodes[0] <= 0 or nodes[-1] >= 1:
            return 1e10
            
        # Compute errors on test functions
        max_err = 0.0
        for k, f in enumerate(test_funcs):
            # Approximate integral
            approx_val = np.dot(weights, f(nodes))
            true_val = true_integrals[k]
            
            # Relative error
            if abs(true_val) < 1e-12:
                rel_err = abs(approx_val)
            else:
                rel_err = abs(approx_val - true_val) / abs(true_val)
                
            if rel_err > max_err:
                max_err = rel_err
                
        return max_err

    # Initial guess: Gauss-Legendre nodes and weights
    nodes_gl, weights_gl = np.polynomial.legendre.leggauss(8)
    nodes_gl = (nodes_gl + 1.0) / 2.0
    weights_gl = weights_gl / 2.0
    
    # Encode GL nodes to our gap parameterization (midpoints of 9 gaps)
    # We need to find y such that midpoints match nodes_gl.
    # This is an inverse problem.
    # Approximation: Uniform gaps give nodes at 1/16, 3/16, ..., 15/16.
    # GL nodes are clustered near edges.
    # Let's just initialize with uniform gaps (y=0) and log(weights_gl) for v.
    # This might not be perfect, but the optimizer can move it.
    
    y_init = np.zeros(9)
    v_init = np.log(weights_gl)
    
    x0 = np.concatenate([y_init, v_init])
    
    # Bounds
    # y: [-10, 10]
    # v: [-10, 10]
    bounds = [(-10.0, 10.0)] * 17
    
    best_x = x0
    best_val = objective(best_x)
    
    # Optimization
    # Use Nelder-Mead or L-BFGS-B. Nelder-Mead is robust for small dimensions.
    # L-BFGS-B requires gradient. We can use numerical gradient or just Nelder-Mead.
    # Given 17 dimensions, Nelder-Mead is feasible.
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
    except Exception:
        pass
        
    # If time permits, do a second pass with L-BFGS-B for polish
    if time.time() < deadline - 10:
        try:
            res2 = minimize(objective, best_x, method='L-BFGS-B', bounds=bounds, 
                            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            if res2.fun < best_val:
                best_val = res2.fun
                best_x = res2.x
        except Exception:
            pass

    # Decode final result
    nodes, weights = decode_params(best_x)
    
    # Ensure weights sum to 1
    weights = weights / np.sum(weights)
    
    # Check validity
    if (not np.all(np.diff(nodes) > 1e-6) or not np.all(weights > 0) or 
        abs(np.sum(weights) - 1.0) > 1e-10 or nodes[0] <= 0 or nodes[-1] >= 1):
        # Fallback to GL
        nodes = nodes_gl.copy()
        weights = weights_gl.copy()
    
    return {"nodes": nodes, "weights": weights}
