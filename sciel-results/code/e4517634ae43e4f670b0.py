"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.integrate import quad


def _time_left(deadline):
    return max(0.0, deadline - time.time())


def _E_cpl(a, w0, wa):
    """Hubble parameter H(a)/H0 for CPL model in flat universe."""
    z = 1.0 / a - 1.0
    base = 1.0 + w0 + wa * z
    base = np.maximum(base, 1e-10)
    
    denom = 1.0 + w0
    if abs(denom) < 1e-10:
        exponent = 0.0
    else:
        exponent = wa / denom
        
    return a**(-3.0) * base**exponent


def _integrand(z, w0, wa):
    """1/E(z) for CPL model."""
    a = 1.0 / (1.0 + z)
    return 1.0 / _E_cpl(a, w0, wa)


def _exact_integral(z, w0, wa):
    """Compute exact integral_0^z dz'/E(z') using adaptive quadrature."""
    result, _ = quad(lambda zp: _integrand(zp, w0, wa), 0, z, limit=300, epsabs=1e-12, epsrel=1e-12)
    return result


def _params_to_nodes_weights(params):
    """Convert raw parameters to strictly increasing nodes and positive weights."""
    raw_nodes = np.exp(params[:8])
    nodes = np.cumsum(raw_nodes) / np.sum(raw_nodes)
    log_weights = params[8:]
    weights = np.exp(log_weights)
    weights = weights / np.sum(weights)
    return nodes, weights


def _compute_relative_errors(nodes, weights, models, z_test, exact_integrals):
    """Compute vector of relative errors for all model-redshift pairs."""
    n_models = len(models)
    n_z = len(z_test)
    errors = np.zeros(n_models * n_z)
    
    idx = 0
    for i_model, (w0, wa) in enumerate(models):
        denom = 1.0 + w0
        if abs(denom) < 1e-10:
            exponent = 0.0
        else:
            exponent = wa / denom
            
        for i_z, z in enumerate(z_test):
            z_nodes = z * nodes
            a_nodes = 1.0 / (1.0 + z_nodes)
            z_calc = 1.0 / a_nodes - 1.0
            base = 1.0 + w0 + wa * z_calc
            base = np.maximum(base, 1e-10)
            E_vals = a_nodes**(-3.0) * base**exponent
            vals = 1.0 / E_vals
            
            approx = z * np.dot(weights, vals)
            exact = exact_integrals[idx]
            idx += 1
            
            if abs(exact) < 1e-15:
                errors[i_model * n_z + i_z] = 0.0
            else:
                errors[i_model * n_z + i_z] = abs((approx - exact) / exact)
                
    return errors


def _soft_max(errors, temperature=0.01):
    """Smooth approximation of max using log-sum-exp."""
    if len(errors) == 0:
        return 0.0
    max_err = np.max(errors)
    if max_err < 1e-15:
        return 0.0
    # Scale to avoid numerical issues
    scaled = errors / max_err
    # Use log-sum-exp trick for stability
    log_sum = np.log(np.sum(np.exp(scaled / temperature))) + np.log(temperature)
    return max_err * np.exp(log_sum - np.log(len(errors)) - np.log(temperature))


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s deadline, 5s safety margin

    # Representative CPL models
    models = [
        (0.0, 0.0),
        (-1.0, 0.0),
        (-0.5, -0.5),
        (-1.0, 0.5),
        (-0.8, -0.5),
        (-1.2, 0.3),
        (-0.3, -0.3),
        (-1.0, -0.5),
    ]

    # Test redshifts
    z_test = np.array([0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0])

    # Pre-compute exact integrals to speed up objective function
    exact_integrals = []
    for w0, wa in models:
        for z in z_test:
            try:
                val = _exact_integral(z, w0, wa)
                exact_integrals.append(val)
            except Exception:
                exact_integrals.append(1.0) # Fallback if quad fails
    exact_integrals = np.array(exact_integrals)
    
    model_params = [(w0, wa) for w0, wa in models]
    z_vals = z_test
    
    # Smooth surrogate objective using soft-max
    temperature = 0.01
    
    def objective_smooth(params):
        if _time_left(deadline) < 5.0:
            return 1e10
            
        nodes, weights = _params_to_nodes_weights(params)
        errors = _compute_relative_errors(nodes, weights, model_params, z_vals, exact_integrals)
        return _soft_max(errors, temperature)
    
    # Hard max objective for final evaluation
    def objective_hard(params):
        if _time_left(deadline) < 5.0:
            return 1e10
            
        nodes, weights = _params_to_nodes_weights(params)
        errors = _compute_relative_errors(nodes, weights, model_params, z_vals, exact_integrals)
        return np.max(errors)

    # Initial guess: Gauss-Legendre
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(8)
    nodes0 = 0.5 * (gl_nodes + 1.0)
    weights0 = 0.5 * gl_weights
    
    # Convert initial nodes/weights to raw parameter space
    raw_nodes_init = np.diff(np.concatenate([[0.0], nodes0]))
    params0 = np.concatenate([np.log(raw_nodes_init + 1e-15), np.log(weights0 + 1e-15)])
    
    bounds = []
    for _ in range(8):
        bounds.append((-5.0, 5.0))
    for _ in range(8):
        bounds.append((-10.0, 10.0))

    best_params = params0.copy()
    best_score = objective_hard(params0)
    
    # Phase 1: Differential evolution with smooth surrogate
    try:
        result = differential_evolution(
            objective_smooth,
            bounds,
            maxiter=80,
            popsize=15,
            tol=1e-6,
            seed=42,
            polish=False
        )
        
        if result.fun < 1e10:
            hard_score = objective_hard(result.x)
            if hard_score < best_score:
                best_score = hard_score
                best_params = result.x.copy()
                
    except Exception:
        pass

    # Phase 2: Local polish with L-BFGS-B on hard objective
    if _time_left(deadline) > 10.0:
        try:
            polish_result = minimize(
                objective_hard,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if polish_result.fun < best_score:
                best_score = polish_result.fun
                best_params = polish_result.x.copy()
                
        except Exception:
            pass

    # Phase 3: Final refinement with tighter tolerance
    if _time_left(deadline) > 5.0:
        try:
            final_result = minimize(
                objective_hard,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            if final_result.fun < best_score:
                best_score = final_result.fun
                best_params = final_result.x.copy()
                
        except Exception:
            pass

    # Decode final parameters
    nodes_final, weights_final = _params_to_nodes_weights(best_params)
    
    # Validate and return
    if (np.all(np.diff(nodes_final) > 0) and 
        np.all(nodes_final > 0) and 
        np.all(nodes_final < 1) and
        np.all(weights_final > 0) and
        abs(np.sum(weights_final) - 1.0) < 1e-12):
        return {"nodes": nodes_final, "weights": weights_final}
    
    # Fallback: Gauss-Legendre
    return {"nodes": nodes0, "weights": weights0}
# EVOLVE-BLOCK-END
