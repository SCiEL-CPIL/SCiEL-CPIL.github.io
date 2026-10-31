"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 100

def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _smooth_max(x, alpha=100.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    return x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (x - x_max))))

def _enforce_constraints(h, target_sum=None):
    """Enforce sum constraint and bounds via iterative projection (more robust)."""
    if target_sum is None:
        target_sum = N / 2.0
    h = h.copy()
    for _ in range(50):
        s = np.sum(h)
        diff = target_sum - s
        if abs(diff) < 1e-15:
            if np.all(h >= -1e-15) and np.all(h <= 1.0 + 1e-15):
                break
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    s = np.sum(h)
    h = h + (target_sum - s) / N
    h = np.clip(h, 0.0, 1.0)
    return h

def _compute_overlap(h):
    shifted = np.zeros(N)
    shifted[:-1] = h[1:]
    return h * (1.0 - shifted)

def _params_to_h(params):
    """Convert parameter vector to step heights using sinusoidal basis.
    
    params = [a1, phi1, a2, phi2, a3, phi3, ...]
    h_i = 0.5 + sum_k a_k * sin(2*pi*k*i/N + phi_k)
    Then project to [0,1] and fix sum to N/2.
    """
    n_harmonics = len(params) // 2
    i = np.arange(N)
    h = np.full(N, 0.5)
    for k in range(n_harmonics):
        a = params[2*k]
        phi = params[2*k+1]
        h += a * np.sin(2 * np.pi * (k+1) * i / N + phi)
    # Project to [0,1]
    h = np.clip(h, 0.0, 1.0)
    # Fix sum
    h = _enforce_constraints(h)
    return h

def _score_from_params(params):
    """Score from parameter vector."""
    h = _params_to_h(params)
    return _score(h)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Start with 2-block solution as initial guess
    A = (3.0 - math.sqrt(5.0)) / 2.0
    B = 1.0 - A
    h0 = np.zeros(N)
    half = N // 2
    h0[:half] = A
    h0[half:] = B
    current_score = _score(h0)
    h_final = h0.copy()

    def _try_update(h_candidate):
        """Try to update best solution if candidate is better and valid."""
        nonlocal h_final, current_score
        h_opt = _enforce_constraints(h_candidate.copy())
        if not (np.all(h_opt >= 0.0) and np.all(h_opt <= 1.0)):
            return False
        new_score = _score(h_opt)
        if new_score < current_score:
            h_final = h_opt.copy()
            current_score = new_score
            return True
        return False

    # Phase 1: SLSQP on raw max-overlap from the 2-block init
    try:
        def objective_raw(h):
            overlap = _compute_overlap(h)
            return float(np.max(overlap))

        constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2.0}]
        bounds = [(0.0, 1.0)] * N

        result = minimize(
            objective_raw,
            h0.copy(),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 2000, 'ftol': 1e-15, 'disp': False}
        )

        if result.fun < current_score:
            _try_update(result.x)

    except Exception:
        pass

    # Phase 2: Global optimization in sinusoidal parameter space
    # This is the structural change: use a low-dimensional parameterized family
    # and optimize over it globally, then polish in full space
    try:
        if time.time() < deadline - 40.0:
            n_harmonics = 3
            n_params = 2 * n_harmonics
            
            def param_objective(params):
                h = _params_to_h(params)
                overlap = _compute_overlap(h)
                return float(np.max(overlap))
            
            # Bounds for parameters: amplitude in [0, 0.45], phase in [0, 2*pi]
            param_bounds = []
            for k in range(n_harmonics):
                param_bounds.append((0.0, 0.45))  # amplitude
                param_bounds.append((0.0, 2 * np.pi))  # phase
            
            # Run differential evolution in parameter space
            result_de = differential_evolution(
                param_objective,
                param_bounds,
                maxiter=100,
                popsize=20,
                tol=1e-12,
                seed=42,
                polish=True,
                polish_args={'method': 'Nelder-Mead', 'options': {'maxiter': 500}},
                update_interval=10
            )
            
            if result_de.fun < current_score:
                h_de = _params_to_h(result_de.x)
                _try_update(h_de)
                
                # Polish the result with SLSQP in full space
                if time.time() < deadline - 20.0:
                    def objective_polish(h):
                        overlap = _compute_overlap(h)
                        return float(np.max(overlap))
                    
                    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2.0}]
                    bounds = [(0.0, 1.0)] * N
                    
                    result_polish = minimize(
                        objective_polish,
                        h_de.copy(),
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints,
                        options={'maxiter': 3000, 'ftol': 1e-15, 'disp': False}
                    )
                    
                    if result_polish.fun < current_score:
                        _try_update(result_polish.x)

    except Exception:
        pass

    # Phase 3: Multi-start sinusoidal optimization with different harmonic counts
    try:
        if time.time() < deadline - 30.0:
            for n_h in [2, 4, 5]:
                if time.time() > deadline - 15.0:
                    break
                
                n_params = 2 * n_h
                
                def param_objective_multi(params, n_h=n_h):
                    h = _params_to_h(params)
                    overlap = _compute_overlap(h)
                    return float(np.max(overlap))
                
                param_bounds = []
                for k in range(n_h):
                    param_bounds.append((0.0, 0.45))
                    param_bounds.append((0.0, 2 * np.pi))
                
                # Multiple random starts
                np.random.seed(123 + n_h)
                best_local = None
                best_local_score = current_score
                
                for trial in range(5):
                    if time.time() > deadline - 10.0:
                        break
                    
                    x0 = np.zeros(n_params)
                    for k in range(n_h):
                        x0[2*k] = np.random.uniform(0.1, 0.4)
                        x0[2*k+1] = np.random.uniform(0, 2*np.pi)
                    
                    result_local = minimize(
                        param_objective_multi,
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10}
                    )
                    
                    if result_local.fun < best_local_score:
                        best_local_score = result_local.fun
                        best_local = result_local.x.copy()
                
                if best_local is not None and best_local_score < current_score:
                    h_best = _params_to_h(best_local)
                    _try_update(h_best)
                    
                    # Polish with SLSQP
                    if time.time() < deadline - 10.0:
                        def objective_polish2(h):
                            overlap = _compute_overlap(h)
                            return float(np.max(overlap))
                        
                        constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2.0}]
                        bounds = [(0.0, 1.0)] * N
                        
                        result_polish2 = minimize(
                            objective_polish2,
                            h_best.copy(),
                            method='SLSQP',
                            bounds=bounds,
                            constraints=constraints,
                            options={'maxiter': 2000, 'ftol': 1e-14, 'disp': False}
                        )
                        
                        if result_polish2.fun < current_score:
                            _try_update(result_polish2.x)

    except Exception:
        pass

    # Phase 4: Final ultra-tight polish with epigraph formulation
    try:
        if time.time() < deadline - 10.0:
            alpha = 500.0
            penalty = 20000.0

            def objective_polish(params):
                h = params[:N]
                t = params[N]
                overlap = _compute_overlap(h)
                smax = _smooth_max(overlap, alpha=alpha)
                return t + penalty * np.maximum(0.0, smax - t)**2

            def constraint_sum(params):
                return np.sum(params[:N]) - N/2.0

            def constraint_epigraph(params):
                h = params[:N]
                t = params[N]
                overlap = _compute_overlap(h)
                return t - np.max(overlap)

            x0_polish = np.concatenate([h_final, [current_score + 0.01]])
            bounds_ext = [(0.0, 1.0)] * N + [(0.0, 1.0)]

            constraints_ext = [
                {'type': 'eq', 'fun': constraint_sum},
                {'type': 'ineq', 'fun': constraint_epigraph}
            ]

            result3 = minimize(
                objective_polish,
                x0_polish,
                method='SLSQP',
                bounds=bounds_ext,
                constraints=constraints_ext,
                options={'maxiter': 3000, 'ftol': 1e-15, 'disp': False}
            )

            if result3.fun < 100.0:
                _try_update(result3.x[:N])

    except Exception:
        pass

    # Phase 5: Final check - ensure we return a valid solution
    h_final = _enforce_constraints(h_final.copy())
    
    # Verify sum is exactly N/2 (within floating point)
    s = np.sum(h_final)
    h_final = h_final + (N/2.0 - s) / N
    h_final = np.clip(h_final, 0.0, 1.0)
    
    return h_final
