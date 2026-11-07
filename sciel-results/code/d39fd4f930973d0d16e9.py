import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 280  # 20s buffer before 300s kill
        
        calls_made = 0
        best_x = None
        best_f = float('inf')
        N = self.dim
        budget = self.budget

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if time.time() - start_time > time_limit:
                raise TimeoutError()
            if calls_made >= self.budget:
                raise RuntimeError()
            
            calls_made += 1
            try:
                f_val = func(x)
            except Exception:
                return best_f if best_f != float('inf') else 0.0
            
            if f_val < best_f:
                best_f = f_val
                best_x = np.array(x)
            return f_val

        # Phase 1: Multi-start L-BFGS-B for global basin identification
        # Use 20 deterministic start points spread across the domain
        # This is much more efficient than a 243-point grid for smooth functions
        
        # Generate start points: corners, center, and midpoints of edges/faces
        start_points = []
        
        # Center
        start_points.append(np.zeros(N))
        
        # Corners of the domain
        for bits in range(2**N):
            x = np.array([(-4.0 if (bits >> i) & 1 else 4.0) for i in range(N)])
            start_points.append(x)
        
        # Midpoints along each axis
        for i in range(N):
            for val in [-2.0, 2.0]:
                x = np.zeros(N)
                x[i] = val
                start_points.append(x)
        
        # Additional scattered points
        for i in range(5):
            x = np.zeros(N)
            for j in range(N):
                x[j] = -3.0 + 1.5 * ((i * 7 + j * 3) % 5)
            start_points.append(x)
        
        # Deduplicate and limit to 20 points
        unique_points = []
        for x in start_points:
            if all(np.linalg.norm(x - y) > 1e-6 for y in unique_points):
                unique_points.append(x)
            if len(unique_points) >= 20:
                break
        
        # Evaluate all start points
        start_results = []
        try:
            for x in unique_points:
                f_val = evaluate(x)
                start_results.append((f_val, x))
        except (TimeoutError, RuntimeError):
            pass
        
        # Sort by fitness
        start_results.sort(key=lambda x: x[0])
        
        # If we've used all budget, return best
        if calls_made >= budget:
            return best_x, best_f
        
        # Phase 2: L-BFGS-B from top 5 basins
        # L-BFGS-B exploits local curvature for fast convergence in Rosenbrock valley
        
        def lbfgs_wrapper(x):
            nonlocal calls_made, best_x, best_f
            if time.time() - start_time > time_limit:
                raise TimeoutError()
            if calls_made >= self.budget:
                raise RuntimeError()
            
            calls_made += 1
            try:
                f_val = func(x)
            except Exception:
                return best_f if best_f != float('inf') else 0.0
            
            if f_val < best_f:
                best_f = f_val
                best_x = np.array(x)
            return f_val
        
        # Run L-BFGS-B from top 5 start points
        top_k = min(5, len(start_results))
        for f_val, start_x in start_results[:top_k]:
            if calls_made >= budget or time.time() - start_time > time_limit:
                break
            
            try:
                res = minimize(
                    lbfgs_wrapper,
                    start_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * N,
                    options={
                        'maxiter': 100,
                        'ftol': 1e-12,
                        'gtol': 1e-8,
                        'maxls': 50
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.array(res.x)
                    
            except (TimeoutError, RuntimeError):
                break
            except Exception:
                pass
        
        # Phase 3: Nelder-Mead polish to escape any local minima
        # Nelder-Mead is derivative-free and can handle non-smooth regions
        
        def nm_wrapper(x):
            nonlocal calls_made, best_x, best_f
            if time.time() - start_time > time_limit:
                raise TimeoutError()
            if calls_made >= self.budget:
                raise RuntimeError()
            
            calls_made += 1
            try:
                f_val = func(x)
            except Exception:
                return best_f if best_f != float('inf') else 0.0
            
            if f_val < best_f:
                best_f = f_val
                best_x = np.array(x)
            return f_val
        
        if calls_made < budget and time.time() - start_time < time_limit:
            try:
                # Create initial simplex around best point
                initial_simplex = [best_x.copy()]
                for i in range(N):
                    x_i = best_x.copy()
                    x_i[i] += 0.1
                    initial_simplex.append(x_i)
                
                res = minimize(
                    nm_wrapper,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 500,
                        'xatol': 1e-6,
                        'fatol': 1e-6,
                        'initial_simplex': initial_simplex,
                        'adaptive': True
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.array(res.x)
                    
            except (TimeoutError, RuntimeError):
                pass
            except Exception:
                pass
        
        # Phase 4: Final L-BFGS-B polish for high precision
        if calls_made < budget and time.time() - start_time < time_limit:
            try:
                res = minimize(
                    lbfgs_wrapper,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * N,
                    options={
                        'maxiter': 200,
                        'ftol': 1e-14,
                        'gtol': 1e-10,
                        'maxls': 50
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.array(res.x)
            except (TimeoutError, RuntimeError):
                pass
            except Exception:
                pass

        return best_x, best_f
