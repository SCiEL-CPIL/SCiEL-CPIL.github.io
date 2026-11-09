"""Root finder for 3 fixed two-dimensional systems using a hybrid approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import root, minimize

def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Try multiple robust root-finding methods from different starting points.
    2. Use Broyden's method as the primary fast solver.
    3. Use scipy's hybrid (Powell) method as a robust fallback.
    4. Use optimization of ||F(x)||^2 as a last resort.
    5. Time-guarded to stay within budget.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin for 150s limit
    
    x0 = np.array(x0, dtype=float)
    
    def time_ok():
        return time.time() < deadline

    def eval_F(x):
        """Safely evaluate F and return norm and value."""
        try:
            fx = F(np.asarray(x, dtype=float))
            fx = np.asarray(fx, dtype=float)
            if fx.shape != (2,) or not np.all(np.isfinite(fx)):
                return np.inf, None
            return np.linalg.norm(fx), fx
        except:
            return np.inf, None

    def update_best(x, best_state):
        """Update best solution if x is better."""
        n, _ = eval_F(x)
        if n < best_state['norm']:
            best_state['norm'] = n
            best_state['x'] = np.array(x, dtype=float)
        return best_state['norm'] <= 1e-8

    def broyden_method(x_init, max_iter=100, tol=1e-12):
        """
        Broyden's Good method for root finding.
        Returns (x, success).
        """
        x = np.array(x_init, dtype=float)
        n, Fx = eval_F(x)
        if not np.isfinite(n):
            return x, False
        
        if n <= tol:
            return x, True

        # Initialize Jacobian using finite differences
        h = 1e-7
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            n_plus, Fx_plus = eval_F(x + e)
            n_minus, Fx_minus = eval_F(x - e)
            if np.isfinite(n_plus) and np.isfinite(n_minus):
                J[:, j] = (Fx_plus - Fx_minus) / (2 * h)
            else:
                J[:, j] = 1.0 
        
        # Check if J is singular initially
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            J = J + 1e-6 * np.eye(2)
            try:
                dx = np.linalg.solve(J, -Fx)
            except:
                return x, False

        for _ in range(max_iter):
            if not time_ok():
                break
            
            # Damped Newton step with Broyden update
            # Line search: find alpha such that ||F(x + alpha*dx)|| < ||F(x)||
            alpha = 1.0
            for _ in range(20):
                x_new = x + alpha * dx
                n_new, Fx_new = eval_F(x_new)
                
                if np.isfinite(n_new) and n_new < n:
                    break
                alpha *= 0.5
            
            if not np.isfinite(n_new) or n_new >= n:
                # If line search fails, try a smaller step or break
                if alpha < 1e-10:
                    break
            
            # Accept step
            # Broyden update: J_new = J + (Fx_new - Fx - J*dx) * (x_new - x)^T / ||x_new - x||^2
            delta_x = x_new - x
            delta_F = Fx_new - Fx
            denom = np.dot(delta_x, delta_x)
            if denom > 1e-14:
                rank1 = (delta_F - J @ delta_x) * delta_x.T / denom
                J = J + rank1
            
            x = x_new
            Fx = Fx_new
            n = n_new
            
            if n <= tol:
                return x, True
            
            # Solve for next step
            try:
                dx = np.linalg.solve(J, -Fx)
                if not np.all(np.isfinite(dx)):
                    raise np.linalg.LinAlgError
            except np.linalg.LinAlgError:
                # Reset Jacobian if it becomes singular
                h = 1e-7
                J = np.zeros((2, 2))
                for j in range(2):
                    e = np.zeros(2)
                    e[j] = h
                    n_plus, Fx_plus = eval_F(x + e)
                    n_minus, Fx_minus = eval_F(x - e)
                    if np.isfinite(n_plus) and np.isfinite(n_minus):
                        J[:, j] = (Fx_plus - Fx_minus) / (2 * h)
                    else:
                        J[:, j] = 1.0
                try:
                    dx = np.linalg.solve(J, -Fx)
                    if not np.all(np.isfinite(dx)):
                        break
                except:
                    break
        
        return x, False

    def newton_polish(x_init, max_iter=10):
        """
        Final polishing with Newton's method.
        """
        x = np.array(x_init, dtype=float)
        h = 1e-8
        for _ in range(max_iter):
            if not time_ok():
                break
            n, Fx = eval_F(x)
            if n <= 1e-12:
                return x
            
            J = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                n_plus, Fx_plus = eval_F(x + e)
                n_minus, Fx_minus = eval_F(x - e)
                if np.isfinite(n_plus) and np.isfinite(n_minus):
                    J[:, j] = (Fx_plus - Fx_minus) / (2 * h)
                else:
                    break
            
            try:
                dx = np.linalg.solve(J, -Fx)
                x_new = x + dx
                n_new, _ = eval_F(x_new)
                if np.isfinite(n_new) and n_new < n:
                    x = x_new
                else:
                    break
            except:
                break
        return x

    best_state = {'x': x0.copy(), 'norm': np.inf}
    update_best(x0, best_state)

    # Generate candidate starting points
    scale = max(1.0, np.linalg.norm(x0))
    candidates = [
        x0,
        x0 + np.array([1.0, 0.0]) * scale,
        x0 + np.array([0.0, 1.0]) * scale,
        x0 + np.array([1.0, 1.0]) * scale / np.sqrt(2),
        x0 + np.array([-1.0, 0.0]) * scale,
        x0 + np.array([0.0, -1.0]) * scale,
        x0 + np.array([1.0, -1.0]) * scale / np.sqrt(2),
        x0 * 0.5,
        x0 * 2.0,
        x0 + np.array([0.1, 0.1]) * scale,
        x0 + np.array([-0.1, -0.1]) * scale,
    ]
    
    # Add some random points
    np.random.seed(42)
    for _ in range(5):
        candidates.append(x0 + np.random.uniform(-scale, scale, size=2))

    # Try Broyden from each candidate
    for c in candidates:
        if not time_ok():
            break
        x_sol, success = broyden_method(c)
        if update_best(x_sol, best_state):
             # Polish
             x_final = newton_polish(x_sol)
             update_best(x_final, best_state)
             return best_state['x']

    # If Broyden didn't find a solution, try scipy's hybrid method
    for c in candidates[:5]:  # Limit number of scipy calls
        if not time_ok():
            break
        try:
            sol = root(F, c, method='hybr', tol=1e-12, maxfev=100)
            if sol.success:
                x_sol = sol.x
                if update_best(x_sol, best_state):
                     # Polish
                     x_final = newton_polish(x_sol)
                     update_best(x_final, best_state)
                     return best_state['x']
        except:
            pass

    # If still no solution, try optimization of ||F(x)||^2
    for c in candidates[:3]:
        if not time_ok():
            break
        try:
            def obj(x):
                n, _ = eval_F(x)
                return n**2
            res = minimize(obj, c, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-12, 'fatol': 1e-24})
            if res.success:
                x_sol = res.x
                if update_best(x_sol, best_state):
                     # Polish
                     x_final = newton_polish(x_sol)
                     update_best(x_final, best_state)
                     return best_state['x']
        except:
            pass

    # Final polish on best found
    if best_state['norm'] < np.inf and best_state['norm'] > 1e-8 and time_ok():
        x_final = newton_polish(best_state['x'])
        update_best(x_final, best_state)

    return best_state['x'] if np.all(np.isfinite(best_state['x'])) else x0

# EVOLVE-BLOCK-END
