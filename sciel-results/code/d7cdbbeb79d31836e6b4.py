"""Root finder for 3 fixed 2-D systems: Nelder-Mead on residual norm."""
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    - Use Nelder-Mead simplex method to minimize ||F(x)||^2.
    - Strictly limit function calls to 490 to ensure we stay within the 500 limit.
    - Return the best point found if the limit is reached or convergence is achieved.
    """
    start_time = time.time()
    time_limit = 110.0  # Safety margin before 120s hard kill
    call_count = 0
    max_calls = 490
    
    def time_ok():
        return (time.time() - start_time) < time_limit

    def objective(x):
        """Minimize 0.5 * ||F(x)||^2."""
        nonlocal call_count
        if call_count >= max_calls:
            return np.inf
        if not time_ok():
            return np.inf
            
        call_count += 1
        try:
            # Ensure x is a numpy array of floats
            x_np = np.asarray(x, dtype=float)
            if not np.all(np.isfinite(x_np)):
                return np.inf
                
            result = F(x_np)
            result = np.asarray(result, dtype=float)
            
            if result.shape != (2,) or not np.all(np.isfinite(result)):
                return np.inf
                
            return 0.5 * np.dot(result, result)
        except Exception:
            return np.inf

    # Initialize Nelder-Mead simplex
    x0 = np.asarray(x0, dtype=float)
    
    # Initial simplex points
    # Start with x0 and two other points
    step_size = 1.0
    if np.linalg.norm(x0) > 1e-6:
        step_size = 0.5 * np.linalg.norm(x0)
    
    x1 = x0.copy()
    x2 = x0.copy()
    x2[0] += step_size
    x3 = x0.copy()
    x3[1] += step_size
    
    # Evaluate initial simplex
    f1 = objective(x1)
    f2 = objective(x2)
    f3 = objective(x3)
    
    # Keep track of best point
    best_x = x1.copy()
    best_f = f1
    if f2 < best_f:
        best_x = x2.copy()
        best_f = f2
    if f3 < best_f:
        best_x = x3.copy()
        best_f = f3

    # Check convergence
    if best_f <= 1e-16: # ||F||^2 <= 1e-16 => ||F|| <= 1e-8
        return best_x

    # Nelder-Mead parameters
    alpha = 1.0  # reflection
    gamma = 2.0  # expansion
    rho = 0.5    # contraction
    sigma = 0.5  # shrink

    for _ in range(1000):  # Max iterations, but will break on call limit
        if call_count >= max_calls or not time_ok():
            break
            
        # Sort vertices
        # Vertices are x1, x2, x3 with functions f1, f2, f3
        pts = [(x1, f1), (x2, f2), (x3, f3)]
        pts.sort(key=lambda p: p[1])
        x1, f1 = pts[0]
        x2, f2 = pts[1]
        x3, f3 = pts[2]
        
        # Update best
        if f1 < best_f:
            best_f = f1
            best_x = x1.copy()
            
        if best_f <= 1e-16:
            break

        # Centroid of the best two points
        centroid = (x1 + x2) / 2.0
        
        # Reflection
        xr = centroid + alpha * (centroid - x3)
        fr = objective(xr)
        
        if fr < f2:
            # Check for expansion
            if fr < f1:
                xe = centroid + gamma * (centroid - x3)
                fe = objective(xe)
                if fe < fr:
                    x3, f3 = xe, fe
                else:
                    x3, f3 = xr, fr
            else:
                x3, f3 = xr, fr
        else:
            # Contraction
            if fr < f3:
                # Outside contraction
                xc = centroid - rho * (centroid - x3)
                fc = objective(xc)
                if fc <= fr:
                    x3, f3 = xc, fc
                else:
                    # Shrink
                    x2 = (x1 + x2) / 2.0
                    f2 = objective(x2)
                    x3 = (x1 + x3) / 2.0
                    f3 = objective(x3)
            else:
                # Inside contraction
                xc = centroid + rho * (x3 - centroid)
                fc = objective(xc)
                if fc < f3:
                    x3, f3 = xc, fc
                else:
                    # Shrink
                    x2 = (x1 + x2) / 2.0
                    f2 = objective(x2)
                    x3 = (x1 + x3) / 2.0
                    f3 = objective(x3)

    return best_x
