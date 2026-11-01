"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 32

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Initial grid placement
    side = int(math.ceil(math.sqrt(N)))
    centers = []
    for i in range(side):
        for j in range(side):
            if len(centers) < N:
                centers.append([(i + 0.5) / side, (j + 0.5) / side])
    
    centers = np.array(centers, dtype=np.float64)
    n = len(centers)
    
    # Compute radii given centers
    def compute_radii(c):
        radii = np.zeros(n)
        for i in range(n):
            r_wall = min(c[i][0], 1 - c[i][0], c[i][1], 1 - c[i][1])
            min_d = r_wall
            for j in range(n):
                if i != j:
                    d = np.linalg.norm(c[i] - c[j])
                    min_d = min(min_d, d / 2)
            radii[i] = min_d
        return radii
    
    def sum_radii(c):
        return np.sum(compute_radii(c))
    
    best_centers = centers.copy()
    best_radii = compute_radii(centers)
    best_sum = sum_radii(centers)
    
    # Smooth objective using soft-min for better gradient behavior
    def smooth_objective(c_flat, beta=50):
        c = c_flat.reshape(n, 2)
        c = np.clip(c, 1e-8, 1 - 1e-8)
        
        total = 0.0
        for i in range(n):
            vals = []
            vals.append(c[i][0])
            vals.append(1 - c[i][0])
            vals.append(c[i][1])
            vals.append(1 - c[i][1])
            for j in range(n):
                if i != j:
                    d = np.linalg.norm(c[i] - c[j]) / 2
                    vals.append(d)
            
            vals = np.array(vals)
            m = np.min(vals)
            soft_min = m - np.log(np.sum(np.exp(beta * (vals - m)))) / beta
            total += soft_min
        
        return -total
    
    # Gradient of smooth objective via central differences
    def smooth_gradient(c_flat, beta=50, eps=1e-6):
        c = c_flat.reshape(n, 2)
        c = np.clip(c, 1e-8, 1 - 1e-8)
        
        f0 = smooth_objective(c_flat, beta)
        grad = np.zeros_like(c_flat)
        
        for k in range(len(c_flat)):
            c_plus = c_flat.copy()
            c_plus[k] += eps
            c_minus = c_flat.copy()
            c_minus[k] -= eps
            f_plus = smooth_objective(c_plus, beta)
            f_minus = smooth_objective(c_minus, beta)
            grad[k] = (f_plus - f_minus) / (2 * eps)
        
        return f0, grad
    
    # Use scipy optimization
    try:
        from scipy.optimize import minimize
        use_scipy = True
    except ImportError:
        use_scipy = False
    
    # Multiple restarts with different initial configurations
    np.random.seed(42)
    restarts = 8
    beta_values = [30, 50, 80]
    
    for restart in range(restarts):
        if time.time() > deadline - 10:
            break
        
        if restart == 0:
            c0 = centers.copy()
        elif restart == 1:
            # Hexagonal packing initial
            c0 = np.zeros((n, 2))
            idx = 0
            for r in range(8):
                for col in range(8):
                    if idx >= n:
                        break
                    x = (col + 0.5 * (r % 2)) * 0.135 + 0.07
                    y = r * 0.135 + 0.07
                    c0[idx] = [x, y]
                    idx += 1
            c0 = np.clip(c0, 0.02, 0.98)
        else:
            # Random perturbation from best or grid
            base = best_centers if restart % 2 == 0 else centers
            c0 = base + np.random.randn(n, 2) * 0.08
            c0 = np.clip(c0, 0.02, 0.98)
        
        beta = beta_values[restart % len(beta_values)]
        
        if use_scipy:
            def obj(x):
                return smooth_objective(x, beta)
            
            def jac(x):
                return smooth_gradient(x, beta)[1]
            
            try:
                result = minimize(
                    obj,
                    c0.flatten(),
                    jac=jac,
                    method='L-BFGS-B',
                    bounds=[(1e-6, 1 - 1e-6)] * (2 * n),
                    options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-9, 'maxls': 50}
                )
                
                c_new = np.clip(result.x.reshape(n, 2), 1e-6, 1 - 1e-6)
                r_new = compute_radii(c_new)
                s_new = np.sum(r_new)
                
                if s_new > best_sum:
                    best_centers = c_new
                    best_radii = r_new
                    best_sum = s_new
                    
            except Exception:
                pass
        else:
            # Fallback: simple gradient descent
            x = c0.flatten().copy()
            lr = 0.001
            for it in range(500):
                if time.time() > deadline - 10:
                    break
                f, g = smooth_gradient(x, beta)
                x -= lr * g
                x = np.clip(x, 1e-6, 1 - 1e-6)
            
            c_new = x.reshape(n, 2)
            r_new = compute_radii(c_new)
            s_new = np.sum(r_new)
            if s_new > best_sum:
                best_centers = c_new
                best_radii = r_new
                best_sum = s_new
    
    # Final high-precision polish: direct local search on exact objective
    if time.time() < deadline - 5:
        # Multi-scale local search
        scales = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
        for scale in scales:
            if time.time() > deadline - 3:
                break
            
            improved = True
            while improved and time.time() < deadline - 3:
                improved = False
                for i in range(n):
                    best_pos = best_centers[i].copy()
                    best_local_sum = best_sum
                    
                    for dx in [-scale, -scale*0.5, 0, scale*0.5, scale]:
                        for dy in [-scale, -scale*0.5, 0, scale*0.5, scale]:
                            if dx == 0 and dy == 0:
                                continue
                            new_pos = best_centers[i] + np.array([dx, dy])
                            if new_pos[0] < 0.001 or new_pos[0] > 0.999:
                                continue
                            if new_pos[1] < 0.001 or new_pos[1] > 0.999:
                                continue
                            
                            new_c = best_centers.copy()
                            new_c[i] = new_pos
                            new_sum = sum_radii(new_c)
                            
                            if new_sum > best_local_sum + 1e-12:
                                best_local_sum = new_sum
                                best_pos = new_pos.copy()
                                improved = True
                    
                    if improved:
                        best_centers[i] = best_pos
                        best_radii = compute_radii(best_centers)
                        best_sum = best_local_sum
    
    # Very fine polish
    if time.time() < deadline - 2:
        for _ in range(3):
            if time.time() > deadline - 1:
                break
            for i in range(n):
                best_pos = best_centers[i].copy()
                best_local_sum = best_sum
                for dx in np.linspace(-0.001, 0.001, 5):
                    for dy in np.linspace(-0.001, 0.001, 5):
                        if abs(dx) < 1e-10 and abs(dy) < 1e-10:
                            continue
                        new_pos = best_centers[i] + np.array([dx, dy])
                        if new_pos[0] < 0.001 or new_pos[0] > 0.999:
                            continue
                        if new_pos[1] < 0.001 or new_pos[1] > 0.999:
                            continue
                        new_c = best_centers.copy()
                        new_c[i] = new_pos
                        new_sum = sum_radii(new_c)
                        if new_sum > best_local_sum + 1e-12:
                            best_local_sum = new_sum
                            best_pos = new_pos.copy()
                best_centers[i] = best_pos
            best_radii = compute_radii(best_centers)
            best_sum = np.sum(best_radii)
    
    # Build result
    result = np.zeros((n, 3))
    for i in range(n):
        result[i] = [best_centers[i][0], best_centers[i][1], best_radii[i]]
    
    return result
# EVOLVE-BLOCK-END
