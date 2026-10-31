import numpy as np
import time
import random
from scipy.optimize import differential_evolution, minimize

N = 600
DEADLINE = 140

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    def compute_autoconv_max(h):
        """Compute max of autoconvolution of h (assumed sum=1)."""
        size = 2 * N
        h_pad = np.zeros(size)
        h_pad[:N] = h
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real
        return np.max(conv)
    
    # Gaussian basis representation
    # h(x) = sum_{j=1}^{K} w_j * exp(-((x - c_j)^2) / (2 * s_j^2))
    # where x is in [0, N-1] (discrete grid)
    # Parameters: w_j > 0, c_j in [0, N-1], s_j > 0
    # Total parameters: 3*K
    
    K = 10  # Number of Gaussian components
    
    i = np.arange(N, dtype=float)
    
    def params_to_h(params):
        """Convert parameter vector to height array."""
        h = np.zeros(N)
        for j in range(K):
            w_j = params[3*j]
            c_j = params[3*j+1]
            s_j = params[3*j+2]
            if s_j < 1e-6:
                s_j = 1e-6
            h += w_j * np.exp(-((i - c_j)**2) / (2 * s_j**2))
        return h
    
    def objective(params):
        """Objective function: max autoconvolution of normalized h."""
        if time_left() < 1:
            return 1e10
        h = params_to_h(params)
        s = np.sum(h)
        if s < 1e-10:
            return 1e10
        h_norm = h / s
        return compute_autoconv_max(h_norm)
    
    # Define bounds for parameters
    # w_j in [0.01, 10], c_j in [0, N-1], s_j in [1, N/2]
    bounds = []
    for j in range(K):
        bounds.append((0.01, 10.0))   # w_j
        bounds.append((0.0, N-1))     # c_j
        bounds.append((1.0, N/2))     # s_j
    
    # Generate diverse initial populations
    def generate_initial_pop(size):
        """Generate diverse initial parameter vectors."""
        pop = np.zeros((size, 3*K))
        for idx in range(size):
            # Random strategy
            strategy = random.randint(0, 3)
            
            if strategy == 0:
                # Bimodal: two large Gaussians at edges
                for j in range(K):
                    if j < K//2:
                        pop[idx, 3*j] = random.uniform(0.5, 2.0)
                        pop[idx, 3*j+1] = random.uniform(0, N*0.2)
                        pop[idx, 3*j+2] = random.uniform(N/10, N/4)
                    else:
                        pop[idx, 3*j] = random.uniform(0.5, 2.0)
                        pop[idx, 3*j+1] = random.uniform(N*0.8, N-1)
                        pop[idx, 3*j+2] = random.uniform(N/10, N/4)
            elif strategy == 1:
                # Spread out
                for j in range(K):
                    pop[idx, 3*j] = random.uniform(0.1, 1.0)
                    pop[idx, 3*j+1] = j * (N-1) / (K-1) + random.uniform(-N/10, N/10)
                    pop[idx, 3*j+2] = random.uniform(N/8, N/3)
            elif strategy == 2:
                # Central concentration
                for j in range(K):
                    pop[idx, 3*j] = random.uniform(0.1, 0.5)
                    pop[idx, 3*j+1] = random.uniform(N*0.3, N*0.7)
                    pop[idx, 3*j+2] = random.uniform(N/6, N/2)
            else:
                # Fully random
                for j in range(K):
                    pop[idx, 3*j] = random.uniform(0.01, 5.0)
                    pop[idx, 3*j+1] = random.uniform(0, N-1)
                    pop[idx, 3*j+2] = random.uniform(2, N/2)
        return pop
    
    best_h = np.ones(N) / N
    best_val = compute_autoconv_max(best_h)
    
    # Stage 1: Differential Evolution
    if time_left() > 60:
        max_iter = int(min(100, time_left() * 0.5))
        pop_size = min(50, max(20, int(time_left() * 0.3)))
        
        try:
            result = differential_evolution(
                objective,
                bounds=bounds,
                maxiter=max_iter,
                popsize=pop_size,
                tol=1e-6,
                seed=42,
                polish=False,
                init='sobol'
            )
            
            if result.success or result.fun < best_val:
                h_de = params_to_h(result.x)
                s = np.sum(h_de)
                if s > 1e-10:
                    h_de = h_de / s
                    val_de = compute_autoconv_max(h_de)
                    if val_de < best_val:
                        best_val = val_de
                        best_h = h_de.copy()
        except Exception:
            pass
    
    # Stage 2: Local refinement with L-BFGS-B
    if time_left() > 30:
        # Get current best params from best_h via least squares fit
        # Actually, let's just do local search from multiple starting points
        
        # Create several starting points near the best
        start_points = []
        
        # Point 1: fit Gaussians to best_h
        # Simple approach: use the DE result if available
        if 'result' in dir() and hasattr(result, 'x'):
            start_points.append(result.x.copy())
        
        # Point 2: bimodal guess
        p2 = np.zeros(3*K)
        for j in range(K):
            if j < K//2:
                p2[3*j] = 1.0
                p2[3*j+1] = N*0.15
                p2[3*j+2] = N/8
            else:
                p2[3*j] = 1.0
                p2[3*j+1] = N*0.85
                p2[3*j+2] = N/8
        start_points.append(p2)
        
        # Point 3: uniform-ish
        p3 = np.zeros(3*K)
        for j in range(K):
            p3[3*j] = 0.5
            p3[3*j+1] = j * (N-1) / (K-1) if K > 1 else N/2
            p3[3*j+2] = N/4
        start_points.append(p3)
        
        # Point 4: random
        p4 = np.zeros(3*K)
        for j in range(K):
            p4[3*j] = random.uniform(0.1, 2.0)
            p4[3*j+1] = random.uniform(0, N-1)
            p4[3*j+2] = random.uniform(N/10, N/3)
        start_points.append(p4)
        
        for sp in start_points:
            if time_left() < 15:
                break
            
            try:
                res = minimize(
                    objective,
                    sp,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if res.fun < best_val:
                    h_res = params_to_h(res.x)
                    s = np.sum(h_res)
                    if s > 1e-10:
                        h_res = h_res / s
                        val_res = compute_autoconv_max(h_res)
                        if val_res < best_val:
                            best_val = val_res
                            best_h = h_res.copy()
            except Exception:
                pass
    
    # Stage 3: Simulated annealing refinement on the grid
    if time_left() > 20:
        current_h = best_h.copy()
        current_val = best_val
        best_h_sa = best_h.copy()
        best_val_sa = best_val
        
        T = 0.005
        T_min = 1e-7
        cooling = 0.998
        n_iter = int(min(3000, time_left() * 40))
        
        for k in range(n_iter):
            if time_left() < 5:
                break
            
            n_perturb = random.randint(1, 3)
            indices = random.sample(range(N), n_perturb)
            
            new_h = current_h.copy()
            for idx in indices:
                noise = random.gauss(0, T * 0.05)
                new_h[idx] = max(0, current_h[idx] + noise)
            
            s = np.sum(new_h)
            if s < 1e-10:
                continue
            new_h /= s
            
            new_val = compute_autoconv_max(new_h)
            
            if new_val < current_val or random.random() < np.exp(-(new_val - current_val) / T):
                current_h = new_h
                current_val = new_val
                
                if current_val < best_val_sa:
                    best_val_sa = current_val
                    best_h_sa = current_h.copy()
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        if best_val_sa < best_val:
            best_val = best_val_sa
            best_h = best_h_sa
    
    # Stage 4: Fine coordinate descent
    if time_left() > 10:
        current_h = best_h.copy()
        current_val = best_val
        improved = True
        max_rounds = int(min(10, time_left() // 2))
        
        for round_num in range(max_rounds):
            if time_left() < 5:
                break
            
            improved = False
            order = np.random.permutation(N)
            for idx in order:
                if time_left() < 3:
                    break
                
                for delta in [0.05, 0.02, 0.01, -0.05, -0.02, -0.01]:
                    new_h = current_h.copy()
                    new_h[idx] = max(0, current_h[idx] + delta * current_h[idx])
                    s = np.sum(new_h)
                    if s < 1e-10:
                        continue
                    new_h /= s
                    
                    new_val = compute_autoconv_max(new_h)
                    if new_val < current_val - 1e-12:
                        current_h = new_h
                        current_val = new_val
                        improved = True
                        break
            
            if not improved:
                break
        
        if current_val < best_val:
            best_val = current_val
            best_h = current_h
    
    return best_h.tolist()
