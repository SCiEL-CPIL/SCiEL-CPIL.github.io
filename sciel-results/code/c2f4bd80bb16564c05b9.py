import numpy as np
import time

def construct():
        N = 200
        start_time = time.time()
        
        # 1. Initialization: Grid
        m = int(np.ceil(np.sqrt(N)))
        # Adjust grid to be more square-ish if N is not a square
        # 200 is close to 14x14=196 or 15x15=225. 
        # 14x15 = 210. 
        # Let's use a rectangular grid that fits N best.
        # Find w, h such that w*h >= N and aspect ratio close to 1.
        best_w, best_h = 14, 15 # 210 spots
        # Actually, let's just use the seed's method but ensure it's valid.
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        C = grid[:N].copy()
        
        # Add small random jitter to break symmetry
        rng = np.random.default_rng(42)
        C += rng.uniform(-0.001, 0.001, size=C.shape)
        
        # Calculate initial score
        def get_radius(C):
            # Pairwise distances
            diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dist, np.inf)
            min_pair = np.min(dist) / 2.0
            
            # Boundary distances
            d_x = 0.5 - np.abs(C[:, 0])
            d_y = 0.5 - np.abs(C[:, 1])
            min_bound = np.min(np.minimum(d_x, d_y))
            
            return min(min_pair, min_bound)
        
        r_best = get_radius(C)
        C_best = C.copy()
        
        # 2. Simulated Annealing
        T = 0.01
        T_min = 1e-5
        cooling = 0.9995
        
        # We can also use a step size that scales with T
        step_scale = 0.01
        
        while T > T_min:
            if time.time() - start_time > 110:
                break
            
            # Pick random index
            i = rng.integers(0, N)
            
            # Generate move
            delta = rng.normal(0, step_scale * T, size=2)
            new_C = C.copy()
            new_C[i] += delta
            
            # Ensure it doesn't fly too far out? 
            # The score function handles boundaries, but if it goes far, 
            # min_bound becomes negative, score drops, unlikely to accept.
            
            r_new = get_radius(new_C)
            
            if r_new > r_best:
                C = new_C
                C_best = new_C.copy()
                r_best = r_new
            else:
                # Accept with probability
                if r_new < 0: 
                    # If invalid (negative radius), skip acceptance logic? 
                    # Actually min_bound can be negative if outside. 
                    # Score is min(pair_dist/2, bound). If bound < 0, score < 0.
                    # We want to maximize.
                    pass
                delta_r = r_new - r_best
                if delta_r < 0:
                    p = np.exp(delta_r / T)
                    if rng.random() < p:
                        C = new_C
                        # Do not update C_best unless r_new > r_best
                else:
                    C = new_C
                    C_best = new_C.copy()
                    r_best = r_new
                    
            T *= cooling
        
        return C_best
