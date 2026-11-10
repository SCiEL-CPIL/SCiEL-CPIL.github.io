# EVOLVE-BLOCK-START
import numpy as np
import time

def _area_batch(x_batch, n=64, M=2000):
    """
    Compute union area for a batch of configurations.
    x_batch: (B, n)
    Returns: (B,) array of areas.
    """
    a = (np.arange(n, dtype=np.float64) + 1.0) / n
    y = (np.arange(M, dtype=np.float64) + 0.5) / M
    w = (1.0 - y) / n
    
    # L shape: (B, M, n)
    # x_batch[:, None, :] -> (B, 1, n)
    # a[None, None, :] * y[None, :, None] -> (1, M, n)
    L = x_batch[:, None, :] + a[None, None, :] * y[None, :, None]
    
    # Sort along the last axis (n)
    Ls = np.sort(L, axis=2)
    
    # Gaps: (B, M, n-1)
    gaps = Ls[:, :, 1:] - Ls[:, :, :-1]
    
    # Union length: w + sum(min(gaps, w))
    # w shape: (M,) -> (1, M, 1)
    w_b = w[None, :, None]
    union_len = w_b + np.sum(np.minimum(gaps, w_b), axis=2)
    
    # Area is mean over M
    areas = np.mean(union_len, axis=1)
    return areas

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    deadline = time.time() + 610.0  # Safety margin
    
    # Precompute constants
    a = (np.arange(n, dtype=np.float64) + 1.0) / n
    
    # Initial best: Aligned construction
    # x_i = 0.5 - a_i
    x_init = 0.5 - a
    best_x = x_init.copy()
    
    # Evaluate initial best with high accuracy first? 
    # No, use the batch evaluator for consistency during search.
    # We will do a final high-accuracy check at the end.
    
    # Use M=2000 for the search loop. It's a good balance.
    M_search = 2000
    
    # Get initial area
    init_areas = _area_batch(best_x[None, :], n, M_search)
    best_area = float(init_areas[0])
    current_x = best_x.copy()
    current_area = best_area
    
    # Simulated Annealing Parameters
    # We want to explore the space. 
    # Start with a larger step size and cool down.
    # The landscape is likely smooth but has many local minima due to the sorting/union operation.
    # Random restarts + SA within each restart might be too slow if restarts are expensive.
    # Instead, do a long SA run with occasional large jumps or just a very long SA run.
    # Given the 600s budget, a single long SA run with a good cooling schedule is often robust.
    # Or, multiple short SA runs from different starts.
    
    # Let's try multiple short SA runs.
    # Each run: 
    # 1. Start from a random or heuristic position.
    # 2. Run SA for T steps.
    # 3. Update global best.
    
    # How many steps per run? 
    # _area_batch with B=1 is equivalent to single eval.
    # Let's benchmark _area_batch speed.
    # (1, 2000, 64) sort is fast.
    # Let's assume ~1ms per eval. 600s / 1ms = 600,000 evals.
    # That's a lot. We can afford many steps.
    
    # Strategy:
    # 1. Generate a few diverse initial points.
    # 2. For each, run a short SA.
    # 3. Keep track of global best.
    
    # Diverse inits:
    # 1. Aligned (current best)
    # 2. Random uniform in [0, 1]
    # 3. Random with some structure?
    
    inits = [
        x_init,
        np.random.rand(n),
        np.random.rand(n) * 2 - 0.5,
        # Maybe a "spread" init?
        np.linspace(0, 1, n)
    ]
    
    # Add some random inits
    for _ in range(5):
        inits.append(np.random.rand(n) * 0.5)
        
    # SA Parameters
    T_start = 0.1
    T_end = 0.001
    alpha = 0.9995 # Cooling rate
    # Steps per run: 
    # We have ~600s. 
    # Let's divide time among runs.
    # Say 10 runs. 60s each.
    # If 1ms/eval, 60,000 evals per run.
    # That's plenty for convergence to a local min.
    
    # Let's implement a dynamic step size SA.
    # Or fixed step size with scaling.
    
    for init_idx, x0 in enumerate(inits):
        if time.time() > deadline:
            break
            
        # Check if this init is already better
        x0_areas = _area_batch(x0[None, :], n, M_search)
        x0_area = float(x0_areas[0])
        
        # If this init is worse than current best, we still try to optimize it, 
        # but we might skip if it's significantly worse and we have time pressure?
        # No, SA can escape. But if it's way worse, it might take long to get back.
        # Let's just run it.
        
        x_curr = x0.copy()
        a_curr = x0_area
        
        # If this start is better than global best, update
        if a_curr < best_area:
            best_area = a_curr
            best_x = x_curr.copy()
            
        # Run SA
        # Determine time budget for this run
        # If we have 10 inits, and 600s total, ~60s per run.
        # But earlier runs might be faster if we break early? No, fixed steps or fixed time.
        # Fixed time is safer.
        
        run_start = time.time()
        run_time_budget = 50.0 # 50 seconds per run
        
        T = T_start
        # Estimate steps:
        # We can't know steps exactly without running.
        # Let's just loop until time runs out for this run.
        
        # Step size: 
        # Start with 0.05, decrease?
        # Or use a fixed step size and rely on T to accept/reject.
        # Fixed step size 0.01 is reasonable for [0,1] scale.
        step_size = 0.05
        
        while time.time() - run_start < run_time_budget:
            # Generate neighbor
            # Randomly pick a few coordinates to perturb? Or all?
            # Perturbing all is more mixing.
            delta = np.random.normal(0, step_size, size=n)
            x_next = x_curr + delta
            
            # Evaluate
            a_next = float(_area_batch(x_next[None, :], n, M_search)[0])
            
            # Metropolis criterion
            if a_next < a_curr:
                x_curr = x_next
                a_curr = a_next
                if a_curr < best_area:
                    best_area = a_curr
                    best_x = x_curr.copy()
            else:
                # Accept with probability exp(-(a_next - a_curr) / T)
                # a_next - a_curr is positive (worse)
                diff = a_next - a_curr
                # If T is small, exp is ~0
                if T > 1e-10:
                    prob = np.exp(-diff / T)
                    if np.random.rand() < prob:
                        x_curr = x_next
                        a_curr = a_next
                        if a_curr < best_area:
                            best_area = a_curr
                            best_x = x_curr.copy()
            
            # Cool down
            T *= alpha
            
            # Optional: Reduce step size as T reduces
            # step_size = max(0.001, 0.05 * (T / T_start))
            # This helps fine-tuning at the end.
            step_size = max(0.001, 0.05 * (T / T_start))
            
            if T < T_end:
                break
                
    # Final Polish
    # Use the best_x found.
    # Do a quick coordinate descent or just return it?
    # The SA should have done a good job.
    # Let's do a few iterations of coordinate descent with line search on the best_x to squeeze out more.
    
    if time.time() < deadline - 10:
        # Quick CD
        M_cd = 2000
        y_cd = (np.arange(M_cd, dtype=np.float64) + 0.5) / M_cd
        w_cd = (1.0 - y_cd) / n
        
        def _area_cd(x, i, delta):
            L = x[None, :] + a[None, :] * y_cd[:, None]
            L[:, i] += delta
            Ls = np.sort(L, axis=1)
            gaps = Ls[:, 1:] - Ls[:, :-1]
            union_len = w_cd + np.sum(np.minimum(gaps, w_cd[:, None]), axis=1)
            return float(np.mean(union_len))
        
        current_area_cd = _area_cd(best_x, 0, 0) # Just to get a baseline, though we have best_area from M_search
        # Note: best_area is from M_search (2000). _area_cd uses M_cd (2000). So they match.
        
        for _ in range(2): # 2 passes
            for i in range(n):
                if time.time() > deadline - 5:
                    break
                # Simple golden section or ternary search?
                # Or just try a few deltas.
                # Since SA already optimized, small steps are enough.
                # Let's try delta in [-0.01, 0, 0.01]
                best_delta = 0
                best_local_area = current_area_cd
                
                # Ternary search on [-0.05, 0.05]
                lo = -0.05
                hi = 0.05
                for _ in range(10):
                    m1 = lo + (hi - lo) / 3
                    m2 = hi - (hi - lo) / 3
                    a1 = _area_cd(best_x, i, m1)
                    a2 = _area_cd(best_x, i, m2)
                    if a1 < a2:
                        hi = m2
                    else:
                        lo = m1
                        
                opt_delta = (lo + hi) / 2
                new_area = _area_cd(best_x, i, opt_delta)
                
                if new_area < current_area_cd - 1e-9:
                    best_x[i] += opt_delta
                    current_area_cd = new_area
                    if current_area_cd < best_area:
                        best_area = current_area_cd
        
    # Final high-accuracy check is not needed for return, just return the positions.
    # The evaluator will compute the score.
    
    return best_x.astype(np.float64)

# EVOLVE-BLOCK-END
