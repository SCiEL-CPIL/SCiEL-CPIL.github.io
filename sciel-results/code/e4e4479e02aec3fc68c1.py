import numpy as np
import time
import math

def construct():
    """
    Finds 24 coefficients in {+1, -1} minimizing the sup norm of P(z) on |z|=1.
    Uses a hybrid approach:
    1. Fast local search (1-3 bit flips) using a dense grid + golden section refinement.
    2. A Genetic Algorithm (GA) to escape local minima, using the same fast evaluation.
    3. Final high-precision polish on the best candidate found.
    
    The objective is to minimize max_{|z|=1} |P(z)| / sqrt(24).
    The raw value minimized is max_{|z|=1} |P(z)|.
    """
    N_COEFFS = 24
    start_time = time.time()
    # The evaluator kills at 150s. We need to return well before.
    # Let's aim to finish by 145s to be safe.
    TIME_LIMIT = 145.0
    
    def time_left():
        return TIME_LIMIT - (time.time() - start_time)

    # --- Evaluation Functions ---
    
    # Precompute a base grid for fast evaluation
    # 2048 points is a good balance. The peaks of these polynomials are usually
    # captured well enough by this grid for the purpose of local search.
    N_GRID = 2048
    thetas = np.linspace(0, 2 * np.pi, N_GRID, endpoint=False)
    
    # Precompute powers of z for the grid
    # Z[i, k] = exp(i * theta_i * k)
    # Shape: (N_GRID, N_COEFFS)
    Z = np.zeros((N_GRID, N_COEFFS), dtype=np.complex128)
    Z[:, 0] = 1.0
    for k in range(1, N_COEFFS):
        Z[:, k] = Z[:, k-1] * np.exp(1j * thetas)

    def calc_peak_fast(coeffs, refine_top=5, refine_iters=20):
        """
        Calculate the peak magnitude of P(z) on the unit circle.
        Uses dense sampling and golden-section refinement for top peaks.
        coeffs: numpy array of shape (N_COEFFS,) with values +1/-1
        """
        # P = Z @ coeffs
        P = Z.dot(coeffs)
        mag2 = np.abs(P) ** 2
        peak_val = np.max(mag2)
        
        if refine_top <= 0 or N_GRID <= 10:
            return math.sqrt(peak_val)
            
        # Find indices of top peaks
        # argsort is O(N log N), N=2048, so it's fast.
        top_indices = np.argsort(mag2)[-refine_top:][::-1]
        best_peak = peak_val
        
        # Precompute powers for refinement
        powers = np.arange(N_COEFFS)
        
        for idx in top_indices:
            delta = 2.0 * np.pi / N_GRID
            lo = thetas[idx] - delta
            hi = thetas[idx] + delta
            
            # Golden Section Search for max of |P(z)|^2
            # f(a) = |sum c_k exp(i k a)|^2
            # We can compute this efficiently.
            
            gr = (math.sqrt(5) - 1) / 2
            x1 = hi - gr * (hi - lo)
            x2 = lo + gr * (hi - lo)
            
            def f(a):
                z_a = np.exp(1j * a)
                # z_powers[k] = z_a^k
                # Instead of power, we can use recurrence or just np.power
                # np.power is fast for small N
                z_powers = np.power(z_a, powers)
                s = np.dot(coeffs, z_powers)
                return s.real * s.real + s.imag * s.imag

            f1 = f(x1)
            f2 = f(x2)
            
            for _ in range(refine_iters):
                if f1 < f2:
                    hi = x2
                    x2 = x1
                    f2 = f1
                    x1 = hi - gr * (hi - lo)
                    f1 = f(x1)
                else:
                    lo = x1
                    x1 = x2
                    f1 = f2
                    x2 = lo + gr * (hi - lo)
                    f2 = f(x2)
            
            local_peak = max(f1, f2)
            if local_peak > best_peak:
                best_peak = local_peak
        
        return math.sqrt(best_peak)

    def calc_peak_high_accuracy(coeffs, M=4096, refine_top=10, refine_iters=40):
        """
        Higher accuracy peak calculation for final verification.
        """
        thetas_ha = np.linspace(0, 2 * np.pi, M, endpoint=False)
        Z_ha = np.zeros((M, N_COEFFS), dtype=np.complex128)
        Z_ha[:, 0] = 1.0
        for k in range(1, N_COEFFS):
            Z_ha[:, k] = Z_ha[:, k-1] * np.exp(1j * thetas_ha)
            
        P = Z_ha.dot(coeffs)
        mag2 = np.abs(P) ** 2
        peak_val = np.max(mag2)
        
        if refine_top <= 0 or M <= 10:
            return math.sqrt(peak_val)
            
        top_indices = np.argsort(mag2)[-refine_top:][::-1]
        best_peak = peak_val
        powers = np.arange(N_COEFFS)
        
        for idx in top_indices:
            delta = 2.0 * np.pi / M
            lo = thetas_ha[idx] - delta
            hi = thetas_ha[idx] + delta
            
            gr = (math.sqrt(5) - 1) / 2
            x1 = hi - gr * (hi - lo)
            x2 = lo + gr * (hi - lo)
            
            def f(a):
                z_a = np.exp(1j * a)
                z_powers = np.power(z_a, powers)
                s = np.dot(coeffs, z_powers)
                return s.real * s.real + s.imag * s.imag

            f1 = f(x1)
            f2 = f(x2)
            
            for _ in range(refine_iters):
                if f1 < f2:
                    hi = x2
                    x2 = x1
                    f2 = f1
                    x1 = hi - gr * (hi - lo)
                    f1 = f(x1)
                else:
                    lo = x1
                    x1 = x2
                    f1 = f2
                    x2 = lo + gr * (hi - lo)
                    f2 = f(x2)
            
            local_peak = max(f1, f2)
            if local_peak > best_peak:
                best_peak = local_peak
        
        return math.sqrt(best_peak)

    # --- Search Strategies ---

    def local_search(c, max_time=10.0):
        """
        Performs local search starting from c.
        Tries 1-bit, then 2-bit, then 3-bit flips.
        Returns the best c found.
        """
        current_c = c.copy()
        current_val = calc_peak_fast(current_c)
        best_c = current_c.copy()
        best_val = current_val
        
        start_ls = time.time()
        
        while time.time() - start_ls < max_time and time_left() > 5.0:
            improved = False
            
            # 1. Try 1-bit flips
            for k in range(N_COEFFS):
                if time.time() - start_ls > max_time or time_left() < 5.0:
                    break
                current_c[k] = -current_c[k]
                trial_val = calc_peak_fast(current_c, refine_top=3, refine_iters=15)
                if trial_val < current_val - 1e-12:
                    current_val = trial_val
                    improved = True
                    if trial_val < best_val - 1e-12:
                        best_val = trial_val
                        best_c = current_c.copy()
                else:
                    current_c[k] = -current_c[k]
            
            if improved:
                continue
            
            # 2. Try 2-bit flips
            double_improved = False
            for j in range(N_COEFFS):
                if time.time() - start_ls > max_time or time_left() < 5.0:
                    break
                for k in range(j + 1, N_COEFFS):
                    if time.time() - start_ls > max_time or time_left() < 5.0:
                        break
                    current_c[j] = -current_c[j]
                    current_c[k] = -current_c[k]
                    trial_val = calc_peak_fast(current_c, refine_top=3, refine_iters=15)
                    if trial_val < current_val - 1e-12:
                        current_val = trial_val
                        double_improved = True
                        improved = True
                        if trial_val < best_val - 1e-12:
                            best_val = trial_val
                            best_c = current_c.copy()
                    else:
                        current_c[j] = -current_c[j]
                        current_c[k] = -current_c[k]
            
            if improved:
                continue
            
            # 3. Try 3-bit flips
            triple_improved = False
            for j in range(N_COEFFS):
                if time.time() - start_ls > max_time or time_left() < 5.0:
                    break
                for k in range(j + 1, N_COEFFS):
                    if time.time() - start_ls > max_time or time_left() < 5.0:
                        break
                    for l in range(k + 1, N_COEFFS):
                        if time.time() - start_ls > max_time or time_left() < 5.0:
                            break
                        current_c[j] = -current_c[j]
                        current_c[k] = -current_c[k]
                        current_c[l] = -current_c[l]
                        trial_val = calc_peak_fast(current_c, refine_top=3, refine_iters=15)
                        if trial_val < current_val - 1e-12:
                            current_val = trial_val
                            triple_improved = True
                            improved = True
                            if trial_val < best_val - 1e-12:
                                best_val = trial_val
                                best_c = current_c.copy()
                        else:
                            current_c[j] = -current_c[j]
                            current_c[k] = -current_c[k]
                            current_c[l] = -current_c[l]
            
            if not improved:
                break
                
        return best_c, best_val

    def genetic_algorithm(pop_size=40, generations=20, max_time=30.0):
        """
        Simple Genetic Algorithm to explore the space.
        Fitness: lower peak is better.
        """
        # Initialize population
        pop = []
        # Include best found so far
        pop.append(best_c_overall.copy() if best_c_overall is not None else np.random.choice([-1, 1], size=N_COEFFS))
        
        for _ in range(pop_size - 1):
            c = np.random.choice([-1, 1], size=N_COEFFS)
            pop.append(c)
            
        # Evaluate initial population
        pop_vals = [calc_peak_fast(c, refine_top=3, refine_iters=15) for c in pop]
        
        start_ga = time.time()
        
        for gen in range(generations):
            if time.time() - start_ga > max_time or time_left() < 10.0:
                break
                
            # Sort by fitness (ascending)
            combined = list(zip(pop_vals, pop))
            combined.sort(key=lambda x: x[0])
            
            # Keep top half as parents
            parents = [c for _, c in combined[:pop_size // 2]]
            
            # Create new population
            new_pop = []
            # Elitism: keep the best few
            for _, c in combined[:5]:
                new_pop.append(c.copy())
                
            # Fill rest with children
            while len(new_pop) < pop_size:
                if time.time() - start_ga > max_time or time_left() < 10.0:
                    break
                    
                # Tournament selection
                def select():
                    idx1 = np.random.randint(0, len(parents))
                    idx2 = np.random.randint(0, len(parents))
                    # We need their values too, but we lost them in sort.
                    # Let's just pick random from parents, or re-evaluate?
                    # Simpler: pick 2 random from parents, compare values (expensive).
                    # Even simpler: just pick random parents.
                    return parents[np.random.randint(0, len(parents))]
                    
                p1 = select()
                p2 = select()
                
                # Crossover: uniform
                mask = np.random.choice([True, False], size=N_COEFFS)
                child = np.where(mask, p1, p2).astype(int)
                
                # Mutation: flip a few bits
                num_mut = np.random.randint(0, 4)
                for _ in range(num_mut):
                    idx = np.random.randint(0, N_COEFFS)
                    child[idx] = -child[idx]
                    
                new_pop.append(child)
                
            # Evaluate new population
            pop = new_pop
            pop_vals = [calc_peak_fast(c, refine_top=3, refine_iters=15) for c in pop]
            
            # Update global best
            min_val_idx = np.argmin(pop_vals)
            if pop_vals[min_val_idx] < best_val_overall - 1e-12:
                best_val_overall = pop_vals[min_val_idx]
                best_c_overall = pop[min_val_idx].copy()

    # --- Main Execution Flow ---

    # Initial candidates
    def get_gr_sequence(n):
        p, q = [1], [1]
        while len(p) < n:
            p, q = p + q, p + [-c for c in q]
        return np.array(p[:n])
        
    initial_candidates = []
    initial_candidates.append(get_gr_sequence(N_COEFFS))
    
    for _ in range(20):
        initial_candidates.append(np.random.choice([-1, 1], size=N_COEFFS))
        
    initial_candidates.append(np.array([1 if i % 2 == 0 else -1 for i in range(N_COEFFS)]))
    initial_candidates.append(np.ones(N_COEFFS, dtype=int))
    
    best_c_overall = None
    best_val_overall = np.inf
    
    # Phase 1: Local Search from initial candidates
    # We have ~145s.
    # Local search is fast. Let's run it on all initial candidates.
    for c_init in initial_candidates:
        if time_left() < 15.0:
            break
        c_res, val_res = local_search(c_init, max_time=5.0)
        if val_res < best_val_overall:
            best_val_overall = val_res
            best_c_overall = c_res
            
    # Phase 2: More random restarts with local search
    # Each local search takes ~0.1-0.5s depending on depth.
    # Let's do as many as we can.
    while time_left() > 40.0:
        c_init = np.random.choice([-1, 1], size=N_COEFFS)
        c_res, val_res = local_search(c_init, max_time=2.0)
        if val_res < best_val_overall:
            best_val_overall = val_res
            best_c_overall = c_res

    # Phase 3: Genetic Algorithm
    # Use remaining time for GA to jump out of local minima.
    if time_left() > 20.0 and best_c_overall is not None:
        # Pass current best to GA
        genetic_algorithm(pop_size=30, generations=10, max_time=time_left() - 15.0)

    # Phase 4: Final Polish
    if best_c_overall is not None and time_left() > 5.0:
        # Do a final local search on the best candidate found, with more time
        c_res, val_res = local_search(best_c_overall, max_time=time_left() - 2.0)
        if val_res < best_val_overall:
            best_val_overall = val_res
            best_c_overall = c_res
            
        # High accuracy check
        final_val = calc_peak_high_accuracy(best_c_overall, M=4096, refine_top=10, refine_iters=40)
        # The evaluator computes the score as max_norm / sqrt(24).
        # We just need to return the coefficients. The evaluator will compute the score.
        # But we want to make sure we are returning the best one.
        
    if best_c_overall is None:
        # Fallback
        return get_gr_sequence(N_COEFFS).tolist()
        
    return best_c_overall.tolist()

# Test run
if __name__ == "__main__":
    import time
    start = time.time()
    coeffs = construct()
    end = time.time()
    print(f"Time taken: {end - start:.2f}s")
    print(f"Coefficients: {coeffs}")
    # Calculate score for verification
    N = 24
    thetas = np.linspace(0, 2*np.pi, 10000, endpoint=False)
    z = np.exp(1j * thetas)
    P = sum(c * z**k for k, c in enumerate(coeffs))
    max_norm = np.max(np.abs(P))
    score = max_norm / np.sqrt(N)
    print(f"Score: {score}")
