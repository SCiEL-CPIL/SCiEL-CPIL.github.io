import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin

        calls_made = 0
        best_x = np.zeros(dim)
        best_f = np.inf
        visited = set()  # Store tuples of rounded coordinates

        def counted_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            x = np.asarray(x, dtype=float).copy()
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            if not np.isfinite(fx):
                fx = 1e6
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def local_search(x_start, max_evals=40):
            """
            Perform coordinate-wise local search to find the nearest local minimum.
            Uses a window of +/- 1.5 around the current coordinate.
            """
            x = np.clip(x_start.copy(), -5.0, 5.0)
            evals_used = 0
            max_passes = 3
            
            for _ in range(max_passes):
                if evals_used >= max_evals:
                    break
                for i in range(dim):
                    if evals_used >= max_evals:
                        break
                    # Define search window
                    lo = x[i] - 1.5
                    hi = x[i] + 1.5
                    lo = max(lo, -5.0)
                    hi = min(hi, 5.0)
                    
                    # Golden Section Search
                    gr = (np.sqrt(5) - 1) / 2
                    a = lo
                    b = hi
                    c = b - gr * (b - a)
                    d = a + gr * (b - a)
                    
                    fc = counted_func(np.concatenate([x[:i], [c], x[i+1:]]))
                    fd = counted_func(np.concatenate([x[:i], [d], x[i+1:]]))
                    evals_used += 2
                    
                    iter_count = 0
                    while (b - a) > 1e-5 and evals_used < max_evals and iter_count < 15:
                        if fc < fd:
                            b = d
                            d = c
                            fd = fc
                            c = b - gr * (b - a)
                            fc = counted_func(np.concatenate([x[:i], [c], x[i+1:]]))
                            evals_used += 1
                        else:
                            a = c
                            c = d
                            fc = fd
                            d = a + gr * (b - a)
                            fd = counted_func(np.concatenate([x[:i], [d], x[i+1:]]))
                            evals_used += 1
                        iter_count += 1
                    
                    # Update x[i] to the best point found
                    if fc < fd:
                        x[i] = c
                    else:
                        x[i] = d
            
            # Final evaluation of the converged point
            fx = counted_func(x)
            return x, fx

        # Phase 1: Initial local search from origin
        x0, f0 = local_search(np.zeros(dim))
        key0 = tuple(np.round(x0, 2))
        visited.add(key0)
        
        # Priority queue: list of (f_value, x_coords)
        # We will use a simple list and sort, as the number of candidates is small
        candidates = [(f0, x0)]
        
        # Phase 2: Lattice Walk (Best-First Search)
        # We explore neighbors of the best candidates
        # Each local search costs ~40-50 evals.
        # Budget 500. Initial took ~50. Remaining ~450.
        # We can do ~9 more searches.
        
        while calls_made < budget - 50 and time.time() < deadline:
            # Sort candidates by function value
            candidates.sort(key=lambda item: item[0])
            
            # Take the best candidate
            curr_f, curr_x = candidates[0]
            
            # Try to expand to neighbors
            # To avoid infinite loops, we check if a neighbor was already visited
            # We generate start points for neighbor basins
            
            expanded = False
            for i in range(dim):
                for sign in [1.0, -1.0]:
                    if calls_made >= budget - 50 or time.time() > deadline:
                        break
                    
                    # Perturb current best in direction i
                    x_start = curr_x.copy()
                    x_start[i] += sign * 0.6
                    
                    # Run local search from this start
                    x_new, f_new = local_search(x_start, max_evals=40)
                    
                    key_new = tuple(np.round(x_new, 2))
                    
                    if key_new not in visited:
                        visited.add(key_new)
                        candidates.append((f_new, x_new))
                        expanded = True
                        if f_new < best_f:
                            # best_f is updated in counted_func, but we can also track it here
                            pass 
                            
            if not expanded:
                # If no new neighbors found, remove the current candidate to avoid re-processing
                # But we might have added some, so we just continue
                # Actually, if we can't expand, we should mark this as processed
                # Simple heuristic: just break if we've processed many
                break
                
            # Limit the number of candidates to keep memory/processing low
            if len(candidates) > 20:
                candidates = candidates[:20]

        return best_x, best_f
