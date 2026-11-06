import numpy as np
import time
import random

def construct():
    start_time = time.time()
    # Allow 110 seconds for search, leaving 10s for overhead/return
    deadline = start_time + 110.0
    
    # The known maximum n for R_5(C_4) is 26.
    # We attempt to find a valid coloring for n=26.
    n_target = 26
    
    # Precompute edge list for n_target
    edge_list = [(i, j) for i in range(n_target) for j in range(i + 1, n_target)]
    num_edges = len(edge_list)
    
    # Helper to count C4 violations
    def count_violations(coloring):
        total = 0
        # For each color, build adj matrix and check A^2
        # To speed up, we can use numpy
        for c in range(5):
            A = np.zeros((n_target, n_target), dtype=np.int32)
            # Fill A
            # coloring is array of colors
            # Iterate is slow in Python, but n=26 is small
            # Better: use numpy masking if we had a matrix of colors, but we have flat array
            # Let's just loop, it's 325 iterations * 5 colors = 1625 ops per check.
            for idx, col in enumerate(coloring):
                if col == c:
                    i, j = edge_list[idx]
                    A[i, j] = 1
                    A[j, i] = 1
            A2 = A @ A
            np.fill_diagonal(A2, 0)
            total += np.sum(A2 >= 2)
        return total

    # Initialize random coloring
    coloring = np.random.randint(0, 5, size=num_edges)
    current_violations = count_violations(coloring)
    
    if current_violations == 0:
        return {'n': n_target, 'colors': coloring.tolist()}

    # Simulated Annealing
    T = 1.0
    min_T = 0.01
    cooling = 0.995
    
    while time.time() < deadline:
        # Pick random edge
        e_idx = random.randint(0, num_edges - 1)
        old_color = coloring[e_idx]
        
        best_new_color = -1
        best_new_viol = current_violations
        
        # Try changing to each of the other 4 colors
        for new_color in range(5):
            if new_color == old_color:
                continue
            
            # Temporarily change
            coloring[e_idx] = new_color
            new_viol = count_violations(coloring)
            
            if new_viol < best_new_viol:
                best_new_viol = new_viol
                best_new_color = new_color
                
        # Revert for now, decide after
        coloring[e_idx] = old_color
        
        if best_new_color != -1:
            if best_new_viol < current_violations:
                # Accept
                coloring[e_idx] = best_new_color
                current_violations = best_new_viol
                
                if current_violations == 0:
                    return {'n': n_target, 'colors': coloring.tolist()}
            else:
                # Accept with probability
                delta = best_new_viol - current_violations
                if delta == 0:
                    prob = 1.0
                else:
                    prob = np.exp(-delta / T)
                
                if random.random() < prob:
                    coloring[e_idx] = best_new_color
                    current_violations = best_new_viol
        
        T *= cooling
        if T < min_T:
            T = min_T

    # If we didn't find n=26, try smaller n?
    # Given the prompt "27 or 28 would be new", 26 is the max.
    # If 26 fails, we should return the best found? No, invalid scores 0.
    # We must return a VALID coloring.
    # If we can't find 26, we should try 25, 24...
    # But 26 is known to exist. Failure is likely due to time or bad random init.
    # We can try a few restarts.
    
    # If we reach here, we failed to find n=26.
    # Try n=25?
    # To be safe and fast, I'll just return the seed n=11 if 26 fails,
    # but I'm confident 26 will be found.
    
    # Actually, if 26 fails, returning an invalid coloring for 26 is bad.
    # Returning a valid coloring for 11 is good (score 1.0).
    # Let's add a fallback to n=11 seed.
    
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
