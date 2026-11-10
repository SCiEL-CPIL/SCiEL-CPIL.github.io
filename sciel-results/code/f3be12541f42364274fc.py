import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # Target n values. Published construction is 34.
    # We will try to find a valid coloring for the largest possible n.
    # 34 is the target. If we fail, we should fall back to smaller n.
    # However, finding a valid 34 is the main goal.
    
    # Strategy:
    # 1. Efficient structure generation for n=34.
    # 2. Vectorized Simulated Annealing / Local Search with incremental updates.
    # 3. Multiple random restarts if time permits.
    # 4. Fallback to smaller n if 34 is not solved.

    def solve_for_n(n, time_limit):
        """
        Solves the coloring problem for a given n within time_limit.
        Returns a list of colors if a valid coloring is found, else None.
        """
        triples = list(combinations(range(n), 3))
        M = len(triples)
        triple_index = {t: i for i, t in enumerate(triples)}
        
        # Precompute K4 and K5 structures
        k4_triple_indices = []
        triple_to_k4s = [[] for _ in range(M)]
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            t1 = triple_index[(a, b, c)]
            t2 = triple_index[(a, b, d)]
            t3 = triple_index[(a, c, d)]
            t4 = triple_index[(b, c, d)]
            idx = len(k4_triple_indices)
            k4_triple_indices.append((t1, t2, t3, t4))
            triple_to_k4s[t1].append(idx)
            triple_to_k4s[t2].append(idx)
            triple_to_k4s[t3].append(idx)
            triple_to_k4s[t4].append(idx)
            
        k5_triple_indices = []
        triple_to_k5s = [[] for _ in range(M)]
        for pent in combinations(range(n), 5):
            indices = []
            for t in combinations(pent, 3):
                indices.append(triple_index[t])
            idx = len(k5_triple_indices)
            k5_triple_indices.append(tuple(indices))
            for t in indices:
                triple_to_k5s[t].append(idx)
                
        N_k4 = len(k4_triple_indices)
        N_k5 = len(k5_triple_indices)
        
        # Convert to numpy arrays for faster access if needed, but lists of tuples are often faster for iteration in Python if we don't vectorize the whole update.
        # Let's stick to lists for the main loop but optimize the inner loop.
        
        # Initial Coloring: Random
        colors = np.random.randint(0, 2, M, dtype=np.int8)
        
        # Calculate initial violations
        # Red K4: all 4 triples are 0
        # Blue K5: all 10 triples are 1
        
        # We will track violation counts incrementally.
        # v4[k4_idx] = 1 if red K4 violated else 0
        # v5[k5_idx] = 1 if blue K5 violated else 0
        
        v4 = np.zeros(N_k4, dtype=np.int8)
        v5 = np.zeros(N_k5, dtype=np.int8)
        
        # Count red K4s
        for i, (t1, t2, t3, t4) in enumerate(k4_triple_indices):
            if colors[t1] == 0 and colors[t2] == 0 and colors[t3] == 0 and colors[t4] == 0:
                v4[i] = 1
                
        # Count blue K5s
        for i, indices in enumerate(k5_triple_indices):
            # Check if all 10 are 1
            # Summing is faster in numpy, but we only have 10 elements.
            # Let's just check directly.
            is_blue = True
            for t in indices:
                if colors[t] == 0:
                    is_blue = False
                    break
            if is_blue:
                v5[i] = 1
                
        total_violations = v4.sum() + v5.sum()
        
        if total_violations == 0:
            return colors.tolist()
            
        best_violations = total_violations
        best_colors = colors.copy()
        
        # Simulated Annealing Parameters
        T = 1.0
        T_min = 0.01
        alpha = 0.9995
        
        current_time = time.time()
        remaining_time = time_limit - (current_time - start_time)
        
        # Main Loop
        while time.time() < start_time + time_limit - 1.0 and T > T_min:
            # Pick a random triple to flip
            t_idx = np.random.randint(0, M)
            
            old_color = colors[t_idx]
            new_color = 1 - old_color
            
            delta_v = 0
            
            # Update K4 violations
            for k4_idx in triple_to_k4s[t_idx]:
                t1, t2, t3, t4 = k4_triple_indices[k4_idx]
                # Check current state of the other 3
                # If old_color was 0, it was contributing to red.
                # If old_color was 1, it was blocking red.
                
                # Count how many of the other 3 are 0
                # We can precompute or just check.
                # Since M is small (34 choose 3 = 5984), and each triple is in few K4s, this is fast.
                
                others = [t1, t2, t3, t4]
                others.remove(t_idx)
                
                # Count zeros in others
                zeros_in_others = sum(1 for x in others if colors[x] == 0)
                
                # Old state:
                # If old_color == 0:
                #   If zeros_in_others == 3, then this K4 was red (violation).
                #   If zeros_in_others < 3, it was not red.
                # If old_color == 1:
                #   It was never red.
                
                if old_color == 0:
                    if zeros_in_others == 3:
                        # Was red, now will not be red (since new_color=1)
                        delta_v -= 1
                else: # old_color == 1
                    if zeros_in_others == 3:
                        # Was not red, now will be red (since new_color=0)
                        delta_v += 1
                        
            # Update K5 violations
            for k5_idx in triple_to_k5s[t_idx]:
                indices = k5_triple_indices[k5_idx]
                others = [x for x in indices if x != t_idx]
                
                # Count ones in others
                ones_in_others = sum(1 for x in others if colors[x] == 1)
                
                # Old state:
                # If old_color == 1:
                #   If ones_in_others == 9, then this K5 was blue (violation).
                #   If ones_in_others < 9, it was not blue.
                # If old_color == 0:
                #   It was never blue.
                
                if old_color == 1:
                    if ones_in_others == 9:
                        # Was blue, now will not be blue (since new_color=0)
                        delta_v -= 1
                else: # old_color == 0
                    if ones_in_others == 9:
                        # Was not blue, now will be blue (since new_color=1)
                        delta_v += 1
                        
            new_violations = total_violations + delta_v
            
            # Acceptance
            if delta_v < 0 or random.random() < np.exp(-delta_v / T):
                colors[t_idx] = new_color
                total_violations = new_violations
                
                if total_violations < best_violations:
                    best_violations = total_violations
                    best_colors = colors.copy()
                    if total_violations == 0:
                        return colors.tolist()

            T *= alpha
            
        # Return best found if it's valid
        if best_violations == 0:
            return best_colors.tolist()
        else:
            return None

    # Main logic
    # Try n=34 first
    # Allocate most of the time to n=34
    time_alloc_34 = 130.0
    sol_34 = solve_for_n(34, time_alloc_34)
    
    if sol_34 is not None:
        return {'n': 34, 'colors': sol_34}
        
    # If 34 failed, try smaller n with remaining time
    # Check time
    elapsed = time.time() - start_time
    remaining = 145.0 - elapsed
    
    if remaining > 5.0:
        # Try 33
        time_alloc_33 = min(remaining - 2.0, 10.0)
        if time_alloc_33 > 1.0:
            sol_33 = solve_for_n(33, time_alloc_33)
            if sol_33 is not None:
                return {'n': 33, 'colors': sol_33}
                
    # Fallback to a known valid small n
    # n=9 is a safe bet with the seed strategy
    n = 9
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
    return {'n': n, 'colors': colors}

if __name__ == "__main__":
    res = construct()
    print(res['n'])
