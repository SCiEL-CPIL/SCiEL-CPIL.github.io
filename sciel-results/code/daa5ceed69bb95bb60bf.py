"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # We want to maximize n. Known bound is 34.
    # We will try to find a valid coloring for n=34, then 33, etc.
    # If we find one for n, we return it.
    
    # We'll use a local search (Simulated Annealing / WalkSAT style)
    
    def try_find_coloring(n, time_limit):
        """
        Attempts to find a valid 2-coloring of triples for n vertices.
        Returns None if not found within time_limit.
        """
        if n < 1:
            return None
            
        # Precompute triples
        triples = list(itertools.combinations(range(n), 3))
        N_triples = len(triples)
        
        # Map triple to index
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Precompute K4s and K5s
        # K4: 4 vertices, 4 triples. Forbidden if all 4 red.
        # K5: 5 vertices, 10 triples. Forbidden if all 10 blue.
        
        k4s = []
        k4_indices = []
        for combo in itertools.combinations(range(n), 4):
            # The 4 triples in a K4
            inds = []
            for i in range(4):
                t = tuple(sorted(combo[:i] + combo[i+1:]))
                inds.append(triple_to_idx[t])
            k4s.append(tuple(inds))
            k4_indices.append(len(k4s)-1)
            
        k5s = []
        k5_indices = []
        for combo in itertools.combinations(range(n), 5):
            # The 10 triples in a K5
            inds = []
            for i in range(5):
                for j in range(i+1, 5):
                    # Select 3 vertices from the 5, excluding i and j? No.
                    # A triple is a subset of size 3.
                    # We need all subsets of size 3 from the 5 vertices.
                    pass
            
            # Simpler: generate all combinations of 3 from the 5
            inds = [triple_to_idx[tuple(sorted(t))] for t in itertools.combinations(combo, 3)]
            k5s.append(tuple(inds))
            k5_indices.append(len(k5s)-1)
            
        N_k4 = len(k4s)
        N_k5 = len(k5s)
        
        # Maps: triple index -> list of k4 indices, list of k5 indices
        triple_to_k4s = [[] for _ in range(N_triples)]
        triple_to_k5s = [[] for _ in range(N_triples)]
        
        for i, inds in enumerate(k4s):
            for t_idx in inds:
                triple_to_k4s[t_idx].append(i)
                
        for i, inds in enumerate(k5s):
            for t_idx in inds:
                triple_to_k5s[t_idx].append(i)
                
        # State
        # colors: 0=Red, 1=Blue
        colors = [random.randint(0, 1) for _ in range(N_triples)]
        
        # Counts
        # red_counts[k4_idx]: number of red triples in this K4
        red_counts = [0] * N_k4
        # blue_counts[k5_idx]: number of blue triples in this K5
        blue_counts = [0] * N_k5
        
        # Initialize counts
        for t_idx, c in enumerate(colors):
            if c == 0: # Red
                for k4_idx in triple_to_k4s[t_idx]:
                    red_counts[k4_idx] += 1
            else: # Blue
                for k5_idx in triple_to_k5s[t_idx]:
                    blue_counts[k5_idx] += 1
                    
        # Calculate initial violations
        def calc_violations():
            v = 0
            for c in red_counts:
                if c == 4:
                    v += 1
            for c in blue_counts:
                if c == 10:
                    v += 1
            return v
            
        current_violations = calc_violations()
        
        # Local Search Loop
        # We use a simple hill-climbing with random restarts or simulated annealing.
        # Given time constraints, simple hill-climbing with random restarts is robust.
        
        best_violations = current_violations
        best_colors = colors[:]
        
        temp = 1.0
        cooling_rate = 0.999
        
        while time.time() < time_limit:
            if current_violations == 0:
                return colors[:]
                
            # Pick a random triple to flip
            t_idx = random.randint(0, N_triples - 1)
            
            # Calculate change in violations if we flip t_idx
            old_color = colors[t_idx]
            new_color = 1 - old_color
            
            delta_violations = 0
            
            if old_color == 0: # Was Red, becomes Blue
                # Red counts decrease, Blue counts increase
                for k4_idx in triple_to_k4s[t_idx]:
                    old_c = red_counts[k4_idx]
                    new_c = old_c - 1
                    # Violation if old_c == 4 (was violating, now not) -> -1
                    # Violation if new_c == 4 (impossible as new_c < old_c)
                    if old_c == 4:
                        delta_violations -= 1
                    # No new violations created by decreasing red count
                    
                for k5_idx in triple_to_k5s[t_idx]:
                    old_c = blue_counts[k5_idx]
                    new_c = old_c + 1
                    # Violation if old_c == 10 (was violating, now not? No, increasing)
                    # Violation if new_c == 10 (was not, now is) -> +1
                    if new_c == 10:
                        delta_violations += 1
            else: # Was Blue, becomes Red
                # Blue counts decrease, Red counts increase
                for k5_idx in triple_to_k5s[t_idx]:
                    old_c = blue_counts[k5_idx]
                    new_c = old_c - 1
                    if old_c == 10:
                        delta_violations -= 1
                    # No new violations created by decreasing blue count
                    
                for k4_idx in triple_to_k4s[t_idx]:
                    old_c = red_counts[k4_idx]
                    new_c = old_c + 1
                    if new_c == 4:
                        delta_violations += 1
            
            new_violations = current_violations + delta_violations
            
            # Acceptance criterion
            if new_violations < current_violations:
                # Accept
                colors[t_idx] = new_color
                if old_color == 0:
                    for k4_idx in triple_to_k4s[t_idx]:
                        red_counts[k4_idx] -= 1
                    for k5_idx in triple_to_k5s[t_idx]:
                        blue_counts[k5_idx] += 1
                else:
                    for k5_idx in triple_to_k5s[t_idx]:
                        blue_counts[k5_idx] -= 1
                    for k4_idx in triple_to_k4s[t_idx]:
                        red_counts[k4_idx] += 1
                current_violations = new_violations
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors[:]
            else:
                # Accept with probability exp(-delta / temp)
                if delta_violations > 0:
                    prob = math.exp(-delta_violations / temp)
                    if random.random() < prob:
                        colors[t_idx] = new_color
                        if old_color == 0:
                            for k4_idx in triple_to_k4s[t_idx]:
                                red_counts[k4_idx] -= 1
                            for k5_idx in triple_to_k5s[t_idx]:
                                blue_counts[k5_idx] += 1
                        else:
                            for k5_idx in triple_to_k5s[t_idx]:
                                blue_counts[k5_idx] -= 1
                            for k4_idx in triple_to_k4s[t_idx]:
                                red_counts[k4_idx] += 1
                        current_violations = new_violations
                        if current_violations < best_violations:
                            best_violations = current_violations
                            best_colors = colors[:]
                        
                # Cooling
                temp *= cooling_rate
                if temp < 0.01:
                    # Restart with random initialization
                    colors = [random.randint(0, 1) for _ in range(N_triples)]
                    red_counts = [0] * N_k4
                    blue_counts = [0] * N_k5
                    for t_idx, c in enumerate(colors):
                        if c == 0:
                            for k4_idx in triple_to_k4s[t_idx]:
                                red_counts[k4_idx] += 1
                        else:
                            for k5_idx in triple_to_k5s[t_idx]:
                                blue_counts[k5_idx] += 1
                    current_violations = calc_violations()
                    temp = 1.0
        
        # If we didn't find 0 violations, we might return the best found if it's valid?
        # No, must be valid.
        if best_violations == 0:
            return best_colors
        return None

    # Try n from 34 down to 10
    # Allocate time for each n
    # We have ~110s total.
    # Try 34 for 30s. If fail, 33 for 20s, etc.
    
    n_to_try = list(range(34, 9, -1))
    # Distribute time
    # Simple strategy: 20s for first, 10s for next, 5s for next...
    # Or just loop until deadline.
    
    for n in n_to_try:
        if time.time() > deadline - 5:
            break
            
        # Estimate time to spend
        # If n is large, spend more time.
        # Let's just try each for a short burst, say 5-10s.
        # If we find a valid one, return immediately.
        
        # For n=34, 5984 vars. Local search should be fast if it works.
        # Let's give it 10s.
        time_limit = time.time() + 10
        if time_limit > deadline:
            time_limit = deadline - 1
            
        result = try_find_coloring(n, time_limit)
        if result is not None:
            return {'n': n, 'colors': result}
            
    # Fallback to seed if all fail
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in itertools.combinations(range(n), 3)]}

# EVOLVE-BLOCK-END
