"""Rewritten program for hypergraph Ramsey R(4,5;3) construction."""
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 130  # seconds, leaving safety margin before 150s kill
    
    def elapsed():
        return time.time() - start_time
    
    def check_validity(n, colors, k4s, k5s, triple_indices, red_in_k4, blue_in_k5):
        """Check if coloring is valid: no red K4^(3) and no blue K5^(3)."""
        # Check red K4s
        for i, k4 in enumerate(k4s):
            if red_in_k4[i] == 4:
                return False
        # Check blue K5s
        for i, k5 in enumerate(k5s):
            if blue_in_k5[i] == 10:
                return False
        return True
    
    def build_structures(n):
        """Build all K4s, K5s, and their triple mappings."""
        k4s = list(combinations(range(n), 4))
        k5s = list(combinations(range(n), 5))
        
        # For each triple, which K4s and K5s contain it
        triple_to_k4s = [[] for _ in range(len(list(combinations(range(n), 3))))]
        triple_to_k5s = [list() for _ in range(len(list(combinations(range(n), 3))))]
        
        triple_idx_map = {}
        triples = list(combinations(range(n), 3))
        for idx, t in enumerate(triples):
            triple_idx_map[t] = idx
        
        for k4_idx, k4 in enumerate(k4s):
            for i in range(4):
                for j in range(i+1, 4):
                    for k in range(j+1, 4):
                        t = (k4[i], k4[j], k4[k])
                        triple_to_k4s[triple_idx_map[t]].append(k4_idx)
        
        for k5_idx, k5 in enumerate(k5s):
            for i in range(5):
                for j in range(i+1, 5):
                    for k in range(j+1, 5):
                        t = (k5[i], k5[j], k5[k])
                        triple_to_k5s[triple_idx_map[t]].append(k5_idx)
        
        return k4s, k5s, triples, triple_idx_map, triple_to_k4s, triple_to_k5s
    
    def count_violations(n, colors, k4s, k5s, triples, triple_idx_map, triple_to_k4s, triple_to_k5s):
        """Count red K4 and blue K5 violations."""
        red_in_k4 = np.zeros(len(k4s), dtype=np.int32)
        blue_in_k5 = np.zeros(len(k5s), dtype=np.int32)
        
        for t_idx, t in enumerate(triples):
            c = colors[t_idx]
            for k4_idx in triple_to_k4s[t_idx]:
                if c == 0:  # red
                    red_in_k4[k4_idx] += 1
            for k5_idx in triple_to_k5s[t_idx]:
                if c == 1:  # blue
                    blue_in_k5[k5_idx] += 1
        
        violations = np.sum(red_in_k4 == 4) + np.sum(blue_in_k5 == 10)
        return violations, red_in_k4, blue_in_k5
    
    def random_coloring(n):
        """Generate a random valid coloring."""
        triples = list(combinations(range(n), 3))
        colors = np.random.randint(0, 2, size=len(triples))
        return colors
    
    def local_search(n, colors, k4s, k5s, triples, triple_idx_map, triple_to_k4s, triple_to_k5s, 
                     max_iter=50000, time_limit=30):
        """Local search to eliminate violations."""
        violations, red_in_k4, blue_in_k5 = count_violations(n, colors, k4s, k5s, triples, triple_idx_map, triple_to_k4s, triple_to_k5s)
        
        if violations == 0:
            return colors.copy(), True
        
        start_ls = time.time()
        for it in range(max_iter):
            if time.time() - start_ls > time_limit:
                break
            if elapsed() > DEADLINE - 5:
                break
            
            # Pick a violating triple
            # Find triples that are part of violations
            violating_triples = set()
            for k4_idx, cnt in enumerate(red_in_k4):
                if cnt == 4:
                    k4 = k4s[k4_idx]
                    for i in range(4):
                        for j in range(i+1, 4):
                            for k in range(j+1, 4):
                                violating_triples.add(triple_idx_map[(k4[i], k4[j], k4[k])])
            for k5_idx, cnt in enumerate(blue_in_k5):
                if cnt == 10:
                    k5 = k5s[k5_idx]
                    for i in range(5):
                        for j in range(i+1, 5):
                            for k in range(j+1, 5):
                                violating_triples.add(triple_idx_map[(k5[i], k5[j], k5[k])])
            
            if not violating_triples:
                return colors, True
            
            t_idx = random.choice(list(violating_triples))
            old_val = colors[t_idx]
            new_val = 1 - old_val
            colors[t_idx] = new_val
            
            # Update counts
            for k4_idx in triple_to_k4s[t_idx]:
                if old_val == 0:  # was red, now blue
                    red_in_k4[k4_idx] -= 1
                elif old_val == 1:  # was blue, now red
                    red_in_k4[k4_idx] += 1
            
            for k5_idx in triple_to_k5s[t_idx]:
                if old_val == 1:  # was blue, now red
                    blue_in_k5[k5_idx] -= 1
                elif old_val == 0:  # was red, now blue
                    blue_in_k5[k5_idx] += 1
            
            violations = np.sum(red_in_k4 == 4) + np.sum(blue_in_k5 == 10)
            
            if violations == 0:
                return colors, True
            
            # If violations increased, revert with some probability
            if random.random() < 0.3:
                colors[t_idx] = old_val
                for k4_idx in triple_to_k4s[t_idx]:
                    if old_val == 0:
                        red_in_k4[k4_idx] += 1
                    elif old_val == 1:
                        red_in_k4[k4_idx] -= 1
                for k5_idx in triple_to_k5s[t_idx]:
                    if old_val == 1:
                        blue_in_k5[k5_idx] += 1
                    elif old_val == 0:
                        blue_in_k5[k5_idx] -= 1
                violations = np.sum(red_in_k4 == 4) + np.sum(blue_in_k5 == 10)
        
        return colors, False
    
    def try_construct(n, num_restarts=10):
        """Try to find a valid coloring for given n."""
        if n < 4:
            # Trivial: no K4 possible
            triples = list(combinations(range(n), 3))
            return [0] * len(triples)
        
        k4s, k5s, triples, triple_idx_map, triple_to_k4s, triple_to_k5s = build_structures(n)
        
        for restart in range(num_restarts):
            if elapsed() > DEADLINE - 5:
                break
            
            # Start with a biased random coloring
            # More blue to avoid red K4 (since red K4 is the tighter constraint for small n)
            colors = np.random.choice([0, 1], size=len(triples), p=[0.4, 0.6])
            
            colors, valid = local_search(n, colors, k4s, k5s, triples, triple_idx_map, triple_to_k4s, triple_to_k5s, 
                                         max_iter=20000, time_limit=min(20, DEADLINE - 5 - elapsed()))
            
            if valid:
                return colors.tolist()
        
        return None
    
    # Try from high n down
    best_n = 1
    best_colors = [0]
    
    for n in range(34, 20, -1):
        if elapsed() > DEADLINE - 10:
            break
        
        colors = try_construct(n, num_restarts=5)
        if colors is not None:
            best_n = n
            best_colors = colors
            break
    
    # If nothing found, use the seed construction
    if best_n < 9:
        n = 9
        best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
        best_n = n
    
    return {'n': best_n, 'colors': best_colors}
