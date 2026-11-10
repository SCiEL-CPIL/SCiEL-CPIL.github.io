import numpy as np
import time
import random
from itertools import combinations

def construct():
    """
    Find a valid 2-coloring of triples for n=34 such that 
    there is no red K4^(3) and no blue K5^(3).
    """
    n = 34
    deadline = time.time() + 110.0  # Safety margin for 120s limit
    
    # Precompute triple indices
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    # Precompute 4-sets and their triples
    four_sets = list(combinations(range(n), 4))
    num_4sets = len(four_sets)
    # For each 4-set, store the 4 triple indices
    # Shape: (num_4sets, 4)
    four_set_triples = np.zeros((num_4sets, 4), dtype=np.int32)
    for i, s in enumerate(four_sets):
        # s is a tuple of 4 vertices (a,b,c,d)
        # Triples are (a,b,c), (a,b,d), (a,c,d), (b,c,d)
        four_set_triples[i, 0] = triple_to_idx[(s[0], s[1], s[2])]
        four_set_triples[i, 1] = triple_to_idx[(s[0], s[1], s[3])]
        four_set_triples[i, 2] = triple_to_idx[(s[0], s[2], s[3])]
        four_set_triples[i, 3] = triple_to_idx[(s[1], s[2], s[3])]
        
    # Precompute 5-sets and their triples
    five_sets = list(combinations(range(n), 5))
    num_5sets = len(five_sets)
    # For each 5-set, store the 10 triple indices
    # Shape: (num_5sets, 10)
    five_set_triples = np.zeros((num_5sets, 10), dtype=np.int32)
    # We can generate these efficiently
    for i, s in enumerate(five_sets):
        # s is (a,b,c,d,e)
        # Triples are all combinations of 3 from 5
        # 0: abc, 1: abd, 2: abe, 3: acd, 4: ace, 5: ade, 6: bcd, 7: bce, 8: bde, 9: cde
        five_set_triples[i, 0] = triple_to_idx[(s[0], s[1], s[2])]
        five_set_triples[i, 1] = triple_to_idx[(s[0], s[1], s[3])]
        five_set_triples[i, 2] = triple_to_idx[(s[0], s[1], s[4])]
        five_set_triples[i, 3] = triple_to_idx[(s[0], s[2], s[3])]
        five_set_triples[i, 4] = triple_to_idx[(s[0], s[2], s[4])]
        five_set_triples[i, 5] = triple_to_idx[(s[0], s[3], s[4])]
        five_set_triples[i, 6] = triple_to_idx[(s[1], s[2], s[3])]
        five_set_triples[i, 7] = triple_to_idx[(s[1], s[2], s[4])]
        five_set_triples[i, 8] = triple_to_idx[(s[1], s[3], s[4])]
        five_set_triples[i, 9] = triple_to_idx[(s[2], s[3], s[4])]

    # For fast updates, we need to know which 4-sets and 5-sets contain a given triple.
    # This is a sparse matrix or list of lists.
    # triple_to_4sets: list of lists of 4-set indices
    # triple_to_5sets: list of lists of 5-set indices
    triple_to_4sets = [[] for _ in range(num_triples)]
    triple_to_5sets = [[] for _ in range(num_triples)]
    
    for i in range(num_4sets):
        for t_idx in four_set_triples[i]:
            triple_to_4sets[t_idx].append(i)
            
    for i in range(num_5sets):
        for t_idx in five_set_triples[i]:
            triple_to_5sets[t_idx].append(i)

    def run_sa():
        # Initialize colors randomly
        colors = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
        
        # Calculate initial counts
        # red_count_4[i] = sum of colors (red=0? No, let's say red=0, blue=1? 
        # Task says color 0 is red, 1 is blue.
        # Red K4 means all 4 triples are 0.
        # Blue K5 means all 10 triples are 1.
        # Let's define:
        # is_red[t] = 1 if colors[t] == 0 else 0
        # is_blue[t] = 1 if colors[t] == 1 else 0
        
        # It's easier to track counts of "bad" states.
        # For a 4-set, violation if count_red == 4.
        # For a 5-set, violation if count_blue == 10.
        
        # Let's maintain:
        # red_counts_4: array of size num_4sets, count of red triples in each 4-set
        # blue_counts_5: array of size num_5sets, count of blue triples in each 5-set
        
        red_counts_4 = np.zeros(num_4sets, dtype=np.int32)
        blue_counts_5 = np.zeros(num_5sets, dtype=np.int32)
        
        # Efficiently calculate initial counts
        # For each triple, if red, add 1 to all 4-sets containing it.
        # If blue, add 1 to all 5-sets containing it.
        # This is O(num_triples * avg_degree).
        # avg_degree for 4-sets: each triple is in n-3 = 31 4-sets.
        # avg_degree for 5-sets: each triple is in (n-3)(n-4)/2 = 31*30/2 = 465 5-sets.
        # Total ops ~ 6000 * (31 + 465) ~ 3M. Acceptable.
        
        for t in range(num_triples):
            if colors[t] == 0: # Red
                for s_idx in triple_to_4sets[t]:
                    red_counts_4[s_idx] += 1
            else: # Blue
                for s_idx in triple_to_5sets[t]:
                    blue_counts_5[s_idx] += 1
            
        # Calculate initial violations
        violations = np.sum(red_counts_4 == 4) + np.sum(blue_counts_5 == 10)
        
        if violations == 0:
            return colors.tolist()
            
        T = 1.0
        cooling = 0.9999
        min_T = 1e-5
        
        while T > min_T and time.time() < deadline:
            # Pick a random triple to flip
            t = np.random.randint(0, num_triples)
            
            old_color = colors[t]
            new_color = 1 - old_color
            
            # Calculate delta violations
            delta_v = 0
            
            # Update 4-sets
            # For each 4-set containing t:
            # If old_color was 0 (Red), red_count decreases by 1.
            # If old_color was 1 (Blue), red_count unchanged.
            # New contribution: 1 if new_count == 4 else 0.
            # Old contribution: 1 if old_count == 4 else 0.
            
            for s_idx in triple_to_4sets[t]:
                old_count = red_counts_4[s_idx]
                if old_color == 0:
                    new_count = old_count - 1
                else:
                    new_count = old_count
                    
                old_viol = 1 if old_count == 4 else 0
                new_viol = 1 if new_count == 4 else 0
                delta_v += new_viol - old_viol
                red_counts_4[s_idx] = new_count
                
            # Update 5-sets
            for s_idx in triple_to_5sets[t]:
                old_count = blue_counts_5[s_idx]
                if old_color == 1:
                    new_count = old_count - 1
                else:
                    new_count = old_count
                    
                old_viol = 1 if old_count == 10 else 0
                new_viol = 1 if new_count == 10 else 0
                delta_v += new_viol - old_viol
                blue_counts_5[s_idx] = new_count
                
            # Acceptance criterion
            if delta_v < 0 or (delta_v == 0) or (random.random() < np.exp(-delta_v / T)):
                colors[t] = new_color
                violations += delta_v
                if violations == 0:
                    return colors.tolist()
            else:
                # Revert changes
                for s_idx in triple_to_4sets[t]:
                    if old_color == 0:
                        red_counts_4[s_idx] += 1
                    # else unchanged
                for s_idx in triple_to_5sets[t]:
                    if old_color == 1:
                        blue_counts_5[s_idx] += 1
                    # else unchanged
            
            T *= cooling
            
        return None

    # Try to find a solution for n=34
    # We can try multiple restarts if the first one fails
    best_colors = None
    for restart in range(5):
        if time.time() > deadline:
            break
        res = run_sa()
        if res is not None:
            best_colors = res
            break
            
    if best_colors:
        return {'n': n, 'colors': best_colors}
        
    # If n=34 fails, try n=33?
    # But the task says "Maximize n". The published construction has 34.
    # If we can't find 34, we should try to find a valid coloring for a smaller n
    # to get a score > 1.0.
    # Score is n. Seed is 9.
    
    # Fallback: try n=30 with simpler heuristic or just return best effort
    # Given the difficulty, let's just return n=9 if we fail, but we expect success.
    
    # Actually, let's try n=32 if 34 fails, as it might be easier.
    # But implementing a generic solver for any n requires re-precomputing.
    # For now, if 34 fails, we return the seed result to be safe.
    
    return {'n': 9, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]}

