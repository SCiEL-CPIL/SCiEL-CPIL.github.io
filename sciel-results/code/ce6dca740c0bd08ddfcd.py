"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    def time_left():
        return deadline - time.time()
    
    # Precompute triple indices for a given n
    def get_triple_indices(n):
        """Return array of shape (num_triples, 3) with all triples (a,b,c) a<b<c"""
        triples = list(combinations(range(n), 3))
        return np.array(triples, dtype=np.int32)
    
    def get_4set_indices(n):
        """Return array of shape (num_4sets, 4) with all 4-sets"""
        sets4 = list(combinations(range(n), 4))
        return np.array(sets4, dtype=np.int32)
    
    def get_5set_indices(n):
        """Return array of shape (num_5sets, 5) with all 5-sets"""
        sets5 = list(combinations(range(n), 5))
        return np.array(sets5, dtype=np.int32)
    
    def triple_index_map(n):
        """Map (a,b,c) -> index in the flat color array"""
        # For triple (a,b,c) with a<b<c, index in lex order
        # Precompute: for each (a,b), the starting index
        idx = {}
        count = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    idx[(a,b,c)] = count
                    count += 1
        return idx
    
    def count_violations(colors, n, triples, four_sets, five_sets, idx_map):
        """Count red K4 violations and blue K5 violations."""
        num_violations = 0
        
        # Red K4 violations: all 4 triples in a 4-set are red (color 0)
        if len(four_sets) > 0:
            # For each 4-set, get the 4 triple indices
            # 4-set (a,b,c,d) has triples: (a,b,c), (a,b,d), (a,c,d), (b,c,d)
            tri_idx = np.zeros((len(four_sets), 4), dtype=np.int32)
            for i, (a,b,c,d) in enumerate(four_sets):
                tri_idx[i, 0] = idx_map[(a,b,c)]
                tri_idx[i, 1] = idx_map[(a,b,d)]
                tri_idx[i, 2] = idx_map[(a,c,d)]
                tri_idx[i, 3] = idx_map[(b,c,d)]
            # Check if all 4 are red (0)
            all_red = np.all(colors[tri_idx] == 0, axis=1)
            num_violations += np.sum(all_red)
        
        # Blue K5 violations: all 10 triples in a 5-set are blue (color 1)
        if len(five_sets) > 0:
            # 5-set (a,b,c,d,e) has 10 triples
            tri_idx = np.zeros((len(five_sets), 10), dtype=np.int32)
            for i, (a,b,c,d,e) in enumerate(five_sets):
                tri_idx[i, 0] = idx_map[(a,b,c)]
                tri_idx[i, 1] = idx_map[(a,b,d)]
                tri_idx[i, 2] = idx_map[(a,b,e)]
                tri_idx[i, 3] = idx_map[(a,c,d)]
                tri_idx[i, 4] = idx_map[(a,c,e)]
                tri_idx[i, 5] = idx_map[(a,d,e)]
                tri_idx[i, 6] = idx_map[(b,c,d)]
                tri_idx[i, 7] = idx_map[(b,c,e)]
                tri_idx[i, 8] = idx_map[(b,d,e)]
                tri_idx[i, 9] = idx_map[(c,d,e)]
            # Check if all 10 are blue (1)
            all_blue = np.all(colors[tri_idx] == 1, axis=1)
            num_violations += np.sum(all_blue)
        
        return num_violations
    
    def try_coloring(n, max_steps=200000, initial_temp=1.0, cooling=0.9999):
        """Try to find a valid coloring for n vertices using simulated annealing."""
        if time_left() < 5.0:
            return None, None
        
        triples = get_triple_indices(n)
        num_triples = len(triples)
        four_sets = get_4set_indices(n)
        five_sets = get_5set_indices(n)
        idx_map = triple_index_map(n)
        
        # Initialize with random colors
        colors = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
        
        # Precompute triple indices for 4-sets and 5-sets
        if len(four_sets) > 0:
            tri_idx_4 = np.zeros((len(four_sets), 4), dtype=np.int32)
            for i, (a,b,c,d) in enumerate(four_sets):
                tri_idx_4[i, 0] = idx_map[(a,b,c)]
                tri_idx_4[i, 1] = idx_map[(a,b,d)]
                tri_idx_4[i, 2] = idx_map[(a,c,d)]
                tri_idx_4[i, 3] = idx_map[(b,c,d)]
        else:
            tri_idx_4 = None
        
        if len(five_sets) > 0:
            tri_idx_5 = np.zeros((len(five_sets), 10), dtype=np.int32)
            for i, (a,b,c,d,e) in enumerate(five_sets):
                tri_idx_5[i, 0] = idx_map[(a,b,c)]
                tri_idx_5[i, 1] = idx_map[(a,b,d)]
                tri_idx_5[i, 2] = idx_map[(a,b,e)]
                tri_idx_5[i, 3] = idx_map[(a,c,d)]
                tri_idx_5[i, 4] = idx_map[(a,c,e)]
                tri_idx_5[i, 5] = idx_map[(a,d,e)]
                tri_idx_5[i, 6] = idx_map[(b,c,d)]
                tri_idx_5[i, 7] = idx_map[(b,c,e)]
                tri_idx_5[i, 8] = idx_map[(b,d,e)]
                tri_idx_5[i, 9] = idx_map[(c,d,e)]
        else:
            tri_idx_5 = None
        
        def count_violations_fast(colors):
            nv = 0
            if tri_idx_4 is not None and len(tri_idx_4) > 0:
                all_red = np.all(colors[tri_idx_4] == 0, axis=1)
                nv += np.sum(all_red)
            if tri_idx_5 is not None and len(tri_idx_5) > 0:
                all_blue = np.all(colors[tri_idx_5] == 1, axis=1)
                nv += np.sum(all_blue)
            return nv
        
        current_violations = count_violations_fast(colors)
        best_violations = current_violations
        best_colors = colors.copy()
        
        temp = initial_temp
        step = 0
        
        while step < max_steps and time_left() > 10.0:
            # Pick a random triple to flip
            t_idx = np.random.randint(0, num_triples)
            old_color = colors[t_idx]
            colors[t_idx] = 1 - old_color
            
            new_violations = count_violations_fast(colors)
            delta = new_violations - current_violations
            
            if delta <= 0 or (temp > 0.01 and np.random.random() < np.exp(-delta / temp)):
                current_violations = new_violations
                if new_violations < best_violations:
                    best_violations = new_violations
                    best_colors = colors.copy()
                    if best_violations == 0:
                        return best_colors, n
            else:
                # Revert
                colors[t_idx] = old_color
            
            temp *= cooling
            step += 1
        
        if best_violations == 0:
            return best_colors, n
        return None, None
    
    # Strategy: try to find valid colorings for increasing n
    best_n = 9
    best_colors = None
    
    # Start with the seed solution for n=9
    n_seed = 9
    triples_seed = list(combinations(range(n_seed), 3))
    best_colors = np.array([0 if (a+b+c) % 3 == 0 else 1 for a,b,c in triples_seed], dtype=np.int8)
    
    # Try to extend from n=9 upward
    for n in range(10, 35):
        if time_left() < 15.0:
            break
        
        # Try multiple random initializations
        found = False
        for attempt in range(3):
            if time_left() < 15.0:
                break
            
            result_colors, result_n = try_coloring(n, max_steps=50000, initial_temp=1.0, cooling=0.99995)
            
            if result_colors is not None:
                best_n = n
                best_colors = result_colors
                found = True
                break
        
        if found:
            # Try to go higher
            continue
        else:
            # If we can't find for this n, try a few more times with different params
            for attempt in range(2):
                if time_left() < 15.0:
                    break
                result_colors, result_n = try_coloring(n, max_steps=80000, initial_temp=0.5, cooling=0.9999)
                if result_colors is not None:
                    best_n = n
                    best_colors = result_colors
                    found = True
                    break
            if not found:
                break
    
    # Convert best_colors back to list format
    if best_colors is not None and best_n > 9:
        triples_list = list(combinations(range(best_n), 3))
        color_list = best_colors.tolist()
    else:
        # Fallback to seed
        n_seed = 9
        triples_list = list(combinations(range(n_seed), 3))
        color_list = [0 if (a+b+c) % 3 == 0 else 1 for a,b,c in triples_list]
        best_n = n_seed
    
    return {'n': best_n, 'colors': color_list}

# EVOLVE-BLOCK-END
