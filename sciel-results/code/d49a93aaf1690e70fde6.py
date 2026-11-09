"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def _build_triple_list(n):
    """Build list of triples in lexicographic order."""
    return list(combinations(range(n), 3))

def _build_idx_map(triple_list):
    """Build mapping from triple to index."""
    return {t: i for i, t in enumerate(triple_list)}

def _build_4sets_for_triple(triple_list, idx_map, n):
    """For each triple, find all 4-sets containing it."""
    num_triples = len(triple_list)
    triple_4sets = [[] for _ in range(num_triples)]
    for i, (a, b, c) in enumerate(triple_list):
        for d in range(n):
            if d == a or d == b or d == c:
                continue
            s = tuple(sorted([a, b, c, d]))
            # The 4 triples in this 4-set
            w, x, y, z = s
            triples_in_4 = [(w,x,y), (w,x,z), (w,y,z), (x,y,z)]
            triple_4sets[i].append((s, triples_in_4))
    return triple_4sets

def _build_5sets_for_triple(triple_list, idx_map, n):
    """For each triple, find all 5-sets containing it."""
    num_triples = len(triple_list)
    triple_5sets = [[] for _ in range(num_triples)]
    for i, (a, b, c) in enumerate(triple_list):
        others = [v for v in range(n) if v != a and v != b and v != c]
        for i_d in range(len(others)):
            for i_e in range(i_d + 1, len(others)):
                d = others[i_d]
                e = others[i_e]
                s = tuple(sorted([a, b, c, d, e]))
                w, x, y, z, v = s
                triples_in_5 = [(w,x,y),(w,x,z),(w,x,v),(w,y,z),(w,y,v),(w,z,v),(x,y,z),(x,y,v),(x,z,v),(y,z,v)]
                triple_5sets[i].append((s, triples_in_5))
    return triple_5sets

def _count_violations_fast(colors_arr, triple_list, idx_map, n):
    """Count total violations using vectorized approach."""
    total = 0
    # Red K4
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    count = 0
                    if colors_arr[idx_map[(i,j,k)]] == 0: count += 1
                    if colors_arr[idx_map[(i,j,l)]] == 0: count += 1
                    if colors_arr[idx_map[(i,k,l)]] == 0: count += 1
                    if colors_arr[idx_map[(j,k,l)]] == 0: count += 1
                    if count == 4:
                        total += 1
    # Blue K5
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    for m in range(l+1, n):
                        count = 0
                        triples_5 = [(i,j,k),(i,j,l),(i,j,m),(i,k,l),(i,k,m),(i,l,m),(j,k,l),(j,k,m),(j,l,m),(k,l,m)]
                        for t in triples_5:
                            if colors_arr[idx_map[t]] == 1:
                                count += 1
                        if count == 10:
                            total += 1
    return total

def _algebraic_init(n, seed=42):
    """Use algebraic construction as initialization."""
    rng = np.random.RandomState(seed)
    triple_list = list(combinations(range(n), 3))
    num_triples = len(triple_list)
    # Use a linear function over GF(2) for initial coloring
    colors = np.zeros(num_triples, dtype=np.int32)
    for i, (a, b, c) in enumerate(triple_list):
        # Simple XOR-based coloring
        colors[i] = (a * 7 + b * 13 + c * 19) % 2
    return colors, triple_list

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    best_n = 0
    best_colors = None
    
    # Try increasing n
    for n in range(5, 35):
        if time.time() > deadline - 15:
            break
        
        triple_list = _build_triple_list(n)
        idx_map = _build_idx_map(triple_list)
        num_triples = len(triple_list)
        
        # Initialize with algebraic construction
        colors, _ = _algebraic_init(n, seed=n)
        
        # Precompute 4-sets and 5-sets for each triple
        triple_4sets = _build_4sets_for_triple(triple_list, idx_map, n)
        triple_5sets = _build_5sets_for_triple(triple_list, idx_map, n)
        
        # Initial violation count
        initial_viol = _count_violations_fast(colors, triple_list, idx_map, n)
        
        if initial_viol == 0:
            best_n = n
            best_colors = colors.tolist()
            continue
        
        # Local search with incremental updates
        improved = True
        max_iterations = 20000
        iter_count = 0
        stagnation = 0
        
        while improved and iter_count < max_iterations:
            if time.time() > deadline - 5:
                break
            improved = False
            iter_count += 1
            
            # Try flipping a random triple
            t_idx = random.randint(0, num_triples - 1)
            old_color = colors[t_idx]
            new_color = 1 - old_color
            
            delta_red = 0
            delta_blue = 0
            
            # Update red K4 violations
            for s, triples_in_4 in triple_4sets[t_idx]:
                red_count = 0
                for t4 in triples_in_4:
                    if colors[idx_map[t4]] == 0:
                        red_count += 1
                
                if old_color == 0:  # was red, becoming blue
                    if red_count == 4:
                        delta_red -= 1
                else:  # was blue, becoming red
                    if red_count == 3:
                        delta_red += 1
            
            # Update blue K5 violations
            for s, triples_in_5 in triple_5sets[t_idx]:
                blue_count = 0
                for t5 in triples_in_5:
                    if colors[idx_map[t5]] == 1:
                        blue_count += 1
                
                if old_color == 1:  # was blue, becoming red
                    if blue_count == 10:
                        delta_blue -= 1
                else:  # was red, becoming blue
                    if blue_count == 9:
                        delta_blue += 1
            
            total_delta = delta_red + delta_blue
            
            if total_delta < 0:
                colors[t_idx] = new_color
                improved = True
                stagnation = 0
            else:
                stagnation += 1
                if stagnation > 500:
                    # Try random restart
                    for _ in range(50):
                        t_idx2 = random.randint(0, num_triples - 1)
                        old_c = colors[t_idx2]
                        new_c = 1 - old_c
                        
                        dr = 0
                        db = 0
                        for s, triples_in_4 in triple_4sets[t_idx2]:
                            rc = 0
                            for t4 in triples_in_4:
                                if colors[idx_map[t4]] == 0:
                                    rc += 1
                            if old_c == 0:
                                if rc == 4: dr -= 1
                            else:
                                if rc == 3: dr += 1
                        
                        for s, triples_in_5 in triple_5sets[t_idx2]:
                            bc = 0
                            for t5 in triples_in_5:
                                if colors[idx_map[t5]] == 1:
                                    bc += 1
                            if old_c == 1:
                                if bc == 10: db -= 1
                            else:
                                if bc == 9: db += 1
                        
                        if dr + db < 0:
                            colors[t_idx2] = new_c
                            improved = True
                            stagnation = 0
                            break
        
        # Check if we found a valid coloring
        final_viol = _count_violations_fast(colors, triple_list, idx_map, n)
        if final_viol == 0:
            best_n = n
            best_colors = colors.tolist()
            continue
        elif best_n == 0:
            best_n = n
            best_colors = colors.tolist()
    
    if best_colors is None:
        n = 9
        best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
        best_n = n
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
