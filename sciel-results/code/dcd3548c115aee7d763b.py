"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def _triple_index(n, a, b, c):
    """Compute the lexicographic index of triple (a,b,c) in the list of all triples on n vertices."""
    idx = 0
    for ap in range(a):
        idx += (n - 1 - ap) * (n - 2 - ap) // 2
    for bp in range(a + 1, b):
        idx += (n - 1 - a - bp)
    idx += c - b - 1
    return idx

def _build_triple_list(n):
    """Build list of all triples and reverse index map."""
    triples = []
    idx_map = {}
    idx = 0
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                triples.append((a, b, c))
                idx_map[(a, b, c)] = idx
                idx += 1
    return triples, idx_map

def _build_conflict_structures(n, triples, idx_map):
    """
    For each triple, precompute:
    - red_conflicts[ti]: list of triples that with this triple form a red K4 if both red
    - blue_conflicts[ti]: list of triples that with this triple form a blue K5 if both blue
    
    Actually, let's think differently. For incremental violation counting:
    - For red K4: a 4-set {a,b,c,d} is a red K4 iff all 4 triples are red.
    - For blue K5: a 5-set is a blue K5 iff all 10 triples are blue.
    
    Precompute for each 4-set: which triples it contains.
    Precompute for each 5-set: which triples it contains.
    
    Then violation count = number of 4-sets where all 4 triples are red + number of 5-sets where all 10 are blue.
    
    For incremental update when flipping triple t from color c to 1-c:
    - For each 4-set containing t: if t was red and becomes blue, the 4-set can no longer be all-red.
      If t was blue and becomes red, the 4-set might become all-red.
    - Similarly for 5-sets.
    """
    # Precompute all 4-sets and their triples
    four_sets = []
    four_set_triples = []  # list of lists of triple indices
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        t_indices = [idx_map[(a,b,c)], idx_map[(a,b,d)], idx_map[(a,c,d)], idx_map[(b,c,d)]]
        four_sets.append(quad)
        four_set_triples.append(t_indices)
    
    # Precompute all 5-sets and their triples
    five_sets = []
    five_set_triples = []
    for quint in combinations(range(n), 5):
        t_indices = []
        for i in range(5):
            for j in range(i+1, 5):
                for k in range(j+1, 5):
                    t_indices.append(idx_map[(quint[i], quint[j], quint[k])])
        five_sets.append(quint)
        five_set_triples.append(t_indices)
    
    return four_sets, four_set_triples, five_sets, five_set_triples

def _count_violations_fast(n, colors, four_set_triples, five_set_triples):
    """Count violations using precomputed structures."""
    red_k4 = 0
    for t_indices in four_set_triples:
        if all(colors[t] == 0 for t in t_indices):
            red_k4 += 1
    
    blue_k5 = 0
    for t_indices in five_set_triples:
        if all(colors[t] == 1 for t in t_indices):
            blue_k5 += 1
    
    return red_k4 + blue_k5, red_k4, blue_k5

def _validate_fast(n, colors):
    """Fast validation. Returns True if valid."""
    triples, idx_map = _build_triple_list(n)
    four_sets, four_set_triples, five_sets, five_set_triples = _build_conflict_structures(n, triples, idx_map)
    total, _, _ = _count_violations_fast(n, colors, four_set_triples, five_set_triples)
    return total == 0

def _local_search(n, colors, timeout=30.0, verbose=False):
    """
    Local search with incremental violation counting.
    Uses conflict-guided flipping.
    """
    start = time.time()
    n_triples = len(colors)
    triples, idx_map = _build_triple_list(n)
    four_sets, four_set_triples, five_sets, five_set_triples = _build_conflict_structures(n, triples, idx_map)
    
    # Build reverse mapping: for each triple, which 4-sets and 5-sets it belongs to
    triple_to_fours = [[] for _ in range(n_triples)]
    for i, t_indices in enumerate(four_set_triples):
        for t in t_indices:
            triple_to_fours[t].append(i)
    
    triple_to_fives = [[] for _ in range(n_triples)]
    for i, t_indices in enumerate(five_set_triples):
        for t in t_indices:
            triple_to_fives[t].append(i)
    
    # Initialize violation counts per 4-set and 5-set
    # four_red_count[i] = number of red triples in 4-set i
    # five_blue_count[i] = number of blue triples in 5-set i
    four_red_count = np.zeros(len(four_set_triples), dtype=np.int32)
    five_blue_count = np.zeros(len(five_set_triples), dtype=np.int32)
    
    for i, t_indices in enumerate(four_set_triples):
        for t in t_indices:
            if colors[t] == 0:
                four_red_count[i] += 1
    
    for i, t_indices in enumerate(five_set_triples):
        for t in t_indices:
            if colors[t] == 1:
                five_blue_count[i] += 1
    
    cur_viol = int(np.sum(four_red_count == 4)) + int(np.sum(five_blue_count == 10))
    best_viol = cur_viol
    best_colors = colors.copy()
    
    if verbose:
        print(f"  Initial violations: {cur_viol}")
    
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() < start + timeout:
        if cur_viol == 0:
            break
        if no_improve_count > max_no_improve:
            break
        
        # Choose a triple to flip: prefer triples in violating structures
        if cur_viol > 0 and random.random() < 0.7:
            # Pick a random violating structure and a triple in it
            if random.random() < 0.5:
                # Red K4 violation
                violating_fours = [i for i in range(len(four_set_triples)) if four_red_count[i] == 4]
                if violating_fours:
                    fs_idx = random.choice(violating_fours)
                    t = random.choice(four_set_triples[fs_idx])
                else:
                    t = random.randint(0, n_triples - 1)
            else:
                # Blue K5 violation
                violating_fives = [i for i in range(len(five_set_triples)) if five_blue_count[i] == 10]
                if violating_fives:
                    fs_idx = random.choice(violating_fives)
                    t = random.choice(five_set_triples[fs_idx])
                else:
                    t = random.randint(0, n_triples - 1)
        else:
            t = random.randint(0, n_triples - 1)
        
        old_color = colors[t]
        new_color = 1 - old_color
        
        # Compute delta in violations
        delta = 0
        
        # Update 4-sets containing t
        for fs_idx in triple_to_fours[t]:
            if old_color == 0:  # was red, becoming blue
                four_red_count[fs_idx] -= 1
                if four_red_count[fs_idx] == 3:
                    delta -= 1  # was a violation, now not
                # if four_red_count was 4 before, it's now 3, so we remove a violation
                # if it was <4, no change in violation status
            else:  # was blue, becoming red
                four_red_count[fs_idx] += 1
                if four_red_count[fs_idx] == 4:
                    delta += 1  # becomes a violation
        
        # Update 5-sets containing t
        for fs_idx in triple_to_fives[t]:
            if old_color == 1:  # was blue, becoming red
                five_blue_count[fs_idx] -= 1
                if five_blue_count[fs_idx] == 9:
                    delta -= 1
            else:  # was red, becoming blue
                five_blue_count[fs_idx] += 1
                if five_blue_count[fs_idx] == 10:
                    delta += 1
        
        new_viol = cur_viol + delta
        
        if new_viol < cur_viol:
            cur_viol = new_viol
            colors[t] = new_color
            no_improve_count = 0
            if cur_viol < best_viol:
                best_viol = cur_viol
                best_colors = colors.copy()
                if verbose and best_viol % 100 == 0:
                    print(f"  Best violations: {best_viol} at t={time.time()-start:.1f}s")
        else:
            # Revert
            for fs_idx in triple_to_fours[t]:
                if old_color == 0:
                    four_red_count[fs_idx] += 1
                else:
                    four_red_count[fs_idx] -= 1
            for fs_idx in triple_to_fives[t]:
                if old_color == 1:
                    five_blue_count[fs_idx] += 1
                else:
                    five_blue_count[fs_idx] -= 1
            no_improve_count += 1
    
    return best_colors, best_viol

def _try_modular(n, m):
    """Try modular coloring: triple (a,b,c) is red if (a+b+c) % m == 0, else blue."""
    num_triples = n * (n - 1) * (n - 2) // 6
    colors = np.ones(num_triples, dtype=np.int8)
    idx = 0
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                if (a + b + c) % m == 0:
                    colors[idx] = 0
                idx += 1
    return colors

def _try_modular2(n, m1, m2):
    """Try: red if (a+b+c) % m1 == 0 AND (a*b+c) % m2 == 0, else blue."""
    num_triples = n * (n - 1) * (n - 2) // 6
    colors = np.ones(num_triples, dtype=np.int8)
    idx = 0
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                if (a + b + c) % m1 == 0 and (a * b + c) % m2 == 0:
                    colors[idx] = 0
                idx += 1
    return colors

def _try_modular3(n, m):
    """Try: red if (a+b+c) % m in {0,1}, else blue."""
    num_triples = n * (n - 1) * (n - 2) // 6
    colors = np.ones(num_triples, dtype=np.int8)
    idx = 0
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                if (a + b + c) % m in (0, 1):
                    colors[idx] = 0
                idx += 1
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    best_n = 0
    best_colors = None
    
    # Precompute structures for each n we might try
    structures_cache = {}
    
    def get_structures(n):
        if n not in structures_cache:
            triples, idx_map = _build_triple_list(n)
            four_sets, four_set_triples, five_sets, five_set_triples = _build_conflict_structures(n, triples, idx_map)
            structures_cache[n] = (triples, idx_map, four_sets, four_set_triples, five_sets, five_set_triples)
        return structures_cache[n]
    
    # Phase 1: Try algebraic constructions for high n
    for n in range(34, 25, -1):
        if time.time() > deadline - 20:
            break
        
        # Try various modular constructions
        candidates = []
        for m in range(2, n + 20):
            candidates.append(_try_modular(n, m))
        for m1 in range(2, 15):
            for m2 in range(2, 15):
                candidates.append(_try_modular2(n, m1, m2))
        for m in range(2, n + 20):
            candidates.append(_try_modular3(n, m))
        
        # Also try random colorings
        num_triples = n * (n - 1) * (n - 2) // 6
        for _ in range(10):
            candidates.append(np.random.randint(0, 2, size=num_triples, dtype=np.int8))
        
        # Validate each candidate
        triples, idx_map, four_sets, four_set_triples, five_sets, five_set_triples = get_structures(n)
        
        for colors in candidates:
            if time.time() > deadline - 15:
                break
            total, _, _ = _count_violations_fast(n, colors, four_set_triples, five_set_triples)
            if total == 0:
                best_n = n
                best_colors = colors.tolist()
                break
        
        if best_n == n:
            break
    
    # Phase 2: Local search from best algebraic candidate or random
    if best_n > 0:
        return {'n': best_n, 'colors': best_colors}
    
    for n in range(34, 25, -1):
        if time.time() > deadline - 10:
            break
        
        triples, idx_map, four_sets, four_set_triples, five_sets, five_set_triples = get_structures(n)
        num_triples = n * (n - 1) * (n - 2) // 6
        
        # Try multiple starting points
        best_viol = float('inf')
        best_result = None
        
        for attempt in range(8):
            if time.time() > deadline - 8:
                break
            
            # Start from a modular coloring or random
            if attempt < 4:
                colors = _try_modular(n, 7 + attempt)
            else:
                colors = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
            
            # Run local search
            remaining_time = deadline - 8 - time.time()
            if remaining_time < 2:
                break
            
            improved, viol = _local_search(n, colors, timeout=min(remaining_time, 15.0))
            
            if viol < best_viol:
                best_viol = viol
                best_result = improved
            
            if viol == 0:
                best_n = n
                best_colors = improved.tolist()
                break
        
        if best_n == n:
            break
        
        # If we got close, keep the best result for this n
        if best_result is not None and best_viol < 50:
            # Try more aggressive search
            remaining_time = deadline - 5 - time.time()
            if remaining_time > 5:
                improved, viol = _local_search(n, best_result, timeout=remaining_time)
                if viol == 0:
                    best_n = n
                    best_colors = improved.tolist()
                    break
    
    if best_n > 0 and best_colors is not None:
        return {'n': best_n, 'colors': best_colors}
    
    # Fallback: small valid coloring
    n = 9
    colors = [0 if (a + b + c) % 3 == 0 else 1 
              for a in range(n) for b in range(a+1, n) for c in range(b+1, n)]
    return {'n': n, 'colors': colors}

# EVOLVE-BLOCK-END
