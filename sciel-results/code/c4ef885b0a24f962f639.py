"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from collections import Counter

# EVOLVE-BLOCK-START
def _count_additive_relations(s):
    """Count the number of distinct 4-tuples (i,j,k,l) with i<j, k<l, i!=k,
    such that s[i]+s[j] == s[k]+s[l]. This measures additive density."""
    n = len(s)
    sum_to_pairs = {}
    for i in range(n):
        for j in range(i + 1, n):
            sm = s[i] + s[j]
            if sm not in sum_to_pairs:
                sum_to_pairs[sm] = []
            sum_to_pairs[sm].append((i, j))
    
    count = 0
    for sm, pairs in sum_to_pairs.items():
        m = len(pairs)
        if m >= 2:
            count += m * (m - 1) // 2
    return count

def _max_sidon_subset_size(s, time_limit=3.0):
    """Find the largest subset with all pair sums (i<=j) unique."""
    n = len(s)
    start = time.time()
    
    for size in range(n, 1, -1):
        if time.time() - start > time_limit:
            return size + 1  # Upper bound
        for subset in itertools.combinations(range(n), size):
            if time.time() - start > time_limit:
                return size + 1
            vals = [s[i] for i in subset]
            seen = set()
            ok = True
            for i in range(len(vals)):
                for j in range(i, len(vals)):
                    sm = vals[i] + vals[j]
                    if sm in seen:
                        ok = False
                        break
                    seen.add(sm)
                if not ok:
                    break
            if ok:
                return size
    return 1

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    random.seed(12345)
    
    best_set = None
    best_rel_count = -1
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Arithmetic progression
    candidates.append(list(range(12)))
    
    # 2. Two interleaved APs
    candidates.append(sorted([0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11]))
    
    # 3. Clustered
    candidates.append(sorted([0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15]))
    
    # 4. Spread with clusters
    candidates.append(sorted([0, 1, 2, 20, 21, 22, 40, 41, 42, 60, 61, 62]))
    
    # 5. Quadratic-like
    candidates.append(sorted([(i * i) % 129 for i in range(12)]))
    
    # 6. Random
    for _ in range(5):
        candidates.append(sorted(random.sample(range(129), 12)))
    
    # Evaluate all candidates
    for cand in candidates:
        if len(set(cand)) != 12:
            continue
        if any(x < 0 or x > 128 for x in cand):
            continue
        rel = _count_additive_relations(cand)
        if rel > best_rel_count:
            best_rel_count = rel
            best_set = list(cand)
    
    if best_set is None:
        best_set = list(range(12))
    
    current = list(best_set)
    current_rel = _count_additive_relations(current)
    
    # Local search with swaps
    no_improve_count = 0
    
    while time.time() < deadline:
        improved = False
        no_improve_count += 1
        
        # Try random single-element replacements
        for _ in range(500):
            if time.time() > deadline:
                break
            idx = random.randint(0, 11)
            old_val = current[idx]
            new_val = random.randint(0, 128)
            if new_val == old_val or (new_val in current and new_val != old_val):
                continue
            
            current[idx] = new_val
            new_rel = _count_additive_relations(current)
            
            if new_rel > current_rel:
                current_rel = new_rel
                improved = True
                no_improve_count = 0
            else:
                current[idx] = old_val
        
        if not improved:
            # Try a random restart with some probability
            if random.random() < 0.3:
                new_set = sorted(random.sample(range(129), 12))
                new_rel = _count_additive_relations(new_set)
                if new_rel > current_rel:
                    current = new_set
                    current_rel = new_rel
                    improved = True
                    no_improve_count = 0
        
        if no_improve_count > 20:
            # Do a random restart
            current = sorted(random.sample(range(129), 12))
            current_rel = _count_additive_relations(current)
            no_improve_count = 0
    
    return current
# EVOLVE-BLOCK-END
