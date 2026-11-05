import time
import random
from itertools import combinations
import numpy as np

# Precompute all 4-subsets of [1, 16]
ALL_SETS = [frozenset(c) for c in combinations(range(1, 17), 4)]
N_SETS = len(ALL_SETS)

# Convert to bitmasks for fast intersection
ALL_MASKS = [0] * N_SETS
for i, s in enumerate(ALL_SETS):
    m = 0
    for bit in s:
        m |= (1 << (bit - 1))
    ALL_MASKS[i] = m

# Precompute intersection matrix
INTER = np.zeros((N_SETS, N_SETS), dtype=np.int32)
for i in range(N_SETS):
    mi = ALL_MASKS[i]
    for j in range(i, N_SETS):
        v = mi & ALL_MASKS[j]
        INTER[i, j] = v
        INTER[j, i] = v

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Fast can_add using numpy vectorization
    def can_add(idx_new, family):
        """Check if idx_new can be added to family (list of indices)."""
        n = len(family)
        if n < 2:
            return True
        
        # Get intersections of idx_new with all family members
        x_inter = INTER[idx_new, family]  # shape (n,)
        
        # For each pair (i, j) in family, check if INTER[x,i] == INTER[i,j] == INTER[x,j]
        # This is equivalent to: for all i<j, x_inter[i] == INTER[family[i], family[j]] == x_inter[j]
        # We need: NOT (x_inter[i] == x_inter[j] == INTER[family[i], family[j]])
        
        # Vectorized approach:
        # Create pairwise intersection matrix of family
        fam_arr = np.array(family, dtype=np.int32)
        # Pairwise intersections: INTER[fam_arr[:,None], fam_arr[None,:]]
        # But this is O(n^2) memory. For n up to ~250, that's 62500 entries, fine.
        
        # Actually, let's just do the nested loop but in C-speed via numpy isn't straightforward
        # for the triple equality check. Let's optimize the inner loop.
        
        # Alternative: for each i, check if there exists j > i such that
        # x_inter[i] == x_inter[j] == INTER[family[i], family[j]]
        
        # Let's group by x_inter value
        # For each value v, get indices i where x_inter[i] == v
        # For each such group, check if any pair (i,j) in the group has INTER[family[i],family[j]] == v
        
        # This could be faster if groups are small.
        
        # Simpler: just do the double loop but break early
        for i in range(n):
            xi = x_inter[i]
            ai = family[i]
            for j in range(i + 1, n):
                xj = x_inter[j]
                if xi != xj:
                    continue
                # Now check if INTER[ai, family[j]] == xi
                if INTER[ai, family[j]] == xi:
                    return False
        return True

    # Greedy with "freedom" heuristic: prefer sets that block fewer future additions
    def greedy_with_freedom(order):
        family = []
        # Precompute for each set, how many pairs in the current family it would conflict with
        # Actually, let's just use the standard greedy but with a better ordering
        
        for idx in order:
            if time.time() > deadline - 3.0:
                break
            if can_add(idx, family):
                family.append(idx)
        return family

    # Local search with add, 1-swap, 2-swap
    def local_search(family, max_iters=100):
        improved = True
        iters = 0
        while improved and time.time() < deadline - 5.0 and iters < max_iters:
            improved = False
            iters += 1
            in_family = set(family)
            
            # Try to add
            not_in_family = [i for i in range(N_SETS) if i not in in_family]
            random.shuffle(not_in_family)
            for idx_new in not_in_family:
                if time.time() > deadline - 5.0:
                    return family
                if can_add(idx_new, family):
                    family.append(idx_new)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try 1-swap
            if family:
                idx_remove = random.choice(family)
                temp_family = [x for x in family if x != idx_remove]
                not_in_family = [i for i in range(N_SETS) if i not in set(temp_family)]
                random.shuffle(not_in_family)
                
                for idx_add in not_in_family:
                    if time.time() > deadline - 5.0:
                        return family
                    if can_add(idx_add, temp_family):
                        family = temp_family + [idx_add]
                        improved = True
                        break
            
            if improved:
                continue
            
            # Try 2-swap
            if len(family) >= 2:
                to_remove = random.sample(family, 2)
                in_remove = set(to_remove)
                temp_family = [x for x in family if x not in in_remove]
                not_in_family = [i for i in range(N_SETS) if i not in set(family)]
                random.shuffle(not_in_family)
                
                found = False
                # Try to find two that can be added
                for i in range(min(50, len(not_in_family))):
                    if time.time() > deadline - 5.0:
                        return family
                    a = not_in_family[i]
                    if not can_add(a, temp_family):
                        continue
                    for j in range(i + 1, min(50, len(not_in_family))):
                        if time.time() > deadline - 5.0:
                            return family
                        b = not_in_family[j]
                        if can_add(b, temp_family + [a]):
                            family = temp_family + [a, b]
                            found = True
                            improved = True
                            break
                    if found:
                        break
        
        return family

    # Main
    random.seed(42)
    
    best_family = []
    best_size = 0
    
    # Phase 1: Multiple greedy restarts with different orderings
    # Use different random seeds and also try structured orderings
    for restart in range(80):
        if time.time() > deadline - 15.0:
            break
        
        if restart % 3 == 0:
            # Pure random
            order = list(range(N_SETS))
            random.shuffle(order)
        elif restart % 3 == 1:
            # Sort by number of 1s in mask (all have 4, so this doesn't help)
            # Instead, sort by some other criterion - e.g., sum of elements
            order = sorted(range(N_SETS), key=lambda i: sum(ALL_SETS[i]))
            random.shuffle(order)
        else:
            # Another random
            order = list(range(N_SETS))
            random.shuffle(order)
        
        family = greedy_with_freedom(order)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
        
        # Local search on promising families
        if len(family) >= best_size - 5 and time.time() < deadline - 10.0:
            improved = local_search(family, max_iters=50)
            if len(improved) > best_size:
                best_size = len(improved)
                best_family = improved[:]
    
    # Phase 2: Intensive local search on best family
    if best_size > 0 and time.time() < deadline - 5.0:
        current = best_family[:]
        for _ in range(30):
            if time.time() > deadline - 2.0:
                break
            improved = local_search(current, max_iters=30)
            if len(improved) > best_size:
                best_size = len(improved)
                best_family = improved[:]
                current = improved[:]
            else:
                # Perturb: remove 2-4 random elements
                if len(current) > 4:
                    k = random.randint(2, 4)
                    to_remove = random.sample(current, k)
                    current = [x for x in current if x not in set(to_remove)]
                else:
                    break
    
    return [sorted(list(ALL_SETS[idx])) for idx in best_family]
