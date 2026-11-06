import time
import random
from itertools import combinations

def construct():
    # Generate all 4-subsets of [1, 16] as bitmasks
    all_sets = []
    for i in range(16):
        for j in range(i+1, 16):
            for k in range(j+1, 16):
                for l in range(k+1, 16):
                    mask = (1 << i) | (1 << j) | (1 << k) | (1 << l)
                    all_sets.append(mask)
    
    n_sets = len(all_sets) # 1820
    
    def can_add_fast(C, F, pairs_by_core):
        # Check submasks of C
        sub = C
        while sub:
            if sub in pairs_by_core:
                for A, B in pairs_by_core[sub]:
                    if (C & (A ^ sub)) == 0 and (C & (B ^ sub)) == 0:
                        return False
            sub = (sub - 1) & C
        # Check empty core
        if 0 in pairs_by_core:
            for A, B in pairs_by_core[0]:
                if (C & A) == 0 and (C & B) == 0:
                    return False
        return True

    def add_set(C, F, F_set, pairs_by_core):
        F.append(C)
        F_set.add(C)
        for A in F[:-1]:
            K = A & C
            if K not in pairs_by_core:
                pairs_by_core[K] = []
            pairs_by_core[K].append((A, C))

    def remove_set(C, F, F_set, pairs_by_core):
        F.remove(C)
        F_set.remove(C)
        for K in list(pairs_by_core.keys()):
            # Filter out pairs involving C
            new_list = []
            for A, B in pairs_by_core[K]:
                if A != C and B != C:
                    new_list.append((A, B))
            if new_list:
                pairs_by_core[K] = new_list
            else:
                del pairs_by_core[K]

    deadline = time.time() + 85.0
    best_family = []
    best_size = 0
    
    # Initial greedy
    random.seed(42)
    candidates = all_sets[:]
    random.shuffle(candidates)
    F = []
    F_set = set()
    pairs_by_core = {}
    for C in candidates:
        if can_add_fast(C, F, pairs_by_core):
            add_set(C, F, F_set, pairs_by_core)
    
    if len(F) > best_size:
        best_size = len(F)
        best_family = F[:]

    # Local Search
    while time.time() < deadline:
        added = False
        # Try to add a random set
        for _ in range(50):
            C = random.choice(all_sets)
            if C in F_set:
                continue
            if can_add_fast(C, F, pairs_by_core):
                add_set(C, F, F_set, pairs_by_core)
                added = True
                if len(F) > best_size:
                    best_size = len(F)
                    best_family = F[:]
                break
        
        if not added:
            # Try to swap: remove one, add one
            if F:
                A_remove = random.choice(F)
                remove_set(A_remove, F, F_set, pairs_by_core)
                
                added_any = False
                for _ in range(50):
                    C = random.choice(all_sets)
                    if C in F_set:
                        continue
                    if can_add_fast(C, F, pairs_by_core):
                        add_set(C, F, F_set, pairs_by_core)
                        added_any = True
                        if len(F) > best_size:
                            best_size = len(F)
                            best_family = F[:]
                        break
                
                if not added_any:
                    # Restore
                    add_set(A_remove, F, F_set, pairs_by_core)
                
                if len(F) > best_size:
                    best_size = len(F)
                    best_family = F[:]

    # Convert best_family to result format
    result = []
    for mask in best_family:
        elems = []
        for i in range(16):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(elems)
        
    return result
