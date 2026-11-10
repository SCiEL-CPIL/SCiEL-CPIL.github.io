# EVOLVE-BLOCK-START
import numpy as np


def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    k = 2
    
    # Compute forbidden differences: nonzero quadratic residues mod 205
    forbidden = set()
    for x in range(1, m):
        val = pow(x, k, m)
        if val != 0:
            forbidden.add(val)
    
    # Precompute conflict matrix or use set-based checking
    # Use CRT structure: 205 = 5 * 41
    # Map each element x to (x % 5, x % 41)
    
    # Try multiple greedy orderings and keep the best
    def greedy(order):
        selected = []
        selected_set = set()
        for cand in order:
            ok = True
            for s in selected:
                d1 = (cand - s) % m
                if d1 in forbidden:
                    ok = False
                    break
            if ok:
                selected.append(cand)
                selected_set.add(cand)
        return selected
    
    # Different orderings to try
    orderings = []
    
    # Natural order
    orderings.append(list(range(m)))
    
    # By mod 5 residue, then mod 41
    by_mod5 = []
    for r5 in range(5):
        for r41 in range(41):
            # Find x such that x % 5 == r5 and x % 41 == r41
            # CRT: x = r5 * 41 * inv(41 mod 5) + r41 * 5 * inv(5 mod 41) mod 205
            # 41 mod 5 = 1, inv(1) = 1
            # 5 mod 41 = 5, inv(5) mod 41: 5*33=165=165-4*41=165-164=1, so inv=33
            x = (r5 * 41 * 1 + r41 * 5 * 33) % 205
            by_mod5.append(x)
    orderings.append(by_mod5)
    
    # By mod 41 residue, then mod 5
    by_mod41 = []
    for r41 in range(41):
        for r5 in range(5):
            x = (r5 * 41 * 1 + r41 * 5 * 33) % 205
            by_mod41.append(x)
    orderings.append(by_mod41)
    
    # Reverse natural
    orderings.append(list(range(m - 1, -1, -1)))
    
    # Random shuffles
    import random
    rng = random.Random(42)
    for _ in range(10):
        order = list(range(m))
        rng.shuffle(order)
        orderings.append(order)
    
    best = []
    for order in orderings:
        result = greedy(order)
        if len(result) > len(best):
            best = result
    
    # Try local improvement on the best
    # For each element not in best, try to add it and remove one that conflicts
    # Or try swapping
    best_set = set(best)
    improved = True
    while improved:
        improved = False
        for cand in range(m):
            if cand in best_set:
                continue
            # Check if cand can be added
            conflicts = []
            for s in best:
                d1 = (cand - s) % m
                if d1 in forbidden:
                    conflicts.append(s)
            if not conflicts:
                best.append(cand)
                best_set.add(cand)
                improved = True
            elif len(conflicts) == 1:
                # Try swapping: remove the one conflict, add cand
                # But we need to check that removing it doesn't break other constraints
                # Actually if we remove one, the rest are still independent
                # And cand only conflicts with that one, so it's safe
                remove_elem = conflicts[0]
                # Verify cand doesn't conflict with others
                # We already know it only conflicts with remove_elem
                best.remove(remove_elem)
                best_set.remove(remove_elem)
                best.append(cand)
                best_set.add(cand)
                improved = True
    
    return np.array(best, dtype=np.int64)


# EVOLVE-BLOCK-END
