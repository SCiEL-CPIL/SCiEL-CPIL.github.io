"""Find maximum pairwise-disjoint residue classes with distinct moduli 2..128."""
import math
import time
import random

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill
    
    MOD_MAX = 128
    moduli = list(range(2, MOD_MAX + 1))
    n = len(moduli)
    
    # Precompute gcd for all pairs
    gcd_table = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            g = math.gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    def omega(m):
        """Number of distinct prime factors."""
        count = 0
        d = 2
        mm = m
        while d * d <= mm:
            if mm % d == 0:
                count += 1
                while mm % d == 0:
                    mm //= d
            d += 1
        if mm > 1:
            count += 1
        return count
    
    def greedy_independent_set(order):
        """Greedy: add nodes in given order if compatible with all selected."""
        selected = []
        for idx in order:
            compatible = True
            for sel in selected:
                if gcd_table[idx][sel] == 1:
                    compatible = False
                    break
            if compatible:
                selected.append(idx)
        return selected
    
    def local_search(selected, time_limit=2.0):
        """Improve independent set via swaps."""
        selected = set(selected)
        not_selected = set(range(n)) - selected
        t0 = time.time()
        
        improved = True
        while improved and time.time() - t0 < time_limit:
            improved = False
            # Try all single swaps
            for sel in list(selected):
                if time.time() - t0 >= time_limit:
                    break
                for nsel in list(not_selected):
                    # nsel must be compatible with selected - {sel}
                    ok = True
                    for s in selected:
                        if s == sel:
                            continue
                        if gcd_table[nsel][s] == 1:
                            ok = False
                            break
                    if ok:
                        selected.discard(sel)
                        selected.add(nsel)
                        not_selected = set(range(n)) - selected
                        improved = True
                        break
                if time.time() - t0 >= time_limit:
                    break
        return list(selected)
    
    def assign_residues(selected_indices):
        """Assign residues to make all pairs disjoint."""
        assigned = []  # list of (residue, modulus_idx)
        
        # Sort by modulus for determinism
        for idx in sorted(selected_indices):
            m = moduli[idx]
            found = False
            for r in range(m):
                ok = True
                for (r2, m2_idx) in assigned:
                    g = gcd_table[idx][m2_idx]
                    if g > 1 and (r - r2) % g == 0:
                        ok = False
                        break
                if ok:
                    assigned.append((r, idx))
                    found = True
                    break
            if not found:
                assigned.append((0, idx))
        
        return [[r, moduli[idx]] for (r, idx) in assigned]
    
    def verify(result):
        """Verify all pairs are disjoint."""
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                r1, m1 = result[i]
                r2, m2 = result[j]
                g = math.gcd(m1, m2)
                if g > 1 and (r1 - r2) % g == 0:
                    return False
        return True
    
    # Strategy 1: Greedy by omega (descending)
    order1 = sorted(range(n), key=lambda i: (-omega(moduli[i]), moduli[i]))
    set1 = greedy_independent_set(order1)
    set1 = local_search(set1, 2.0)
    
    best_set = set1
    best_size = len(set1)
    
    # Strategy 2: Multiple random orderings
    random.seed(42)
    for attempt in range(50):
        if time.time() > deadline - 10.0:
            break
        
        # Random ordering with some bias toward high-omega
        weights = [omega(moduli[i]) + 1 for i in range(n)]
        order2 = []
        remaining = list(range(n))
        while remaining:
            total_w = sum(weights[i] for i in remaining)
            r = random.random() * total_w
            cum = 0
            for i, idx in enumerate(remaining):
                cum += weights[idx]
                if r <= cum:
                    order2.append(idx)
                    remaining.pop(i)
                    break
        
        set2 = greedy_independent_set(order2)
        if len(set2) > best_size:
            set2 = local_search(set2, 1.0)
            best_set = set2
            best_size = len(set2)
        
        # Also try local search on set2 even if not larger
        if len(set2) >= best_size - 1:
            set2b = local_search(set2, 1.0)
            if len(set2b) > best_size:
                best_set = set2b
                best_size = len(set2b)
    
    # Try more aggressive local search on best
    if time.time() < deadline - 5.0:
        best_set = local_search(best_set, deadline - time.time() - 5.0)
    
    # Assign residues
    result = assign_residues(best_set)
    
    # Verify
    if not verify(result):
        # Fallback to simple power-of-2 solution
        result = [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]
    
    return result
