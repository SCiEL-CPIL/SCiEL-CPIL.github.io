import time
import math
import random

def _is_disjoint(r1, m1, r2, m2, gcd_table=None):
    """Check if two residue classes are disjoint using exact gcd criterion."""
    if gcd_table is not None:
        g = gcd_table[m1][m2]
    else:
        g = math.gcd(m1, m2)
    return (r1 - r2) % g != 0

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    MAX_MOD = 128
    
    moduli_list = list(range(2, MAX_MOD + 1))
    
    # Precompute gcd table
    gcd_table = [[0] * (MAX_MOD + 1) for _ in range(MAX_MOD + 1)]
    for i in range(2, MAX_MOD + 1):
        for j in range(i, MAX_MOD + 1):
            g = math.gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    best_count = 0
    best_result = None
    
    def try_add(chosen, m, gcd_table):
        """Try to find a valid residue for modulus m given current chosen set."""
        for r in range(m):
            ok = True
            for (r2, m2) in chosen:
                g = gcd_table[m][m2]
                if (r - r2) % g == 0:
                    ok = False
                    break
            if ok:
                return r
        return None
    
    def greedy_construct(order, gcd_table):
        """Greedy: add moduli in given order, picking first valid residue."""
        chosen = []
        for m in order:
            r = try_add(chosen, m, gcd_table)
            if r is not None:
                chosen.append((r, m))
        return chosen
    
    def local_search(chosen, gcd_table, deadline):
        """Try to improve by removing one and adding two, or swapping."""
        chosen = list(chosen)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            n = len(chosen)
            # Try removing one element and adding two new ones
            for i in range(n):
                if time.time() > deadline:
                    break
                remaining = [c for j, c in enumerate(chosen) if j != i]
                removed = chosen[i]
                # Try adding any two unused moduli
                used_moduli = set(m for _, m in remaining)
                unused = [m for m in range(2, MAX_MOD + 1) if m not in used_moduli]
                added = 0
                new_chosen = list(remaining)
                for m in unused:
                    r = try_add(new_chosen, m, gcd_table)
                    if r is not None:
                        new_chosen.append((r, m))
                        added += 1
                    if added >= 2:
                        break
                if added >= 2:
                    chosen = new_chosen
                    improved = True
                    break
            # Also try single swap: remove one, add one different modulus
            if not improved:
                for i in range(n):
                    if time.time() > deadline:
                        break
                    remaining = [c for j, c in enumerate(chosen) if j != i]
                    used_moduli = set(m for _, m in remaining)
                    unused = [m for m in range(2, MAX_MOD + 1) if m not in used_moduli]
                    for m in unused:
                        r = try_add(remaining, m, gcd_table)
                        if r is not None:
                            # Check if this is better (it's same size, so skip unless we want diversity)
                            pass
        return chosen
    
    # Multiple random restarts with different orderings
    num_restarts = 500
    
    for it in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Create a random ordering
        order = moduli_list[:]
        random.shuffle(order)
        
        # Bias: sometimes sort by descending modulus, sometimes by number of divisors
        if it % 3 == 0:
            order = sorted(moduli_list, key=lambda m: random.random())
        elif it % 3 == 1:
            order = sorted(moduli_list, reverse=True)
        else:
            order = sorted(moduli_list, key=lambda m: (-m.bit_count(), random.random()))
        
        chosen = greedy_construct(order, gcd_table)
        chosen = local_search(chosen, gcd_table, deadline)
        
        count = len(chosen)
        if count > best_count:
            best_count = count
            best_result = [[r, m] for r, m in chosen]
    
    # Fallback
    fallback = [[1, 2], [2, 4], [4, 8], [8, 16], [16, 32], [32, 64], [64, 128]]
    if best_result is None or len(best_result) < len(fallback):
        best_result = fallback
        
    return best_result
