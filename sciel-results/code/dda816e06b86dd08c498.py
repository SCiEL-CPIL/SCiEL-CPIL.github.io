import math
import sys
from itertools import combinations

def construct():
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    
    # Precompute gcds
    gcds = {}
    for i in range(len(moduli)):
        for j in range(i+1, len(moduli)):
            m1, m2 = moduli[i], moduli[j]
            g = math.gcd(m1, m2)
            gcds[(i, j)] = g
            if g == 1:
                # These two cannot be in the same set
                pass
    
    # Heuristic: Sort moduli by number of divisors (descending)
    # More divisors -> more likely to share factors with others
    def num_divisors(n):
        count = 0
        d = 2
        while d * d <= n:
            if n % d == 0:
                count += 2
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    
    sorted_indices = sorted(range(len(moduli)), key=lambda i: -num_divisors(moduli[i]))
    
    best_set = []
    
    # Backtracking search
    # To speed up, we can use a greedy initialization and then local search or limited backtracking
    # Given the small size (127), and strong constraints, a greedy with restarts might be best.
    
    def are_disjoint(r1, m1, r2, m2):
        g = math.gcd(m1, m2)
        return (r1 % g) != (r2 % g)

    def try_greedy(order):
        chosen = []
        for idx in order:
            m = moduli[idx]
            placed = False
            for r in range(m):
                ok = True
                for (r2, m2) in chosen:
                    if not are_disjoint(r, m, r2, m2):
                        ok = False
                        break
                if ok:
                    chosen.append((r, m))
                    placed = True
                    break
            # If not placed, skip
        return chosen

    # Try several orderings
    orders = [
        sorted_indices,
        sorted(range(len(moduli))), # increasing
        sorted(range(len(moduli)), reverse=True), # decreasing
    ]
    
    # Add random permutations? No, deterministic is better for reproducibility.
    # Try ordering by "conflict potential": moduli that are coprime to many others should be placed later or avoided?
    # Actually, moduli that are coprime to many others are "bad" to include if they block others.
    # But we MUST have gcd > 1 for all pairs.
    # So if we include m, all other included moduli must share a factor with m.
    
    best = try_greedy(sorted_indices)
    
    # Try to improve by local search?
    # Or try specific "clique" structures.
    # Let's try to build a set based on multiples of 6.
    multiples_of_6 = [m for m in moduli if m % 6 == 0]
    if multiples_of_6:
        idxs_6 = [moduli.index(m) for m in multiples_of_6]
        # Order by decreasing value?
        order_6 = sorted(idxs_6, key=lambda i: -moduli[i])
        res_6 = try_greedy(order_6)
        if len(res_6) > len(best):
            best = res_6
            
    # Try multiples of 30
    multiples_of_30 = [m for m in moduli if m % 30 == 0]
    if multiples_of_30:
        idxs_30 = [moduli.index(m) for m in multiples_of_30]
        order_30 = sorted(idxs_30, key=lambda i: -moduli[i])
        res_30 = try_greedy(order_30)
        if len(res_30) > len(best):
            best = res_30
            
    # Try a more sophisticated backtracking for the top candidates
    # We'll stick to greedy for now to ensure it finishes fast.
    
    if not best:
        return [[0, 2]]
        
    return [[r, m] for r, m in best]
