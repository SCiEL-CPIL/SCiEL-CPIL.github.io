import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin (limit is 90s)
    
    # Helper: check if (r_new, m_new) is disjoint from all in existing
    # Disjoint iff gcd(m_new, m_existing) != 1 AND (r_new - r_existing) % gcd != 0
    # If gcd == 1, they always intersect (so NOT disjoint)
    def is_disjoint(r1, m1, r2, m2):
        g = math.gcd(m1, m2)
        if g == 1:
            return False
        return (r1 - r2) % g != 0
    
    def can_add(r, m, selected):
        for (r_j, m_j) in selected:
            if not is_disjoint(r, m, r_j, m_j):
                return False
        return True
    
    # All candidate moduli
    moduli = list(range(2, 129))
    
    best_result = None
    best_count = 0
    
    # Precompute gcd table for speed
    n_mod = len(moduli)
    mod_to_idx = {m: i for i, m in enumerate(moduli)}
    gcd_table = [[0] * n_mod for _ in range(n_mod)]
    for i in range(n_mod):
        for j in range(i + 1, n_mod):
            g = math.gcd(moduli[i], moduli[j])
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Multiple orderings to try
    orderings = []
    
    # 1. By descending number of divisors
    def num_divisors(m):
        count = 0
        for d in range(1, int(math.isqrt(m)) + 1):
            if m % d == 0:
                count += 1
                if d != m // d:
                    count += 1
        return count
    
    orderings.append(sorted(moduli, key=lambda m: (-num_divisors(m), m)))
    orderings.append(sorted(moduli, reverse=True))
    orderings.append(sorted(moduli))
    
    # 2. By descending number of prime factors (with multiplicity)
    def num_prime_factors(m):
        count = 0
        mm = m
        d = 2
        while d * d <= mm:
            while mm % d == 0:
                count += 1
                mm //= d
            d += 1
        if mm > 1:
            count += 1
        return count
    
    orderings.append(sorted(moduli, key=lambda m: (-num_prime_factors(m), m)))
    
    # 3. Grouped by v2 (2-adic valuation), processing groups in order
    # This is the structural idea from the current program
    v2_groups = {}
    for m in moduli:
        k = 0
        temp = m
        while temp % 2 == 0:
            temp //= 2
            k += 1
        if k not in v2_groups:
            v2_groups[k] = []
        v2_groups[k].append(m)
    
    # Create an ordering that processes groups by v2, and within group by modulus
    v2_ordering = []
    for k in sorted(v2_groups.keys()):
        v2_ordering.extend(sorted(v2_groups[k]))
    orderings.append(v2_ordering)
    
    # 4. Reverse v2 ordering
    v2_ordering_rev = []
    for k in sorted(v2_groups.keys(), reverse=True):
        v2_ordering_rev.extend(sorted(v2_groups[k]))
    orderings.append(v2_ordering_rev)
    
    # 5. Random-ish orderings (deterministic seeds)
    import random
    for seed in [42, 123, 456, 789, 101]:
        rng = random.Random(seed)
        shuffled = list(moduli)
        rng.shuffle(shuffled)
        orderings.append(shuffled)
    
    # Try each ordering with greedy
    for ordering in orderings:
        if time.time() > deadline:
            break
        
        selected = []
        selected_moduli = set()
        
        for m in ordering:
            if time.time() > deadline:
                break
            
            # Try residues in a smart order
            # Start with 0, then odd, then even, etc.
            # Actually, just try all residues 0..m-1
            found = False
            for r in range(m):
                if can_add(r, m, selected):
                    selected.append((r, m))
                    selected_moduli.add(m)
                    found = True
                    break
            
            # If not found, try a different residue ordering
            if not found:
                # Try residues in reverse
                for r in range(m - 1, -1, -1):
                    if can_add(r, m, selected):
                        selected.append((r, m))
                        selected_moduli.add(m)
                        found = True
                        break
            
            # If still not found, skip this modulus for now
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_result = [[r, m] for (r, m) in selected]
    
    # Second phase: try to add remaining moduli to the best result
    if best_result and time.time() < deadline - 5:
        selected_set = set(m for _, m in best_result)
        remaining = [m for m in moduli if m not in selected_set]
        
        improved = list(best_result)
        # Sort remaining by descending divisor count for better chances
        remaining.sort(key=lambda m: (-num_divisors(m), m))
        
        for m in remaining:
            if time.time() > deadline:
                break
            for r in range(m):
                if can_add(r, m, improved):
                    improved.append((r, m))
                    break
        
        if len(improved) > best_count:
            best_count = len(improved)
            best_result = improved
    
    # Third phase: local search - try swapping residues to free up space
    if best_result and time.time() < deadline - 10:
        # Try to improve by reassigning residues
        improved = list(best_result)
        improved_set = set(m for _, m in improved)
        remaining = [m for m in moduli if m not in improved_set]
        
        # Try adding each remaining modulus with more thorough residue search
        for m in remaining:
            if time.time() > deadline:
                break
            
            # Try all residues
            found = False
            for r in range(m):
                if can_add(r, m, improved):
                    improved.append((r, m))
                    found = True
                    break
            
            # If not found, try to remove one existing modulus to make room
            if not found:
                for i in range(len(improved) - 1, -1, -1):
                    if time.time() > deadline:
                        break
                    r_old, m_old = improved[i]
                    # Try removing this and adding the new one
                    temp_improved = improved[:i] + improved[i+1:]
                    for r in range(m):
                        if can_add(r, m, temp_improved):
                            temp_improved.append((r, m))
                            # Check if this is better (it's the same size, but might enable future additions)
                            # For now, just accept if it allows adding more later
                            # Actually, same size, so skip for now
                            break
    
    if best_result is None:
        # Fallback
        return [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]
    
    return best_result
