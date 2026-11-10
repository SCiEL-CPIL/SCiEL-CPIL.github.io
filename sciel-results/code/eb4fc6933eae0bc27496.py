"""Rewritten program: randomized greedy maximization of pairwise disjoint residue classes."""
# EVOLVE-BLOCK-START
import time
import math
import random

def _is_disjoint(r1, m1, r2, m2):
    """Check if two residue classes are disjoint using exact gcd criterion."""
    g = math.gcd(m1, m2)
    return (r1 - r2) % g != 0

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    MAX_MOD = 128
    
    moduli_list = list(range(2, MAX_MOD + 1))
    
    best_count = 0
    best_result = None
    
    # Precompute gcds for all pairs of moduli to speed up disjointness checks
    # gcd_table[m1][m2] = gcd(m1, m2)
    gcd_table = {}
    for m1 in moduli_list:
        for m2 in moduli_list:
            if m1 not in gcd_table:
                gcd_table[m1] = {}
            if m2 not in gcd_table[m1]:
                gcd_table[m1][m2] = math.gcd(m1, m2)
    
    def is_disjoint_fast(r1, m1, r2, m2):
        """Fast disjointness check using precomputed gcd table."""
        g = gcd_table[m1][m2]
        return (r1 - r2) % g != 0
    
    # Strategy: multiple randomized greedy passes with different heuristics
    # 1. Random order of moduli, random order of residues
    # 2. Descending moduli first (larger moduli have more residue choices)
    # 3. Ascending moduli first
    # 4. Prime moduli first (fewer conflicts due to coprime moduli)
    # 5. Composite moduli first
    
    # Precompute which moduli are prime
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    prime_moduli = [m for m in moduli_list if is_prime(m)]
    composite_moduli = [m for m in moduli_list if not is_prime(m)]
    
    # Different modulus orderings to try
    def get_modulus_orderings():
        orderings = []
        
        # Random shuffles
        for _ in range(30):
            mo = moduli_list[:]
            random.shuffle(mo)
            orderings.append(mo)
        
        # Descending
        orderings.append(sorted(moduli_list, reverse=True))
        
        # Ascending
        orderings.append(sorted(moduli_list))
        
        # Prime first, then composite
        orderings.append(prime_moduli + composite_moduli)
        
        # Composite first, then prime
        orderings.append(composite_moduli + prime_moduli)
        
        # Large primes first
        large_primes = [m for m in prime_moduli if m >= 64]
        small_primes = [m for m in prime_moduli if m < 64]
        large_composites = [m for m in composite_moduli if m >= 64]
        small_composites = [m for m in composite_moduli if m < 64]
        orderings.append(large_primes + small_primes + large_composites + small_composites)
        
        # Random prime/composite interleaving
        for _ in range(10):
            mo = moduli_list[:]
            random.shuffle(mo)
            orderings.append(mo)
        
        return orderings
    
    orderings = get_modulus_orderings()
    
    # For each ordering, try multiple residue strategies
    # Residue strategies:
    # 1. Random order
    # 2. Start from 0
    # 3. Start from 1
    # 4. Try residues that are coprime to the modulus first
    
    def get_residue_orderings(m):
        """Get different orderings of residues for modulus m."""
        orderings = []
        
        # Random
        ro = list(range(m))
        random.shuffle(ro)
        orderings.append(ro)
        
        # Another random
        ro2 = list(range(m))
        random.shuffle(ro2)
        orderings.append(ro2)
        
        # Ascending
        orderings.append(list(range(m)))
        
        # Descending
        orderings.append(list(range(m - 1, -1, -1)))
        
        # Coprime first
        coprime = [r for r in range(m) if math.gcd(r, m) == 1]
        non_coprime = [r for r in range(m) if math.gcd(r, m) != 1]
        orderings.append(coprime + non_coprime)
        
        return orderings
    
    # Greedy fill function
    def greedy_fill(modulus_order, residue_strategy_func):
        """Fill using the given modulus order and residue strategy."""
        chosen = []
        chosen_moduli = set()
        
        for m in modulus_order:
            if m in chosen_moduli:
                continue
            
            # Get residue candidates in the specified order
            candidates = residue_strategy_func(m)
            
            for r in candidates:
                ok = True
                for (r2, m2) in chosen:
                    if not is_disjoint_fast(r, m, r2, m2):
                        ok = False
                        break
                if ok:
                    chosen.append((r, m))
                    chosen_moduli.add(m)
                    break
        
        return chosen
    
    # Try multiple combinations
    iteration = 0
    for mo in orderings:
        if time.time() > deadline:
            break
        
        # Try several residue strategies for this modulus ordering
        for res_strategy_idx in range(5):
            if time.time() > deadline:
                break
            
            if res_strategy_idx == 0:
                def res_func(m):
                    ro = list(range(m))
                    random.shuffle(ro)
                    return ro
            elif res_strategy_idx == 1:
                def res_func(m):
                    return list(range(m))
            elif res_strategy_idx == 2:
                def res_func(m):
                    return list(range(m - 1, -1, -1))
            elif res_strategy_idx == 3:
                def res_func(m):
                    coprime = [r for r in range(m) if math.gcd(r, m) == 1]
                    non_coprime = [r for r in range(m) if math.gcd(r, m) != 1]
                    random.shuffle(coprime)
                    random.shuffle(non_coprime)
                    return coprime + non_coprime
            else:
                def res_func(m):
                    ro = list(range(m))
                    random.shuffle(ro)
                    return ro
            
            result = greedy_fill(mo, res_func)
            count = len(result)
            
            if count > best_count:
                best_count = count
                best_result = [[r, m] for r, m in result]
        
        iteration += 1
    
    # Additional randomized restarts if we have time
    while time.time() < deadline:
        mo = moduli_list[:]
        random.shuffle(mo)
        
        def res_func(m):
            ro = list(range(m))
            random.shuffle(ro)
            return ro
        
        result = greedy_fill(mo, res_func)
        count = len(result)
        
        if count > best_count:
            best_count = count
            best_result = [[r, m] for r, m in result]
        
        # Also try a variant where we sort by some heuristic after random shuffle
        mo2 = moduli_list[:]
        random.shuffle(mo2)
        mo2.sort(key=lambda x: -x)  # descending
        result2 = greedy_fill(mo2, res_func)
        count2 = len(result2)
        if count2 > best_count:
            best_count = count2
            best_result = [[r, m] for r, m in result2]
    
    # Fallback if random search failed or found very few
    fallback = [[1, 2], [2, 4], [4, 8], [8, 16], [16, 32], [32, 64], [64, 128]]
    if best_result is None or len(best_result) < len(fallback):
        best_result = fallback
    
    return best_result
# EVOLVE-BLOCK-END
