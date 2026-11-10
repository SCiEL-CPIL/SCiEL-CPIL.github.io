"""Rewritten program: randomized greedy maximization with local search for pairwise disjoint residue classes."""
# EVOLVE-BLOCK-START
import time
import math
import random

def _is_disjoint(r1, m1, r2, m2):
    """Check if two residue classes are disjoint using exact gcd criterion."""
    g = math.gcd(m1, m2)
    return (r1 - r2) % g != 0

def _can_add(r, m, chosen):
    """Check if (r, m) can be added to chosen list."""
    for (r2, m2) in chosen:
        if not _is_disjoint(r, m, r2, m2):
            return False
    return True

def _greedy(moduli_order, residue_order_fn, deadline):
    """Run greedy selection with given moduli order and residue ordering function."""
    chosen = []
    for m in moduli_order:
        if time.time() > deadline:
            break
        candidates = list(range(m))
        residue_order_fn(candidates)
        for r in candidates:
            if _can_add(r, m, chosen):
                chosen.append((r, m))
                break
    return chosen

def _local_search(chosen, deadline):
    """Try local search moves: swap residues, remove one and add two."""
    chosen = list(chosen)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(chosen)
        
        # Try swapping residues of pairs
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                r1, m1 = chosen[i]
                r2, m2 = chosen[j]
                # Try swapping residues
                if r1 != r2:
                    new_i = (r2, m1)
                    new_j = (r1, m2)
                    # Check if new_i is disjoint from all others except j
                    ok_i = True
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        if not _is_disjoint(r2, m1, chosen[k][0], chosen[k][1]):
                            ok_i = False
                            break
                    if ok_i:
                        ok_j = True
                        for k in range(n):
                            if k == i or k == j:
                                continue
                            if not _is_disjoint(r1, m2, chosen[k][0], chosen[k][1]):
                                ok_j = False
                                break
                        if ok_j:
                            # Also check new_i and new_j are disjoint from each other
                            if _is_disjoint(r2, m1, r1, m2):
                                chosen[i] = new_i
                                chosen[j] = new_j
                                improved = True
        
        # Try removing one and trying to add a different modulus/residue
        if time.time() > deadline:
            break
        for i in range(n):
            if time.time() > deadline:
                break
            r_remove, m_remove = chosen[i]
            temp = chosen[:i] + chosen[i+1:]
            
            # Try to find a better replacement
            best_replace = None
            for m_new in range(2, 129):
                if time.time() > deadline:
                    break
                if any(m_new == cm for _, cm in temp):
                    continue
                for r_new in range(m_new):
                    if _can_add(r_new, m_new, temp):
                        # Check if this gives us more or same count but different composition
                        # For now, just accept any valid replacement to explore
                        best_replace = (r_new, m_new)
                        break
                if best_replace:
                    break
            
            if best_replace:
                new_chosen = temp + [best_replace]
                # Only accept if it doesn't reduce count (it shouldn't, we removed 1 added 1)
                chosen = new_chosen
                improved = True
    
    return chosen

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    MAX_MOD = 128
    
    moduli_list = list(range(2, MAX_MOD + 1))
    
    best_count = 0
    best_result = None
    
    # Strategy 1: Multiple randomized greedy runs with different orderings
    strategies = []
    
    # Random shuffle
    for _ in range(10):
        mo = moduli_list[:]
        random.shuffle(mo)
        strategies.append((mo, lambda c: random.shuffle(c)))
    
    # Descending moduli (larger moduli have more residue choices)
    mo_desc = sorted(moduli_list, reverse=True)
    strategies.append((mo_desc, lambda c: random.shuffle(c)))
    
    # Ascending moduli
    mo_asc = sorted(moduli_list)
    strategies.append((mo_asc, lambda c: random.shuffle(c)))
    
    # Prime-first
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(math.isqrt(n)) + 1):
            if n % i == 0:
                return False
        return True
    
    primes = [m for m in moduli_list if is_prime(m)]
    composites = [m for m in moduli_list if not is_prime(m)]
    strategies.append((primes + composites, lambda c: random.shuffle(c)))
    
    # Composite-first
    strategies.append((composites + primes, lambda c: random.shuffle(c)))
    
    # Run greedy for each strategy
    for mo, res_fn in strategies:
        if time.time() > deadline:
            break
        chosen = _greedy(mo, res_fn, deadline)
        count = len(chosen)
        if count > best_count:
            best_count = count
            best_result = [[r, m] for r, m in chosen]
    
    # Strategy 2: Local search on best result
    if best_result and time.time() < deadline:
        chosen_tuples = [(r, m) for r, m in best_result]
        improved = _local_search(chosen_tuples, deadline)
        count = len(improved)
        if count > best_count:
            best_count = count
            best_result = [[r, m] for r, m in improved]
    
    # Strategy 3: More randomized greedy with local search
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        num_extra = min(50, int(remaining_time / 0.5))
        for _ in range(num_extra):
            if time.time() > deadline:
                break
            mo = moduli_list[:]
            random.shuffle(mo)
            chosen = _greedy(mo, lambda c: random.shuffle(c), deadline)
            if len(chosen) >= best_count - 1:  # Only do local search if close
                improved = _local_search(chosen, deadline)
                count = len(improved)
                if count > best_count:
                    best_count = count
                    best_result = [[r, m] for r, m in improved]
    
    # Fallback
    fallback = [[1, 2], [2, 4], [4, 8], [8, 16], [16, 32], [32, 64], [64, 128]]
    if best_result is None or len(best_result) < len(fallback):
        best_result = fallback
        
    return best_result
# EVOLVE-BLOCK-END
