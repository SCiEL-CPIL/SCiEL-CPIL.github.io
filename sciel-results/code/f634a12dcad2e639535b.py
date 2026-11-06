"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

MAX_MOD = 128
DEADLINE_MARGIN = 5.0

def _disjoint(r1, m1, r2, m2):
    g = math.gcd(m1, m2)
    return (r2 - r1) % g != 0

def _valid(pairs):
    n = len(pairs)
    for i in range(n):
        for j in range(i + 1, n):
            if not _disjoint(pairs[i][0], pairs[i][1], pairs[j][0], pairs[j][1]):
                return False
    return True

def _prime_construction():
    """Construct a solution using all primes in [2,128] with residue 0.
    All primes are pairwise coprime, so any two prime-modulus classes are disjoint."""
    primes = []
    for m in range(2, MAX_MOD + 1):
        is_prime = True
        for d in range(2, int(math.isqrt(m)) + 1):
            if m % d == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(m)
    return [(0, p) for p in primes]

def _backtrack_search(moduli_list, time_limit):
    """Backtracking search with forward checking.
    Tries to build a set of (residue, modulus) pairs that are pairwise disjoint.
    Uses most-constrained-first ordering for efficiency."""
    best = []
    best_len = 0
    current = []  # list of (r, m)
    
    # Precompute gcd for all pairs of moduli
    n_mods = len(moduli_list)
    gcd_table = {}
    for i in range(n_mods):
        for j in range(i + 1, n_mods):
            g = math.gcd(moduli_list[i], moduli_list[j])
            gcd_table[(i, j)] = g
    
    # Sort moduli by number of divisors (more constrained first)
    def constraint_score(m):
        # Count how many other moduli share a gcd > 1 with m
        count = 0
        for other in moduli_list:
            if other != m and math.gcd(m, other) > 1:
                count += 1
        return count
    
    ordered_mods = sorted(moduli_list, key=constraint_score, reverse=True)
    
    # Map from modulus value to index in ordered_mods
    mod_to_idx = {m: i for i, m in enumerate(ordered_mods)}
    
    def backtrack(idx, current_pairs):
        nonlocal best, best_len
        
        if time.time() > time_limit:
            return
        
        if len(current_pairs) > best_len:
            best = list(current_pairs)
            best_len = len(current_pairs)
        
        if idx >= len(ordered_mods):
            return
        
        m = ordered_mods[idx]
        
        # Try to add modulus m with some residue r
        found = False
        # Try residues in order 0, 1, ..., m-1
        for r in range(m):
            # Check if (r, m) is disjoint from all current pairs
            ok = True
            for (r2, m2) in current_pairs:
                if not _disjoint(r, m, r2, m2):
                    ok = False
                    break
            if ok:
                current_pairs.append((r, m))
                backtrack(idx + 1, current_pairs)
                current_pairs.pop()
                found = True
                # If we found one, we can stop trying other residues for this modulus
                # (since we just need to maximize count, not optimize residues)
                break
        
        # Also try skipping this modulus
        backtrack(idx + 1, current_pairs)
    
    backtrack(0, [])
    return best

def _local_search_improve(pairs, time_limit, all_mods):
    """Improve a solution via local search: try swapping or adding/removing pairs."""
    best = [[r, m] for r, m in pairs]
    best_len = len(best)
    all_mods = list(all_mods)
    
    for iteration in range(2000):
        if time.time() > time_limit:
            break
        
        current = best[:]
        
        # Random perturbation: remove 1-2 random pairs
        if current:
            k = min(len(current), random.randint(1, 2))
            indices = random.sample(range(len(current)), k)
            indices.sort(reverse=True)
            for idx in indices:
                current.pop(idx)
        
        # Greedily add back pairs from remaining moduli
        used_mods = set(m for _, m in current)
        remaining = [m for m in all_mods if m not in used_mods]
        
        # Order remaining by constraint (most constrained first)
        def constraint_score(m):
            count = 0
            for other in all_mods:
                if other != m and math.gcd(m, other) > 1:
                    count += 1
            return count
        
        remaining.sort(key=constraint_score, reverse=True)
        
        new_pairs = current[:]
        for m in remaining:
            if time.time() > time_limit:
                break
            # Try to find a valid residue for m
            # Use randomized order to escape local optima
            residues = list(range(m))
            random.shuffle(residues)
            for r in residues:
                ok = True
                for (r2, m2) in new_pairs:
                    if not _disjoint(r, m, r2, m2):
                        ok = False
                        break
                if ok:
                    new_pairs.append([r, m])
                    break
        
        if len(new_pairs) > len(current):
            best = new_pairs
            best_len = len(new_pairs)
    
    return best

def construct():
    t0 = time.time()
    deadline = t0 + 60.0 - DEADLINE_MARGIN
    
    all_mods = list(range(2, MAX_MOD + 1))
    
    # Start with prime construction as baseline
    prime_sol = _prime_construction()
    best = prime_sol
    best_len = len(prime_sol)
    
    # Try backtracking search on a subset of moduli (full set is too large for exhaustive)
    # Use backtracking on all moduli but with early termination
    if time.time() < deadline - 10:
        bt_sol = _backtrack_search(all_mods, deadline - 5)
        if len(bt_sol) > best_len:
            best = bt_sol
            best_len = len(bt_sol)
    
    # Local search improvement
    if time.time() < deadline - 2:
        improved = _local_search_improve(best, deadline, all_mods)
        if len(improved) > best_len:
            best = improved
            best_len = len(improved)
    
    # Convert to required format
    result = [[r, m] for r, m in best]
    return result
# EVOLVE-BLOCK-END
