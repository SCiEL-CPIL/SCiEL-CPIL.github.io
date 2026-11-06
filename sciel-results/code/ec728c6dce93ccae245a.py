"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random
from itertools import combinations

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)

def _are_disjoint(r1, m1, r2, m2):
    """Two residue classes r1 mod m1 and r2 mod m2 are disjoint iff gcd(m1,m2) does not divide (r2-r1)."""
    g = _gcd(m1, m2)
    return (r2 - r1) % g != 0

def _validate(pairs):
    """Check all pairs are disjoint."""
    n = len(pairs)
    for i in range(n):
        for j in range(i+1, n):
            if not _are_disjoint(pairs[i][0], pairs[i][1], pairs[j][0], pairs[j][1]):
                return False
    return True

def _primes_up_to(n):
    sieve = [True] * (n+1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5)+1):
        if sieve[i]:
            for j in range(i*i, n+1, i):
                sieve[j] = False
    return [i for i in range(2, n+1) if sieve[i]]

def construct():
    deadline = time.time() + 85  # 85s safety margin before 90s kill
    
    # Start with all primes up to 128
    primes = _primes_up_to(128)
    # For primes, any residue works; use 0 for simplicity
    best_pairs = [[0, p] for p in primes]
    used_moduli = set(primes)
    
    # Available composite moduli from 2..128 not already used
    all_moduli = list(range(2, 129))
    available_composites = [m for m in all_moduli if m not in used_moduli]
    
    # Try to add composite moduli one by one using greedy + backtracking
    def try_add_modulus(pairs, moduli_to_try, used_mods):
        """Try to add as many moduli as possible from moduli_to_try."""
        current = list(pairs)
        used = set(used_mods)
        added = 0
        
        # Sort by "difficulty" - prefer moduli with more prime factors (harder to place)
        # Actually, prefer moduli that share few factors with existing ones
        def difficulty(m):
            # Count how many existing moduli share a factor > 1 with m
            count = 0
            for _, em in current:
                if _gcd(m, em) > 1:
                    count += 1
            return count
        
        remaining = list(moduli_to_try)
        remaining.sort(key=lambda m: difficulty(m))
        
        for m in remaining:
            if time.time() > deadline:
                break
            if m in used:
                continue
            
            # Try to find a residue r in [0, m) such that (r, m) is disjoint from all current pairs
            found = False
            for r in range(m):
                ok = True
                for r2, m2 in current:
                    if not _are_disjoint(r, m, r2, m2):
                        ok = False
                        break
                if ok:
                    current.append([r, m])
                    used.add(m)
                    added += 1
                    found = True
                    break
            
            if not found:
                continue
        
        return current, used, added
    
    # Greedy pass to add composites
    best_pairs, used_moduli, _ = try_add_modulus(best_pairs, available_composites, used_moduli)
    
    # Local search: try to swap residues or remove/add moduli to improve
    def local_search(pairs, used_mods, max_iter=2000):
        """Hill climbing: try removing one pair and re-adding with different residue, or swapping moduli."""
        current = [list(p) for p in pairs]
        used = set(used_mods)
        best_count = len(current)
        all_moduli_set = set(range(2, 129))
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try to add a new modulus not in used
            candidates = [m for m in range(2, 129) if m not in used]
            random.shuffle(candidates)
            
            for m in candidates[:5]:  # Try a few
                for r in range(m):
                    ok = True
                    for r2, m2 in current:
                        if not _are_disjoint(r, m, r2, m2):
                            ok = False
                            break
                    if ok:
                        current.append([r, m])
                        used.add(m)
                        best_count = len(current)
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                # Try removing a pair and re-adding with different residue
                if len(current) > 1:
                    idx = random.randint(0, len(current)-1)
                    r_old, m_old = current[idx]
                    current.pop(idx)
                    used.discard(m_old)
                    
                    # Try to re-add with a different residue
                    found = False
                    for r in range(m_old):
                        if r == r_old:
                            continue
                        ok = True
                        for r2, m2 in current:
                            if not _are_disjoint(r, m_old, r2, m2):
                                ok = False
                                break
                        if ok:
                            current.append([r, m_old])
                            used.add(m_old)
                            found = True
                            break
                    
                    if not found:
                        # Restore
                        current.append([r_old, m_old])
                        used.add(m_old)
        
        return current, used
    
    best_pairs, used_moduli = local_search(best_pairs, used_moduli)
    
    # Multiple restarts with randomization
    best_overall = [list(p) for p in best_pairs]
    best_count = len(best_overall)
    
    restarts = 0
    while time.time() < deadline - 5:
        restarts += 1
        
        # Randomized greedy: start with primes, add composites in random order
        current = [[0, p] for p in primes]
        used = set(primes)
        
        remaining = [m for m in range(2, 129) if m not in used]
        random.shuffle(remaining)
        
        for m in remaining:
            if time.time() > deadline - 5:
                break
            for r in range(m):
                ok = True
                for r2, m2 in current:
                    if not _are_disjoint(r, m, r2, m2):
                        ok = False
                        break
                if ok:
                    current.append([r, m])
                    used.add(m)
                    break
        
        # Local search
        current, used = local_search(current, used, max_iter=500)
        
        if len(current) > best_count:
            best_count = len(current)
            best_overall = [list(p) for p in current]
        
        # Early stop if we've used all moduli
        if best_count >= 127:
            break
    
    # Final validation
    if not _validate(best_overall):
        # Fallback to just primes
        best_overall = [[0, p] for p in primes]
    
    return best_overall
# EVOLVE-BLOCK-END
