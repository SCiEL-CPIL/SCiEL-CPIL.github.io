import math
import random
import time
from fractions import Fraction

# ---- primes up to 1000 (fixed) ----
def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            step = i
            start = i * i
            sieve[start:n + 1:step] = [False] * len(sieve[start:n + 1:step])
    return [i for i, v in enumerate(sieve) if v]

_PRIMES = _primes_upto(1000)
NUM_PRIMES = len(_PRIMES)
_COSTS = [Fraction(1, p) for p in _PRIMES]
MAX_CHECK = 100000


def _seed_solution():
    """The i-th prime from 13 upward covers the integer i."""
    primes_from_13 = [p for p in _PRIMES if p >= 13]
    assign = [0] * NUM_PRIMES
    for i, p in enumerate(primes_from_13):
        idx = _PRIMES.index(p)
        r = (i + 1) % p
        if r == 0:
            r = 1
        assign[idx] = r
    return assign


def _coverage_length(assign):
    covered = bytearray(MAX_CHECK + 1)
    for idx, r in enumerate(assign):
        if r == 0:
            continue
        p = _PRIMES[idx]
        for n in range(r, MAX_CHECK + 1, p):
            covered[n] = 1
    L = 0
    for n in range(1, MAX_CHECK + 1):
        if covered[n]:
            L = n
        else:
            break
    return L


def _is_valid(assign):
    c = Fraction(0)
    for idx, r in enumerate(assign):
        if r != 0:
            c += _COSTS[idx]
    return c <= Fraction(1, 1)


def _compute_L_from_bits(bits):
    """Compute coverage length from a list of bitarrays (one per position 0..MAX_CHECK)."""
    L = 0
    for n in range(1, MAX_CHECK + 1):
        if any(bits[n]):
            L = n
        else:
            break
    return L


def _branch_and_bound(deadline):
    """
    Branch-and-bound search over primes sorted in descending order.
    At each level, try: skip prime, or use prime with each residue.
    Use bitset coverage to quickly detect gaps.
    """
    start = time.time()
    
    # Sort primes in descending order (larger primes first for better pruning)
    # Actually, let's sort by cost ascending (small primes first) for tighter bounds
    # No - for branch and bound, we want to explore high-value branches first
    # Let's use a different strategy: process primes in order, use greedy + local search
    
    # Better approach: Greedy construction with backtracking
    # Sort primes by 1/p descending (small primes first)
    sorted_indices = list(range(NUM_PRIMES))
    sorted_indices.sort(key=lambda i: -_COSTS[i])  # largest cost first (smallest prime)
    
    best_L = 0
    best_assign = [0] * NUM_PRIMES
    best_cost = Fraction(0)
    
    # Use a recursive approach with memoization of coverage state
    # But that's too expensive. Instead, use iterative deepening with pruning.
    
    # Alternative: Greedy with local improvement
    # Start with no primes, add primes one by one choosing best residue
    current_assign = [0] * NUM_PRIMES
    current_cost = Fraction(0)
    covered = bytearray(MAX_CHECK + 1)
    
    # For each prime in order of decreasing cost (smallest primes first),
    # decide whether to include it and with what residue
    for idx in sorted_indices:
        p = _PRIMES[idx]
        if current_cost + _COSTS[idx] > Fraction(1, 1):
            continue
        
        # Try all residues and pick the one that maximizes new coverage
        best_r = 0
        best_new_L = -1
        best_new_covered = None
        
        # Quick check: compute current L
        cur_L = 0
        for n in range(1, min(200, MAX_CHECK + 1)):
            if covered[n]:
                cur_L = n
            else:
                break
        
        for r in range(1, p + 1):
            # Check if this residue helps
            # Count how many new positions it covers in the first L+100
            new_count = 0
            for n in range(r, min(cur_L + 200, MAX_CHECK + 1), p):
                if not covered[n]:
                    new_count += 1
            
            if new_count > best_new_L:
                best_new_L = new_count
                best_r = r
        
        if best_r > 0 and best_new_L > 0:
            # Add this prime
            for n in range(best_r, MAX_CHECK + 1, p):
                covered[n] = 1
            current_assign[idx] = best_r
            current_cost += _COSTS[idx]
    
    # Compute final L
    L = 0
    for n in range(1, MAX_CHECK + 1):
        if covered[n]:
            L = n
        else:
            break
    
    return current_assign, L, current_cost


def _greedy_with_backtracking(deadline):
    """
    Greedy construction with backtracking.
    Process primes in order, at each step try all residues.
    Use coverage tracking to guide decisions.
    """
    start = time.time()
    
    # Sort primes: small primes first (higher cost, more flexible)
    sorted_indices = list(range(NUM_PRIMES))
    sorted_indices.sort(key=lambda i: _PRIMES[i])
    
    best_assign = [0] * NUM_PRIMES
    best_L = 0
    best_cost = Fraction(0)
    
    current_assign = [0] * NUM_PRIMES
    current_cost = Fraction(0)
    covered = bytearray(MAX_CHECK + 1)
    
    def compute_L():
        L = 0
        for n in range(1, MAX_CHECK + 1):
            if covered[n]:
                L = n
            else:
                break
        return L
    
    def try_add_prime(idx, r):
        nonlocal current_assign, current_cost
        p = _PRIMES[idx]
        for n in range(r, MAX_CHECK + 1, p):
            covered[n] = 1
        current_assign[idx] = r
        current_cost += _COSTS[idx]
    
    def try_remove_prime(idx, r):
        nonlocal current_assign, current_cost
        p = _PRIMES[idx]
        for n in range(r, MAX_CHECK + 1, p):
            covered[n] = 0
        current_assign[idx] = 0
        current_cost -= _COSTS[idx]
    
    # Greedy phase: add primes one by one
    for idx in sorted_indices:
        if time.time() > deadline:
            break
        if current_cost + _COSTS[idx] > Fraction(1, 1):
            continue
        
        p = _PRIMES[idx]
        cur_L = compute_L()
        
        # Find best residue
        best_r = 0
        best_gain = 0
        for r in range(1, p + 1):
            gain = 0
            for n in range(r, min(cur_L + 100, MAX_CHECK + 1), p):
                if not covered[n]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_r = r
        
        if best_r > 0 and best_gain > 0:
            try_add_prime(idx, best_r)
    
    L = compute_L()
    if L > best_L:
        best_L = L
        best_assign = current_assign[:]
        best_cost = current_cost
    
    # Local improvement phase
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try changing residues of used primes
        used = [i for i in range(NUM_PRIMES) if current_assign[i] != 0]
        for idx in used:
            if time.time() > deadline:
                break
            p = _PRIMES[idx]
            old_r = current_assign[idx]
            
            for r in range(1, p + 1):
                if r == old_r:
                    continue
                
                # Remove old
                for n in range(old_r, MAX_CHECK + 1, p):
                    covered[n] = 0
                # Add new
                for n in range(r, MAX_CHECK + 1, p):
                    covered[n] = 1
                
                new_L = compute_L()
                if new_L > L:
                    L = new_L
                    current_assign[idx] = r
                    improved = True
                    if L > best_L:
                        best_L = L
                        best_assign = current_assign[:]
                        best_cost = current_cost
                    break
                else:
                    # Revert
                    for n in range(r, MAX_CHECK + 1, p):
                        covered[n] = 0
                    for n in range(old_r, MAX_CHECK + 1, p):
                        covered[n] = 1
        
        # Try adding unused primes
        if not improved:
            unused = [i for i in range(NUM_PRIMES) if current_assign[i] == 0]
            for idx in unused:
                if time.time() > deadline:
                    break
                if current_cost + _COSTS[idx] > Fraction(1, 1):
                    continue
                
                p = _PRIMES[idx]
                cur_L = L
                best_r = 0
                best_gain = 0
                for r in range(1, p + 1):
                    gain = 0
                    for n in range(r, min(cur_L + 100, MAX_CHECK + 1), p):
                        if not covered[n]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_r = r
                
                if best_r > 0 and best_gain > 0:
                    for n in range(best_r, MAX_CHECK + 1, p):
                        covered[n] = 1
                    current_assign[idx] = best_r
                    current_cost += _COSTS[idx]
                    L = compute_L()
                    improved = True
                    if L > best_L:
                        best_L = L
                        best_assign = current_assign[:]
                        best_cost = current_cost
    
    return best_assign, best_L


def _systematic_search(deadline):
    """
    Systematic search: try different orderings and strategies.
    Use multiple random restarts with different prime orderings.
    """
    start = time.time()
    best_assign = [0] * NUM_PRIMES
    best_L = 0
    
    # Try multiple random orderings
    for attempt in range(10):
        if time.time() > deadline - 5:
            break
        
        # Random permutation of primes
        indices = list(range(NUM_PRIMES))
        random.shuffle(indices)
        
        current_assign = [0] * NUM_PRIMES
        current_cost = Fraction(0)
        covered = bytearray(MAX_CHECK + 1)
        
        def compute_L():
            L = 0
            for n in range(1, MAX_CHECK + 1):
                if covered[n]:
                    L = n
                else:
                    break
            return L
        
        # Greedy add
        for idx in indices:
            if time.time() > deadline - 5:
                break
            if current_cost + _COSTS[idx] > Fraction(1, 1):
                continue
            
            p = _PRIMES[idx]
            cur_L = compute_L()
            
            best_r = 0
            best_gain = 0
            for r in range(1, p + 1):
                gain = 0
                for n in range(r, min(cur_L + 50, MAX_CHECK + 1), p):
                    if not covered[n]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            if best_r > 0 and best_gain > 0:
                for n in range(best_r, MAX_CHECK + 1, p):
                    covered[n] = 1
                current_assign[idx] = best_r
                current_cost += _COSTS[idx]
        
        L = compute_L()
        
        # Quick local improvement
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            used = [i for i in range(NUM_PRIMES) if current_assign[i] != 0]
            for idx in used:
                if time.time() > deadline - 5:
                    break
                p = _PRIMES[idx]
                old_r = current_assign[idx]
                
                # Try a few random residues
                for _ in range(min(5, p)):
                    r = random.randint(1, p)
                    if r == old_r:
                        continue
                    
                    for n in range(old_r, MAX_CHECK + 1, p):
                        covered[n] = 0
                    for n in range(r, MAX_CHECK + 1, p):
                        covered[n] = 1
                    
                    new_L = compute_L()
                    if new_L > L:
                        L = new_L
                        current_assign[idx] = r
                        improved = True
                        break
                    else:
                        for n in range(r, MAX_CHECK + 1, p):
                            covered[n] = 0
                        for n in range(old_r, MAX_CHECK + 1, p):
                            covered[n] = 1
            
            # Try adding a few unused primes
            if not improved:
                unused = [i for i in range(NUM_PRIMES) if current_assign[i] == 0]
                random.shuffle(unused)
                for idx in unused[:5]:
                    if time.time() > deadline - 5:
                        break
                    if current_cost + _COSTS[idx] > Fraction(1, 1):
                        continue
                    
                    p = _PRIMES[idx]
                    r = random.randint(1, p)
                    for n in range(r, MAX_CHECK + 1, p):
                        covered[n] = 1
                    current_assign[idx] = r
                    current_cost += _COSTS[idx]
                    L = compute_L()
                    improved = True
        
        if L > best_L:
            best_L = L
            best_assign = current_assign[:]
    
    return best_assign, best_L


def construct():
    start = time.time()
    deadline = start + 140  # Safety margin
    
    # Phase 1: Seed solution
    seed_assign = _seed_solution()
    seed_L = _coverage_length(seed_assign)
    
    best_assign = seed_assign[:]
    best_L = seed_L
    
    # Phase 2: Greedy with backtracking (different ordering)
    if time.time() < deadline - 60:
        greedy_assign, greedy_L = _greedy_with_backtracking(deadline - 60)
        if greedy_L > best_L and _is_valid(greedy_assign):
            best_assign = greedy_assign
            best_L = greedy_L
    
    # Phase 3: Systematic search with multiple random orderings
    if time.time() < deadline - 30:
        sys_assign, sys_L = _systematic_search(deadline - 30)
        if sys_L > best_L and _is_valid(sys_assign):
            best_assign = sys_assign
            best_L = sys_L
    
    # Phase 4: Final local search on best
    if time.time() < deadline - 5:
        # Quick local improvement
        covered = bytearray(MAX_CHECK + 1)
        for idx, r in enumerate(best_assign):
            if r != 0:
                p = _PRIMES[idx]
                for n in range(r, MAX_CHECK + 1, p):
                    covered[n] = 1
        
        cur_L = 0
        for n in range(1, MAX_CHECK + 1):
            if covered[n]:
                cur_L = n
            else:
                break
        
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            used = [i for i in range(NUM_PRIMES) if best_assign[i] != 0]
            for idx in used:
                if time.time() > deadline - 3:
                    break
                p = _PRIMES[idx]
                old_r = best_assign[idx]
                
                for r in range(1, p + 1):
                    if r == old_r:
                        continue
                    
                    for n in range(old_r, MAX_CHECK + 1, p):
                        covered[n] = 0
                    for n in range(r, MAX_CHECK + 1, p):
                        covered[n] = 1
                    
                    new_L = 0
                    for n in range(1, MAX_CHECK + 1):
                        if covered[n]:
                            new_L = n
                        else:
                            break
                    
                    if new_L > cur_L:
                        cur_L = new_L
                        best_assign[idx] = r
                        improved = True
                        if cur_L > best_L:
                            best_L = cur_L
                        break
                    else:
                        for n in range(r, MAX_CHECK + 1, p):
                            covered[n] = 0
                        for n in range(old_r, MAX_CHECK + 1, p):
                            covered[n] = 1
    
    # Final validation
    if _is_valid(best_assign):
        return [[_PRIMES[i], r] for i, r in enumerate(best_assign) if r != 0]
    else:
        return [[_PRIMES[i], r] for i, r in enumerate(seed_assign) if r != 0]
