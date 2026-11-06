import time
import numpy as np
from fractions import Fraction
from math import isqrt

# Precompute primes up to 1000
def _primes_up_to(n):
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

_PRIMES = _primes_up_to(1000)
_PRIME_SET = set(_PRIMES)
_LCM_DENOM = 1  # we'll use Fraction for exactness
_MAX_L = 100000

def _coverage_length(classes, cap=_MAX_L):
    """Compute largest L such that every integer 1..L is covered.
    classes: list of [p, r]
    """
    if not classes:
        return 0
    # Use numpy for speed
    n = cap + 1  # 1-indexed, index 0 unused
    covered = np.zeros(n, dtype=bool)
    for p, r in classes:
        # numbers x in [1, cap] with x % p == r
        # smallest x >= 1: if r == 0, x = p, 2p, ...; else x = r, r+p, ...
        start = r if r >= 1 else p
        if start > cap:
            # maybe r=0: start=p
            start = p
            if start > cap:
                continue
        covered[start::p] = True
    # find first gap
    # covered[1:] gives coverage of 1..cap
    c = covered[1:]
    # find first False
    if c.all():
        return cap
    idx = np.argmax(~c)
    return int(idx)

def _exact_reciprocal_sum(classes):
    s = Fraction(0, 1)
    for p, r in classes:
        s += Fraction(1, p)
    return s

def _valid(classes):
    """Check distinct primes, 0<=r<p, sum 1/p <= 1."""
    seen = set()
    s = Fraction(0, 1)
    for p, r in classes:
        if p in seen:
            return False
        if p < 2 or p > 1000 or p not in _PRIME_SET:
            return False
        if r < 0 or r >= p:
            return False
        seen.add(p)
        s += Fraction(1, p)
        if s > 1:
            return False
    return True

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    best_classes = None
    best_L = 0

    # ---- Phase 1: Greedy set-cover to get a good initial solution ----
    # We want to cover 1..L maximally with sum 1/p <= 1.
    # Greedy: repeatedly pick the (p,r) that covers the most new integers
    # among currently-uncovered, subject to budget.
    
    # Candidate pool: for each prime p, all residues r in [0, p-1]
    # But that's huge. Instead, use a smarter heuristic.
    
    # Strategy: use the "natural" covering where prime p covers residue r,
    # and try to maximize coverage.
    
    # Start with a greedy approach:
    # - Maintain covered set
    # - At each step, try adding (p, r) that maximizes new coverage / cost
    # - Stop when budget exceeded or no improvement
    
    # For efficiency, we'll work with a target L and check coverage.
    
    # Initial: use the seed-like construction as a baseline
    # primes from 13 upward, p_i covers i
    primes_list = [p for p in _PRIMES if p >= 13]
    seed_classes = [[p, (i + 1) % p] for i, p in enumerate(primes_list)]
    seed_L = _coverage_length(seed_classes)
    if _valid(seed_classes):
        best_classes = seed_classes
        best_L = seed_L
    
    # Now do local search / greedy improvement
    # We'll try to improve by:
    # 1. Greedy addition of high-value (p,r)
    # 2. Swap moves
    # 3. Perturbation
    
    def _greedy_add(classes, budget_remaining, covered_mask, cap):
        """Try to add (p,r) that gives best new coverage per cost."""
        best_gain = 0
        best_pr = None
        best_cost = None
        n = cap + 1
        for p in _PRIMES:
            if any(c[0] == p for c in classes):
                continue
            cost = Fraction(1, p)
            if cost > budget_remaining:
                continue
            # For this p, try all residues? Too many for large p.
            # Heuristic: try residue that covers the first uncovered number
            # that is congruent to something mod p
            # Actually, try a few promising residues
            # The residue that covers the most uncovered in [1, cap]
            # For speed, try: r = 0, and r = (first_uncovered % p)
            # Let's try the residue that aligns with the first gap
            # Find first uncovered
            if not covered_mask[1:].all():
                first_gap = int(np.argmax(~covered_mask[1:])) + 1
            else:
                first_gap = cap + 1
            candidates_r = set()
            candidates_r.add(0)
            candidates_r.add(first_gap % p)
            # Also try a random residue
            import random
            candidates_r.add(random.randint(0, p - 1))
            for r in candidates_r:
                if r < 0 or r >= p:
                    continue
                # Compute new coverage
                start = r if r >= 1 else p
                if start > cap:
                    continue
                # Count new covered
                idx = np.arange(start, cap + 1, p)
                new_count = int(np.sum(~covered_mask[idx]))
                gain = new_count
                if gain > best_gain:
                    best_gain = gain
                    best_pr = (p, r)
                    best_cost = cost
        return best_pr, best_gain, best_cost
    
    # Phase 2: Iterative improvement
    # We'll use a simple local search with swaps
    import random
    random.seed(42)
    
    current_classes = list(best_classes) if best_classes else []
    current_L = best_L
    
    # Build covered mask for current
    def _build_mask(classes, cap):
        n = cap + 1
        covered = np.zeros(n, dtype=bool)
        for p, r in classes:
            start = r if r >= 1 else p
            if start > cap:
                continue
            covered[start::p] = True
        return covered
    
    # Try to improve by adding classes
    for iteration in range(200):
        if time.time() > deadline - 10:
            break
        if not current_classes:
            break
        budget = 1 - _exact_reciprocal_sum(current_classes)
        if budget <= 0:
            break
        cap = max(current_L + 1000, 2000)
        covered_mask = _build_mask(current_classes, cap)
        # Try to add
        best_pr, best_gain, best_cost = _greedy_add(current_classes, budget, covered_mask, cap)
        if best_pr is None or best_gain <= 0:
            # Try swap: remove one, add two
            # For simplicity, try removing the least useful and adding
            break
        p, r = best_pr
        new_classes = current_classes + [[p, r]]
        if _exact_reciprocal_sum(new_classes) <= 1:
            new_L = _coverage_length(new_classes)
            if new_L > current_L:
                current_classes = new_classes
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_classes = list(current_classes)
            else:
                break
        else:
            break
    
    # Phase 3: Local search with swaps
    for iteration in range(500):
        if time.time() > deadline - 5:
            break
        if not current_classes:
            break
        budget = 1 - _exact_reciprocal_sum(current_classes)
        cap = max(current_L + 500, 1000)
        covered_mask = _build_mask(current_classes, cap)
        
        # Try swap: remove one class, add one different class
        improved = False
        for i in range(len(current_classes)):
            if time.time() > deadline - 5:
                break
            p_remove, r_remove = current_classes[i]
            # Temporarily remove
            remaining = current_classes[:i] + current_classes[i+1:]
            new_budget = 1 - _exact_reciprocal_sum(remaining)
            # Try to add a new class
            for p_add in _PRIMES:
                if any(c[0] == p_add for c in remaining):
                    continue
                cost = Fraction(1, p_add)
                if cost > new_budget:
                    continue
                # Try a few residues
                if not covered_mask[1:].all():
                    first_gap = int(np.argmax(~covered_mask[1:])) + 1
                else:
                    first_gap = cap + 1
                candidate_rs = {0, first_gap % p_add, random.randint(0, p_add-1)}
                for r_add in candidate_rs:
                    if r_add < 0 or r_add >= p_add:
                        continue
                    new_classes = remaining + [[p_add, r_add]]
                    new_L = _coverage_length(new_classes, cap)
                    if new_L > current_L:
                        current_classes = new_classes
                        current_L = new_L
                        improved = True
                        if current_L > best_L:
                            best_L = current_L
                            best_classes = list(current_classes)
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            # Try perturbation: randomly change a residue
            if current_classes:
                i = random.randint(0, len(current_classes) - 1)
                p, r = current_classes[i]
                new_r = random.randint(0, p - 1)
                if new_r != r:
                    current_classes[i] = [p, new_r]
                    new_L = _coverage_length(current_classes)
                    if new_L >= current_L:
                        current_L = new_L
                        if current_L > best_L:
                            best_L = current_L
                            best_classes = list(current_classes)
    
    # Final validation
    if best_classes and _valid(best_classes):
        final_L = _coverage_length(best_classes)
        if final_L > best_L:
            best_L = final_L
        return best_classes
    elif current_classes and _valid(current_classes):
        final_L = _coverage_length(current_classes)
        if final_L > best_L:
            best_L = final_L
            best_classes = current_classes
        return current_classes
    else:
        # Fallback to seed
        return seed_classes

# Keep the interface exactly as required
def construct():
    return _construct_impl()

def _construct_impl():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    best_classes = None
    best_L = 0

    # Precompute
    primes_list = [p for p in _PRIMES if p >= 13]
    seed_classes = [[p, (i + 1) % p] for i, p in enumerate(primes_list)]
    if _valid(seed_classes):
        best_classes = seed_classes
        best_L = _coverage_length(seed_classes)
    
    import random
    random.seed(12345)
    
    def _build_mask(classes, cap):
        n = cap + 1
        covered = np.zeros(n, dtype=bool)
        for p, r in classes:
            start = r if r >= 1 else p
            if start <= cap:
                covered[start::p] = True
        return covered
    
    def _coverage_from_mask(covered_mask, cap):
        c = covered_mask[1:]
        if c.all():
            return cap
        return int(np.argmax(~c))
    
    current_classes = list(best_classes) if best_classes else []
    current_L = best_L
    
    # Greedy addition phase
    for iteration in range(100):
        if time.time() > deadline - 15:
            break
        if not current_classes:
            break
        budget = 1 - _exact_reciprocal_sum(current_classes)
        if budget <= 0:
            break
        cap = max(current_L + 2000, 3000)
        covered_mask = _build_mask(current_classes, cap)
        
        best_gain = 0
        best_pr = None
        for p in _PRIMES:
            if any(c[0] == p for c in current_classes):
                continue
            cost = Fraction(1, p)
            if cost > budget:
                continue
            if not covered_mask[1:].all():
                first_gap = int(np.argmax(~covered_mask[1:])) + 1
            else:
                first_gap = cap + 1
            candidate_rs = {0, first_gap % p}
            for r in candidate_rs:
                if r < 0 or r >= p:
                    continue
                start = r if r >= 1 else p
                if start > cap:
                    continue
                idx = np.arange(start, cap + 1, p)
                new_count = int(np.sum(~covered_mask[idx]))
                if new_count > best_gain:
                    best_gain = new_count
                    best_pr = (p, r)
        
        if best_pr is None or best_gain <= 0:
            break
        p, r = best_pr
        new_classes = current_classes + [[p, r]]
        if _exact_reciprocal_sum(new_classes) <= 1:
            new_L = _coverage_length(new_classes)
            if new_L > current_L:
                current_classes = new_classes
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_classes = list(current_classes)
            else:
                break
    
    # Local search: swaps and perturbations
    for iteration in range(300):
        if time.time() > deadline - 5:
            break
        if not current_classes:
            break
        budget = 1 - _exact_reciprocal_sum(current_classes)
        cap = max(current_L + 1000, 1500)
        covered_mask = _build_mask(current_classes, cap)
        
        improved = False
        # Try swaps
        for i in range(len(current_classes)):
            if time.time() > deadline - 5:
                break
            p_remove, r_remove = current_classes[i]
            remaining = current_classes[:i] + current_classes[i+1:]
            new_budget = 1 - _exact_reciprocal_sum(remaining)
            for p_add in _PRIMES:
                if any(c[0] == p_add for c in remaining):
                    continue
                cost = Fraction(1, p_add)
                if cost > new_budget:
                    continue
                if not covered_mask[1:].all():
                    first_gap = int(np.argmax(~covered_mask[1:])) + 1
                else:
                    first_gap = cap + 1
                candidate_rs = {0, first_gap % p_add, random.randint(0, p_add-1)}
                for r_add in candidate_rs:
                    if r_add < 0 or r_add >= p_add:
                        continue
                    new_classes = remaining + [[p_add, r_add]]
                    new_L = _coverage_length(new_classes, cap)
                    if new_L > current_L:
                        current_classes = new_classes
                        current_L = new_L
                        improved = True
                        if current_L > best_L:
                            best_L = current_L
                            best_classes = list(current_classes)
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Perturbation
            if current_classes:
                i = random.randint(0, len(current_classes) - 1)
                p, r = current_classes[i]
                new_r = random.randint(0, p - 1)
                if new_r != r:
                    current_classes[i] = [p, new_r]
                    new_L = _coverage_length(current_classes)
                    if new_L >= current_L:
                        current_L = new_L
                        if current_L > best_L:
                            best_L = current_L
                            best_classes = list(current_classes)
    
    # Final
    if best_classes and _valid(best_classes):
        return best_classes
    elif current_classes and _valid(current_classes):
        return current_classes
    else:
        return seed_classes
