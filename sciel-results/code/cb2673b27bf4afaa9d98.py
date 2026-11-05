"""Structural redesign: greedy construction + multi-start local search with count-based coverage."""
import time
import random
from fractions import Fraction

# Deadline guard: kill at 145s to stay under 150s wall-clock
_DEADLINE = 145.0
_start_time = None

def _time_ok():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return (time.time() - _start_time) < _DEADLINE

def _sieve(n):
    """Return list of primes <= n."""
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

_ALL_PRIMES = _sieve(1000)
# Map prime to its reciprocal as a float for quick budget checks, and Fraction for exact
_PRIME_RECIP_FLOAT = {p: 1.0 / p for p in _ALL_PRIMES}
_PRIME_RECIP_FRAC = {p: Fraction(1, p) for p in _ALL_PRIMES}

def _coverage_length(classes, max_check=100000):
    """Return the largest L such that every integer 1..L is covered by some (p, r) in classes.
    Uses an efficient marking approach."""
    if not classes:
        return 0
    # We need to find the longest prefix [1, L] where every number is covered.
    # A number x is covered if exists (p, r) such that x % p == r.
    # We'll use a difference-array / sweep approach: for each (p, r), mark which numbers in [1, max_check] it covers.
    # But that's O(sum p) which could be large. Instead, we check sequentially and stop at first gap.
    
    # For efficiency, we precompute for each prime p, the set of residues.
    # Actually, the fastest approach for small L: iterate x from 1 upward, check if covered.
    # Each check is O(len(classes)). Total O(L * len(classes)).
    # For L up to ~5000 and ~30 classes, that's 150k ops per evaluation. Fast enough.
    
    L = 0
    for x in range(1, max_check + 1):
        covered = False
        for p, r in classes:
            if x % p == r:
                covered = True
                break
        if covered:
            L = x
        else:
            break
    return L

def _budget_ok(classes):
    """Check sum of 1/p <= 1 using exact fractions."""
    total = Fraction(0, 1)
    for p, r in classes:
        total += _PRIME_RECIP_FRAC[p]
    return total <= 1

def _budget_sum_float(classes):
    """Fast float budget sum."""
    return sum(_PRIME_RECIP_FLOAT[p] for p, r in classes)

def _greedy_construct():
    """Greedy: try to build a covering set. Start with small primes, assign residues to maximize coverage."""
    # Strategy: use primes in increasing order, for each prime choose the residue that covers the most uncovered numbers in [1, L_target].
    # But we don't know L_target. Let's just build a set and evaluate.
    
    # Alternative greedy: start with all primes <= 1000 sorted by 1/p descending (i.e., smallest primes first).
    # For each prime, pick the residue that maximizes coverage of [1, current_best_L + margin].
    
    classes = []
    used_primes = set()
    
    # We'll iterate primes in order of increasing p (largest 1/p first)
    for p in _ALL_PRIMES:
        if p > 1000:
            break
        # Check budget
        current_sum = _budget_sum_float(classes)
        if current_sum + 1.0 / p > 1.0:
            continue
        # Choose best residue for this prime
        # Try to cover as much of [1, 5000] as possible
        best_r = 0
        best_cov = -1
        # Sample: for each residue, count how many x in [1, 5000] are covered
        # But we need to count only those NOT already covered by existing classes.
        # For speed, let's just pick residue 0 or random and refine later.
        # Actually, let's pick the residue that covers 1 (i.e., r = 1 % p) to start with.
        best_r = 1 % p
        classes.append([p, best_r])
        used_primes.add(p)
    
    return classes

def _local_search(classes, max_iters=50000):
    """Local search: try swapping residues, adding/removing primes."""
    best_classes = [c[:] for c in classes]
    best_L = _coverage_length(best_classes)
    best_budget = _budget_sum_float(best_classes)
    
    primes_in_set = [p for p, r in best_classes]
    residues = {p: r for p, r in best_classes}
    
    for it in range(max_iters):
        if not _time_ok():
            break
        if random.random() < 0.3:
            # Try changing a residue
            if not primes_in_set:
                continue
            p = random.choice(primes_in_set)
            old_r = residues[p]
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue
            residues[p] = new_r
            new_classes = [[pp, residues[pp]] for pp in primes_in_set]
            if _budget_sum_float(new_classes) <= 1.0:
                L = _coverage_length(new_classes)
                if L > best_L:
                    best_L = L
                    best_classes = new_classes
                else:
                    residues[p] = old_r
            else:
                residues[p] = old_r
        elif random.random() < 0.5:
            # Try adding a prime
            candidates = [p for p in _ALL_PRIMES if p <= 1000 and p not in residues]
            if not candidates:
                continue
            p = random.choice(candidates)
            if best_budget + 1.0 / p > 1.0:
                continue
            r = random.randint(0, p - 1)
            residues[p] = r
            new_classes = [[pp, residues[pp]] for pp in primes_in_set + [p]]
            L = _coverage_length(new_classes)
            if L > best_L:
                best_L = L
                best_classes = new_classes
                primes_in_set.append(p)
                best_budget += 1.0 / p
            else:
                del residues[p]
        else:
            # Try removing a prime
            if len(primes_in_set) <= 2:
                continue
            p = random.choice(primes_in_set)
            del residues[p]
            primes_in_set.remove(p)
            best_budget -= 1.0 / p
            new_classes = [[pp, residues[pp]] for pp in primes_in_set]
            L = _coverage_length(new_classes)
            if L > best_L:
                best_L = L
                best_classes = new_classes
            else:
                # Revert
                primes_in_set.append(p)
                residues[p] = random.randint(0, p - 1)
                best_budget += 1.0 / p
    
    return best_classes, best_L

def construct():
    global _start_time
    _start_time = time.time()
    
    # Start with a greedy construction
    base_classes = _greedy_construct()
    if not _budget_ok(base_classes):
        # Fallback to simple seed
        primes = [p for p in range(13, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
        base_classes = [[p, (i + 1) % p] for i, p in enumerate(primes)]
    
    best_L = _coverage_length(base_classes)
    best_classes = [c[:] for c in base_classes]
    
    # Multi-start local search
    random.seed(42)
    num_starts = 20
    for s in range(num_starts):
        if not _time_ok():
            break
        # Perturb the base
        perturbed = [c[:] for c in base_classes]
        # Randomly change some residues
        for _ in range(min(5, len(perturbed))):
            if perturbed:
                idx = random.randint(0, len(perturbed) - 1)
                p = perturbed[idx][0]
                perturbed[idx][1] = random.randint(0, p - 1)
        # Maybe add/remove a prime
        if random.random() < 0.5 and len(perturbed) > 3:
            idx = random.randint(0, len(perturbed) - 1)
            perturbed.pop(idx)
        elif random.random() < 0.3:
            candidates = [p for p in _ALL_PRIMES if p <= 1000 and p not in [c[0] for c in perturbed]]
            if candidates:
                p = random.choice(candidates)
                if _budget_sum_float(perturbed) + 1.0 / p <= 1.0:
                    perturbed.append([p, random.randint(0, p - 1)])
        
        if not _budget_ok(perturbed):
            continue
        
        result, L = _local_search(perturbed, max_iters=10000)
        if L > best_L:
            best_L = L
            best_classes = result
    
    # Final check
    if not _budget_ok(best_classes):
        # Fallback
        primes = [p for p in range(13, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
        best_classes = [[p, (i + 1) % p] for i, p in enumerate(primes)]
    
    return best_classes
