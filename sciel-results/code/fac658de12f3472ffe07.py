"""Suite task: large integer sets with no two-other-product divisibility."""
import time
import random
from math import isqrt

NMAX = 256

def _valid_addition(x, S):
    """Check if x can be added to set S without violating the constraint.
    Constraint: for all distinct a,b,c in final set, a does not divide b*c.
    When adding x to S, we need to check:
    1. For all pairs (b,c) in S, b!=c: x does not divide b*c  (i.e., (b*c) % x != 0)
    2. For all b in S: for all c in S, c != b: b does not divide x*c  (i.e., (x*c) % b != 0)
       AND c does not divide x*b  (i.e., (x*b) % c != 0) -- but this is the same check with roles swapped.
    Actually, let's be precise. The triples involving x are:
    (x, b, c) where b,c in S, b!=c: check x | b*c
    (b, x, c) where b,c in S, b!=c: check b | x*c
    (c, x, b) where b,c in S, b!=c: check c | x*b
    But (b, x, c) and (c, x, b) are distinct triples if b!=c.
    So for each unordered pair {b,c} from S, we check:
    - x | b*c
    - b | x*c
    - c | x*b
    If any of these is true, x cannot be added.
    """
    S_list = list(S)
    n = len(S_list)
    for i in range(n):
        b = S_list[i]
        for j in range(n):
            if i == j:
                continue
            c = S_list[j]
            # Check triple (x, b, c): does x divide b*c?
            if (b * c) % x == 0:
                return False
            # Check triple (b, x, c): does b divide x*c?
            if (x * c) % b == 0:
                return False
            # Check triple (c, x, b): does c divide x*b?
            if (x * b) % c == 0:
                return False
    return True

def _greedy_largest_first(deadline):
    """Greedy: try numbers from largest to smallest."""
    chosen = []
    chosen_set = set()
    
    for x in range(NMAX, 1, -1):
        if time.time() > deadline:
            break
        if x in chosen_set:
            continue
        if _valid_addition(x, chosen):
            chosen.append(x)
            chosen_set.add(x)
    
    return chosen

def _greedy_randomized(deadline, seed=42):
    """Greedy with randomized order."""
    rng = random.Random(seed)
    candidates = list(range(2, NMAX + 1))
    rng.shuffle(candidates)
    
    chosen = []
    chosen_set = set()
    
    for x in candidates:
        if time.time() > deadline:
            break
        if x in chosen_set:
            continue
        if _valid_addition(x, chosen):
            chosen.append(x)
            chosen_set.add(x)
    
    return chosen

def _greedy_hybrid(deadline, seed=42):
    """Greedy: start with large numbers, then fill with smaller ones."""
    rng = random.Random(seed)
    
    # First, add all large numbers (above some threshold) in descending order
    threshold = 128
    chosen = []
    chosen_set = set()
    
    for x in range(NMAX, threshold, -1):
        if time.time() > deadline:
            break
        if _valid_addition(x, chosen):
            chosen.append(x)
            chosen_set.add(x)
    
    # Then try smaller numbers in randomized order
    smaller = list(range(2, threshold + 1))
    rng.shuffle(smaller)
    
    for x in smaller:
        if time.time() > deadline:
            break
        if x in chosen_set:
            continue
        if _valid_addition(x, chosen):
            chosen.append(x)
            chosen_set.add(x)
    
    return chosen

def _local_search(S, deadline, max_iters=500):
    """Local search: try adding elements, removing elements, swapping."""
    S = list(S)
    S_set = set(S)
    best = S[:]
    best_size = len(S)
    rng = random.Random(12345)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Try adding a random element
        if len(S) < NMAX:
            candidates = [x for x in range(2, NMAX + 1) if x not in S_set]
            if candidates:
                x = rng.choice(candidates)
                if _valid_addition(x, S):
                    S.append(x)
                    S_set.add(x)
                    if len(S) > best_size:
                        best = S[:]
                        best_size = len(S)
                    continue
        
        # Try swapping: remove one, add another
        if S:
            i = rng.randrange(len(S))
            removed = S[i]
            S_temp = S[:i] + S[i+1:]
            S_temp_set = set(S_temp)
            candidates = [x for x in range(2, NMAX + 1) if x not in S_temp_set]
            if candidates:
                # Try a few random candidates
                rng.shuffle(candidates)
                improved = False
                for x in candidates[:30]:
                    if _valid_addition(x, S_temp):
                        S = S_temp + [x]
                        S_set = set(S)
                        if len(S) > best_size:
                            best = S[:]
                            best_size = len(S)
                        improved = True
                        break
                if not improved:
                    # Restore
                    S = S_temp + [removed]
                    S_set = set(S)
            else:
                S = S_temp + [removed]
                S_set = set(S)
    
    return best

def _local_search_aggressive(S, deadline):
    """More aggressive local search with multiple restarts."""
    best = S[:]
    best_size = len(S)
    
    # Try multiple random restarts of local search
    for restart in range(5):
        if time.time() > deadline:
            break
        
        # Perturb the current best
        S_perturbed = best[:]
        rng = random.Random(1000 + restart)
        
        # Remove 1-3 random elements
        n_remove = rng.randint(1, min(3, len(S_perturbed)))
        indices = rng.sample(range(len(S_perturbed)), n_remove)
        indices.sort(reverse=True)
        for idx in indices:
            S_perturbed.pop(idx)
        
        # Run local search on perturbed set
        S_result = _local_search(S_perturbed, deadline, max_iters=100)
        
        if len(S_result) > best_size:
            best = S_result
            best_size = len(S_result)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    best = None
    best_size = 0
    
    # Candidate 1: greedy largest first (should find ~128 elements)
    c = _greedy_largest_first(deadline - 5)
    if c and len(c) > best_size:
        best = c
        best_size = len(c)
    
    # Candidate 2: greedy hybrid (large first, then randomized small)
    if time.time() < deadline - 10:
        c = _greedy_hybrid(deadline - 10, seed=42)
        if c and len(c) > best_size:
            best = c
            best_size = len(c)
    
    # Candidate 3: randomized greedy with different seeds
    for seed in [1, 7, 13, 99, 256]:
        if time.time() > deadline - 15:
            break
        c = _greedy_randomized(deadline - 15, seed=seed)
        if c and len(c) > best_size:
            best = c
            best_size = len(c)
    
    # Local search to improve
    if best and time.time() < deadline - 10:
        c = _local_search(best, deadline - 10, max_iters=300)
        if c and len(c) > best_size:
            best = c
            best_size = len(c)
    
    # Aggressive local search from multiple starts
    for seed in [2, 5, 11, 23, 42, 100, 200, 300]:
        if time.time() > deadline - 5:
            break
        c = _greedy_hybrid(deadline - 5, seed=seed)
        if c:
            c2 = _local_search_aggressive(c, deadline - 5)
            if c2 and len(c2) > best_size:
                best = c2
                best_size = len(c2)
    
    if best is None:
        # Fallback: all primes
        def is_prime(n):
            if n < 2:
                return False
            if n < 4:
                return True
            if n % 2 == 0 or n % 3 == 0:
                return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0:
                    return False
                i += 6
            return True
        best = [x for x in range(2, NMAX + 1) if is_prime(x)]
    
    # Ensure we return a list of Python ints
    return [int(x) for x in best]
