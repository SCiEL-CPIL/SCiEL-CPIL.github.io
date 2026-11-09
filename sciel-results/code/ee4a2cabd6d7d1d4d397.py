"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _objective(A):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers."""
    n = len(A)
    sums = set()
    prods = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sums.add(ai + aj)
            prods.add(ai * aj)
    return max(len(sums), len(prods))

def _is_valid(A, n=20, maximum=10000):
    """Check that A has n distinct positive integers <= maximum."""
    if len(A) != n:
        return False
    if len(set(A)) != n:
        return False
    if any(x < 1 or x > maximum for x in A):
        return False
    return True

def _generate_structured_candidates():
    """Generate candidate sets using structured patterns (AP, GP, mixed)."""
    candidates = []
    random.seed(42)
    
    # 1. Pure arithmetic progressions: a, a+d, a+2d, ..., a+19d
    for a in [1, 2, 3, 5, 10]:
        for d in [1, 2, 3, 5, 7, 10, 13]:
            A = [a + i * d for i in range(20)]
            if max(A) <= 10000:
                candidates.append(sorted(A))
    
    # 2. Pure geometric progressions: a, a*r, a*r^2, ..., a*r^19 (integer ratios)
    for a in [1, 2, 3]:
        for r in [2, 3]:
            A = [a * (r ** i) for i in range(20)]
            if max(A) <= 10000:
                candidates.append(sorted(A))
    
    # 3. Mixed: first k from AP, rest from GP (or vice versa)
    for k in range(10, 16):
        for a1 in [1, 2, 3, 5]:
            for d in [1, 2, 3, 5]:
                for a2 in [10, 20, 50, 100]:
                    for r in [2, 3]:
                        try:
                            part1 = [a1 + i * d for i in range(k)]
                            part2 = [a2 * (r ** i) for i in range(20 - k)]
                            A = sorted(set(part1 + part2))
                            if len(A) == 20 and max(A) <= 10000:
                                candidates.append(A)
                        except OverflowError:
                            continue
    
    # 4. Two interleaved APs
    for a1 in [1, 2, 3]:
        for d1 in [2, 4, 6, 8]:
            for a2 in [2, 4, 6, 8]:
                for d2 in [3, 5, 7, 9]:
                    part1 = [a1 + i * d1 for i in range(10)]
                    part2 = [a2 + i * d2 for i in range(10)]
                    A = sorted(set(part1 + part2))
                    if len(A) == 20 and max(A) <= 10000:
                        candidates.append(A)
    
    # 5. Random perturbations of best structured sets
    # (will be added later after evaluating initial candidates)
    
    return candidates

def _local_search(A, time_budget, rng):
    """Local search: try swapping elements to reduce objective."""
    A = list(A)
    current_score = _objective(A)
    best_score = current_score
    best_A = list(A)
    start_time = time.time()
    
    while time.time() - start_time < time_budget:
        # Pick a random element to replace
        idx = rng.randint(0, 19)
        old_val = A[idx]
        
        # Try a few candidate replacements
        found_improvement = False
        for _ in range(50):
            # Generate a new value near existing elements or in a structured way
            if rng.random() < 0.5:
                # Near an existing element
                base = A[rng.randint(0, 19)]
                new_val = base + rng.randint(-10, 10)
            else:
                # Random in range
                new_val = rng.randint(1, 10000)
            
            if new_val <= 0 or new_val > 10000 or new_val == old_val:
                continue
            
            # Check for duplicates
            if new_val in A and new_val != old_val:
                continue
            
            A[idx] = new_val
            if _is_valid(A):
                new_score = _objective(A)
                if new_score < current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_A = list(A)
                    found_improvement = True
                    break
            A[idx] = old_val
        
        if not found_improvement:
            # Try a small random perturbation of the whole set
            for _ in range(10):
                idx2 = rng.randint(0, 19)
                A[idx2] = max(1, min(10000, A[idx2] + rng.randint(-5, 5)))
            
            if _is_valid(A):
                new_score = _objective(A)
                if new_score < best_score:
                    best_score = new_score
                    best_A = list(A)
    
    return best_A, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    rng = random.Random(12345)
    
    # Phase 1: Evaluate structured candidates
    candidates = _generate_structured_candidates()
    
    best_score = float('inf')
    best_A = None
    
    for A in candidates:
        if time.time() > deadline - 20:
            break
        if not _is_valid(A):
            continue
        score = _objective(A)
        if score < best_score:
            best_score = score
            best_A = list(A)
    
    # Phase 2: Local search from best candidate
    if best_A is not None:
        remaining = deadline - time.time() - 5  # Leave 5s margin
        if remaining > 5:
            best_A, best_score = _local_search(best_A, remaining, rng)
    
    # Phase 3: Try more structured candidates if time permits
    if best_A is None or time.time() < deadline - 10:
        # Generate additional random structured candidates
        for _ in range(100):
            if time.time() > deadline - 5:
                break
            
            # Random mix of AP and GP
            k = rng.randint(5, 15)
            a1 = rng.randint(1, 20)
            d = rng.randint(1, 10)
            a2 = rng.randint(5, 200)
            r = rng.choice([2, 3])
            
            part1 = [a1 + i * d for i in range(k)]
            part2 = []
            for i in range(20 - k):
                val = a2 * (r ** i)
                if val > 10000:
                    break
                part2.append(val)
            
            if len(part2) < 20 - k:
                continue
            
            A = sorted(set(part1 + part2))
            if len(A) != 20 or max(A) > 10000:
                continue
            
            score = _objective(A)
            if score < best_score:
                best_score = score
                best_A = list(A)
    
    if best_A is None:
        # Fallback
        best_A = list(range(1, 21))
    
    return best_A
# EVOLVE-BLOCK-END
