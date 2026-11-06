"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
import numpy as np

# EVOLVE-BLOCK-START

def _evaluate(A):
    """Return max(|A+A|, |A*A|) for a set A of distinct positive integers."""
    A = list(A)
    n = len(A)
    # Sum set: all a_i + a_j for i <= j (diagonal allowed)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            sums.add(A[i] + A[j])
    # Product set: all a_i * a_j for i <= j (diagonal allowed)
    prods = set()
    for i in range(n):
        for j in range(i, n):
            prods.add(A[i] * A[j])
    return max(len(sums), len(prods))


def _evaluate_fast(A):
    """Vectorized evaluation using numpy where possible, with set for products."""
    A = np.array(A, dtype=np.int64)
    n = len(A)
    # Sum set via broadcasting (n=20 so 20x20=400 elements, fine)
    sums = np.add.outer(A, A)
    sum_set = set(sums.tolist())
    # Product set
    prods = np.multiply.outer(A, A)
    prod_set = set(prods.tolist())
    return max(len(sum_set), len(prod_set))


def _gen_geometric(ratio_num, ratio_den, start, n=20, max_val=10000):
    """Generate n distinct positive integers following a geometric-like progression."""
    result = []
    val = start
    for i in range(n):
        if val > max_val:
            return None
        if val in result:
            return None
        result.append(val)
        # Next value: val * ratio_num / ratio_den, rounded
        next_val = (val * ratio_num) // ratio_den
        if next_val <= val:
            next_val = val + 1
        val = next_val
    if len(result) != n or len(set(result)) != n:
        return None
    if any(x <= 0 or x > max_val for x in result):
        return None
    return result


def _local_search(A, time_limit, rng):
    """Hill-climbing with random perturbations. Returns best A found."""
    A = list(A)
    best_val = _evaluate(A)
    best_A = list(A)
    deadline = time.time() + time_limit
    n = len(A)
    max_val = 10000
    
    while time.time() < deadline:
        # Random perturbation: pick 1-3 elements and adjust them
        num_changes = rng.randint(1, 3)
        candidates = list(A)
        for _ in range(num_changes):
            idx = rng.randint(0, n - 1)
            # Try several offsets
            found = False
            for _ in range(20):
                delta = rng.randint(-50, 50)
                new_val = candidates[idx] + delta
                if 1 <= new_val <= max_val and new_val not in candidates:
                    candidates[idx] = new_val
                    found = True
                    break
            if not found:
                # Try random value
                for _ in range(10):
                    new_val = rng.randint(1, max_val)
                    if new_val not in candidates:
                        candidates[idx] = new_val
                        found = True
                        break
                if not found:
                    candidates = list(A)
                    break
        
        if len(set(candidates)) != n:
            continue
        
        val = _evaluate(candidates)
        if val < best_val:
            best_val = val
            best_A = list(candidates)
            A = list(candidates)
    
    return best_A, best_val


def _gen_structured_candidates(rng, n=20, max_val=10000):
    """Generate diverse structured candidates."""
    candidates = []
    
    # 1. Geometric progressions with rational ratios
    for _ in range(200):
        start = rng.randint(1, 50)
        rn = rng.randint(2, 12)
        rd = rng.randint(1, 11)
        if rn == rd:
            continue
        c = _gen_geometric(rn, rd, start, n, max_val)
        if c:
            candidates.append(c)
    
    # 2. Power-of-2 based with perturbations
    for _ in range(100):
        base = rng.choice([2, 3, 4, 5])
        c = []
        for i in range(n):
            v = base ** i
            if v > max_val:
                v = max_val - rng.randint(0, 50)
            if v not in c and 1 <= v <= max_val:
                c.append(v)
        if len(c) == n:
            candidates.append(c)
    
    # 3. Arithmetic progressions
    for _ in range(100):
        start = rng.randint(1, 20)
        step = rng.randint(1, 10)
        c = [start + i * step for i in range(n)]
        if all(1 <= x <= max_val for x in c) and len(set(c)) == n:
            candidates.append(c)
    
    # 4. Random sets with small values (dense sum/product structure)
    for _ in range(200):
        c = sorted(rng.sample(range(1, 200), n))
        candidates.append(c)
    
    # 5. Random sets with geometric bias
    for _ in range(200):
        vals = []
        current = rng.randint(1, 5)
        for i in range(n):
            vals.append(current)
            current = current * rng.randint(1, 3) + rng.randint(0, 2)
            if current > max_val:
                current = max_val - rng.randint(0, 10)
        c = sorted(set(vals))
        while len(c) < n:
            c.append(rng.randint(1, max_val))
        c = c[:n]
        if len(set(c)) == n:
            candidates.append(c)
    
    # 6. Multiplicative structure: products of small primes
    primes = [2, 3, 5, 7, 11, 13]
    for _ in range(100):
        c = set()
        for _ in range(n * 3):
            v = 1
            for p in primes:
                v *= p ** rng.randint(0, 3)
            if 1 <= v <= max_val:
                c.add(v)
        if len(c) >= n:
            c = sorted(list(c)[:n])
            candidates.append(c)
    
    return candidates


def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin for 90s limit
    rng = random.Random(42)
    n = 20
    max_val = 10000
    
    # Phase 1: Generate and evaluate structured candidates
    candidates = _gen_structured_candidates(rng, n, max_val)
    
    best_A = None
    best_val = float('inf')
    
    # Evaluate all candidates
    for c in candidates:
        if time.time() > deadline:
            break
        val = _evaluate(c)
        if val < best_val:
            best_val = val
            best_A = list(c)
    
    # Also try the simple sequential set as baseline
    simple = list(range(1, n + 1))
    val = _evaluate(simple)
    if val < best_val:
        best_val = val
        best_A = simple
    
    if best_A is None:
        return list(range(1, n + 1))
    
    # Phase 2: Local search from the best candidate
    remaining = deadline - time.time()
    if remaining > 5:
        best_A, best_val = _local_search(best_A, remaining - 5, rng)
    
    # Phase 3: Try multiple local searches from different starting points
    remaining = deadline - time.time()
    if remaining > 10:
        # Pick a few diverse starting points
        starts = [best_A]
        # Add a few random ones
        for _ in range(min(3, len(candidates))):
            idx = rng.randint(0, len(candidates) - 1)
            starts.append(candidates[idx])
        
        for s in starts:
            if time.time() > deadline:
                break
            rem = deadline - time.time()
            if rem < 3:
                break
            A_local, val_local = _local_search(s, rem - 2, rng)
            if val_local < best_val:
                best_val = val_local
                best_A = A_local
    
    # Final check
    if time.time() > deadline - 2:
        # Emergency: return best found
        pass
    
    return best_A

# EVOLVE-BLOCK-END
