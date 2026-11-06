import time
import random
import itertools
import math

def _compute_objective(A):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers."""
    if len(A) != 20:
        return float('inf')
    if any(x < 1 or x > 10000 for x in A):
        return float('inf')
    if len(set(A)) != 20:
        return float('inf')
    
    # Sumset A+A (with equal summands allowed)
    sumset = set()
    for i in range(20):
        for j in range(20):
            sumset.add(A[i] + A[j])
    
    # Product set A*A (with equal factors allowed)
    prodset = set()
    for i in range(20):
        for j in range(20):
            prodset.add(A[i] * A[j])
    
    return max(len(sumset), len(prodset))


def _local_search(A, time_limit=50.0, rng=None):
    """Local search to minimize max(|A+A|, |A*A|)."""
    if rng is None:
        rng = random.Random(42)
    
    start_time = time.time()
    A = sorted(A)
    best_obj = _compute_objective(A)
    best_A = list(A)
    
    if best_obj == float('inf'):
        return best_A
    
    # Precompute some helper data
    n = 20
    
    # Try to improve by swapping elements or moving elements
    # We'll do a simulated annealing / hill climbing hybrid
    
    current_obj = best_obj
    current_A = list(A)
    
    # Temperature for simulated annealing
    T = 10.0
    T_min = 0.1
    cooling = 0.995
    
    while time.time() - start_time < time_limit:
        # Generate a neighbor: swap two elements, or replace one element
        op = rng.random()
        
        if op < 0.3:
            # Swap two elements
            i, j = rng.sample(range(n), 2)
            new_A = current_A.copy()
            new_A[i], new_A[j] = new_A[j], new_A[i]
            if len(set(new_A)) != n:
                continue
            if any(x < 1 or x > 10000 for x in new_A):
                continue
        elif op < 0.6:
            # Replace one element with a random one
            i = rng.randrange(n)
            old_val = current_A[i]
            new_val = rng.randint(1, 10000)
            if new_val == old_val:
                continue
            # Check if new_val is already in the set
            if new_val in current_A:
                continue
            new_A = current_A.copy()
            new_A[i] = new_val
        elif op < 0.8:
            # Move one element by a small delta
            i = rng.randrange(n)
            delta = rng.randint(-50, 50)
            if delta == 0:
                continue
            new_val = current_A[i] + delta
            if new_val < 1 or new_val > 10000:
                continue
            if new_val in current_A:
                continue
            new_A = current_A.copy()
            new_A[i] = new_val
        else:
            # Replace two elements
            i, j = rng.sample(range(n), 2)
            new_val_i = rng.randint(1, 10000)
            new_val_j = rng.randint(1, 10000)
            if new_val_i == current_A[i] and new_val_j == current_A[j]:
                continue
            if new_val_i in current_A or new_val_j in current_A:
                continue
            if new_val_i == new_val_j:
                continue
            new_A = current_A.copy()
            new_A[i] = new_val_i
            new_A[j] = new_val_j
            if len(set(new_A)) != n:
                continue
        
        new_obj = _compute_objective(new_A)
        
        if new_obj < current_obj:
            current_obj = new_obj
            current_A = new_A
            if new_obj < best_obj:
                best_obj = new_obj
                best_A = list(new_A)
        else:
            # Accept with probability exp(-delta/T)
            delta = new_obj - current_obj
            if delta < 0:
                prob = 1.0
            else:
                prob = math.exp(-delta / max(T, 1e-10))
            if rng.random() < prob:
                current_obj = new_obj
                current_A = new_A
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_A


def _try_structural_constructions(rng):
    """Try various structural constructions."""
    candidates = []
    
    # 1. Geometric progression-like
    base = rng.randint(2, 5)
    for shift in range(3):
        A = [base ** i + shift for i in range(20)]
        if all(1 <= x <= 10000 for x in A) and len(set(A)) == 20:
            candidates.append(A)
    
    # 2. Exponential with small base
    for base in [2, 3]:
        for offset in range(5):
            A = [base ** i + offset for i in range(20)]
            if all(1 <= x <= 10000 for x in A) and len(set(A)) == 20:
                candidates.append(A)
    
    # 3. Quadratic sequence
    for a in range(1, 10):
        for b in range(0, 20):
            A = [a * i * i + b * i + 1 for i in range(20)]
            if all(1 <= x <= 10000 for x in A) and len(set(A)) == 20:
                candidates.append(A)
    
    # 4. Small integers with structure
    # Use first 20 integers but with some gaps
    for gap in range(1, 10):
        A = [i * gap + 1 for i in range(20)]
        if all(1 <= x <= 10000 for x in A) and len(set(A)) == 20:
            candidates.append(A)
    
    # 5. Mixed: some small, some spread
    for _ in range(5):
        A = sorted(rng.sample(range(1, 10001), 20))
        candidates.append(A)
    
    # 6. Powers of 2 with offsets
    for offset in range(3):
        A = [2 ** i + offset for i in range(20)]
        if all(1 <= x <= 10000 for x in A) and len(set(A)) == 20:
            candidates.append(A)
    
    # 7. Fibonacci-like
    for _ in range(3):
        a, b = rng.randint(1, 10), rng.randint(1, 10)
        A = [a, b]
        while len(A) < 20:
            A.append(A[-1] + A[-2])
        if all(1 <= x <= 10000 for x in A) and len(set(A)) == 20:
            candidates.append(A)
    
    return candidates


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5 second safety margin
    
    rng = random.Random(12345)
    
    # Generate initial candidates
    candidates = []
    
    # Basic seed
    candidates.append(list(range(1, 21)))
    
    # Structural constructions
    candidates.extend(_try_structural_constructions(rng))
    
    # Random initial sets
    for _ in range(10):
        candidates.append(sorted(rng.sample(range(1, 10001), 20)))
    
    # Evaluate all candidates
    best_obj = float('inf')
    best_A = None
    
    for A in candidates:
        if time.time() > deadline:
            break
        obj = _compute_objective(A)
        if obj < best_obj:
            best_obj = obj
            best_A = list(A)
    
    if best_A is None:
        return list(range(1, 21))
    
    # Local search from the best candidate
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        best_A = _local_search(best_A, time_limit=remaining_time - 5.0, rng=rng)
    
    # Final check
    final_obj = _compute_objective(best_A)
    if final_obj == float('inf'):
        return list(range(1, 21))
    
    return sorted(best_A)
