import time
import random
import numpy as np

MAX_N = 20
MAX_VAL = 10000
TIME_LIMIT = 85  # safety margin before 90s kill

def _objective(A):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers."""
    A_arr = np.array(sorted(A), dtype=np.int64)
    n = len(A_arr)
    i_idx, j_idx = np.triu_indices(n)
    sums = A_arr[i_idx] + A_arr[j_idx]
    products = A_arr[i_idx] * A_arr[j_idx]
    return max(len(np.unique(sums)), len(np.unique(products)))

def _objective_parts(A):
    """Compute |A+A| and |A*A| separately."""
    A_arr = np.array(sorted(A), dtype=np.int64)
    n = len(A_arr)
    i_idx, j_idx = np.triu_indices(n)
    sums = A_arr[i_idx] + A_arr[j_idx]
    products = A_arr[i_idx] * A_arr[j_idx]
    return len(np.unique(sums)), len(np.unique(products))

def _is_valid(A):
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    for x in A:
        if not isinstance(x, int) or x < 1 or x > 10000:
            return False
    return True

def _smooth_numbers(limit, count):
    """Generate the 'count' smallest smooth numbers <= limit."""
    smooths = set()
    i_max = int(limit.bit_length())
    j_max = int(limit.bit_length())
    k_max = int(limit.bit_length())
    
    for i in range(i_max + 1):
        pow2 = 1 << i
        if pow2 > limit: break
        for j in range(j_max + 1):
            pow3 = pow2 * (3 ** j)
            if pow3 > limit: break
            for k in range(k_max + 1):
                val = pow3 * (5 ** k)
                if val > limit: break
                smooths.add(val)
    
    sorted_smooths = sorted(smooths)
    return sorted_smooths[:count]

def _construct_candidates():
    """Generate a list of candidate starting sets."""
    candidates = []
    
    # 1. Smooth numbers (powers of 2, 3, 5)
    try:
        smooths = _smooth_numbers(10000, 20)
        if len(smooths) == 20:
            candidates.append(smooths)
    except:
        pass
        
    # 2. Geometric Progressions with more ratios
    for ratio in [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.2, 2.5]:
        for start in [1, 2, 3, 4]:
            A = []
            current = float(start)
            for _ in range(20):
                val = int(round(current))
                if val in A:
                    val += 1
                    while val in A:
                        val += 1
                if val > 10000:
                    break
                A.append(val)
                current *= ratio
            if len(A) == 20 and _is_valid(A):
                candidates.append(A)
                
    # 3. Arithmetic Progressions
    for start in [1, 2, 5]:
        for step in [1, 2, 3]:
            A = [start + i * step for i in range(20)]
            if max(A) <= 10000 and _is_valid(A):
                candidates.append(A)
                
    # 4. Random
    candidates.append(random.sample(range(1, 10001), 20))
    candidates.append(random.sample(range(1, 1001), 20))
    candidates.append(random.sample(range(1, 500), 20))
    
    return candidates

def _structure_aware_moves(A, rng):
    """Generate structure-aware moves targeting product collisions and sum compactness."""
    A_sorted = sorted(A)
    n = len(A_sorted)
    moves = []
    
    # Target product collisions: try to replace an element to create more product overlaps
    # For each pair (i,j), the product A[i]*A[j] should collide with other products
    # Try replacing elements to create multiplicative relationships
    
    # 1. Try to create geometric-like structures
    for idx in range(n):
        # Try values that are multiples or divisors of existing elements
        for other in A_sorted:
            if other == A_sorted[idx]:
                continue
            # Multiples
            for mult in [2, 3, 4]:
                new_val = other * mult
                if 1 <= new_val <= 10000 and new_val not in A_sorted:
                    moves.append((idx, new_val))
            # Divisors
            for div in [2, 3, 4, 5]:
                if other % div == 0:
                    new_val = other // div
                    if 1 <= new_val <= 10000 and new_val not in A_sorted:
                        moves.append((idx, new_val))
    
    # 2. Try to create additive structures (arithmetic progressions)
    for idx in range(n):
        # Try values that fit in arithmetic progressions with existing elements
        for i in range(n):
            for j in range(i+1, n):
                if i == idx or j == idx:
                    continue
                # Try to extend or fit into AP
                a, b = A_sorted[i], A_sorted[j]
                d = b - a
                if d > 0:
                    # Try a-d, b+d
                    for new_val in [a - d, b + d]:
                        if 1 <= new_val <= 10000 and new_val not in A_sorted:
                            moves.append((idx, new_val))
    
    # 3. Small perturbations for local refinement
    for idx in range(n):
        old_val = A_sorted[idx]
        for delta in range(-10, 11):
            if delta == 0:
                continue
            new_val = old_val + delta
            if 1 <= new_val <= 10000 and new_val not in A_sorted:
                moves.append((idx, new_val))
    
    # 4. Random moves
    for _ in range(20):
        idx = rng.randint(0, n-1)
        new_val = rng.randint(1, 10000)
        if new_val not in A_sorted:
            moves.append((idx, new_val))
    
    return moves

def _local_search(A, time_limit, rng):
    """Local search with simulated annealing and structure-aware moves."""
    A = sorted(A)
    current_obj = _objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    temperature = 30.0
    cooling_rate = 0.997
    
    while temperature > 0.05:
        if time.time() - start_time > time_limit:
            break
        
        # Use structure-aware moves
        moves = _structure_aware_moves(A, rng)
        if not moves:
            temperature *= cooling_rate
            continue
            
        rng.shuffle(moves)
        accepted = False
        
        for idx, new_val in moves[:30]:  # Limit moves per iteration
            if time.time() - start_time > time_limit:
                break
            
            old_val = A[idx]
            A[idx] = new_val
            new_obj = _objective(A)
            A[idx] = old_val
            
            delta = new_obj - current_obj
            if delta <= 0 or rng.random() < min(1.0, max(0.0, -delta / (temperature + 1e-10))):
                A[idx] = new_val
                current_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_A = list(A)
                accepted = True
                break
                
        temperature *= cooling_rate
        
    return best_A, best_obj

def _polish(A, time_limit, rng):
    """Final high-precision polish pass with tighter tolerance."""
    A = sorted(A)
    current_obj = _objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    
    # Iterative improvement: keep trying small changes until no improvement
    while time.time() - start_time < time_limit:
        improved = False
        for idx in range(20):
            if time.time() - start_time > time_limit:
                break
            old_val = A[idx]
            for new_val in range(max(1, old_val - 5), min(10000, old_val + 5) + 1):
                if new_val == old_val or new_val in A:
                    continue
                A[idx] = new_val
                new_obj = _objective(A)
                A[idx] = old_val
                if new_obj < current_obj:
                    A[idx] = new_val
                    current_obj = new_obj
                    if new_obj < best_obj:
                        best_obj = new_obj
                        best_A = list(A)
                    improved = True
                    break
        if not improved:
            break
            
    return best_A, best_obj

def construct():
    start_time = time.time()
    time_budget = TIME_LIMIT - 5
    
    candidates = _construct_candidates()
    
    best_A = None
    best_obj = float('inf')
    
    rng = random.Random(42)
    
    # First pass: local search on each candidate
    for A in candidates:
        if time.time() - start_time > time_budget:
            break
        if not _is_valid(A):
            continue
        elapsed = time.time() - start_time
        remaining_for_this = max(1.0, time_budget - elapsed)
        refined_A, obj = _local_search(A, remaining_for_this, rng)
        if obj < best_obj:
            best_obj = obj
            best_A = refined_A
            
    if best_A is None:
        best_A = list(range(1, 21))
        best_obj = _objective(best_A)
        
    # Second pass: more intensive refinement on best solution
    remaining = TIME_LIMIT - 5 - (time.time() - start_time)
    if remaining > 2:
        try:
            refined_A, refined_obj = _local_search(best_A, remaining * 0.6, rng)
            if refined_obj < best_obj:
                best_A = refined_A
                best_obj = refined_obj
        except Exception:
            pass
    
    # Final polish: very tight local search
    remaining = TIME_LIMIT - 5 - (time.time() - start_time)
    if remaining > 1:
        try:
            # Use a different RNG seed for diversity
            rng2 = random.Random(123)
            refined_A, refined_obj = _polish(best_A, remaining, rng2)
            if refined_obj < best_obj:
                best_A = refined_A
                best_obj = refined_obj
        except Exception:
            pass

    if not _is_valid(best_A):
        best_A = list(range(1, 21))
        
    return sorted(best_A)
