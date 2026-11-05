import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 256
    
    def is_valid_fast(S):
        """Fast validation using NumPy vectorization."""
        if len(S) < 3:
            return True
        arr = np.array(sorted(S), dtype=np.int64)
        m = len(arr)
        # For each a (index i), check all pairs (b, c) with i < j <= k
        # such that a does NOT divide (b + c)
        for i in range(m):
            a = arr[i]
            if a == 0:
                continue
            # b and c are from arr[i+1:]
            rest = arr[i+1:]
            if len(rest) < 2:
                continue
            # Create all pairs (b, c) where b <= c
            # Use broadcasting: b = rest[:, None], c = rest[None, :]
            # We need b <= c, so we take upper triangle including diagonal
            b_mat = rest[:, None]
            c_mat = rest[None, :]
            # Only consider b <= c
            mask = b_mat <= c_mat
            if not mask.any():
                continue
            b_vals = b_mat[mask]
            c_vals = c_mat[mask]
            sums = b_vals + c_vals
            # Check if any sum is divisible by a
            if np.any(sums % a == 0):
                return False
        return True

    def count_violations_fast(S):
        """Count violating triples using vectorization."""
        if len(S) < 3:
            return 0
        arr = np.array(sorted(S), dtype=np.int64)
        m = len(arr)
        count = 0
        for i in range(m):
            a = arr[i]
            if a == 0:
                continue
            rest = arr[i+1:]
            if len(rest) < 2:
                continue
            b_mat = rest[:, None]
            c_mat = rest[None, :]
            mask = b_mat <= c_mat
            if not mask.any():
                continue
            b_vals = b_mat[mask]
            c_vals = c_mat[mask]
            sums = b_vals + c_vals
            count += int(np.sum(sums % a == 0))
        return count

    def get_conflicts(x, S):
        """Return list of pairs (b,c) in S such that x divides b+c and x < b <= c."""
        conflicts = []
        S_list = sorted(S)
        for b in S_list:
            if b <= x:
                continue
            rem = (-b) % x
            for c in S_list:
                if c < b:
                    continue
                if (b + c) % x == 0:
                    conflicts.append((b, c))
        return conflicts

    # Strategy 1: Upper tail
    S_upper = set(range(171, 257))
    
    # Strategy 2: Greedy from high to low
    def greedy_high_to_low():
        S = set()
        for x in range(n, 0, -1):
            candidate = S | {x}
            if is_valid_fast(candidate):
                S = candidate
        return S

    # Strategy 3: Greedy from low to high
    def greedy_low_to_high():
        S = set()
        for x in range(1, n + 1):
            candidate = S | {x}
            if is_valid_fast(candidate):
                S = candidate
        return S

    # Strategy 4: Random restarts with greedy
    def random_greedy(seed):
        rng = random.Random(seed)
        order = list(range(1, n + 1))
        rng.shuffle(order)
        S = set()
        for x in order:
            candidate = S | {x}
            if is_valid_fast(candidate):
                S = candidate
        return S

    # Local improvement: try to add elements by removing conflicting ones
    def local_improve(S, max_iterations=100):
        S = set(S)
        improved = True
        iterations = 0
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            # Try to add each missing element
            for x in range(1, n + 1):
                if x in S:
                    continue
                conflicts = get_conflicts(x, S)
                if not conflicts:
                    S.add(x)
                    improved = True
                    break
                # Try to resolve by removing one element from each conflict pair
                # Simple heuristic: try removing each involved element
                involved = set()
                for b, c in conflicts:
                    involved.add(b)
                    involved.add(c)
                for elem in involved:
                    new_S = S - {elem} | {x}
                    if is_valid_fast(new_S):
                        S = new_S
                        improved = True
                        break
                if improved:
                    break
            # Also try swapping two elements for better packing
            S_list = sorted(S)
            for i in range(len(S_list)):
                for j in range(i + 1, len(S_list)):
                    a, b = S_list[i], S_list[j]
                    # Try swapping a with a smaller number not in S
                    for x in range(1, a):
                        if x in S:
                            continue
                        new_S = S - {a} | {x}
                        if is_valid_fast(new_S) and len(new_S) == len(S):
                            # Check if this opens up more additions
                            S = new_S
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return S

    # Generate candidate sets
    candidates = []
    
    # Upper tail
    candidates.append(S_upper)
    
    # Greedy high to low
    candidates.append(greedy_high_to_low())
    
    # Greedy low to high
    candidates.append(greedy_low_to_high())
    
    # Random restarts
    for seed in range(20):
        if time.time() > deadline - 20:
            break
        candidates.append(random_greedy(seed))
    
    # Local improve each candidate
    best_S = set()
    best_size = 0
    
    for S in candidates:
        if time.time() > deadline - 15:
            break
        S_improved = local_improve(S, max_iterations=50)
        if len(S_improved) > best_size:
            best_S = S_improved
            best_size = len(S_improved)
    
    # Final polish: try to add any remaining elements
    for x in range(1, n + 1):
        if time.time() > deadline - 5:
            break
        if x in best_S:
            continue
        new_S = best_S | {x}
        if is_valid_fast(new_S):
            best_S = new_S
    
    return list(best_S)
