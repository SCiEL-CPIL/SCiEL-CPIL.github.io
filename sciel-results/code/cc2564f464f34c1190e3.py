import time
import math
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115

    def generate_square_free(limit):
        """Sieve for square-free numbers up to limit."""
        if limit < 1:
            return []
        is_sq_free = [True] * (limit + 1)
        is_sq_free[0] = False
        p = 2
        while p * p <= limit:
            if is_sq_free[p]:
                sq = p * p
                for j in range(sq, limit + 1, sq):
                    is_sq_free[j] = False
            p += 1
        return [i for i in range(1, limit + 1) if is_sq_free[i]]

    def generate_powerful(limit):
        """Generate all powerful numbers up to limit."""
        powerful = set()
        max_v = int(limit ** (1/3)) + 1
        sq_free_v = generate_square_free(max_v)
        
        for v in sq_free_v:
            v3 = v ** 3
            if v3 > limit:
                break
            max_u = int(math.isqrt(limit // v3))
            for u in range(1, max_u + 1):
                x = u * u * v3
                if x <= limit:
                    powerful.add(x)
        return sorted(powerful)

    best_result = {'N': 2, 'pairs': [[1, 1]]}
    best_score = 1

    # Generate powerful numbers up to a moderate limit
    M = 200_000
    P = generate_powerful(M)
    L = len(P)
    
    if L < 10:
        return best_result

    # Use a dictionary to count pair sums, but only for a subset to manage time
    # Process in chunks to avoid O(L^2) memory
    sum_counts = defaultdict(int)
    
    # For each powerful number a, add counts for N = a + b for all b >= a
    # But this is still O(L^2). Instead, let's use a smarter approach:
    # Sample candidate N values and count their representations.
    
    # First, let's try the numpy approach with a smaller set
    P_np = np.array(P, dtype=np.int64)
    
    # Check if we can do the full outer sum
    mem_est = L * L * 8
    if mem_est < 2 * 1024**3:  # 2GB limit for safety
        try:
            sums = np.add.outer(P_np, P_np)
            max_sum_val = int(sums.max())
            counts = np.bincount(sums.ravel(), minlength=max_sum_val + 1)
            max_count_idx = np.argmax(counts)
            best_N = int(max_count_idx)
            
            P_set = set(P)
            pairs = []
            for a in P:
                b = best_N - a
                if b < a:
                    continue
                if b in P_set:
                    pairs.append([a, b])
            
            current_score = len(pairs)
            if current_score > best_score:
                best_score = current_score
                best_result = {'N': best_N, 'pairs': pairs}
        except MemoryError:
            pass
        except Exception:
            pass
    
    # If numpy approach didn't work well or time is running out, try targeted search
    if time.time() < deadline - 30:
        # Generate more powerful numbers for the targeted search
        M2 = 1_000_000
        P2 = generate_powerful(M2)
        L2 = len(P2)
        P2_set = set(P2)
        
        # Sample candidate N values: try N = 2*a for random a, and N = a+b for random pairs
        import random
        random.seed(42)
        
        candidates = set()
        # Add some structured candidates
        for i in range(0, min(L2, 1000), 10):
            candidates.add(P2[i] * 2)
            if i + 1 < L2:
                candidates.add(P2[i] + P2[i+1])
        
        # Add random pairs
        for _ in range(500):
            i = random.randint(0, L2 - 1)
            j = random.randint(0, L2 - 1)
            candidates.add(P2[i] + P2[j])
        
        # Also try N values around 2*10^6 to 10^7
        for n in range(2_000_000, 10_000_000, 100_000):
            candidates.add(n)
        
        # For each candidate N, count representations
        for N in candidates:
            if time.time() > deadline - 15:
                break
            
            count = 0
            pairs = []
            for a in P2:
                if a > N:
                    break
                b = N - a
                if b < a:
                    continue
                if b in P2_set:
                    count += 1
                    pairs.append([a, b])
            
            if count > best_score:
                best_score = count
                best_result = {'N': N, 'pairs': pairs}
    
    return best_result
