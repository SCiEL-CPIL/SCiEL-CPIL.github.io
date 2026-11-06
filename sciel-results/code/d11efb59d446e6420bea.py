import time
import math

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit."""
    powerful = set()
    powerful.add(1)
    
    max_y = int(limit ** (1/3)) + 1
    
    is_sq_free = [True] * (max_y + 1)
    is_sq_free[0] = False
    for i in range(2, int(math.isqrt(max_y)) + 1):
        if is_sq_free[i]:
            for j in range(i*i, max_y + 1, i*i):
                is_sq_free[j] = False
    
    sq_free = [y for y in range(1, max_y + 1) if is_sq_free[y]]
    
    for y in sq_free:
        y_cubed = y ** 3
        if y_cubed > limit:
            continue
        max_x = int(math.isqrt(limit // y_cubed))
        for x in range(1, max_x + 1):
            val = x * x * y_cubed
            if val <= limit:
                powerful.add(val)
    
    return powerful

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_result = None
    best_count = 0
    
    # Try several limits, larger first
    for limit in [2000000, 1000000, 500000, 200000, 100000]:
        try:
            if time.time() > deadline - 5:
                break
            
            powerful = _generate_powerful(limit)
            P = sorted(powerful)
            P_set = powerful
            n = len(P)
            
            if n < 2:
                continue
            
            # Pairwise sum counting with early break
            sum_map = {}
            
            for i in range(n):
                a = P[i]
                for j in range(i, n):
                    b = P[j]
                    s = a + b
                    if s > 2 * limit:
                        break
                    if s not in sum_map:
                        sum_map[s] = []
                    sum_map[s].append((a, b))
            
            # Find N with most representations
            max_pairs = []
            max_N = 0
            for N, pairs in sum_map.items():
                if len(pairs) > len(max_pairs):
                    max_N = N
                    max_pairs = pairs
            
            if max_N >= 2 and len(max_pairs) > best_count:
                verified = [[a, b] for a, b in max_pairs if a <= b and a + b == max_N and a in P_set and b in P_set]
                if len(verified) > best_count:
                    best_count = len(verified)
                    best_result = {'N': max_N, 'pairs': verified}
                    
        except Exception:
            continue
    
    if best_result is not None:
        return best_result
    
    # Fallback: ensure at least one valid pair
    return {'N': 2, 'pairs': [[1, 1]]}
