import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer for safety
    
    def is_square_free(n):
        if n < 2:
            return True
        for p in range(2, int(math.isqrt(n)) + 1):
            if (n % (p * p)) == 0:
                return False
        return True

    def generate_powerful(limit):
        powerful = set()
        # x = u^2 * v^3
        # v is square-free
        max_v = int(limit ** (1/3)) + 1
        # Precompute square-free status up to max_v
        # Actually, just check on the fly for small max_v
        for v in range(1, max_v + 1):
            if not is_square_free(v):
                continue
            v3 = v ** 3
            if v3 > limit:
                break
            max_u = int(math.isqrt(limit // v3))
            for u in range(1, max_u + 1):
                x = u * u * v3
                if x <= limit:
                    powerful.add(x)
        return sorted(powerful)

    # Start with a moderate limit to ensure we finish
    # We can try to increase it if time allows
    # But generate_powerful is fast. The bottleneck is the pair summation.
    
    # Let's estimate the limit based on time.
    # If we have 100s, and pair summation takes ~0.01s per 10000 pairs?
    # 10^8 pairs -> 100s.
    # So we want < 10^8 pairs.
    # Pairs ~ |P|^2 / 2.
    # |P| ~ 2.6 * sqrt(M).
    # (2.6 * sqrt(M))^2 / 2 < 10^8
    # 3.38 * M < 10^8
    # M < 3 * 10^7.
    
    # Let's try M = 5 * 10^7 first.
    M = 50000000
    
    # Generate
    P = generate_powerful(M)
    
    if time.time() > deadline:
        # Fallback to seed behavior if generation is slow (unlikely)
        return {'N': 2, 'pairs': [[1, 1]]}
        
    L = len(P)
    
    # Pair summation
    counts = {}
    best_N = 0
    best_count = 0
    
    # Optimization: use local variables for speed
    P_list = P
    L_val = L
    
    for i in range(L_val):
        ai = P_list[i]
        # j must be >= i
        for j in range(i, L_val):
            s = ai + P_list[j]
            # We don't strictly need to cap s, but it helps if we want to restrict N
            # However, larger N tends to have more representations.
            # So we allow all sums.
            c = counts.get(s, 0) + 1
            counts[s] = c
            if c > best_count:
                best_count = c
                best_N = s
                
        if time.time() > deadline:
            break

    if best_N == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # Collect pairs for best_N
    pairs = []
    for i in range(L_val):
        a = P_list[i]
        b = best_N - a
        if b < a:
            continue
        # Check if b is in P
        # Since P is sorted, we can use binary search or a set
        # Using set is O(1)
        # Let's create a set of P for lookup
        if not hasattr(construct, 'P_set'):
            construct.P_set = set(P_list)
        if b in construct.P_set:
            pairs.append([a, b])
            
    return {'N': best_N, 'pairs': pairs}
