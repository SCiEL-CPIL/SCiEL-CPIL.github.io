import time
import random

def construct():
    t0 = time.time()
    DEADLINE = t0 + 115.0

    def solve(N, time_limit):
        t_start = time.time()
        t_end = t_start + time_limit
        
        # Edge index mapping
        edge_to_idx = {}
        idx_to_edge = []
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                edge_to_idx[(i, j)] = idx
                idx_to_edge.append((i, j))
                idx += 1
        
        edges = list(range(idx))
        
        for restart in range(50): # Max restarts, will break on time
            if time.time() > t_end:
                break
            
            # Random initialization
            colors = [random.randint(0, 3) for _ in range(idx)]
            
            # Build neighbor bitmasks
            # neighbors[c][v] is a bitmask of vertices u such that C(u,v) == c
            neighbors = [[0] * N for _ in range(4)]
            
            for k in range(idx):
                u, v = idx_to_edge[k]
                c = colors[k]
                neighbors[c][u] |= (1 << v)
                neighbors[c][v] |= (1 << u)
                
            # Calculate initial total triangles
            total_T = 0
            for c in range(4):
                for u in range(N):
                    # For each pair (u, v) in color c, count common neighbors
                    # This is O(N^3 / 64) roughly, or O(N * degree^2)
                    # Better: sum over u<v of (neighbors[c][u] & neighbors[c][v]).bit_count()
                    # But we can just sum over all u, v where u<v and C(u,v)=c
                    # Actually, total triangles is 1/3 * sum_{c} sum_{u} (neighbors[c][u] & neighbors[c][v])? No.
                    # A triangle (u,v,w) is counted 3 times in sum_{u} (neighbors[c][u] & neighbors[c][v])?
                    # No, let's just iterate edges.
                    pass
            
            # Efficient initial T calculation:
            # T = sum_{c} sum_{u < v, C(u,v)=c} (neighbors[c][u] & neighbors[c][v]).bit_count()
            # Note: neighbors[c][u] contains v if C(u,v)=c.
            # So for edge (u,v) with color c, the term is (neighbors[c][u] & neighbors[c][v]).bit_count().
            # This counts w such that C(u,w)=c and C(v,w)=c.
            # Since C(u,v)=c, this is exactly the number of monochromatic triangles in color c containing edge (u,v).
            # Each triangle has 3 edges, so it is counted 3 times in the sum over edges?
            # No, the sum is over edges. Each triangle contributes 1 to the count for each of its 3 edges.
            # So total_T = (1/3) * sum_{edges} count.
            # We want total_T == 0, so we can just minimize the sum.
            
            sum_T = 0
            for k in range(idx):
                u, v = idx_to_edge[k]
                c = colors[k]
                sum_T += (neighbors[c][u] & neighbors[c][v]).bit_count()
                
            # sum_T is 3 * T. We minimize sum_T.
            
            if sum_T == 0:
                return colors
            
            while time.time() < t_end:
                if sum_T == 0:
                    return colors
                    
                # Pick random edge
                k = random.choice(edges)
                u, v = idx_to_edge[k]
                c_old = colors[k]
                
                best_c = -1
                best_delta = float('inf')
                
                for c_new in range(4):
                    if c_new == c_old:
                        continue
                    
                    # Delta = (neighbors[c_new][u] & neighbors[c_new][v]).bit_count() 
                    #        - (neighbors[c_old][u] & neighbors[c_old][v]).bit_count()
                    
                    # Note: neighbors[c_old][u] contains v, neighbors[c_old][v] contains u.
                    # The intersection does NOT contain u or v (no self loops).
                    # So the counts are correct.
                    
                    delta = (neighbors[c_new][u] & neighbors[c_new][v]).bit_count() \
                            - (neighbors[c_old][u] & neighbors[c_old][v]).bit_count()
                            
                    if delta < best_delta:
                        best_delta = delta
                        best_c = c_new
                        
                if best_delta < 0:
                    # Update
                    colors[k] = best_c
                    
                    # Update neighbors
                    neighbors[c_old][u] &= ~(1 << v)
                    neighbors[c_old][v] &= ~(1 << u)
                    neighbors[best_c][u] |= (1 << v)
                    neighbors[best_c][v] |= (1 << u)
                    
                    sum_T += best_delta
                else:
                    # Local minimum, restart
                    break
                    
        return None

    # Try increasing n
    # Start with 36, go up to 50
    # Use time budget wisely
    remaining_time = DEADLINE - time.time()
    
    # Estimate time per n. Smaller n is faster.
    # Let's allocate time proportional to difficulty? Or just fixed slots.
    # 36 is easy. 50 is hard.
    
    n_targets = [36, 38, 40, 42, 44, 46, 48, 50]
    
    # We can also try to extend a solution from n to n+1?
    # That's complex. Random restarts are simpler.
    
    # To maximize score, we should prioritize larger n if we have time,
    # but we need to find a solution first.
    # Let's try 50 first? If we fail, we get 0 for that attempt.
    # The evaluator measures the MAX n found.
    # So we should try the largest n we can within the time limit?
    # Or try smaller n first to guarantee a baseline, then try larger?
    # The seed gives 32. If we find 36, score is 1.125.
    # If we find 40, score is 1.25.
    # If we fail to find 50 but find 40, score is 1.25.
    # If we spend all time on 50 and fail, score is 1.0 (from fallback? No, construct returns dict).
    # If construct returns {'n': 32, ...}, score is 1.0.
    
    # Strategy:
    # 1. Try n=50 with a large chunk of time.
    # 2. If fail, try n=48 with remaining time.
    # ...
    # 3. Fallback to 32.
    
    # However, finding 50 is very hard. Finding 36 is easy.
    # Maybe we should try 36 first to "lock in" a high score, then try 40, 44, etc?
    # But construct() only returns ONE result.
    # So we should return the BEST n found.
    
    best_n = 32
    best_colors = None
    
    # Fallback 32 construction (from seed)
    def build_seed_32():
        def mul(a, b):
            r = 0
            for i in range(4):
                if b >> i & 1:
                    r ^= a << i
            for i in range(7, 3, -1):
                if r >> i & 1:
                    r ^= 0b10011 << (i - 4)
            return r

        log, x = {}, 1
        for e in range(15):
            log[x] = e
            x = mul(x, 2)

        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                if i // 16 == j // 16:
                    colors.append(log[(i % 16) ^ (j % 16)] % 3)
                else:
                    colors.append(3)
        return colors

    seed_32 = build_seed_32()
    best_colors = seed_32

    # Try larger n
    # We will iterate n from 50 down to 36?
    # Or 36 up to 50?
    # If we try 50 first and it takes 100s, we have no time for 36.
    # If 50 fails, we return 32. Score 1.0.
    # If we try 36 first, we find it in 1s. Score 1.125.
    # Then we try 40, find it in 5s. Score 1.25.
    # Then we try 44...
    # This seems safer.
    
    for n in [36, 38, 40, 42, 44, 46, 48, 50]:
        if time.time() > DEADLINE - 5:
            break
            
        # Allocate time for this n
        # Smaller n needs less time.
        # Let's give 10s to 36, 10s to 38, ... 
        # Or dynamic: if we have 50s left and 5 targets left, 10s each.
        
        # Simple: 15s per target, but stop if deadline approaching.
        time_limit = 15.0
        if time.time() + time_limit > DEADLINE:
            time_limit = DEADLINE - time.time() - 1.0
            
        if time_limit <= 0:
            break
            
        res = solve(n, time_limit)
        if res:
            best_n = n
            best_colors = res
            # If we found a solution, we can continue to try larger n
            # But we should keep the best one.
            
    return {'n': best_n, 'colors': best_colors}
