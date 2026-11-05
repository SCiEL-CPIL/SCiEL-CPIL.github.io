import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    def get_index(i, j, n):
        # i < j
        return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)

    def set_color(colors, n, i, j, c):
        idx = get_index(i, j, n)
        colors[idx] = c

    def get_color(colors, n, i, j):
        if i > j: i, j = j, i
        return colors[get_index(i, j, n)]

    def count_violations(colors, n):
        # Build adjacency bitsets for color 0 and 1
        adj = [[0] * n for _ in range(2)]
        
        # Fill adj
        for i in range(n):
            for j in range(i + 1, n):
                c = get_color(colors, n, i, j)
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
        
        violations = 0
        
        for c in range(2):
            # Iterate over all pairs (u, v) as the first pair P1
            for u in range(n):
                for v in range(u + 1, n):
                    # S is the set of vertices w such that (u,w) and (v,w) are color c
                    S = adj[c][u] & adj[c][v]
                    # Exclude u and v themselves (though they shouldn't be in their own adj)
                    S &= ~(1 << u)
                    S &= ~(1 << v)
                    
                    # If S has less than 2 vertices, we can't form a P2
                    if S.bit_count() < 2:
                        continue
                        
                    # Extract bits of S to iterate over pairs (x, y)
                    bits = []
                    temp = S
                    while temp:
                        lsb = temp & -temp
                        idx = lsb.bit_length() - 1
                        bits.append(idx)
                        temp ^= lsb
                    
                    # Iterate over pairs (x, y) from bits
                    len_bits = len(bits)
                    for i_idx in range(len_bits):
                        x = bits[i_idx]
                        for j_idx in range(i_idx + 1, len_bits):
                            y = bits[j_idx]
                            
                            # P1 = {u, v}, P2 = {x, y}
                            # We know u-x, u-y, v-x, v-y are color c.
                            # Now we need P3 = {z, w} such that:
                            # 1. z, w are in S (connected to u, v)
                            # 2. z, w are connected to x, y with color c
                            
                            S2 = S & adj[c][x] & adj[c][y]
                            # Exclude u, v, x, y from S2
                            S2 &= ~(1 << u)
                            S2 &= ~(1 << v)
                            S2 &= ~(1 << x)
                            S2 &= ~(1 << y)
                            
                            count_zw = S2.bit_count()
                            if count_zw >= 2:
                                # Number of ways to choose 2 vertices from S2
                                violations += count_zw * (count_zw - 1) // 2
        
        return violations

    def local_search(n, max_time=50.0):
        # Initialize random coloring
        colors = [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]
        
        current_cost = count_violations(colors, n)
        best_cost = current_cost
        best_colors = colors[:]
        
        # Simulated Annealing
        temperature = 1.0
        cooling_rate = 0.999
        min_temp = 0.01
        
        t_start = time.time()
        
        while time.time() - t_start < max_time:
            if current_cost == 0:
                return colors, 0
            
            # Pick a random edge to flip
            edge_idx = random.randint(0, n * (n - 1) // 2 - 1)
            
            # Find i, j from edge_idx
            # Inverse of get_index
            # i * (n-1) - i(i+1)/2 + (j-i-1) = edge_idx
            # Approx i ~ sqrt(2*edge_idx)
            i = 0
            while True:
                next_i = i + 1
                # Number of edges before i is i*(n-1) - i(i+1)//2
                # Number of edges before i+1 is (i+1)*(n-1) - (i+1)(i+2)//2
                # If edge_idx < edges_before(i+1), then i is correct
                edges_before_next = (i + 1) * (n - 1) - (i + 1) * (i + 2) // 2
                if edge_idx < edges_before_next:
                    break
                i += 1
            
            # j - i - 1 = edge_idx - (i * (n - 1) - i * (i + 1) // 2)
            offset = i * (n - 1) - i * (i + 1) // 2
            j = i + 1 + (edge_idx - offset)
            
            # Flip
            old_color = colors[edge_idx]
            new_color = 1 - old_color
            colors[edge_idx] = new_color
            
            new_cost = count_violations(colors, n)
            
            # Acceptance criterion
            if new_cost < current_cost:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_colors = colors[:]
            else:
                # Metropolis acceptance
                delta = new_cost - current_cost
                if temperature > 0:
                    prob = math.exp(-delta / temperature)
                    if random.random() < prob:
                        current_cost = new_cost
                    else:
                        # Revert
                        colors[edge_idx] = old_color
            
            temperature *= cooling_rate
            if temperature < min_temp:
                temperature = min_temp
                
        return best_colors, best_cost

    # Try increasing n
    best_n = 1
    best_colors = []
    
    # Start from a smaller n where it's easier to find a solution, or just go up
    # Known: R(K2,2,2) is likely 30 or 31.
    # Let's try n=25 first to ensure we have a valid base, then expand.
    # Or just try n=29 directly with SA.
    
    target_ns = [25, 27, 29, 30]
    
    for n in target_ns:
        if time.time() > deadline - 10:
            break
            
        # Run local search for this n
        # Give more time to larger n if previous failed?
        # Or just fixed time per n?
        # Let's use remaining time.
        time_left = deadline - time.time()
        if time_left < 5:
            break
            
        # Allocate time proportional to n? Or just fixed 30s per attempt?
        # Let's try 25s per attempt.
        max_time_for_n = min(25.0, time_left - 5.0)
        
        colors, cost = local_search(n, max_time=max_time_for_n)
        
        if cost == 0:
            best_n = n
            best_colors = colors
            # If we found a valid coloring for n, we can try n+1
            # But we are iterating specific targets.
            # If we find 29, we should try 30.
            # If we find 30, we stop (or try 31? Limit is 31).
        else:
            # If we couldn't find a valid coloring for n, 
            # it's likely we won't for n+1 either, but maybe we should try?
            # Actually, if n=25 fails, n=29 will likely fail.
            # But random restarts might help.
            # Let's continue to next n to see if luck strikes, 
            # but we should probably break if n is small and fails badly?
            # No, just continue.
            pass

    # If we didn't find any, return the best effort from the last run?
    # The loop above overwrites best_n only if cost==0.
    # If no n succeeded, best_n remains 1 (empty or single vertex).
    # We should return the best valid coloring found.
    # If none found, we must return *some* valid coloring.
    # n=1 is always valid.
    # Let's ensure we return a valid coloring.
    
    # If best_n is 1, we return n=1.
    # But we want to maximize n.
    # If the search failed to find 0 violations for any n > 1, we should probably return n=1?
    # Or try a greedy construction for a small n?
    # n=5 is easy.
    
    if best_n == 1:
        # Fallback: Try to construct a small valid coloring greedily
        # Or just return n=1.
        # Let's try n=5 with SA quickly.
        if time.time() < deadline - 5:
            colors, cost = local_search(5, max_time=5.0)
            if cost == 0:
                best_n = 5
                best_colors = colors
            else:
                # n=1 is always valid
                best_n = 1
                best_colors = [] # No edges
        else:
            best_n = 1
            best_colors = []

    return {'n': best_n, 'colors': best_colors}

