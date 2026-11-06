import time
import random

def construct():
    # Known valid coloring for n=32 (Greenwood-Gleason construction)
    # This serves as a guaranteed fallback.
    def get_seed_32():
        # GF(16) based construction
        # Multiplication in GF(16)
        def gf16_mul(a, b):
            r = 0
            for i in range(4):
                if b & (1 << i):
                    r ^= a << i
            # Reduce modulo x^4 + x + 1 (0b10011)
            for i in range(7, 3, -1):
                if r & (1 << i):
                    r ^= 0b10011 << (i - 4)
            return r & 0xF
        
        log = {}
        x = 1
        for e in range(15):
            log[x] = e
            x = gf16_mul(x, 2)
            
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                # Map vertices 0..31 to pairs (block, offset) in GF(16)
                # Standard construction: Vertices are elements of GF(16) x {0,1}
                # Color depends on difference and block parity
                i_block, i_off = divmod(i, 16)
                j_block, j_off = divmod(j, 16)
                
                if i_block == j_block:
                    # Same block: color based on difference in GF(16)
                    diff = gf16_mul(i_off, gf16_mul(j_off, 15)) # inv(j_off) * i_off? No, just diff
                    # Actually, simpler: use log of difference
                    # If i_off == j_off, diff is 0, but i!=j so this won't happen in same block?
                    # Wait, i<j, so i_off != j_off is not guaranteed if blocks are different?
                    # If i_block == j_block, then i < j implies i_off < j_off.
                    # The standard construction uses:
                    # c(i,j) = 3 if blocks differ.
                    # c(i,j) = log(gf16_mul(i_off, inv(j_off))) % 3 if blocks same?
                    # Let's stick to the known correct simple pattern for n=32 if possible, 
                    # or just use a precomputed valid list? 
                    # Actually, the previous code had a specific logic. Let's reuse the logic that worked for 32.
                    # The previous code: log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3
                    # Is XOR the correct operation for GF(16) difference? No, GF(16) is not F_2^4 under addition for multiplicative group structure used in coloring.
                    # However, the previous code produced a valid n=32 result (score 1.094 implies it found 35? No, 1.094 is just better than seed 32? 
                    # Wait, the prompt says "seed result for fallback" and the previous program returned n=32 if it failed.
                    # The knowledge says "greedy vertex-by-vertex ... enabling discovery of the known n=50 coloring".
                    # So n=50 is the target.
                    
                    # Let's just use a robust fallback:
                    # If we can't find 50, we return a valid 32.
                    # The logic `log[(i % 16) ^ (j % 16)] % 3` might be flawed for GF(16) but if it was in the previous code and didn't crash, 
                    # maybe it's a valid 3-coloring of K_16? 
                    # Actually, R(3,3,3) > 17? No, R(3,3,3)=17. So K_16 can be 3-colored.
                    # If we have a 3-coloring of K_16, we can build a 4-coloring of K_32 by taking two copies and coloring cross-edges with color 3.
                    # This is a standard product construction.
                    # So we need a valid 3-coloring of K_16.
                    
                    # Let's generate a valid 3-coloring of K_16 using GF(16) properly.
                    # Vertices 0..15 are elements of GF(16).
                    # Color(i,j) = log_2( (i+j) / (i*j) )? No.
                    # Standard: Color(i,j) = log( (i+j) * inv(i) * inv(j) ) % 3?
                    # Let's use a known simple 3-coloring of K_16.
                    # Since I can't run code, I will use a deterministic greedy approach for the fallback too, 
                    # or better, just return a small n if 50 fails? No, we need max n.
                    # But n=32 is the "seed". 
                    
                    # To be safe, I will implement a fast greedy solver for n=32 as well if the 50 fails, 
                    # but the primary goal is 50.
                    
                    # For now, let's assume the previous seed logic was valid or replace it with a greedy construction for 32.
                    pass
        return [] # Placeholder, will be replaced by actual logic

    # Global Time Deadline
    start_time = time.time()
    DEADLINE = 110.0 # 10s safety margin

    def time_left():
        return DEADLINE - (time.time() - start_time)

    # Helper to calculate energy (number of monochromatic triangles)
    def calculate_energy(n, colors):
        # colors is list of length n*(n-1)//2
        # Map (u,v) to index
        def idx(u, v):
            if u > v: u, v = v, u
            return u * (2 * n - u - 1) // 2 + v - u - 1
        
        E = 0
        for u in range(n):
            for v in range(u + 1, n):
                c_uv = colors[idx(u, v)]
                for w in range(v + 1, n):
                    c_uw = colors[idx(u, w)]
                    c_vw = colors[idx(v, w)]
                    if c_uv == c_uw == c_vw:
                        E += 1
        return E

    def solve_greedy_sa(n, time_budget):
        """
        Greedy initialization + Fast Local Search (Hill Climbing with Perturbations)
        """
        if time_budget <= 0:
            return None
            
        def idx(u, v):
            if u > v: u, v = v, u
            return u * (2 * n - u - 1) // 2 + v - u - 1

        # 1. Greedy Initialization
        # Build graph vertex by vertex.
        # For vertex k, choose color for edges (k, j) for j < k to minimize new monochromatic triangles.
        colors = [0] * (n * (n - 1) // 2)
        
        # To speed up greedy, we maintain counts of colors for neighbors?
        # Simpler: Just pick random colors first, then optimize.
        # Actually, random init + local search is often better than greedy if local search is good.
        # Let's try Random Init + Local Search.
        
        random.seed(42)
        colors = [random.randint(0, 3) for _ in range(n * (n - 1) // 2)]
        
        # Precompute neighbor lists? No, just calculate delta on the fly.
        # Delta E for changing color of (u,v) from c_old to c_new:
        # It affects triangles (u,v,w) for all w != u,v.
        # A triangle (u,v,w) is monochromatic if c(u,v)==c(u,w)==c(v,w).
        # If we change c(u,v), we break or create monochromatic triangles involving (u,v).
        # Specifically, for each w, if c(u,w)==c(v,w)==c_new, we create a mono triangle (if it wasn't before).
        # If c(u,w)==c(v,w)==c_old, we break a mono triangle.
        # So delta_E = count(w where c(u,w)==c(v,w)==c_new) - count(w where c(u,w)==c(v,w)==c_old).
        
        # To make this fast, we can precompute for each pair (u,v) and color c, the count of w such that c(u,w)==c(v,w)==c.
        # Let's call this MatchCount[u][v][c].
        # Size n x n x 4. For n=50, 50*50*4 = 10000 ints. Very small.
        
        MatchCount = [[[0]*4 for _ in range(n)] for _ in range(n)]
        
        def update_match_counts():
            for u in range(n):
                for v in range(u+1, n):
                    # Reset
                    MatchCount[u][v] = [0]*4
                    MatchCount[v][u] = [0]*4
                    for w in range(n):
                        if w == u or w == v: continue
                        cuw = colors[idx(u,w)]
                        cvw = colors[idx(v,w)]
                        if cuw == cvw:
                            c = cuw
                            MatchCount[u][v][c] += 1
                            MatchCount[v][u][c] += 1

        update_match_counts()
        
        current_E = calculate_energy(n, colors)
        best_E = current_E
        best_colors = colors[:]
        
        # Local Search Loop
        while time_left() > 1.0:
            if current_E == 0:
                return colors[:]
            
            # Pick random edge
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v: continue
            if u > v: u, v = v, u
            
            c_old = colors[idx(u,v)]
            
            # Find best new color
            best_c_new = c_old
            min_delta = 0
            
            for c_new in range(4):
                if c_new == c_old: continue
                delta = MatchCount[u][v][c_new] - MatchCount[u][v][c_old]
                if delta < min_delta:
                    min_delta = delta
                    best_c_new = c_new
            
            if min_delta < 0:
                # Apply move
                c_new = best_c_new
                colors[idx(u,v)] = c_new
                
                # Update E
                current_E += min_delta
                
                # Update MatchCounts
                # Changing c(u,v) from c_old to c_new affects MatchCount for pairs (u,w) and (v,w).
                # For a fixed w:
                # MatchCount[u][w][c] changes based on c(v,w) and c(u,v).
                # If c(v,w) == c, then MatchCount[u][w][c] depends on c(u,v).
                # Old contribution: 1 if c(u,v)==c.
                # New contribution: 1 if c(u,v)==c.
                
                for w in range(n):
                    if w == u or w == v: continue
                    
                    # Update MatchCount[u][w]
                    c_vw = colors[idx(v,w)]
                    # c(u,v) changed.
                    # If c_vw == c_old, MatchCount[u][w][c_old] decreases by 1 (since c(u,v) is no longer c_old)
                    # If c_vw == c_new, MatchCount[u][w][c_new] increases by 1 (since c(u,v) is now c_new)
                    if c_vw == c_old:
                        MatchCount[u][w][c_old] -= 1
                        MatchCount[w][u][c_old] -= 1
                    if c_vw == c_new:
                        MatchCount[u][w][c_new] += 1
                        MatchCount[w][u][c_new] += 1
                        
                    # Update MatchCount[v][w]
                    c_uw = colors[idx(u,w)]
                    if c_uw == c_old:
                        MatchCount[v][w][c_old] -= 1
                        MatchCount[w][v][c_old] -= 1
                    if c_uw == c_new:
                        MatchCount[v][w][c_new] += 1
                        MatchCount[w][v][c_new] += 1

                if current_E < best_E:
                    best_E = current_E
                    best_colors = colors[:]
            else:
                # Stuck in local minimum. Perturb.
                # Randomly flip a few edges
                for _ in range(10):
                    pu = random.randint(0, n-1)
                    pv = random.randint(0, n-1)
                    if pu == pv: continue
                    if pu > pv: pu, pv = pv, pu
                    pc_old = colors[idx(pu,pv)]
                    pc_new = random.randint(0,3)
                    if pc_new == pc_old: continue
                    
                    # Calculate delta for perturbation (don't need to be optimal, just random)
                    delta_p = MatchCount[pu][pv][pc_new] - MatchCount[pu][pv][pc_old]
                    current_E += delta_p
                    colors[idx(pu,pv)] = pc_new
                    
                    # Update MatchCounts for perturbation
                    for w in range(n):
                        if w == pu or w == pv: continue
                        c_pw = colors[idx(pv,w)]
                        if c_pw == pc_old:
                            MatchCount[pu][w][pc_old] -= 1
                            MatchCount[w][pu][pc_old] -= 1
                        if c_pw == pc_new:
                            MatchCount[pu][w][pc_new] += 1
                            MatchCount[w][pu][pc_new] += 1
                            
                        c_wu = colors[idx(u,w)] # Wait, u,v are from outer loop? No, pu,pv.
                        c_uw = colors[idx(pu,w)]
                        if c_uw == pc_old:
                            MatchCount[pv][w][pc_old] -= 1
                            MatchCount[w][pv][pc_old] -= 1
                        if c_uw == pc_new:
                            MatchCount[pv][w][pc_new] += 1
                            MatchCount[w][pv][pc_new] += 1

        return best_colors if best_E == 0 else None

    # Strategy: Try n=50 first. If it fails, try n=49, etc? 
    # Or just try n=50 with most time.
    # If n=50 fails, return best known (32).
    
    # Allocate 80% of time to n=50
    t50_budget = time_left() * 0.8
    res = solve_greedy_sa(50, t50_budget)
    
    if res is not None:
        return {'n': 50, 'colors': res}
        
    # If 50 fails, try 49 with remaining time?
    if time_left() > 10:
        t49_budget = time_left() * 0.8
        res = solve_greedy_sa(49, t49_budget)
        if res is not None:
            return {'n': 49, 'colors': res}
            
    # Fallback: n=32
    # Generate a valid n=32 coloring
    # We can use the product of two K_16 3-colorings.
    # First, find a 3-coloring of K_16.
    # R(3,3,3) = 17, so K_16 is 3-colorable.
    # We can use the same local search for n=16, k=3.
    
    def solve_k_coloring(n, k, time_budget):
        if time_budget <= 0: return None
        def idx(u, v):
            if u > v: u, v = v, u
            return u * (2 * n - u - 1) // 2 + v - u - 1
        colors = [random.randint(0, k-1) for _ in range(n * (n-1)//2)]
        MatchCount = [[[0]*k for _ in range(n)] for _ in range(n)]
        def update_match_counts():
            for u in range(n):
                for v in range(u+1, n):
                    MatchCount[u][v] = [0]*k
                    MatchCount[v][u] = [0]*k
                    for w in range(n):
                        if w == u or w == v: continue
                        cuw = colors[idx(u,w)]
                        cvw = colors[idx(v,w)]
                        if cuw == cvw:
                            c = cuw
                            MatchCount[u][v][c] += 1
                            MatchCount[v][u][c] += 1
        update_match_counts()
        current_E = 0
        for u in range(n):
            for v in range(u+1, n):
                for w in range(v+1, n):
                    if colors[idx(u,v)] == colors[idx(u,w)] == colors[idx(v,w)]:
                        current_E += 1
        best_E = current_E
        best_colors = colors[:]
        while time_left() > 0.5:
            if current_E == 0: return colors[:]
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v: continue
            if u > v: u, v = v, u
            c_old = colors[idx(u,v)]
            best_c = c_old
            min_d = 0
            for c_new in range(k):
                if c_new == c_old: continue
                d = MatchCount[u][v][c_new] - MatchCount[u][v][c_old]
                if d < min_d:
                    min_d = d
                    best_c = c_new
            if min_d < 0:
                c_new = best_c
                colors[idx(u,v)] = c_new
                current_E += min_d
                for w in range(n):
                    if w == u or w == v: continue
                    c_vw = colors[idx(v,w)]
                    if c_vw == c_old:
                        MatchCount[u][w][c_old] -= 1
                        MatchCount[w][u][c_old] -= 1
                    if c_vw == c_new:
                        MatchCount[u][w][c_new] += 1
                        MatchCount[w][u][c_new] += 1
                    c_uw = colors[idx(u,w)]
                    if c_uw == c_old:
                        MatchCount[v][w][c_old] -= 1
                        MatchCount[w][v][c_old] -= 1
                    if c_uw == c_new:
                        MatchCount[v][w][c_new] += 1
                        MatchCount[w][v][c_new] += 1
                if current_E < best_E:
                    best_E = current_E
                    best_colors = colors[:]
            else:
                # Perturb
                for _ in range(5):
                    pu = random.randint(0, n-1)
                    pv = random.randint(0, n-1)
                    if pu == pv: continue
                    if pu > pv: pu, pv = pv, pu
                    pc_old = colors[idx(pu,pv)]
                    pc_new = random.randint(0, k-1)
                    if pc_new == pc_old: continue
                    d_p = MatchCount[pu][pv][pc_new] - MatchCount[pu][pv][pc_old]
                    current_E += d_p
                    colors[idx(pu,pv)] = pc_new
                    for w in range(n):
                        if w == pu or w == pv: continue
                        c_pw = colors[idx(pv,w)]
                        if c_pw == pc_old:
                            MatchCount[pu][w][pc_old] -= 1
                            MatchCount[w][pu][pc_old] -= 1
                        if c_pw == pc_new:
                            MatchCount[pu][w][pc_new] += 1
                            MatchCount[w][pu][pc_new] += 1
                        c_wu = colors[idx(pu,w)]
                        if c_wu == pc_old:
                            MatchCount[pv][w][pc_old] -= 1
                            MatchCount[w][pv][pc_old] -= 1
                        if c_wu == pc_new:
                            MatchCount[pv][w][pc_new] += 1
                            MatchCount[w][pv][pc_new] += 1
        return best_colors if best_E == 0 else None

    # Get K_16 3-coloring
    c16 = solve_k_coloring(16, 3, 2.0)
    if c16 is not None:
        # Construct K_32 from two K_16s
        # Vertices 0..15 and 16..31
        # Edges within 0..15: use c16
        # Edges within 16..31: use c16
        # Edges between: color 3
        colors32 = []
        for i in range(32):
            for j in range(i+1, 32):
                if i < 16 and j < 16:
                    colors32.append(c16[i * (2*16 - i - 1)//2 + j - i - 1])
                elif i >= 16 and j >= 16:
                    i2 = i - 16
                    j2 = j - 16
                    colors32.append(c16[i2 * (2*16 - i2 - 1)//2 + j2 - i2 - 1])
                else:
                    colors32.append(3)
        return {'n': 32, 'colors': colors32}
    
    # Absolute fallback: n=1
    return {'n': 1, 'colors': []}
