import random
import time

_START_TIME = time.time()
DEADLINE = 115.0  # Leave 5s margin for 120s limit

def _time_left():
    return DEADLINE - (time.time() - _START_TIME)

def _is_k4_free(adj, n=18):
    """Check if graph is K4-free using bitmasks."""
    for a in range(n):
        for b in range(a + 1, n):
            if not (adj[a] >> b) & 1:
                continue
            common = adj[a] & adj[b]
            c = common
            while c:
                lsb = c & -c
                x = lsb.bit_length() - 1
                c -= lsb
                if (adj[x] & common) != 0:
                    return False
    return True

def _max_triangle_free_induced(adj, n=18):
    """
    Corrected Branch and Bound for largest triangle-free induced subgraph.
    Returns the size of the largest subset S such that G[S] is triangle-free.
    """
    best = [0]
    
    def popcount(x):
        return bin(x).count('1')
    
    # Precompute common neighbors
    common = [[0] * n for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            c = adj[u] & adj[v]
            common[u][v] = c
            common[v][u] = c
            
    def bb(idx, current_mask, forbidden_mask):
        if time.time() - _START_TIME > DEADLINE:
            return
            
        # Upper bound
        remaining_count = 0
        for i in range(idx, n):
            if not (forbidden_mask >> i) & 1:
                remaining_count += 1
                
        cur_sz = popcount(current_mask)
        if cur_sz + remaining_count <= best[0]:
            return
            
        if idx == n:
            if cur_sz > best[0]:
                best[0] = cur_sz
            return
            
        v = idx
        bit = 1 << v
        
        # Option 1: Include v
        if not (forbidden_mask & bit):
            # Check if v creates a triangle with existing vertices in current_mask
            can_include = True
            cm = current_mask
            while cm:
                lsb = cm & -cm
                u = lsb.bit_length() - 1
                cm -= lsb
                if (adj[v] >> u) & 1:
                    # v is connected to u. Check if any common neighbor of v and u is in current_mask
                    if common[v][u] & current_mask:
                        can_include = False
                        break
                        
            if can_include:
                new_forbidden = forbidden_mask
                cm = current_mask
                while cm:
                    lsb = cm & -cm
                    u = lsb.bit_length() - 1
                    cm -= lsb
                    if (adj[v] >> u) & 1:
                        new_forbidden |= common[v][u]
                        
                bb(idx + 1, current_mask | bit, new_forbidden)
                
        # Option 2: Exclude v
        bb(idx + 1, current_mask, forbidden_mask)

    # Greedy initial bound
    greedy_set = 0
    greedy_forbidden = 0
    remaining = (1 << n) - 1
    while remaining:
        candidates = remaining & ~greedy_forbidden
        if not candidates:
            break
        best_v = -1
        best_score = n + 1
        c = candidates
        while c:
            lsb = c & -c
            v = lsb.bit_length() - 1
            c -= lsb
            # Check if v can be added
            can_add = True
            gm = greedy_set
            while gm:
                lsb2 = gm & -gm
                u = lsb2.bit_length() - 1
                gm -= lsb2
                if (adj[v] >> u) & 1:
                    if common[v][u] & greedy_set:
                        can_add = False
                        break
            if not can_add:
                continue
            
            new_forb = 0
            gm = greedy_set
            while gm:
                lsb2 = gm & -gm
                u = lsb2.bit_length() - 1
                gm -= lsb2
                if (adj[v] >> u) & 1:
                    new_forb |= common[v][u]
            cnt = popcount(new_forb & remaining)
            if cnt < best_score:
                best_score = cnt
                best_v = v
        if best_v == -1:
            break
        greedy_set |= (1 << best_v)
        remaining &= ~(1 << best_v)
        gm2 = greedy_set & ~(1 << best_v)
        while gm2:
            lsb2 = gm2 & -gm2
            u = lsb2.bit_length() - 1
            gm2 -= lsb2
            if (adj[best_v] >> u) & 1:
                greedy_forbidden |= common[best_v][u]
                
    best[0] = popcount(greedy_set)
    
    bb(0, 0, 0)
    return best[0]

def construct():
    n = 18
    random.seed(42)
    
    # Start with Turan complement (3 disjoint K6 in H, so G is K6,6,6)
    # H is 3 disjoint cliques of size 6
    adj_h = [0] * n
    for i in range(3):
        start = i * 6
        for u in range(start, start + 6):
            for v in range(start, start + 6):
                if u != v:
                    adj_h[u] |= (1 << v)
                    adj_h[v] |= (1 << u)
    
    # G is complement of H
    full = (1 << n) - 1
    adj_g = [0] * n
    for v in range(n):
        adj_g[v] = full & ~(adj_h[v] | (1 << v))
        
    current_g = adj_g[:]
    current_score = _max_triangle_free_induced(current_g, n)
    best_g = current_g[:]
    best_score = current_score
    
    print(f"Initial Score: {best_score}", flush=True)
    
    steps = 0
    while _time_left() > 10 and steps < 100000:
        steps += 1
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
            
        # Flip edge in H
        new_h = current_g[:] # Wait, we are searching in G or H?
        # Let's search in G directly.
        # Flip edge in G
        new_g = current_g[:]
        if (new_g[u] >> v) & 1:
            new_g[u] &= ~(1 << v)
            new_g[v] &= ~(1 << u)
        else:
            new_g[u] |= (1 << v)
            new_g[v] |= (1 << u)
            
        if not _is_k4_free(new_g, n):
            continue
            
        new_score = _max_triangle_free_induced(new_g, n)
        
        if new_score < best_score:
            best_score = new_score
            best_g = new_g
            current_g = new_g
            print(f"Improved to {best_score} at step {steps}", flush=True)
        elif new_score == best_score and random.random() < 0.5:
            current_g = new_g
        else:
            # Annealing: accept worse with some probability?
            # Or just reject.
            pass
            
    # Convert best_g to edge list
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_g[u] >> v) & 1:
                edges.append([u, v])
                
    return edges
