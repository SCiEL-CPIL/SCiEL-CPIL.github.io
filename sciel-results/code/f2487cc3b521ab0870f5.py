import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    n = 18
    FULL = (1 << n) - 1

    # Precompute popcount for all masks
    POPC = [0] * (1 << n)
    for m in range(1, 1 << n):
        POPC[m] = POPC[m >> 1] + (m & 1)

    def edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def has_k4(adj):
        for a in range(n):
            for b in range(a + 1, n):
                if not ((adj[a] >> b) & 1):
                    continue
                common = adj[a] & adj[b]
                mask_c = common & ~((1 << (b + 1)) - 1)
                while mask_c:
                    lsb = mask_c & -mask_c
                    c = lsb.bit_length() - 1
                    mask_d = adj[c] & common & ~((1 << (c + 1)) - 1)
                    if mask_d:
                        return True
                    mask_c ^= lsb
        return False

    def get_triangles(adj):
        triangles = []
        for a in range(n):
            for b in range(a + 1, n):
                if not ((adj[a] >> b) & 1):
                    continue
                common = adj[a] & adj[b] & ~((1 << (b + 1)) - 1)
                while common:
                    lsb = common & -common
                    c = lsb.bit_length() - 1
                    triangles.append((a, b, c))
                    common ^= lsb
        return triangles

    def max_triangle_free_size(adj):
        """
        Find largest subset of vertices inducing no triangle.
        Optimized Branch and Bound.
        """
        triangles = get_triangles(adj)
        if not triangles:
            return n
        
        # Precompute tri_mask[u][v] = bitmask of w such that {u, v, w} is a triangle
        tri_mask = [[0] * n for _ in range(n)]
        for a, b, c in triangles:
            tri_mask[a][b] |= (1 << c)
            tri_mask[b][a] |= (1 << c)
            tri_mask[a][c] |= (1 << b)
            tri_mask[c][a] |= (1 << b)
            tri_mask[b][c] |= (1 << a)
            tri_mask[c][b] |= (1 << a)

        # Greedy initial lower bound
        best = 0
        for _ in range(20):
            random.shuffle(range(n)) # Can't shuffle range, need list
            verts = list(range(n))
            random.shuffle(verts)
            chosen = 0
            for v in verts:
                # Check if adding v is safe
                # v is safe if for all u in chosen, tri_mask[u][v] & chosen == 0
                # Wait, tri_mask[u][v] contains w. We need to check if any w in chosen.
                # So: (tri_mask[u][v] & chosen) == 0 for all u in chosen.
                safe = True
                # Iterate u in chosen
                mask_u = chosen
                while mask_u:
                    lsb_u = mask_u & -mask_u
                    u = lsb_u.bit_length() - 1
                    if tri_mask[u][v] & chosen:
                        safe = False
                        break
                    mask_u ^= lsb_u
                if safe:
                    chosen |= (1 << v)
            if POPC[chosen] > best:
                best = POPC[chosen]
        
        # B&B
        # candidates: bitmask
        # chosen: bitmask
        def dfs(candidates, chosen):
            nonlocal best
            if not candidates:
                if POPC[chosen] > best:
                    best = POPC[chosen]
                return
            
            # Upper bound
            if POPC[chosen] + POPC[candidates] <= best:
                return
            
            # Heuristic: Pick v from candidates that maximizes "conflict" with chosen?
            # Or just pick the one with highest degree in the "conflict graph" of candidates?
            # For simplicity and speed, pick lowest set bit, but try a few options if time permits?
            # Let's stick to lowest set bit for now, but ensure pruning is tight.
            
            lsb = candidates & -candidates
            v = lsb.bit_length() - 1
            new_cand_excl = candidates ^ lsb
            
            # Branch 1: Exclude v
            dfs(new_cand_excl, chosen)
            
            # Branch 2: Include v
            # Check if v can be added (it's in candidates, so it's not in chosen)
            # We must remove any u in new_cand_excl such that tri_mask[u][v] & chosen != 0
            forbidden = 0
            cand_iter = new_cand_excl
            while cand_iter:
                lsb_u = cand_iter & -cand_iter
                u = lsb_u.bit_length() - 1
                if tri_mask[u][v] & chosen:
                    forbidden |= lsb_u
                cand_iter ^= lsb_u
            
            new_cand_incl = new_cand_excl & ~forbidden
            dfs(new_cand_incl, chosen | lsb)

        dfs(FULL, 0)
        return best

    def remove_k4(adj):
        for _ in range(100):
            if not has_k4(adj):
                return True
            found = False
            for a in range(n):
                for b in range(a + 1, n):
                    if not ((adj[a] >> b) & 1): continue
                    common = adj[a] & adj[b]
                    mask_c = common & ~((1 << (b + 1)) - 1)
                    while mask_c:
                        lsb_c = mask_c & -mask_c
                        c = lsb_c.bit_length() - 1
                        mask_d = adj[c] & common & ~((1 << (c + 1)) - 1)
                        if mask_d:
                            lsb_d = mask_d & -mask_d
                            d = lsb_d.bit_length() - 1
                            edges = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
                            e1, e2 = random.choice(edges)
                            adj[e1] &= ~(1 << e2)
                            adj[e2] &= ~(1 << e1)
                            found = True
                            break
                        mask_c ^= lsb_c
                    if found: break
                if found: break
            if not found:
                break
        return not has_k4(adj)

    random.seed(42)
    
    best_adj = None
    best_score = 18
    
    # Generate initial candidates
    candidates = []
    
    # 1. Random Regular-ish graphs
    for _ in range(30):
        if time.time() > deadline - 60: break
        adj = [0] * n
        # Target degree 7
        deg = 7
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges)
        degs = [0] * n
        for u, v in edges:
            if degs[u] < deg and degs[v] < deg:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                degs[u] += 1
                degs[v] += 1
        if remove_k4(adj):
            candidates.append(adj)

    # 2. Structured: Cyclic
    adj_s = [0] * n
    for i in range(n):
        for off in [1, 2, 4, 8]:
            j = (i + off) % n
            if j != i:
                adj_s[i] |= (1 << j)
                adj_s[j] |= (1 << i)
    if remove_k4(adj_s):
        candidates.append(adj_s)

    # Evaluate candidates
    for adj in candidates:
        if time.time() > deadline - 30: break
        score = max_triangle_free_size(adj)
        if score < best_score:
            best_score = score
            best_adj = adj[:]
            
    if best_adj is None:
        best_adj = candidates[0]
        best_score = max_triangle_free_size(best_adj)

    # Local Search (Simulated Annealing)
    adj = best_adj[:]
    current_score = best_score
    temp = 1.0
    cooling = 0.995
    
    # Precompute list of all edges for faster access
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    for i in range(20000):
        if time.time() > deadline - 5: break
        if best_score <= 6: break # 6 is likely optimal
        
        # Random move: flip an edge
        u, v = random.choice(all_edges)
        
        old_adj = adj[:]
        if (adj[u] >> v) & 1:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        if has_k4(adj):
            adj = old_adj
            continue
            
        new_score = max_triangle_free_size(adj)
        
        delta = new_score - current_score
        if delta <= 0 or random.random() < math.exp(-delta / temp):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = adj[:]
        else:
            adj = old_adj
            
        temp *= cooling
        if temp < 0.01: temp = 0.01

    return edges_from_adj(best_adj)
