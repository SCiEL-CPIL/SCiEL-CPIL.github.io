import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Fallback: Paley graph on 17 vertices
    def paley_17():
        squares = {x * x % 17 for x in range(1, 17)}
        edges = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
        return {'n': 17, 'edges': edges}

    # Helper: Check if graph is valid (no K4, no I6)
    # adj is list of bitmasks
    def is_valid(adj, n):
        # Check K4
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    # Check if common contains an edge
                    c = common
                    while c:
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        c ^= lsb
                        # Check if w is connected to any other vertex in common
                        # (common & adj[w]) will include u, v and potentially others
                        # We need to check if there is a vertex x in common, x != u, x != v, x != w, such that w~x
                        # Actually, if adj[w] & common has any bit set that is not u or v, we have a K4.
                        # But adj[w] & common includes u and v always (since w is in common, w~u and w~v).
                        # So we need to check if (adj[w] & common) has a bit set other than u and v.
                        mask_uv = (1 << u) | (1 << v)
                        if (adj[w] & common) & ~mask_uv:
                            return False
        # Check I6 (Independent set of size 6)
        # Equivalent to K6 in complement
        mask = (1 << n) - 1
        comp_adj = [mask ^ adj[i] ^ (1 << i) for i in range(n)]
        
        # Find K6 in comp_adj
        # Use a simple backtracking / greedy check
        # Since n is small (<=35), we can try to find a clique of size 6 in comp.
        # If we find one, it's invalid.
        
        # Heuristic: Check if any 6-subset is a clique in comp.
        # 35 choose 6 is ~1.6M. Too slow for every step in local search, but okay for final validation.
        # For local search, we might use a faster check or just rely on repair.
        
        # Efficient K6 check:
        # For each subset of 6, check if it's a clique in comp.
        # Or, find max clique in comp. If max clique >= 6, invalid.
        
        # Let's use a standard clique finding algorithm (Bron-Kerbosch or simple backtracking)
        # But for speed in the main loop, we might just check for K4 in comp? No, that's K4 in comp = I4 in G.
        # We need I6 in G = K6 in comp.
        
        # A faster check for K6 in comp:
        # For each vertex v, look at its neighbors in comp.
        # If any vertex has < 5 neighbors in comp, it can't be in a K6.
        # Let's just do a randomized check or a limited search.
        
        # For the final validation, we can do a more thorough check.
        # For now, let's assume the repair loop handles it.
        # But is_valid is used to confirm success.
        
        # Let's implement a fast K6 check using bitsets.
        # Try to build a clique of size 6.
        
        # Simple approach:
        # Pick a vertex v0.
        # Its neighbors in comp must contain a K5.
        # Recursively check for K5 in N_comp(v0).
        
        def has_ksize_in_comp(comp_adj, n, k):
            if k <= 1:
                return True
            # Pick a vertex with max degree in the induced subgraph?
            # Or just iterate.
            # To be fast, we can use the fact that if max degree < k-1, no clique.
            
            # Let's use a simple recursive search with bitsets.
            # candidates is a bitmask of vertices.
            # We want to find a clique of size k in comp_adj induced by candidates.
            
            def search(candidates, size_needed):
                if size_needed == 0:
                    return True
                if candidates == 0:
                    return False
                if size_needed == 1:
                    return candidates != 0
                
                # Pick a vertex from candidates
                # Heuristic: pick one with high degree in candidates
                c = candidates
                best_v = -1
                best_deg = -1
                while c:
                    lsb = c & -c
                    v = lsb.bit_length() - 1
                    c ^= lsb
                    deg = (comp_adj[v] & candidates).bit_count()
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
                        # Early exit if degree is too low
                        if deg < size_needed - 1:
                            continue
                
                if best_v == -1:
                    return False
                
                # If the best vertex doesn't have enough neighbors, no clique
                if (comp_adj[best_v] & candidates).bit_count() < size_needed - 1:
                    return False
                
                # Try to find clique of size_needed-1 in neighbors of best_v within candidates
                new_candidates = comp_adj[best_v] & candidates
                if search(new_candidates, size_needed - 1):
                    return True
                
                # If it failed, try another vertex? 
                # For exactness, we should backtrack. But for speed, maybe just one path?
                # No, that's not exact.
                # However, for n=35, K6 is rare.
                # Let's stick to a simpler check for is_valid:
                # Iterate all 6-sets? 1.6M checks. Each check is O(1) with bitmasks?
                # Check if 6-set is a clique in comp:
                # For set S, check if for all u in S, (comp_adj[u] & mask_S) == mask_S ^ (1<<u).
                # This is fast.
                
                return False
            
            # Actually, the 6-set iteration is probably fastest for n=35 in Python if optimized?
            # 1.6M * 6 ops = 10M ops. ~0.1s in C, ~1s in Python.
            # Let's use the 6-set iteration for is_valid.
            
            return False
        
        # Re-implementing is_valid I6 check using 6-set iteration
        for subset in combinations(range(n), 6):
            mask_sub = 0
            for v in subset:
                mask_sub |= (1 << v)
            is_clique = True
            for v in subset:
                # Check if v is connected to all others in subset in comp
                # comp_adj[v] & mask_sub should equal mask_sub ^ (1 << v)
                if (comp_adj[v] & mask_sub) != (mask_sub ^ (1 << v)):
                    is_clique = False
                    break
            if is_clique:
                return False
        
        return True

    # Local Search for n=35
    # Strategy:
    # 1. Random init
    # 2. Repair K4s by removing edges
    # 3. Repair I6s by adding edges
    # 4. If stuck, try flipping random edges
    
    def find_k4(adj, n):
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    c = common
                    while c:
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        c ^= lsb
                        mask_uv = (1 << u) | (1 << v)
                        if (adj[w] & common) & ~mask_uv:
                            # Found K4: u, v, w, x
                            # x is in (adj[w] & common) & ~mask_uv
                            x_mask = (adj[w] & common) & ~mask_uv
                            lsb_x = x_mask & -x_mask
                            x = lsb_x.bit_length() - 1
                            return (u, v, w, x)
        return None

    def find_i6(adj, n):
        mask = (1 << n) - 1
        comp_adj = [mask ^ adj[i] ^ (1 << i) for i in range(n)]
        # Find K6 in comp
        # Use random sampling for speed in repair loop
        # If not found, assume none (risky, but we verify at end)
        for _ in range(50):
            sample = random.sample(range(n), 6)
            mask_sub = 0
            for v in sample:
                mask_sub |= (1 << v)
            is_clique = True
            for v in sample:
                if (comp_adj[v] & mask_sub) != (mask_sub ^ (1 << v)):
                    is_clique = False
                    break
            if is_clique:
                return sample
        return None

    best_result = paley_17()
    
    # Try to find n=35
    n = 35
    found = False
    restarts = 0
    while time.time() < deadline - 5.0 and not found:
        restarts += 1
        adj = [0] * n
        # Init with random graph, density ~0.5
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Repair loop
        for step in range(2000):
            if time.time() > deadline - 2.0:
                break
                
            # Check K4
            k4 = find_k4(adj, n)
            if k4:
                u, v, w, x = k4
                # Remove a random edge from the K4
                edges = [(u,v), (u,w), (u,x), (v,w), (v,x), (w,x)]
                a, b = random.choice(edges)
                adj[a] &= ~(1 << b)
                adj[b] &= ~(1 << a)
                continue
            
            # Check I6
            i6 = find_i6(adj, n)
            if i6:
                # Add a random edge within the I6 to break it
                a, b = random.sample(i6, 2)
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
                continue
            
            # If no violations found by heuristic checks, verify fully
            if is_valid(adj, n):
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
            
            # If heuristic says valid but full check fails, we have a hidden violation.
            # This shouldn't happen often if find_i6 is robust.
            # If it does, we might be stuck. Try flipping a random edge.
            # But is_valid is exact. So if is_valid is False, there IS a violation.
            # Why did find_k4 and find_i6 miss it?
            # find_k4 is exact. So it must be an I6 that find_i6 missed.
            # find_i6 uses random sampling.
            # Let's do a more thorough I6 search.
            
            # Full I6 search
            mask = (1 << n) - 1
            comp_adj = [mask ^ adj[i] ^ (1 << i) for i in range(n)]
            i6_found = None
            # Iterate 6-sets
            # This is slow, but we only do it when stuck.
            for subset in combinations(range(n), 6):
                mask_sub = 0
                for v in subset:
                    mask_sub |= (1 << v)
                is_clique = True
                for v in subset:
                    if (comp_adj[v] & mask_sub) != (mask_sub ^ (1 << v)):
                        is_clique = False
                        break
                if is_clique:
                    i6_found = list(subset)
                    break
            
            if i6_found:
                a, b = random.sample(i6_found, 2)
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
                continue
            
            # If still not found, something is wrong. Break and restart.
            break

    return best_result

