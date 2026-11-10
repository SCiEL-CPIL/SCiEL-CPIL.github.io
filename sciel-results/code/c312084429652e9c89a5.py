import random
import time
import math
import itertools

def construct():
    n = 18
    
    # Precompute all triples as masks for fast triangle detection?
    # Actually, we just need to check K4-free and compute triangles.
    
    def is_k4_free(adj, u, v):
        # Check if adding edge (u, v) creates a K4.
        # A K4 is formed if there is a common neighbor w such that w is connected to some other common neighbor?
        # No, K4 is u, v, w, x.
        # Edges: (u,v), (u,w), (u,x), (v,w), (v,x), (w,x).
        # If we add (u,v), we need to check if there exists w, x such that
        # w, x are common neighbors of u and v, AND (w,x) is an edge.
        common = adj[u] & adj[v]
        c = common
        while c:
            lsb = c & -c
            w = lsb.bit_length() - 1
            c ^= lsb
            # Check if w is connected to any other vertex in common
            # If (w & common) != 0, then there is an x in common connected to w.
            # Note: w is in common, so w & common includes w itself?
            # adj[w] has bit w? No, simple graph, no self loops.
            # So adj[w] & common checks neighbors of w in common.
            if (adj[w] & common) != 0:
                return False
        return True

    def get_triangles(adj):
        tris = []
        for i in range(n):
            for j in range(i+1, n):
                if not (adj[i] >> j) & 1:
                    continue
                common = adj[i] & adj[j]
                c = common
                while c:
                    lsb = c & -c
                    k = lsb.bit_length() - 1
                    c ^= lsb
                    # Triangle i, j, k
                    # Ensure unique representation? Mask is unique.
                    # We don't need to deduplicate if we use masks in a set?
                    # But list is fine if we generate each triangle once.
                    # i < j. k can be < i, between, or > j.
                    # The mask (1<<i)|(1<<j)|(1<<k) is the same regardless of order.
                    # So we will generate each triangle exactly once?
                    # If we iterate i<j, and k from common, we might generate (i,j,k) and (k,i,j) if k<i?
                    # No, the loop is i, j. k is just a neighbor.
                    # The mask is the same.
                    # So we might have duplicates in the list if we are not careful?
                    # Example: Triangle 0,1,2.
                    # i=0, j=1. common has 2. k=2. Mask 0,1,2.
                    # i=0, j=2. common has 1. k=1. Mask 0,2,1 -> same mask.
                    # So yes, duplicates.
                    # We should use a set or ensure i<j<k.
                    if k > j:
                        tris.append((1 << i) | (1 << j) | (1 << k))
                    elif k < i:
                        # This triangle was generated when i'=k, j'=i?
                        # If k < i, then when we iterated i'=k, j'=i, k'=j.
                        # Since k < i < j, k' > j'. So it was added then.
                        # So we skip here to avoid duplicates.
                        pass
                    else:
                        # i < k < j
                        # Generated when i'=i, j'=k, k'=j?
                        # i < k < j.
                        # When i'=i, j'=k. common has j. k'=j. j > k. So added.
                        # So we skip here.
                        pass
        return tris

    def get_heuristic_score(adj, tris):
        # Greedy vertex cover of triangles
        if not tris:
            return n
        # Count incidence of each vertex in triangles
        incidence = [0] * n
        for t in tris:
            # t is a mask with 3 bits
            m = t
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                incidence[v] += 1
                m ^= lsb
        
        cover_size = 0
        remaining_tris = list(tris)
        # Simple greedy: pick vertex with max incidence, remove all its triangles
        # This is O(K^2) worst case, but K is small.
        # To speed up, we can just iterate and mark.
        # Actually, just picking max incidence and removing is fine for K < 500.
        
        # Better: Just estimate.
        # Lower bound: ceil(sum(incidence) / 3) ? No, sum(incidence) = 3 * K.
        # So ceil(K / max_incidence) is a lower bound.
        max_inc = max(incidence) if incidence else 0
        if max_inc == 0:
            return n
        lb = math.ceil(len(tris) / max_inc)
        
        # Heuristic: Assume we can cover with roughly K / avg_incidence?
        # Let's just return n - lb as an optimistic score?
        # Or n - greedy_cover.
        
        # Let's do a quick greedy cover
        tris_set = set(tris)
        covered = 0
        # To avoid O(K^2), we can just do a few steps or use the bound.
        # For SA, we just need a relative measure.
        # Let's use the lower bound as the score estimate?
        # Score = n - tau. Minimize Score => Maximize tau.
        # tau >= lb.
        # So Score <= n - lb.
        # We want to minimize Score. So we want to maximize lb.
        # So we can use -lb as the cost to minimize?
        # Or just use n - lb as the score.
        
        return n - lb

    def get_exact_score(adj, tris):
        if not tris:
            return n
        
        # Check s from n down to 1
        # Optimization: Precompute masks for each size?
        # Just use combinations.
        for s in range(n, 0, -1):
            for comb in itertools.combinations(range(n), s):
                mask = 0
                for v in comb:
                    mask |= (1 << v)
                is_tf = True
                for t in tris:
                    if (mask & t) == t:
                        is_tf = False
                        break
                if is_tf:
                    return s
        return 0

    # Initialize
    # Start with K_{6,6,6}
    adj = [0] * n
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    for i in range(3):
        for j in range(i+1, 3):
            for u in parts[i]:
                for v in parts[j]:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
    
    tris = get_triangles(adj)
    current_score = get_exact_score(adj, tris)
    best_score = current_score
    best_adj = adj[:]
    
    start_time = time.time()
    timeout = 110
    
    T = 10.0
    cooling = 0.995
    
    while time.time() - start_time < timeout:
        u, v = random.sample(range(n), 2)
        edge_exists = (adj[u] >> v) & 1
        
        new_adj = adj[:]
        if edge_exists:
            new_adj[u] &= ~(1 << v)
            new_adj[v] &= ~(1 << u)
        else:
            if not is_k4_free(adj, u, v):
                continue
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            
        new_tris = get_triangles(new_adj)
        
        # Use heuristic for speed
        new_heur_score = get_heuristic_score(new_adj, new_tris)
        curr_heur_score = get_heuristic_score(adj, tris) # Cache this?
        
        # To save time, only compute exact score if heuristic suggests improvement or periodically
        # Let's compute exact score every 10 steps or if heuristic improves.
        
        # For now, let's just use heuristic for acceptance, and track best exact.
        
        # Actually, computing exact score is the bottleneck.
        # Let's try to compute exact score only if the heuristic score is better than current best heuristic?
        # Or just accept based on heuristic, and update best exact periodically.
        
        # Let's compute exact score for the new candidate if it's "promising"
        # Or just compute it always but optimize get_exact_score?
        
        # Let's try a hybrid:
        # If new_heur_score < current_heur_score:
        #    new_exact = get_exact_score(new_adj, new_tris)
        #    if new_exact < best_score:
        #        best_score = new_exact
        #        best_adj = new_adj[:]
        #    # Accept move if new_exact < current_exact or SA condition
        #    # But we don't have current_exact cached.
        
        # Let's just run SA on the heuristic score, and at the end, verify the best_adj.
        # But we need to return the edges of the best graph found.
        
        # Let's accept based on heuristic.
        delta = new_heur_score - curr_heur_score
        if delta < 0 or random.random() < math.exp(-delta / T):
            adj = new_adj
            tris = new_tris
            # Update current heuristic?
            
        # Periodically check exact score
        if random.random() < 0.05:
            exact_s = get_exact_score(adj, tris)
            if exact_s < best_score:
                best_score = exact_s
                best_adj = adj[:]
        
        T *= cooling
        
    # Final check
    exact_s = get_exact_score(best_adj, get_triangles(best_adj))
    if exact_s < best_score:
        best_score = exact_s
        best_adj = best_adj[:] # already is
        
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
                
    return edges
