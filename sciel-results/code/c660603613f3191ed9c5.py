import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # We want to find a graph with no K4 and no Ind 6.
    # Let's try to construct for n=35.
    # If we fail, we try n=34, etc.
    
    # We use a randomized backtracking search.
    # State: adj list of bitmasks.
    # We add vertices 0 to n-1.
    
    def solve(n, seed=0):
        random.seed(seed)
        adj = [0] * n
        # adj[i] is bitmask of neighbors of i among 0..n-1
        # We build incrementally.
        
        # To speed up, we can use a greedy heuristic to initialize
        # or just use backtracking with random ordering of choices.
        
        # We will try to assign adjacency for vertex k to vertices 0..k-1.
        # We need to check:
        # 1. No K4: If k connects to u and v, and u-v is an edge, 
        #    then there must be no w such that w is connected to u, v, and k.
        #    Actually, if we add edges (k,u) and (k,v), and (u,v) exists,
        #    we create a triangle (k,u,v). This is fine.
        #    It becomes a K4 if there is another vertex w connected to k, u, v.
        #    So, when adding edge (k, u), we must ensure that for any v connected to u,
        #    if k is connected to v, then there is no w connected to k, u, v.
        #    This check is local if we process carefully.
        
        # Better check:
        # When setting adjacency of k, let S be the set of neighbors of k among 0..k-1.
        # 1. K4 Check: For any pair u, v in S, if u-v is an edge, then (adj[u] & adj[v] & adj[k]) must be 0?
        #    No. If u, v are in S, and u-v is edge, then {k, u, v} is a triangle.
        #    A K4 is formed if there is a w in S such that w is connected to u and v.
        #    So, for any u, v in S with edge (u,v), we need (adj[u] & adj[v] & S_mask) to have at most 1 bit set?
        #    No, if there is a w in S connected to u and v, then {k, u, v, w} is a K4.
        #    So, for any triangle {u, v, w} in the subgraph induced by S, we have a K4 {k, u, v, w}.
        #    Therefore, the subgraph induced by S must be Triangle-Free (K3-free).
        #    Wait. If S induces a triangle u,v,w, and k is connected to all, then k,u,v,w is a K4.
        #    So, **The neighborhood of any vertex must be Triangle-Free.**
        #    This is a necessary and sufficient condition for K4-freeness?
        #    Yes. A graph is K4-free iff the neighborhood of every vertex is triangle-free.
        
        # 2. Ind 6 Check:
        #    Let T be the set of non-neighbors of k among 0..k-1.
        #    If T contains an independent set of size 5, then adding k creates an Ind 6.
        #    So, the subgraph induced by T must be Ind-5-free.
        #    Equivalently, the complement of the subgraph induced by T must be K5-free.
        
        # So the algorithm for adding vertex k:
        # Choose a subset S of {0..k-1} to be neighbors.
        # Check 1: The subgraph induced by S is Triangle-Free.
        # Check 2: The subgraph induced by T (complement of S) is Ind-5-free.
        
        # We can use backtracking to find such S.
        
        def is_triangle_free(mask, adj):
            # Check if subgraph induced by 'mask' is triangle free.
            # Iterate over edges in mask.
            # For each edge (u, v) in mask, check if (adj[u] & adj[v] & mask) == 0.
            # If any edge (u,v) in mask has a common neighbor in mask, it's a triangle.
            m = mask
            while m:
                # Pick lowest bit u
                lsb = m & -m
                u = lsb.bit_length() - 1
                m -= lsb
                # Neighbors of u in mask
                nu = adj[u] & mask
                # For each neighbor v of u in mask
                nv = nu
                while nv:
                    lsb_v = nv & -nv
                    v = lsb_v.bit_length() - 1
                    nv -= lsb_v
                    if v < u: continue # Avoid double checking? No, mask is static.
                    # Check if u and v have common neighbor in mask
                    # Common neighbors of u and v in mask: adj[u] & adj[v] & mask
                    # If this is not 0, there is a triangle.
                    if (adj[u] & adj[v] & mask):
                        return False
            return True

        def has_ind_5(mask, adj):
            # Check if subgraph induced by 'mask' has an independent set of size 5.
            # Equivalent to: Does complement of G[mask] have a K5?
            # Let's use a simple backtracking for K5 in complement.
            # Or, since n is small (<=35), we can check all 5-subsets?
            # Number of 5-subsets of 34 is ~300k. Too slow for every step.
            # We need a faster check.
            # Let's use a greedy clique finding in complement.
            # If max clique in complement of G[mask] is >= 5, return True.
            
            # We can use a simple recursive clique finder with pruning.
            # Vertices in mask.
            verts = [i for i in range(k) if (mask >> i) & 1]
            if len(verts) < 5:
                return False
            
            # Complement adjacency: u and v are connected in complement if NOT connected in G.
            # comp_adj[i] = ~adj[i] & mask
            # We want to find a clique of size 5 in comp_adj.
            
            def find_clique_size(candidates, size):
                if size == 0:
                    return True
                if len(candidates) < size:
                    return False
                # Greedy: pick first vertex, intersect with neighbors in complement
                # To be fast, we can just try to find *any* clique of size 5.
                # Use Bron-Kerbosch variant?
                
                # Simple heuristic:
                # If the number of edges in complement is too low, no clique.
                # Let's just do a DFS.
                
                # For efficiency, we can limit the search.
                # If we just need to know if size 5 exists.
                
                # Let's use a simple recursive check.
                # Sort candidates by degree in complement?
                
                # To keep it simple and fast enough for n=35:
                # We can check if the complement is K5-free.
                # If the complement has fewer than 5 vertices, False.
                
                # Let's try a simple backtracking for K5.
                # This might be the bottleneck.
                
                # Optimization: If the maximum degree in complement is < 4, no K5.
                # Calculate degrees in complement for vertices in mask.
                
                # Let's just use a standard clique algorithm for small graphs.
                # Since mask size is at most 34, and we do this for each k,
                # it might be slow.
                
                # Alternative: Precompute? No, adj changes.
                
                # Let's try a randomized greedy check first?
                # No, must be exact.
                
                # Let's use the property: Ind 5 in G is K5 in G_bar.
                # We can use a bitset-based clique finder.
                
                # Let's implement a simple recursive clique finder.
                # To speed up, we can stop as soon as we find one.
                
                def _dfs_clique(cand_mask, size_needed):
                    if size_needed == 0:
                        return True
                    if cand_mask.bit_count() < size_needed:
                        return False
                    
                    # Pick a vertex from cand_mask
                    # Heuristic: pick vertex with highest degree in complement within cand_mask
                    # Degree in complement = (cand_mask & ~adj[u]).bit_count()
                    
                    # To avoid complexity, just pick lowest bit?
                    # Or random?
                    
                    # Let's pick lowest bit for determinism in this helper, 
                    # but the outer loop is randomized.
                    
                    # Actually, for pruning, we should pick a vertex.
                    # If we pick u, we must recurse on (cand_mask & ~adj[u] & ~lsb_u)
                    
                    # Let's iterate through vertices in cand_mask.
                    m = cand_mask
                    found = False
                    while m and not found:
                        lsb = m & -m
                        u = lsb.bit_length() - 1
                        m -= lsb
                        
                        # Neighbors of u in complement within cand_mask
                        # ~adj[u] & cand_mask
                        # Note: adj[u] contains u? No, usually not.
                        # We ensure adj[u] doesn't have bit u.
                        
                        next_cand = cand_mask & ~adj[u] & ~lsb
                        if _dfs_clique(next_cand, size_needed - 1):
                            return True
                    return False
                
                return _dfs_clique(mask, 5)

        # Backtracking function
        def backtrack(k):
            if k == n:
                return True
            
            # Try all subsets S of {0..k-1}?
            # 2^k is too large.
            # We need a smarter search.
            # We can fix the order of adding edges?
            
            # Alternative: Randomized Greedy with Local Search is often better for these.
            # Let's stick to the previous inspiration: Local Search.
            # But with Bitmasks.
            
            return False
            
        # The backtracking approach is tricky to implement efficiently in Python for n=35.
        # Let's switch to **Simulated Annealing** with Bitmasks.
        # It is much easier to implement and robust.
        
        return False

    # Let's implement Simulated Annealing properly.
    
    def run_sa(n, seed=42, steps=50000):
        random.seed(seed)
        adj = [0] * n
        
        # Initialize randomly
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        def count_k4():
            count = 0
            for u in range(n):
                for v in range(u+1, n):
                    if (adj[u] >> v) & 1:
                        common = adj[u] & adj[v]
                        # For each common neighbor w, check if u-v-w is triangle?
                        # No, common neighbors of u and v form triangles with u,v.
                        # For each w in common, we need to check if w is connected to any other common neighbor?
                        # No. A K4 is u,v,w,x where all connected.
                        # If we have triangle u,v,w, then any x in (adj[u] & adj[v] & adj[w]) forms a K4.
                        # So for each triangle, count common neighbors.
                        
                        # Let's iterate triangles.
                        # This is O(n^3) per call. Too slow for 50k steps.
                        
                        # We need incremental updates or a faster count.
                        pass
            return 0

        # Faster K4 count:
        # K4 count = 1/4 * sum_{triangles (u,v,w)} popcount(adj[u] & adj[v] & adj[w])
        
        def count_k4_fast():
            total = 0
            for u in range(n):
                # Iterate neighbors v of u
                m = adj[u]
                while m:
                    lsb_v = m & -m
                    v = lsb_v.bit_length() - 1
                    m -= lsb_v
                    if v < u: continue
                    
                    # Iterate neighbors w of v, where w > v? No, just w in adj[v]
                    # We want triangles u,v,w.
                    # Common neighbors of u and v: adj[u] & adj[v]
                    common = adj[u] & adj[v]
                    # For each w in common, we have triangle u,v,w.
                    # We need to add popcount(adj[u] & adj[v] & adj[w])
                    # But this counts each K4 4 times.
                    
                    # To avoid double counting triangles, we can restrict w > v?
                    # If we restrict w > v, we count each triangle once.
                    # But adj[u] & adj[v] includes all common neighbors.
                    # Let's filter common to w > v.
                    common = common & ~((1 << (v+1)) - 1) # Mask out <= v
                    # Actually, just iterate common.
                    
                    cw = common
                    while cw:
                        lsb_w = cw & -cw
                        w = lsb_w.bit_length() - 1
                        cw -= lsb_w
                        
                        # Triangle u,v,w exists.
                        # Count K4s extending this triangle.
                        # Common neighbors of u,v,w:
                        c = adj[u] & adj[v] & adj[w]
                        total += c.bit_count()
                        
            return total // 4

        def count_ind_6():
            # Count K6 in complement.
            # Complement adj: c_adj[i] = (~adj[i]) & ((1<<n)-1)
            # K6 count in c_adj.
            # Same logic as count_k4_fast but on c_adj.
            
            c_adj = [0] * n
            full_mask = (1 << n) - 1
            for i in range(n):
                c_adj[i] = (~adj[i]) & full_mask
                # Remove self
                c_adj[i] &= ~(1 << i)
            
            total = 0
            for u in range(n):
                m = c_adj[u]
                while m:
                    lsb_v = m & -m
                    v = lsb_v.bit_length() - 1
                    m -= lsb_v
                    if v < u: continue
                    
                    common = c_adj[u] & c_adj[v]
                    common = common & ~((1 << (v+1)) - 1)
                    
                    cw = common
                    while cw:
                        lsb_w = cw & -cw
                        w = lsb_w.bit_length() - 1
                        cw -= lsb_w
                        
                        # Triangle u,v,w in complement.
                        c = c_adj[u] & c_adj[v] & c_adj[w]
                        total += c.bit_count()
                        
            return total // 4 # Wait, K6 has 20 triangles? No.
            # A K6 contains C(6,3) = 20 triangles.
            # Each triangle in K6 is counted.
            # For a K6, sum over its 20 triangles of popcount(common neighbors in K6)
            # For a specific triangle in K6, the other 3 vertices are common neighbors.
            # So sum is 20 * 3 = 60.
            # 60 / 4 = 15? No.
            # My formula was for K4.
            # For K_k, each K_k has C(k,3) triangles.
            # For each triangle, there are k-3 common neighbors.
            # Total sum = C(k,3) * (k-3).
            # We divide by C(k,3) to get number of K_k?
            # No, we sum over ALL triangles in the graph.
            # If we find a K6, it contributes 20 * 3 = 60 to the sum.
            # We want to count 1 K6.
            # So we divide by 60?
            # Let's check K4. C(4,3)=4 triangles. k-3=1. Sum=4. Divide by 4. Correct.
            # So for K6, divide by 60.
            
            return total // 60

        # This is still O(n^3) per evaluation.
        # 50000 steps * 35^3 ~ 200M ops. Might be too slow in Python.
        
        # We need to update counts incrementally.
        # When flipping edge (u,v):
        # 1. Update adj[u], adj[v].
        # 2. Recalculate K4 count?
        #    Only K4s involving u or v change.
        #    K4s not involving u or v are unchanged.
        #    This is complex.
        
        # Given the time limit, a simpler approach might be better:
        # **Greedy Construction with Backtracking on Vertex Addition**
        # using the "Neighborhood is Triangle-Free" and "Non-neighborhood is Ind-5-Free" checks.
        # We can optimize the Ind-5 check using a precomputed table or fast bit ops.
        
        # Let's try the Vertex Addition Backtracking again, but with a good heuristic.
        
        def construct_vertex_addition(n, seed=0):
            random.seed(seed)
            adj = [0] * n
            # adj[i] is bitmask of neighbors.
            
            # We add vertices 0..n-1.
            # For vertex k, we choose S (neighbors in 0..k-1).
            # Constraints:
            # 1. G[S] is Triangle-Free.
            # 2. G[T] is Ind-5-Free (where T is complement of S in 0..k-1).
            
            # We will use backtracking to choose S.
            # To make it fast, we can fix the order of edges?
            # No, we choose S as a whole.
            
            # Heuristic: Start with empty S, add edges one by one?
            # Or start with full S, remove edges?
            
            # Let's try: For each k, try all 2^k subsets? No.
            # Use a randomized greedy:
            # For each k, pick a random S that satisfies constraints?
            # If stuck, backtrack.
            
            # This is essentially a CSP.
            
            # Let's use a simple recursive backtracking.
            # State: k, adj[0..k-1].
            # We need to pick adj[k] (mask of neighbors in 0..k-1).
            
            def check_valid_k(k, mask):
                # mask is neighbors of k in 0..k-1.
                # 1. G[mask] is Triangle-Free.
                if not is_triangle_free(mask, adj):
                    return False
                # 2. G[~mask] is Ind-5-Free.
                T = ((1 << k) - 1) ^ mask
                if has_ind_5(T, adj):
                    return False
                return True
            
            def dfs(k):
                if k == n:
                    return True
                
                # Generate candidate masks for adj[k]
                # To speed up, we can use a heuristic to order candidates.
                # Or just iterate 0..2^k-1?
                # k is up to 34. 2^34 is too big.
                
                # We must use a smarter search.
                # We can build the mask bit by bit?
                # Let's try adding neighbors one by one.
                
                # This is getting complicated.
                
                # Let's fall back to **Simulated Annealing** but with **Incremental Updates** for K4 and Ind6 counts.
                # This is the most robust way to get a high score in the time limit.
                
                return False
            
            return dfs(0)
            
        # I will implement the Simulated Annealing with Incremental Updates.
        
        # ... (Implementation of SA with incremental updates) ...
        
        # For the sake of this response, I will provide a clean Simulated Annealing implementation
        # that uses bitmasks and accepts a slower count function if necessary,
        # but optimized with early termination.
        
        # Actually, the previous program scored 1.41.
        # The inspirations scored 1.7.
        # The key is likely **finding the n=35 graph**.
        
        # Let's try a **Greedy Heuristic** that is known to work for Ramsey graphs:
        # Start with a random graph, then iteratively remove edges from K4s and add edges to Ind6s.
        # This is what the seed did, but maybe it was too slow or poor randomization.
        
        # Let's use **Bitmask** for speed in the local search.
        
        adj = [0] * 35
        random.seed(42)
        for i in range(35):
            for j in range(i+1, 35):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        def count_violations(adj, n):
            k4_count = 0
            ind6_count = 0
            
            # K4 count
            for u in range(n):
                for v in range(u+1, n):
                    if (adj[u] >> v) & 1:
                        common = adj[u] & adj[v]
                        cw = common
                        while cw:
                            lsb = cw & -cw
                            w = lsb.bit_length() - 1
                            cw -= lsb
                            if w < v: continue
                            # Triangle u,v,w
                            c = adj[u] & adj[v] & adj[w]
                            k4_count += c.bit_count()
            k4_count //= 4
            
            # Ind6 count (K6 in complement)
            full_mask = (1 << n) - 1
            c_adj = [0] * n
            for i in range(n):
                c_adj[i] = (~adj[i]) & full_mask
                c_adj[i] &= ~(1 << i)
                
            for u in range(n):
                for v in range(u+1, n):
                    if (c_adj[u] >> v) & 1:
                        common = c_adj[u] & c_adj[v]
                        cw = common
                        while cw:
                            lsb = cw & -cw
                            w = lsb.bit_length() - 1
                            cw -= lsb
                            if w < v: continue
                            # Triangle u,v,w in complement
                            c = c_adj[u] & c_adj[v] & c_adj[w]
                            ind6_count += c.bit_count()
            ind6_count //= 60
            
            return k4_count + ind6_count

        # Local Search
        best_viol = count_violations(adj, 35)
        for i in range(5000):
            if time.time() > deadline - 10:
                break
            if best_viol == 0:
                break
            
            # Pick random edge to flip
            u = random.randint(0, 34)
            v = random.randint(0, 34)
            if u == v:
                continue
            
            # Flip
            if (adj[u] >> v) & 1:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
            new_viol = count_violations(adj, 35)
            
            # Accept if better or random (simulated annealing)
            # Simple: accept if better, else revert.
            if new_viol > best_viol:
                # Revert
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            else:
                best_viol = new_viol
        
        if best_viol == 0:
            edges = []
            for i in range(35):
                for j in range(i+1, 35):
                    if (adj[i] >> j) & 1:
                        edges.append([i, j])
            return {'n': 35, 'edges': edges}
            
        # Fallback to smaller n
        # ...
        
        return {'n': 1, 'edges': []}

    # The above SA might be too slow.
    # Let's use the **Vertex Addition** approach with a **Greedy Heuristic** for the mask selection.
    # This is more likely to find the exact solution quickly.
    
    # I will implement the Vertex Addition Backtracking with a **Randomized Greedy Mask Selection**.
    
    def solve_greedy_addition(n, seed=0):
        random.seed(seed)
        adj = [0] * n
        
        def is_tri_free(mask, adj):
            m = mask
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                m -= lsb
                nu = adj[u] & mask
                nv = nu
                while nv:
                    lsb_v = nv & -nv
                    v = lsb_v.bit_length() - 1
                    nv -= lsb_v
                    if (adj[u] & adj[v] & mask):
                        return False
            return True
            
        def has_ind_5(mask, adj):
            # Check if complement of G[mask] has K5.
            # Use a simple recursive clique finder.
            def _dfs(cand_mask, size):
                if size == 0: return True
                if cand_mask.bit_count() < size: return False
                m = cand_mask
                while m:
                    lsb = m & -m
                    u = lsb.bit_length() - 1
                    m -= lsb
                    # Neighbors in complement
                    next_cand = cand_mask & ~adj[u] & ~lsb
                    if _dfs(next_cand, size-1):
                        return True
                return False
            return _dfs(mask, 5)
            
        def dfs(k):
            if k == n:
                return True
            
            # Try to find a valid mask for adj[k]
            # We can try a few random masks, or iterate.
            # Let's try a heuristic: Start with a random mask, check, adjust.
            
            # For speed, we will just try a few random masks.
            # If we fail, we return False and let the outer loop retry with different seed.
            
            for _ in range(100):
                mask = random.getrandbits(k)
                if is_tri_free(mask, adj):
                    T = ((1 << k) - 1) ^ mask
                    if not has_ind_5(T, adj):
                        adj[k] = mask
                        if dfs(k+1):
                            return True
                        # Backtrack
                        adj[k] = 0
            return False
            
        return dfs(0)
        
    # Try n=35
    if solve_greedy_addition(35, 42):
        edges = []
        for i in range(35):
            for j in range(i+1, 35):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return {'n': 35, 'edges': edges}
        
    # Fallback
    return {'n': 1, 'edges': []}

