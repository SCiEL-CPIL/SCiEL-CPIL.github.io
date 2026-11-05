import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n = 18
    
    def get_edges_from_adj_mask(adj_mask):
        edges = []
        for i in range(n):
            m = adj_mask[i]
            while m:
                lsb = m & (-m)
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                m ^= lsb
        return edges
    
    def is_k4_free(adj_mask):
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj_mask[i] & (1 << j)):
                    continue
                common = adj_mask[i] & adj_mask[j] & ~((1 << i) | (1 << j))
                cm = common
                while cm:
                    lsb = cm & (-cm)
                    k = lsb.bit_length() - 1
                    cm ^= lsb
                    rest = common & (adj_mask[k] & ~((1 << i) | (1 << j) | (1 << k)))
                    while rest:
                        lsb2 = rest & (-rest)
                        l = lsb2.bit_length() - 1
                        rest ^= lsb2
                        if adj_mask[k] & (1 << l):
                            return False
        return True
    
    def max_triangle_free_size(adj_mask):
        """Exact branch and bound for max triangle-free induced subgraph size."""
        best = 0
        
        def bt(remaining_mask, current_size, last_vertex):
            nonlocal best
            if current_size >= best:
                # Try to extend
                pass
            
            if not remaining_mask:
                if current_size > best:
                    best = current_size
                return
            
            # Upper bound: current_size + popcount(remaining_mask)
            if current_size + bin(remaining_mask).count('1') <= best:
                return
            
            # Choose next vertex to try including
            # Heuristic: pick vertex with fewest conflicts
            m = remaining_mask
            best_v = -1
            best_conflicts = -1
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                m ^= lsb
                # conflicts = neighbors of v in remaining that would create triangle
                # Actually for triangle-free: if we add v, no two neighbors of v in current set can be adjacent
                # But we're building incrementally, so we need to check against current set
                # Simpler: just try adding v and check if still triangle-free
                if best_v == -1 or True:
                    best_v = v
                    break
            
            # Try including best_v
            # Try both: include and exclude
            # Include
            v_mask = 1 << best_v
            new_remaining = remaining_mask & ~v_mask
            # Remove vertices that would form triangle with best_v and any vertex in current set
            # Since we don't track current set, let's use a different formulation
            
            # Alternative: iterate over all subsets via branch and bound on complement
            # Actually, let me use a cleaner recursive approach
            
            bt_exclude(remaining_mask & ~v_mask, current_size)
            # Include best_v: need to remove from remaining any vertex that forms a triangle with best_v and another vertex
            # A vertex w in remaining forms a triangle with best_v and some u in remaining iff u-w edge and u-best_v edge
            # So for each u in remaining that is neighbor of best_v, remove neighbors of u from remaining
            new_rem2 = new_remaining
            for u_bit in bin(new_remaining):
                pass
            # This is getting complex. Let me use a simpler but effective approach.
            
            bt_exclude(remaining_mask, current_size + 1)  # placeholder, will fix
        
        # Let me rewrite with a cleaner approach
        return _max_tf_exact(adj_mask)
    
    def _max_tf_exact(adj_mask):
        """Maximum size of triangle-free induced subgraph using branch and bound."""
        best = [0]
        
        # Precompute for each pair (u,v) the common neighbors mask
        # For triangle-freeness: a set S is triangle-free iff for every edge (u,v) in S, 
        # no w in S is adjacent to both u and v
        
        def popcount(x):
            return bin(x).count('1')
        
        def bt(candidates, selected_mask, size):
            if size > best[0]:
                best[0] = size
            
            if not candidates:
                return
            
            # Upper bound
            if size + popcount(candidates) <= best[0]:
                return
            
            # Pick a vertex to branch on
            # Heuristic: vertex with highest degree in candidate-induced subgraph
            m = candidates
            best_v = -1
            best_deg = -1
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                m ^= lsb
                deg = popcount(adj_mask[v] & candidates)
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            
            v = best_v
            v_bit = 1 << v
            
            # Option 1: exclude v
            bt(candidates & ~v_bit, selected_mask, size)
            
            # Option 2: include v
            # After including v, we must exclude any pair of neighbors of v that are adjacent to each other
            # More precisely: if u1, u2 are both in candidates, both adjacent to v, and u1-u2 is an edge,
            # then we can't have both u1 and u2 in the final set along with v
            # So when we include v, for each edge (u1,u2) in the subgraph induced by N(v) ∩ candidates,
            # we must exclude at least one of u1, u2
            
            new_cand = candidates & ~v_bit
            # Remove conflicts: for each neighbor u of v in new_cand, 
            # remove from new_cand all neighbors of u that are also neighbors of v
            # Actually: the set of vertices that can coexist with v is those that don't form a triangle with v
            # A vertex w cannot be in the set with v if there exists u in the set such that u-v and u-w edges exist
            # This is complex. Simpler: just try including v and then prune candidates.
            
            # When v is included, any two vertices u1, u2 in the remaining set that are both 
            # neighbors of v and adjacent to each other form a triangle. So we need to ensure
            # the remaining set has no edges among N(v) ∩ remaining.
            # This means: from new_cand, we need to remove vertices to make N(v) ∩ new_cand an independent set.
            
            # For now, just do: include v, and remove from candidates all neighbors of v that have 
            # neighbors also in candidates∩N(v)
            # This is a safe but aggressive pruning
            
            # Simpler safe approach: when including v, remove from candidates any vertex w such that
            # w is adjacent to v AND there exists another vertex u in candidates (u≠w) adjacent to both v and w
            # i.e., w is in a triangle with v
            # But that's not right either - w is only problematic if paired with specific u's
            
            # Let me just do the straightforward thing: include v, and for each pair of neighbors
            # of v that are adjacent, we handle it in subsequent recursion
            # This might be slow but correct
            
            bt(new_cand, selected_mask | v_bit, size + 1)
        
        full_mask = (1 << n) - 1
        bt(full_mask, 0, 0)
        return best[0]
    
    def _max_tf_exact_fast(adj_mask):
        """Faster: use the fact that triangle-free means no K3 induced."""
        best = [0]
        
        def popcount(x):
            return bin(x).count('1')
        
        def bt(candidates, size):
            if size > best[0]:
                best[0] = size
            
            if not candidates:
                return
            
            if size + popcount(candidates) <= best[0]:
                return
            
            # Pick vertex with max degree in candidate subgraph
            m = candidates
            best_v = -1
            best_deg = -1
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                m ^= lsb
                deg = popcount(adj_mask[v] & candidates)
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            
            v = best_v
            v_bit = 1 << v
            
            # Exclude v
            bt(candidates & ~v_bit, size)
            
            # Include v: remove from candidates all vertices that form a triangle with v and some other candidate
            # A vertex w in candidates forms a triangle with v iff w is neighbor of v and 
            # there exists u in candidates\{w} that is neighbor of both v and w
            # Safe pruning: remove all common neighbors of v and any other neighbor of v
            # i.e., if u1, u2 are both neighbors of v and u1-u2 edge, then including v means 
            # we can include at most one of u1, u2
            
            # Aggressive but safe: when including v, for each neighbor u of v in candidates,
            # remove from candidates all neighbors of u (within candidates) that are also neighbors of v
            new_cand = candidates & ~v_bit
            changed = True
            while changed:
                changed = False
                m = new_cand
                while m:
                    lsb = m & (-m)
                    u = lsb.bit_length() - 1
                    m ^= lsb
                    # Common neighbors of v and u in new_cand
                    common = adj_mask[v] & adj_mask[u] & new_cand & ~v_bit & ~(1 << u)
                    if common:
                        # We can't include all of common along with v and u
                        # Remove u's other neighbors that are also v's neighbors
                        # Actually this is getting too complex. Just remove common from new_cand
                        # to ensure no triangle with v
                        new_cand &= ~common
                        changed = True
            
            bt(new_cand, size + 1)
        
        full_mask = (1 << n) - 1
        bt(full_mask, 0)
        return best[0]
    
    def _max_tf_exact_v2(adj_mask):
        """Cleaner exact branch and bound."""
        best = [0]
        
        def popcount(x):
            return bin(x).count('1')
        
        def bt(candidates, size):
            if size > best[0]:
                best[0] = size
            if not candidates or size + popcount(candidates) <= best[0]:
                return
            
            # Choose vertex to branch on: max degree in induced subgraph
            m = candidates
            chosen = -1
            max_deg = -1
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                m ^= lsb
                d = popcount(adj_mask[v] & candidates)
                if d > max_deg:
                    max_deg = d
                    chosen = v
            
            vb = 1 << chosen
            
            # Branch 1: don't include chosen
            bt(candidates & ~vb, size)
            
            # Branch 2: include chosen
            # Remove from candidates any vertex that would create a triangle with chosen
            # and some other vertex in the set
            # Safe removal: for any edge (u,w) in candidates where both u,w are neighbors of chosen,
            # we can keep at most one. To be safe, remove all neighbors of chosen's neighbors that 
            # are also neighbors of chosen... 
            # Simplest safe approach: just remove chosen and let recursion handle it
            # But that's not pruning enough.
            
            # Better: when we include chosen, the remaining set must be such that 
            # N(chosen) ∩ remaining is an independent set in the original graph
            # So from candidates\{chosen}, remove vertices to make N(chosen)∩remaining independent
            # Greedy: iteratively, if u1,u2 in N(chosen)∩remaining are adjacent, remove one
            
            rem = candidates & ~vb
            # Make N(chosen) ∩ rem an independent set
            nmask = adj_mask[chosen] & rem
            # Find edges within nmask and remove vertices
            changed = True
            while changed:
                changed = False
                nm = nmask
                while nm:
                    lsb1 = nm & (-nm)
                    u1 = lsb1.bit_length() - 1
                    nm ^= lsb1
                    # Check neighbors of u1 within nmask
                    neigh_in_nmask = adj_mask[u1] & nmask
                    while neigh_in_nmask:
                        lsb2 = neigh_in_nmask & (-neigh_in_nmask)
                        u2 = lsb2.bit_length() - 1
                        # u1-u2 edge, both neighbors of chosen: can't have both
                        # Remove u2 (or u1) from rem
                        rem &= ~(1 << u2)
                        nmask &= ~(1 << u2)
                        changed = True
                        break
                    if changed:
                        break
                if changed:
                    nmask = adj_mask[chosen] & rem
            
            bt(rem, size + 1)
        
        full_mask = (1 << n) - 1
        bt(full_mask, 0)
        return best[0]
    
    def _max_tf_exact_v3(adj_mask):
        """Most careful implementation."""
        best = [0]
        
        def popcount(x):
            return bin(x).count('1')
        
        def is_tf_subset(mask):
            """Check if mask is triangle-free."""
            m = mask
            while m:
                lsb = m & (-m)
                u = lsb.bit_length() - 1
                m ^= lsb
                # For each neighbor v of u in mask with v > u
                nv = adj_mask[u] & mask & ~((1 << (u + 1)) - 1)  # vertices > u
                while nv:
                    lsb2 = nv & (-nv)
                    v = lsb2.bit_length() - 1
                    nv ^= lsb2
                    # Check common neighbors of u and v in mask
                    if adj_mask[u] & adj_mask[v] & mask:
                        return False
            return True
        
        def bt(candidates, size, current_mask):
            if size > best[0]:
                best[0] = size
            if not candidates or size + popcount(candidates) <= best[0]:
                return
            
            m = candidates
            chosen = -1
            max_deg = -1
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                m ^= lsb
                d = popcount(adj_mask[v] & candidates)
                if d > max_deg:
                    max_deg = d
                    chosen = v
            
            vb = 1 << chosen
            
            # Exclude
            bt(candidates & ~vb, size, current_mask)
            
            # Include: verify triangle-free with current_mask
            new_mask = current_mask | vb
            if is_tf_subset(new_mask):
                bt(candidates & ~vb, size + 1, new_mask)
        
        full_mask = (1 << n) - 1
        bt(full_mask, 0, 0)
        return best[0]
    
    # Strategy: Use structured construction + local search
    
    def build_structured(p_between):
        """Build graph with 6 groups of 3, triangles within groups."""
        adj_mask = [0] * n
        groups = [[3*i, 3*i+1, 3*i+2] for i in range(6)]
        
        # Within each group: complete triangle
        for g in groups:
            for i in range(3):
                for j in range(i+1, 3):
                    u, v = g[i], g[j]
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
        
        # Between groups: random edges with probability p_between
        for gi in range(6):
            for gj in range(gi+1, 6):
                for u in groups[gi]:
                    for v in groups[gj]:
                        if random.random() < p_between:
                            adj_mask[u] |= (1 << v)
                            adj_mask[v] |= (1 << u)
        
        return adj_mask
    
    def remove_k4s(adj_mask):
        """Remove edges to eliminate all K4s."""
        adj = list(adj_mask)
        for _ in range(100):
            # Find a K4
            found = None
            for i in range(n):
                for j in range(i+1, n):
                    if not (adj[i] & (1 << j)):
                        continue
                    common = adj[i] & adj[j] & ~((1 << i) | (1 << j))
                    cm = common
                    while cm:
                        lsb = cm & (-cm)
                        k = lsb.bit_length() - 1
                        cm ^= lsb
                        rest = common & adj[k] & ~((1 << i) | (1 << j) | (1 << k))
                        while rest:
                            lsb2 = rest & (-rest)
                            l = lsb2.bit_length() - 1
                            rest ^= lsb2
                            if adj[k] & (1 << l):
                                found = (i, j, k, l)
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                break
            
            # Remove a random edge from the K4
            i, j, k, l = found
            edges_in_k4 = [(i,j), (i,k), (i,l), (j,k), (j,l), (k,l)]
            u, v = random.choice(edges_in_k4)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        
        return adj
    
    def local_search(adj_mask, max_iter=2000):
        """Local search: try flipping edges to improve score."""
        adj = list(adj_mask)
        current_score = _max_tf_exact_v2(adj)
        best_score = current_score
        best_adj = list(adj)
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Try adding a random non-edge
            i = random.randrange(n)
            j = random.randrange(n)
            if i == j or (adj[i] & (1 << j)):
                continue
            
            new_adj = list(adj)
            new_adj[i] |= (1 << j)
            new_adj[j] |= (1 << i)
            
            if is_k4_free(new_adj):
                s = _max_tf_exact_v2(new_adj)
                if s < current_score:
                    adj = new_adj
                    current_score = s
                    if s < best_score:
                        best_score = s
                        best_adj = list(adj)
                    continue
            
            # Try removing a random edge
            i = random.randrange(n)
            m = adj[i]
            if not m:
                continue
            lsb = m & (-m)
            j = lsb.bit_length() - 1
            
            new_adj = list(adj)
            new_adj[i] &= ~(1 << j)
            new_adj[j] &= ~(1 << i)
            
            s = _max_tf_exact_v2(new_adj)
            if s < current_score:
                adj = new_adj
                current_score = s
                if s < best_score:
                    best_score = s
                    best_adj = list(adj)
        
        return best_adj, best_score
    
    # Main: try multiple constructions
    best_overall = 18
    best_adj_overall = [0] * n
    
    for p in [0.3, 0.4, 0.5, 0.35, 0.45, 0.25]:
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            adj = build_structured(p)
            adj = remove_k4s(adj)
            
            if not is_k4_free(adj):
                continue
            
            adj_ls, score_ls = local_search(adj, max_iter=500)
            
            if score_ls < best_overall:
                best_overall = score_ls
                best_adj_overall = adj_ls
                if best_overall <= 8:
                    break
        
        if best_overall <= 8:
            break
    
    return get_edges_from_adj_mask(best_adj_overall)

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
    n = 18
    adj = [[0]*n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = 1
        adj[v][u] = 1
    
    # Verify K4-free
    k4_found = False
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                for k in range(j+1, n):
                    if adj[i][k] and adj[j][k]:
                        for l in range(k+1, n):
                            if adj[i][l] and adj[j][l] and adj[k][l]:
                                print(f"K4 found: {i},{j},{k},{l}")
                                k4_found = True
    if not k4_found:
        print("K4-free check passed")
    
    # Find max triangle-free induced subgraph
    adj_mask = [0]*n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    # Simple check
    from itertools import combinations
    max_tf = 0
    for k in range(18, 0, -1):
        found = False
        for subset in combinations(range(18), k):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            # Check triangle-free
            tf = True
            for i in range(3):
                for j in range(i+1, 3):
                    for l in range(j+1, 3):
                        u, v, w = subset[i], subset[j], subset[l]
                        if adj_mask[u] & (1<<v) and adj_mask[v] & (1<<w) and adj_mask[u] & (1<<w):
                            tf = False
                            break
                    if not tf:
                        break
                if not tf:
                    break
            if tf:
                max_tf = k
                found = True
                break
        if found:
            break
    print(f"Max triangle-free induced subgraph size: {max_tf}")
