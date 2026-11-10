import time
import numpy as np

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # 1. Deterministic Seed: Paley Graph
    # The Paley graph P_q for prime power q has vertices Z_q.
    # Edge (u, v) exists if (u - v) is a quadratic residue mod q.
    # For q=23, n=20 is a subset. P_23 is known to be K_{3,3}-free?
    # Actually, Paley graphs are strongly regular. 
    # P_13, P_17, P_29...
    # Let's check properties. A Paley graph P_q is K_{3,3}-free if and only if 
    # there are no 3 distinct residues a,b,c such that a-b, a-c, b-c are all squares?
    # No, K_{3,3} in Paley graph corresponds to specific additive structures.
    # It is known that P_q is K_{t,t}-free for large t only if q is large enough.
    # For small q, we can just check.
    # Let's use P_29 restricted to 20 vertices? Or P_17?
    # Actually, a simpler deterministic construction is the incidence graph of a projective plane, 
    # but that's bipartite. We want general graphs.
    
    # Let's try a random restart strategy but with a very fast vectorized core.
    # The bottleneck in the previous code was Python loops for constraint checking.
    # We will use bitmasks (integers) for adjacency lists.
    
    def is_k33_free_mask(adj_masks, n):
        """
        Check if graph is K3,3 free using bitmasks.
        adj_masks[i] is an integer where bit j is 1 if (i,j) is an edge.
        Condition: For all triples i,j,k, popcount(adj[i] & adj[j] & adj[k]) < 3.
        """
        # This is O(n^3 * cost_of_popcount). 
        # n=20, n^3 = 8000. Very fast.
        for i in range(n):
            mi = adj_masks[i]
            for j in range(i+1, n):
                mij = mi & adj_masks[j]
                if mij == 0: continue
                for k in range(j+1, n):
                    if (mij & adj_masks[k]).bit_count() >= 3:
                        return False
        return True

    def can_add_mask(adj_masks, u, v, n):
        """
        Check if adding edge (u,v) violates K3,3.
        Only need to check triples involving u or v.
        Triples {u, a, b}: popcount((adj[u] | (1<<v)) & adj[a] & adj[b]) < 3
        Note: adj[u] doesn't contain u or v yet.
        New common neighbors of {u, a, b} = (old_common) U {v if v in adj[a] and v in adj[b]}
        So: popcount(old_common) + (1 if (adj[a] & adj[b] & (1<<v)) else 0) < 3
        """
        bit_v = 1 << v
        bit_u = 1 << u
        
        # Triples {u, a, b} with a < b, a,b != u,v
        # We iterate a from 0 to n-1, b from a+1 to n-1
        # Skip if a==u or b==u or a==v or b==v
        # To be efficient, we can just check all pairs a,b.
        # n=20, pairs ~ 190. Very fast.
        
        for a in range(n):
            if a == u or a == v: continue
            ma = adj_masks[a]
            for b in range(a+1, n):
                if b == u or b == v: continue
                mb = adj_masks[b]
                
                # Check triple {u, a, b}
                common_ab = ma & mb
                # Common neighbors of u, a, b in current graph:
                common_uab = adj_masks[u] & common_ab
                # Does v become a common neighbor?
                # v is neighbor of a and b?
                v_is_common = (common_ab & bit_v) != 0
                # New count = popcount(common_uab) + (1 if v_is_common else 0)
                # Note: u is not in common_uab because u is not neighbor of itself.
                # v is not in common_uab because v is not neighbor of itself (simple graph).
                # However, we are adding (u,v). 
                # The set of common neighbors of {u,a,b} *after* adding (u,v) is:
                # {x | x~u, x~a, x~b}
                # x can be v? v~u (new), v~a (old), v~b (old). So yes, if v~a and v~b.
                # x can be u? u~u (no).
                # So we just add 1 if v is in N(a) cap N(b).
                
                cnt = common_uab.bit_count()
                if v_is_common:
                    cnt += 1
                if cnt >= 3:
                    return False
                    
                # Check triple {v, a, b}
                common_vab = adj_masks[v] & common_ab
                u_is_common = (common_ab & bit_u) != 0
                cnt_v = common_vab.bit_count()
                if u_is_common:
                    cnt_v += 1
                if cnt_v >= 3:
                    return False
        return True

    best_edges = []
    best_count = 0
    
    # Strategy: 
    # 1. Generate several random initial graphs (dense).
    # 2. "Clean" them by removing edges that cause K3,3.
    # 3. Greedily add edges.
    # 4. Local search (swap).
    
    rng = np.random.default_rng(42)
    
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Initialize with a random graph
        # Try different densities
        p = 0.5 + 0.2 * rng.random()
        adj_masks = [0] * n
        edges = []
        
        for i in range(n):
            for j in range(i+1, n):
                if rng.random() < p:
                    adj_masks[i] |= (1 << j)
                    adj_masks[j] |= (1 << i)
                    edges.append((i, j))
                    
        # Clean phase: Remove edges to make it K3,3 free
        # We want to maximize edges, so we should remove edges that are in *many* K3,3s?
        # Or just remove the first one found?
        # Let's remove the edge (u,v) that minimizes the loss? 
        # Simpler: Just remove edges until valid.
        # To be smart: Iterate edges. If an edge is in a K3,3, remove it.
        # But which one? 
        # Heuristic: Remove edge (u,v) where u,v have highest degrees? No, that might break many things.
        # Remove edge (u,v) where the number of K3,3s containing (u,v) is highest?
        
        # For speed, let's just do a simple removal:
        # Find any K3,3. Remove one edge from it. Repeat.
        # To avoid getting stuck in bad local minima, we can randomize which edge to remove.
        
        changed = True
        while changed:
            if time.time() > deadline - 5:
                break
            changed = False
            # Find a K3,3
            found = False
            for u in range(n):
                if found: break
                mu = adj_masks[u]
                for v in range(u+1, n):
                    if found: break
                    muv = mu & adj_masks[v]
                    if muv.bit_count() < 3: continue
                    for w in range(v+1, n):
                        if found: break
                        muvw = muv & adj_masks[w]
                        if muvw.bit_count() >= 3:
                            # Found K3,3: {u,v,w} and 3 neighbors from muvw
                            # Choose an edge to remove randomly from the 9 edges of the K3,3
                            # Edges are (u,x), (v,x), (w,x) for x in muvw
                            neighbors = [i for i in range(n) if (muvw >> i) & 1]
                            # Pick a random neighbor x
                            x = neighbors[rng.integers(0, len(neighbors))]
                            # Pick a random vertex from {u,v,w} to disconnect from x
                            vertex = [u, v, w][rng.integers(0, 3)]
                            
                            # Remove edge (vertex, x)
                            adj_masks[vertex] &= ~(1 << x)
                            adj_masks[x] &= ~(1 << vertex)
                            # Update edges list (slow, but n is small)
                            # We can just rebuild edges list later or maintain it.
                            # For now, mark as changed.
                            changed = True
                            found = True
                            break
            # Rebuild edges list if changed? 
            # It's better to maintain edges list for the return value.
            # But checking validity is done on masks.
            # We can just extract edges at the end.
            
        # Rebuild edges list from masks
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj_masks[i] >> j) & 1:
                    edges.append((i, j))
        
        current_count = len(edges)
        
        # Greedy addition
        # Try to add all missing edges
        # Order: Random? Or by "safety"?
        # Random is fine with multiple restarts.
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not ((adj_masks[i] >> j) & 1)]
        rng.shuffle(missing)
        
        for u, v in missing:
            if time.time() > deadline - 2:
                break
            if can_add_mask(adj_masks, u, v, n):
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                edges.append((u, v))
                current_count += 1
        
        # Local Search: Edge Swaps
        # Remove one edge, try to add two.
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Shuffle current edges
            edge_list = list(edges)
            rng.shuffle(edge_list)
            
            for u, v in edge_list:
                if time.time() > deadline - 5:
                    break
                
                # Remove (u,v)
                adj_masks[u] &= ~(1 << v)
                adj_masks[v] &= ~(1 << u)
                edges.remove((u, v))
                # Note: if (u,v) is not in edges, something is wrong. It should be.
                
                # Try to add 2 edges
                added = 0
                # Candidates: all missing edges
                # To speed up, we can prioritize edges incident to u or v?
                # Or just random.
                missing_now = [(i, j) for i in range(n) for j in range(i+1, n) if not ((adj_masks[i] >> j) & 1)]
                rng.shuffle(missing_now)
                
                for a, b in missing_now:
                    if added >= 2:
                        break
                    if can_add_mask(adj_masks, a, b, n):
                        adj_masks[a] |= (1 << b)
                        adj_masks[b] |= (1 << a)
                        edges.append((a, b))
                        added += 1
                
                if added > 1:
                    improved = True
                elif added == 1:
                    # Net zero. Keep it? 
                    # It might open up new possibilities. Let's keep it.
                    pass
                else:
                    # Revert
                    adj_masks[u] |= (1 << v)
                    adj_masks[v] |= (1 << u)
                    edges.append((u, v))
            
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
            
            # If we hit a very high number, we might be near optimal
            # ex(20, K3,3) is likely around 50-60.
            if best_count >= 55:
                # Continue a bit more to see if we can go higher, but break early if time is tight
                if time.time() > deadline - 20:
                    break

    # Final verification and formatting
    # Convert list of tuples to list of lists
    return [list(e) for e in best_edges]
