"""K3,3-free graph on 20 vertices - maximize edges via greedy construction."""
import time

def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin before 120s limit
    n = 20
    
    # Bitmask adjacency: adj[i] is a bitmask of neighbors of vertex i
    adj = [0] * n
    edge_count = 0
    
    # Precompute all possible edges as (u, v) with u < v
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def is_valid_addition(u, v):
        """Check if adding edge (u,v) preserves K3,3-freeness."""
        # After adding (u,v), any triple {u,v,w} must have at most 2 common neighbors
        # Common neighbors of u and v (after adding each other): 
        #   neighbors of u that are also neighbors of v, plus w if applicable
        # Actually, we need to check: for any w, the number of common neighbors of u,v,w <= 2
        
        # When we add edge (u,v):
        # For any vertex w:
        #   common_neighbors(u,v) = neighbors of u ∩ neighbors of v (including each other now)
        #   We need |common_neighbors(u,v) ∩ {w}| <= 2 for all w
        #   But also need to check triples where w is one of u or v... 
        #   Actually the constraint is: no 3 vertices have 3 common neighbors.
        #   So for any triple {a,b,c}, |N(a) ∩ N(b) ∩ N(c)| <= 2
        
        # After adding (u,v), the only affected triples are those containing u or v.
        # For a triple containing u and v and some w:
        #   N(u) ∩ N(v) ∩ N(w) must have size <= 2
        # For a triple containing u but not v (and not w that we're considering):
        #   Adding (u,v) changes N(u) to include v, so:
        #   For triple {u, a, b} where a,b ≠ v: N(u)∩N(a)∩N(b) gains v if v∈N(a)∩N(b)
        # For a triple not containing u or v: unchanged
        
        # Check 1: For each w ≠ u,v: |N(u)∩N(v)∩N(w)| <= 2 after adding
        #   N(u) after = old N(u) ∪ {v}
        #   N(v) after = old N(v) ∪ {u}
        #   Common = (old N(u) ∪ {v}) ∩ (old N(v) ∪ {u}) ∩ N(w)
        #   = [old N(u) ∩ old N(v) ∩ N(w)] ∪ [v if v∈N(w) and v∈(old N(v)∪{u})] ∪ [u if u∈N(w) and u∈(old N(u)∪{v})]
        #   Simpler: just compute directly
        
        # For efficiency, let's use bitmasks
        # N(u) after adding v: adj[u] | (1 << v)
        # N(v) after adding u: adj[v] | (1 << u)
        
        nu = adj[u] | (1 << v)  # N(u) after adding edge
        nv = adj[v] | (1 << u)  # N(v) after adding edge
        
        # Check triples containing both u and v: {u, v, w} for w != u,v
        common_uv = nu & nv  # N(u) ∩ N(v) after adding
        # For each w in common_uv... no, we need for each w: |common_uv ∩ N(w)| <= 2
        # Actually the constraint is on the intersection of ALL THREE neighborhoods.
        # |N(u) ∩ N(v) ∩ N(w)| <= 2 for all w
        
        # For triples with u and v: check all w
        for w in range(n):
            if w == u or w == v:
                continue
            intersection = common_uv & adj[w]
            if intersection.bit_count() > 2:
                return False
        
        # Check triples containing u but not v: {u, a, b} where a,b not v
        # Adding (u,v) means v is now in N(u). So N(u)∩N(a)∩N(b) might gain v
        # This happens if v ∈ N(a) and v ∈ N(b)
        # So for each pair (a,b) not including u or v where v ∈ N(a)∩N(b):
        #   check if |N(u)∩N(a)∩N(b)| <= 2, and N(u) now includes v
        #   The old intersection was |old_N(u) ∩ N(a) ∩ N(b)|
        #   New intersection = old + (1 if v ∈ old_N(u)... no, v wasn't in old_N(u)
        #   New intersection = |old_N(u) ∩ N(a) ∩ N(b)| + (1 if v ∈ N(a) ∩ N(b) else 0)
        #   So if v ∈ N(a) ∩ N(b) and old intersection was already 2, it becomes 3.
        
        neighbors_of_u = adj[u]  # old neighbors (before adding v)
        # For each pair (a,b) where a < b, a,b != u, a,b != v
        # if v ∈ N(a) and v ∈ N(b): check if |neighbors_of_u ∩ N(a) ∩ N(b)| == 2
        for a in range(n):
            if a == u or a == v:
                continue
            if not (adj[a] >> v) & 1:  # v not in N(a)
                continue
            for b in range(a + 1, n):
                if b == u or b == v:
                    continue
                if not (adj[b] >> v) & 1:  # v not in N(b)
                    continue
                intersection = neighbors_of_u & adj[a] & adj[b]
                if intersection.bit_count() >= 2:
                    return False
        
        # Check triples containing v but not u: symmetric
        neighbors_of_v = adj[v]  # old neighbors
        for a in range(n):
            if a == u or a == v:
                continue
            if not (adj[a] >> u) & 1:  # u not in N(a)
                continue
            for b in range(a + 1, n):
                if b == u or b == v:
                    continue
                if not (adj[b] >> u) & 1:  # u not in N(b)
                    continue
                intersection = neighbors_of_v & adj[a] & adj[b]
                if intersection.bit_count() >= 2:
                    return False
        
        return True
    
    def add_edge(u, v):
        nonlocal edge_count
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edge_count += 1
    
    # Greedy: try edges in order of how many they add to connectivity
    # Strategy: start with a good seed, then greedily add edges
    # Seed: K_{2,18} plus cycle on the 18 vertices (55 edges as analyzed)
    
    # Actually, let's try a smarter greedy from scratch
    # Order edges by a heuristic: prefer edges that connect vertices with lower degree
    
    # Try multiple random orderings and take the best
    import random
    random.seed(42)
    
    best_edges = []
    best_count = 0
    
    for trial in range(50):
        if time.time() > deadline:
            break
        
        # Create a randomized edge ordering
        edges_order = all_edges[:]
        random.shuffle(edges_order)
        
        # Try to add edges in this order
        trial_adj = [0] * n
        trial_count = 0
        
        for u, v in edges_order:
            if time.time() > deadline:
                break
            
            # Check validity with trial_adj
            nu = trial_adj[u] | (1 << v)
            nv = trial_adj[v] | (1 << u)
            common_uv = nu & nv
            valid = True
            
            # Check triples with u and v
            for w in range(n):
                if w == u or w == v:
                    continue
                if (common_uv & trial_adj[w]).bit_count() > 2:
                    valid = False
                    break
            
            if valid:
                # Check triples with u but not v
                neighbors_of_u = trial_adj[u]
                for a in range(n):
                    if a == u or a == v:
                        continue
                    if not (trial_adj[a] >> v) & 1:
                        continue
                    for b in range(a + 1, n):
                        if b == u or b == v:
                            continue
                        if not (trial_adj[b] >> v) & 1:
                            continue
                        if (neighbors_of_u & trial_adj[a] & trial_adj[b]).bit_count() >= 2:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                # Check triples with v but not u
                if valid:
                    neighbors_of_v = trial_adj[v]
                    for a in range(n):
                        if a == u or a == v:
                            continue
                        if not (trial_adj[a] >> u) & 1:
                            continue
                        for b in range(a + 1, n):
                            if b == u or b == v:
                                continue
                            if not (trial_adj[b] >> u) & 1:
                                continue
                            if (neighbors_of_v & trial_adj[a] & trial_adj[b]).bit_count() >= 2:
                                valid = False
                                break
                        if not valid:
                            break
            
            if valid:
                trial_adj[u] |= (1 << v)
                trial_adj[v] |= (1 << u)
                trial_count += 1
        
        if trial_count > best_count:
            best_count = trial_count
            # Reconstruct edges
            best_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (trial_adj[u] >> v) & 1:
                        best_edges.append([u, v])
    
    return best_edges
