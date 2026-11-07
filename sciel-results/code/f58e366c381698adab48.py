import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        neighbors = {v: [] for v in range(k, n)}
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[v].append(u)
        return neighbors

    def popcount(x):
        return bin(x).count("1")

    def solve_instance(n, k, p, seed):
        neighbors = instance(n, k, p, seed)
        
        # Bitmasks for N(v)
        masks = {}
        for v in range(k, n):
            m = 0
            for u in neighbors[v]:
                m |= (1 << u)
            masks[v] = m
            
        # Sort I vertices by mask size descending
        sorted_v = sorted(range(k, n), key=lambda v: popcount(masks[v]), reverse=True)
        
        A = {} # v -> mask of A_v
        # To speed up conflict check, we might not need to check all previous if we are smart, 
        # but k is small and n is small, O(n^2) is fine.
        
        for v in sorted_v:
            A[v] = masks[v]
            # Check conflicts with all previously selected
            # We need |A[v] & A[prev]| <= 1
            for prev in sorted_v[:sorted_v.index(v)]:
                inter = A[v] & A[prev]
                while popcount(inter) > 1:
                    # Need to remove at least one bit from A[v] that is in inter
                    # Choose a bit to remove
                    # Heuristic: remove bit that is in the fewest other A[prev]?
                    # Or just the first one?
                    # Let's try to remove a bit that is "least useful".
                    # A bit x is in inter.
                    # If we remove x, we lose edges (x, y) for all y in A[v].
                    # We want to keep A[v] large.
                    # So we just need to reduce intersection.
                    # Let's pick a bit from inter.
                    # To be deterministic, pick the lowest set bit in inter?
                    # Or the one that appears in the most previous A's? (to reduce future conflicts?)
                    # Actually, removing it from A[v] doesn't affect previous A's.
                    # It just reduces A[v].
                    # Let's pick the bit with the lowest index for simplicity, 
                    # or the one that has the highest "cost" to remove?
                    # Cost to remove x from A[v] is 1 (size reduction).
                    # All removals have same cost.
                    # Let's just pick the first one.
                    # Find first set bit in inter
                    # (inter & -inter) gives lowest set bit
                    bit = inter & -inter
                    A[v] &= ~bit
                    inter &= ~bit
                    # Re-calculate inter? 
                    # inter was A[v] & A[prev].
                    # We removed bit from A[v].
                    # So new inter = (A[v] & A[prev]).
                    # Since bit was in inter, and we removed it, 
                    # new inter = old_inter & ~bit.
                    # But wait, if A[prev] has other bits in common, they remain.
                    # So inter = A[v] & A[prev] is correct update.
                    
        # Now we have A[v] for all v.
        # Determine covered K-edges
        # An edge (u, w) is covered if there exists v such that u, w in A[v].
        # We can represent covered edges as a set of frozensets or just a set of (u, w) tuples.
        covered_edges = set()
        for v in range(k, n):
            m = A[v]
            # Get list of vertices in m
            verts = [i for i in range(k) if (m >> i) & 1]
            for i in range(len(verts)):
                for j in range(i+1, len(verts)):
                    covered_edges.add((verts[i], verts[j]))
                    
        # Determine remaining K-edges
        all_k_edges = set()
        for u in range(k):
            for w in range(u+1, k):
                all_k_edges.add((u, w))
                
        remaining_edges = all_k_edges - covered_edges
        
        # Build graph for remaining edges
        adj = {u: set() for u in range(k)}
        for u, w in remaining_edges:
            adj[u].add(w)
            adj[w].add(u)
            
        # Greedy Clique Cover for remaining_edges
        rem_cliques = []
        edges_to_cover = set(remaining_edges)
        
        while edges_to_cover:
            # Find a maximal clique in the graph defined by edges_to_cover
            # Start with an arbitrary edge
            u, w = next(iter(edges_to_cover))
            clique = {u, w}
            
            # Candidates are neighbors of u and w in the current graph
            # Note: The graph changes as we remove edges.
            # We should work on a copy or update adj dynamically.
            # Since k is small, we can just re-evaluate or use a local adj.
            
            # Let's use a local adjacency list for the subgraph of edges_to_cover
            # Actually, simpler: 
            # Candidates must be connected to u and w in the CURRENT graph (edges_to_cover).
            # And then to all other added vertices.
            
            # To check if x is connected to y in current graph, 
            # we check if (x,y) or (y,x) is in edges_to_cover.
            
            candidates = set()
            for x in adj[u]:
                if (u, x) in edges_to_cover or (x, u) in edges_to_cover:
                    candidates.add(x)
            for x in adj[w]:
                if (w, x) in edges_to_cover or (x, w) in edges_to_cover:
                    candidates.add(x)
            
            # Remove u, w from candidates if they are in there (they are not, as adj doesn't contain self)
            # But u, w are in clique.
            
            added = []
            # Sort candidates for determinism?
            for c in sorted(candidates):
                if c == u or c == w:
                    continue
                # Check if c is connected to all in clique
                ok = True
                for a in clique:
                    if a == c:
                        continue
                    if (a, c) not in edges_to_cover and (c, a) not in edges_to_cover:
                        ok = False
                        break
                if ok:
                    clique.add(c)
                    added.append(c)
                    
            rem_cliques.append(list(clique))
            
            # Remove all edges within this clique from edges_to_cover
            clique_list = list(clique)
            for i in range(len(clique_list)):
                for j in range(i+1, len(clique_list)):
                    e1 = (clique_list[i], clique_list[j])
                    e2 = (clique_list[j], clique_list[i])
                    if e1 in edges_to_cover:
                        edges_to_cover.remove(e1)
                    if e2 in edges_to_cover:
                        edges_to_cover.remove(e2)
                        
        # Construct final list of cliques
        cliques = []
        
        # 1. Main cliques for v in I
        for v in range(k, n):
            m = A[v]
            verts = [i for i in range(k) if (m >> i) & 1]
            if verts:
                cliques.append(verts + [v])
            # If A[v] is empty, we still need to cover cross edges?
            # Wait, if A[v] is empty, it means we removed all neighbors due to conflicts?
            # No, we only remove if intersection > 1.
            # If N(v) has size 1, A[v] has size 1.
            # If N(v) has size 0, A[v] is 0.
            
        # 2. Extra cliques for leftover cross edges
        for v in range(k, n):
            m_A = A[v]
            m_N = masks[v]
            # Leftover vertices in N(v) not in A(v)
            leftover_mask = m_N & ~m_A
            if leftover_mask:
                # For each u in leftover, create clique {u, v}
                # We can create one clique per u.
                # Or can we bundle them?
                # No, because {v, u1, u2} would imply u1, u2 in A[v].
                # But we defined A[v] as the set of neighbors in the main clique.
                # If we add u1 to main clique, it changes A[v].
                # So we must use separate 2-cliques for leftovers.
                verts = [i for i in range(k) if (leftover_mask >> i) & 1]
                for u in verts:
                    cliques.append([u, v])
                    
        # 3. Add K-only cliques
        for c in rem_cliques:
            cliques.append(c)
            
        return [[int(x) for x in c] for c in cliques]

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    for spec in specs:
        if time.time() > deadline:
            break
        cliques = solve_instance(*spec)
        results.append(cliques)
        
    return results
