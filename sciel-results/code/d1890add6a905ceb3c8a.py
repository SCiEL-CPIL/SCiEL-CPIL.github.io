import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s limit

    def verify_triangle_free(n, adj):
        """Check if graph is triangle-free using bitmasks."""
        for u in range(n):
            # Get neighbors with index > u to avoid double checking
            mask_u = adj[u]
            m = mask_u
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                if v > u:
                    # If u and v are both neighbors of some w, triangle exists
                    # Check if adj[u] & adj[v] has any bit set
                    if adj[u] & adj[v]:
                        return False
                m -= lsb
        return True

    def has_independent_set_10(n, adj):
        """
        Returns True if there is an independent set of size 10.
        Independent set in G = Clique in complement of G.
        Uses branch and bound with bitmasks.
        """
        # Build complement adjacency
        mask_all = (1 << n) - 1
        comp_adj = [0] * n
        for i in range(n):
            comp_adj[i] = mask_all ^ adj[i]
            comp_adj[i] &= ~(1 << i)  # Remove self-loop

        # Bron-Kerbosch variant with pivot for max clique finding
        # We just need to find a clique of size 10
        found = [False]

        def expand(clique_mask, cand_mask, excl_mask):
            if found[0]:
                return
            
            # If we found a clique of size 10
            if clique_mask.bit_count() >= 10:
                found[0] = True
                return

            # Pruning: if clique size + candidates size < 10, no chance
            if clique_mask.bit_count() + cand_mask.bit_count() < 10:
                return

            if cand_mask == 0:
                return

            # Pivot selection: pick vertex with max degree in cand_mask
            # to partition candidates into neighbors and non-neighbors
            # Standard Bron-Kerbosch with pivot:
            # P = cand_mask, X = excl_mask
            # If P and X are both empty, found clique.
            # But we want to find ANY clique of size 10, so we can use a simpler B&B.
            
            # Simple B&B: Pick a vertex v from P.
            # Branch 1: Include v. P' = P & N(v)
            # Branch 2: Exclude v. P' = P - {v}
            
            # Heuristic: Pick v with max degree in P to reduce P' in branch 1
            best_v = -1
            best_deg = -1
            temp = cand_mask
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                deg = (cand_mask & comp_adj[v]).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                temp -= lsb

            if best_v == -1:
                return

            # Branch 1: Include best_v
            new_cand = cand_mask & comp_adj[best_v]
            expand(clique_mask | (1 << best_v), new_cand, excl_mask)
            if found[0]:
                return

            # Branch 2: Exclude best_v
            new_cand2 = cand_mask & ~(1 << best_v)
            expand(clique_mask, new_cand2, excl_mask)
            if found[0]:
                return

        expand(0, mask_all, 0)
        return found[0]

    def verify_graph(n, adj):
        """Returns True if graph is valid (triangle-free and alpha < 10)."""
        if not verify_triangle_free(n, adj):
            return False
        if has_independent_set_10(n, adj):
            return False
        return True

    def make_edges(n, adj):
        edges = []
        for u in range(n):
            m = adj[u]
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append([u, v])
                m -= lsb
        return edges

    def construct_complement_random_regular(n):
        """
        Strategy: Start with a random regular graph (or dense random graph),
        take its complement, and then fix triangles in the complement.
        Actually, the prompt suggests: "Construct a graph using the complement of a random regular graph... and iteratively remove edges from the resulting graph to destroy any triangles".
        
        Let's interpret:
        1. Generate a random graph G0 (e.g., random regular or just random).
        2. Let G = complement(G0).
        3. G might have triangles. We want G to be triangle-free.
        4. So we remove edges from G to break triangles.
        5. Removing edges from G means adding edges to G0.
        6. We want G to have no independent set of size 10.
           Independent set in G = Clique in G0.
           So we want G0 to have no clique of size 10.
        
        Wait, if we remove edges from G, we make G sparser.
        Sparsifying G tends to INCREASE the independence number of G.
        So this strategy seems risky if we just remove edges blindly.
        
        Alternative interpretation:
        Start with a random graph G.
        If G has triangles, remove edges to make it triangle-free.
        If G has an independent set of size 10, add edges to break it (but avoid creating triangles).
        
        The inspiration says: "Construct a graph using the complement of a random regular graph... and iteratively remove edges from the resulting graph to destroy any triangles, while monitoring the independence number via a greedy heuristic to guide which edges to remove".
        
        Let's try:
        1. Generate a random regular graph R (or just a random graph with specific density).
        2. Let G = complement(R).
        3. G is likely to have many triangles if R is sparse. If R is dense, G is sparse.
        4. If R is d-regular, G is (n-1-d)-regular.
        5. We want G to be triangle-free.
        6. So we remove edges from G until it is triangle-free.
        7. We need to ensure alpha(G) < 10.
        
        Let's implement a local search:
        1. Start with a random graph.
        2. While it has triangles:
           - Find a triangle (u, v, w).
           - Remove one edge (u, v) from the triangle.
           - This might create an independent set.
        3. While it has an independent set of size 10:
           - Find an independent set I of size 10.
           - Add an edge (u, v) where u, v in I.
           - This might create a triangle.
        
        This is essentially the current program's approach, but let's refine the heuristics.
        
        Let's try a different structural idea:
        Use a greedy algorithm to build a maximal triangle-free graph, but bias the edge selection to minimize independence number.
        """
        adj = [0] * n
        # Start with a random graph
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Phase 1: Remove triangles
        for _ in range(1000):
            if time.time() > deadline:
                break
            # Find a triangle
            tri_found = False
            for u in range(n):
                m1 = adj[u]
                while m1:
                    lsb1 = m1 & -m1
                    v = lsb1.bit_length() - 1
                    if v > u:
                        # Check if u and v have a common neighbor w > v
                        common = adj[u] & adj[v] & ~((1 << (v + 1)) - 1)
                        if common:
                            lsb2 = common & -common
                            w = lsb2.bit_length() - 1
                            # Remove edge (u, v)
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                            tri_found = True
                            break
                    m1 -= lsb1
                if tri_found:
                    break
            if not tri_found:
                break
        
        # Phase 2: Fix independent set
        for _ in range(1000):
            if time.time() > deadline:
                break
            # Check if there is an independent set of size 10
            if not has_independent_set_10(n, adj):
                return adj
            
            # Find an independent set of size 10 (or close to it)
            # We can reuse the search logic to find a specific set
            # For now, let's just pick a random pair of non-adjacent vertices and add an edge
            # But we must ensure no triangle is created.
            
            # Better: Find a maximal independent set, and if it's >= 10, add an edge within it.
            # Let's find a large independent set using a greedy heuristic
            indep_set = []
            remaining = (1 << n) - 1
            while remaining:
                # Pick vertex with min degree in remaining
                best_v = -1
                best_deg = n
                temp = remaining
                while temp:
                    lsb = temp & -temp
                    v = lsb.bit_length() - 1
                    deg = (adj[v] & remaining).bit_count()
                    if deg < best_deg:
                        best_deg = deg
                        best_v = v
                    temp -= lsb
                
                if best_v == -1:
                    break
                indep_set.append(best_v)
                # Remove best_v and its neighbors from remaining
                remaining &= ~(adj[best_v] | (1 << best_v))
                
                if len(indep_set) >= 10:
                    break
            
            if len(indep_set) >= 10:
                # Pick two random vertices from indep_set and add an edge
                u, v = random.sample(indep_set, 2)
                # Check if adding (u, v) creates a triangle
                # Triangle would be (u, v, w) where w is neighbor of both u and v
                common = adj[u] & adj[v]
                if not common:
                    # Safe to add
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                else:
                    # Adding (u, v) creates triangles.
                    # We must remove some edges.
                    # Let's pick a triangle (u, v, w) and remove (u, w) or (v, w)
                    # This is complex. Let's just skip this pair and try another.
                    # Or, remove an edge from the triangle.
                    # Let's pick a random w from common
                    lsb = common & -common
                    w = lsb.bit_length() - 1
                    # Remove (u, w)
                    adj[u] &= ~(1 << w)
                    adj[w] &= ~(1 << u)
                    # Now add (u, v)
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            else:
                # No independent set of size 10 found by heuristic, but verify_graph says there is one?
                # The heuristic might be weak.
                # Let's break and rely on other restarts.
                break

        return None

    best_n = 0
    best_edges = []

    # Try n from 40 down to 26
    for n in range(40, 25, -1):
        if time.time() > deadline:
            break
        
        found = False
        
        # Strategy 1: Randomized Local Search (Complement/Edge Removal)
        for _ in range(50):
            if time.time() > deadline:
                break
            adj = construct_complement_random_regular(n)
            if adj:
                if verify_graph(n, adj):
                    edges = make_edges(n, adj)
                    best_n = n
                    best_edges = edges
                    found = True
                    break
        
        if found:
            break

    # Fallback: Return a known small valid graph (Andrasfai graph on 26)
    if best_n == 0:
        n = 26
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % n % 3 == 1:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        edges = make_edges(n, adj)
        best_n = n
        best_edges = edges

    return {'n': best_n, 'edges': best_edges}

if __name__ == "__main__":
    res = construct()
    print(f"Found n={res['n']}")
