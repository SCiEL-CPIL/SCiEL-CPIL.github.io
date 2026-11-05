import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit
    
    n = 16
    target_edges = 65
    
    # Strategy: Start with K_{8,8} (64 edges), remove some cross edges,
    # add internal edges to reach 65 total.
    # Key insight: triangles in this family come from internal edges
    # (u,v) in A or B, with common neighbors in the other part.
    # For internal edge (u,v) in A: triangle (u,v,w) where w in B,
    #   sum = deg(u) + deg(v) + deg(w)
    # We want all such sums <= 24 ideally, or at least < 26.
    
    # Let's do a systematic search over small perturbations of K_{8,8}
    
    def evaluate_graph(edges_set):
        """Compute max triangle degree sum for a graph with given edges."""
        if len(edges_set) != target_edges:
            return 999
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges_set:
            adj[u].add(v)
            adj[v].add(u)
        # Find all triangles and compute max degree sum
        max_sum = 0
        for u in range(n):
            for v in adj[u]:
                if v <= u:
                    continue
                for w in adj[u]:
                    if w <= v:
                        continue
                    if w in adj[v]:
                        s = len(adj[u]) + len(adj[v]) + len(adj[w])
                        if s > max_sum:
                            max_sum = s
        return max_sum
    
    # Base: K_{8,8}
    base_edges = set()
    for u in range(8):
        for v in range(8, 16):
            base_edges.add((u, v))
    
    # We need to add 1 edge total (since base has 64, we need 65)
    # But we can also remove some cross edges and add more internal edges
    # to keep total at 65 while potentially reducing max triangle sum.
    
    # Approach: Try removing k cross edges and adding k+1 internal edges
    # for small k (0, 1, 2, 3)
    
    best_score = 999
    best_edges = None
    
    # Precompute: for each internal edge (u,v) in A (u<v, both <8),
    # and each internal edge (u,v) in B (u<v, both >=8),
    # the set of common neighbors and the degree impact.
    
    # For a given configuration, the degree of vertex i is:
    # deg(i) = (# cross edges incident to i) + (# internal edges incident to i)
    
    # Let's parameterize:
    # removed_cross: set of cross edges removed
    # added_internal_A: set of internal edges in A (pairs from {0..7})
    # added_internal_B: set of internal edges in B (pairs from {8..15})
    # Total edges = 64 - len(removed_cross) + len(added_internal_A) + len(added_internal_B) = 65
    # So: len(added_internal_A) + len(added_internal_B) = 1 + len(removed_cross)
    
    A_vertices = list(range(8))
    B_vertices = list(range(8, 16))
    
    all_internal_A = list(itertools.combinations(A_vertices, 2))
    all_internal_B = list(itertools.combinations(B_vertices, 2))
    all_cross = [(u, v) for u in A_vertices for v in B_vertices]
    
    def build_and_eval(removed_cross, int_A, int_B):
        """Build graph and evaluate."""
        edges = set(base_edges)
        for e in removed_cross:
            edges.discard(e)
        for e in int_A:
            edges.add(tuple(e))
        for e in int_B:
            edges.add(tuple(e))
        if len(edges) != target_edges:
            return 999
        return evaluate_graph(edges)
    
    # Case k=0: no cross edges removed, add 1 internal edge
    # This is the baseline family. Best we can do is pick the internal
    # edge that minimizes max triangle sum.
    
    # For internal edge (u,v) in A:
    # deg(u) = 8 + 1 = 9 (if no other internal edges)
    # deg(v) = 8 + 1 = 9
    # deg(w) for w in B: 8 (no internal edges in B)
    # Common neighbors of u and v in B: all 8 vertices in B
    # Triangle sum = 9 + 9 + 8 = 26
    
    # So k=0 always gives 26. No improvement possible.
    
    # Case k=1: remove 1 cross edge, add 2 internal edges
    # This is more complex. Let's try a focused search.
    
    # Key insight: if we remove cross edge (u, w) where u in A, w in B,
    # then deg(u) = 7 + (internal degree in A), deg(w) = 7 + (internal degree in B)
    
    # Let's try: remove (0, 8), add internal edges (0,1) in A and (8,9) in B
    # deg(0) = 7 + 1 = 8, deg(1) = 8 + 1 = 9, deg(8) = 7 + 1 = 8, deg(9) = 8 + 1 = 9
    # Triangle (0,1,w) for w in B: w must be neighbor of both 0 and 1 in B
    # 0's B neighbors: 9..15 (7 vertices, missing 8)
    # 1's B neighbors: 8..15 (8 vertices)
    # Common: 9..15 (7 vertices)
    # For w in 9..15: deg(w) = 8 (if w not 9) or 9 (if w=9)
    # Triangle (0,1,9): sum = 8+9+9 = 26
    # Triangle (0,1,10): sum = 8+9+8 = 25
    
    # Triangle (8,9,u) for u in A: u must be neighbor of both 8 and 9 in A
    # 8's A neighbors: 1..7 (7 vertices, missing 0)
    # 9's A neighbors: 0..7 (8 vertices)
    # Common: 1..7 (7 vertices)
    # For u in 1..7: deg(u) = 8 (if u not 1) or 9 (if u=1)
    # Triangle (8,9,1): sum = 8+9+9 = 26
    # Triangle (8,9,2): sum = 8+9+8 = 25
    
    # So max is still 26. Hmm.
    
    # Let me think differently. What if the two internal edges share a vertex?
    # Remove (0,8), add (0,1) and (0,2) in A.
    # deg(0) = 7 + 2 = 9, deg(1) = 8+1=9, deg(2)=8+1=9
    # Triangle (0,1,w): common B neighbors of 0,1 = 9..15 (7 verts)
    #   sum = 9+9+deg(w). deg(w)=8 for w!=9, deg(9)=8 (no internal in B)
    #   So sum = 9+9+8 = 26
    # Triangle (0,2,w): same, sum = 26
    # Triangle (1,2,w): common B neighbors of 1,2 = 8..15 (8 verts)
    #   sum = 9+9+8 = 26
    # Still 26.
    
    # What if we use internal edges in both parts more cleverly?
    # Remove 2 cross edges, add 3 internal edges?
    
    # Let me try a different approach: local search / simulated annealing
    # on the edge set.
    
    import random
    random.seed(42)
    
    # Start with a reasonable initial configuration
    # K_{8,8} minus 2 cross edges, plus 3 internal edges
    # Try to find one with max triangle sum <= 25
    
    def get_degrees(edges_set):
        deg = [0] * n
        for u, v in edges_set:
            deg[u] += 1
            deg[v] += 1
        return deg
    
    def max_triangle_sum_fast(edges_set):
        if len(edges_set) != target_edges:
            return 999
        adj = [set() for _ in range(n)]
        for u, v in edges_set:
            adj[u].add(v)
            adj[v].add(u)
        deg = [len(adj[i]) for i in range(n)]
        max_s = 0
        for u in range(n):
            for v in adj[u]:
                if v <= u:
                    continue
                for w in adj[u]:
                    if w <= v:
                        continue
                    if w in adj[v]:
                        s = deg[u] + deg[v] + deg[w]
                        if s > max_s:
                            max_s = s
        return max_s
    
    # Local search: start from K_{8,8} + 1 internal edge, try swaps
    best_score = 26
    best_config = None
    
    # Initial: K_{8,8} + edge (0,1)
    init_edges = set(base_edges)
    init_edges.add((0, 1))
    current_score = max_triangle_sum_fast(init_edges)
    best_score = min(best_score, current_score)
    best_config = set(init_edges)
    
    # Local search with swaps: remove one cross edge, add one internal edge
    # (keeping total at 65)
    
    cross_edges = [(u, v) for u in range(8) for v in range(8, 16)]
    internal_A = list(itertools.combinations(range(8), 2))
    internal_B = list(itertools.combinations(range(8, 16), 2))
    
    # Try all single swaps from initial config
    # Remove one cross edge, add one internal edge
    for cross_e in cross_edges:
        for int_e in internal_A + internal_B:
            new_edges = set(init_edges)
            new_edges.discard(cross_e)
            new_edges.add(tuple(int_e))
            if len(new_edges) != target_edges:
                continue
            s = max_triangle_sum_fast(new_edges)
            if s < best_score:
                best_score = s
                best_config = new_edges
                if best_score <= 24:
                    break
        if best_score <= 24:
            break
    
    if best_score <= 24:
        return {'n': n, 'edges': [[u, v] for u, v in best_config]}
    
    # Try 2-swap moves: remove 2 cross edges, add 2 internal edges
    # This is more expensive, so limit the search
    if time.time() < deadline - 30:
        # Sample a subset of 2-swap moves
        random.shuffle(cross_edges)
        random.shuffle(internal_A)
        random.shuffle(internal_B)
        
        all_int = internal_A + internal_B
        cross_sample = cross_edges[:50]
        int_sample = all_int[:50]
        
        for i in range(len(cross_sample)):
            if time.time() > deadline - 20:
                break
            for j in range(i+1, len(cross_sample)):
                if time.time() > deadline - 20:
                    break
                c1 = cross_sample[i]
                c2 = cross_sample[j]
                for k in range(len(int_sample)):
                    if time.time() > deadline - 20:
                        break
                    for l in range(k+1, len(int_sample)):
                        if time.time() > deadline - 20:
                            break
                        i1 = int_sample[k]
                        i2 = int_sample[l]
                        new_edges = set(base_edges)
                        new_edges.discard(c1)
                        new_edges.discard(c2)
                        new_edges.add(tuple(i1))
                        new_edges.add(tuple(i2))
                        if len(new_edges) != target_edges:
                            continue
                        s = max_triangle_sum_fast(new_edges)
                        if s < best_score:
                            best_score = s
                            best_config = new_edges
                            if best_score <= 24:
                                return {'n': n, 'edges': [[u, v] for u, v in best_config]}
    
    # Also try: remove 1 cross edge, add 2 internal edges (total still 65)
    # 64 - 1 + 2 = 65
    if time.time() < deadline - 20 and best_score > 25:
        for cross_e in cross_edges[:40]:
            if time.time() > deadline - 15:
                break
            for i1 in internal_A + internal_B:
                if time.time() > deadline - 15:
                    break
                for i2 in internal_A + internal_B:
                    if i2 <= i1:
                        continue
                    if time.time() > deadline - 15:
                        break
                    new_edges = set(base_edges)
                    new_edges.discard(cross_e)
                    new_edges.add(tuple(i1))
                    new_edges.add(tuple(i2))
                    if len(new_edges) != target_edges:
                        continue
                    s = max_triangle_sum_fast(new_edges)
                    if s < best_score:
                        best_score = s
                        best_config = new_edges
                        if best_score <= 24:
                            return {'n': n, 'edges': [[u, v] for u, v in best_config]}
    
    # If we found something better than 26, return it
    if best_config is not None and best_score < 26:
        return {'n': n, 'edges': [[u, v] for u, v in best_config]}
    
    # Fallback: return the baseline (score 26)
    edges_list = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
    return {'n': n, 'edges': edges_list}
