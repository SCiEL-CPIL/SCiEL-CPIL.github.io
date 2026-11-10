"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin (150s limit)
    
    q = 5
    # Build PG(2, q) points: q^2 + q + 1 = 31 points
    # Points: (1, x, y) for x,y in GF(q); (0, 1, z) for z in GF(q); (0, 0, 1)
    points = []
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    n = len(points)  # should be 31
    
    def dot(p, r):
        return (p[0]*r[0] + p[1]*r[1] + p[2]*r[2]) % q
    
    # Build polarity graph: edge between i,j if dot(points[i], points[j]) == 0
    # and i != j
    polarity_edges = set()
    for i in range(n):
        for j in range(i+1, n):
            if dot(points[i], points[j]) == 0:
                polarity_edges.add((i, j))
    
    # The polarity graph has q^2 + q = 30 edges... wait, let me check
    # Actually for q=5, the polarity graph (orthogonality graph) of PG(2,5) 
    # should have a specific number of edges. Let me count.
    # Each point is orthogonal to q+1 = 6 points (including itself if the polarity is non-degenerate)
    # For a symplectic polarity, each point is orthogonal to itself, so each vertex has degree q+1-1 = q = 5
    # Total edges = 31*5/2 = 77.5... that's not integer, so I need to be more careful.
    
    # Let me just compute it directly
    edge_list = []
    for i in range(n):
        for j in range(i+1, n):
            if dot(points[i], points[j]) == 0:
                edge_list.append((i, j))
    
    num_polarity_edges = len(edge_list)
    
    # We need 91 edges total. Start with polarity edges, then greedily add more.
    current_edges = set(edge_list)
    
    # Build adjacency for fast C4 counting
    def build_adjacency(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def count_c4s(adj):
        """Count 4-cycles in the graph. A 4-cycle is a subgraph isomorphic to C4."""
        # For each pair of vertices (u,v), count common neighbors.
        # A 4-cycle u-w-v-x-u corresponds to pairs (u,v) and (w,x) sharing two common neighbors.
        # More precisely: number of C4 = sum over all pairs {u,v} of C(common_neighbors(u,v), 2) / 2
        # Wait, let me think more carefully.
        # A 4-cycle is u-w-v-x where u-w, w-v, v-x, x-u are edges, and all four distinct.
        # Equivalently, pick two non-adjacent vertices u,v (diagonally opposite), and count
        # the number of common neighbors. If they have k common neighbors, then each pair of
        # common neighbors forms a 4-cycle with u and v.
        # But we need u and v to NOT be adjacent (otherwise it's a triangle+edge, not a simple 4-cycle).
        # Actually, even if u and v are adjacent, the common neighbors still form 4-cycles u-w-v-x-u
        # as long as w != x and w,x are not u or v.
        # 
        # The standard formula: #C4 = (1/2) * sum_{u,v not adjacent} C(d_common(u,v), 2)
        # No wait. Let me use: #C4 = sum_{unordered pairs {u,v}} C(|N(u) ∩ N(v)|, 2) / 2
        # because each C4 is counted twice (once for each pair of opposite vertices).
        
        count = 0
        for u in range(n):
            for v in range(u+1, n):
                common = len(adj[u] & adj[v])
                if common >= 2:
                    count += common * (common - 1) // 2
        # Each 4-cycle is counted exactly twice (once for each pair of opposite vertices)
        return count // 2
    
    def count_c4s_fast(adj):
        """Faster C4 count using common neighbor matrix."""
        # For each vertex, represent neighbors as a bitset or use set intersection
        count = 0
        for u in range(n):
            for v in range(u+1, n):
                common = len(adj[u] & adj[v])
                if common >= 2:
                    count += common * (common - 1) // 2
        return count // 2
    
    # Current C4 count
    adj = build_adjacency(current_edges)
    current_c4 = count_c4s(adj)
    
    # We need to add (91 - num_polarity_edges) more edges
    needed = 91 - num_polarity_edges
    
    if needed <= 0:
        # Already have enough edges, just return
        edges_to_return = list(current_edges)
        return [[u, v] for u, v in edges_to_return]
    
    # Generate all possible edges not yet in the graph
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in current_edges]
    
    # Greedily add edges that minimize new C4s
    for _ in range(needed):
        best_edge = None
        best_new_c4 = float('inf')
        
        # Check each candidate edge
        for u, v in all_possible:
            # Skip if already added
            if (u, v) in current_edges:
                continue
            
            # New C4s introduced by adding edge (u,v):
            # A new C4 involving edge (u,v) must have the form u-w-v-x-u or u-x-v-w-u
            # i.e., w and x are common neighbors of u and v, and w-x is NOT an edge (otherwise it was already counted)
            # Wait, no. If we add edge (u,v), new C4s are those that use this edge.
            # A C4 using edge (u,v) has the form u-v-w-x-u where v-w, w-x, x-u are edges.
            # So w must be a neighbor of v, x must be a neighbor of u, and w-x must be an edge, w != x, w != u, x != v.
            # Actually simpler: new C4s = pairs (w, x) where w in N(v), x in N(u), w != x, 
            # w not in N(u) [no, w can be in N(u)], x not in N(v), and (w,x) is an edge.
            # Hmm, let me think again.
            
            # When we add edge (u,v), a new 4-cycle is created if there exist distinct w, x such that:
            # - w is adjacent to v (w in N(v))
            # - x is adjacent to u (x in N(u))  
            # - w is adjacent to x (w-x is an edge)
            # - w != x, w != u, x != v
            # - The cycle is u-v-w-x-u
            # Note: we don't need w not in N(u) or x not in N(v) for it to be a new C4.
            # But we do need that this specific 4-cycle wasn't already present. Since it uses edge (u,v)
            # which wasn't present before, it's definitely new.
            
            # So: new_c4 = |{(w, x) : w in N(v), x in N(u), w != x, w != u, x != v, (w,x) in E}|
            # But we need to be careful about double counting. Each 4-cycle u-v-w-x-u is distinct from u-v-x-w-u.
            # Actually, for a given edge (u,v), the 4-cycles using it are in bijection with 
            # edges (w,x) where w in N(v) \ {u}, x in N(u) \ {v}, w != x, and (w,x) in E.
            # Wait, w could equal x? No, w and x must be distinct for a 4-cycle.
            # Also w could be in N(u) and x could be in N(v), that's fine.
            
            new_c4_count = 0
            for w in adj[v]:
                if w == u:
                    continue
                for x in adj[u]:
                    if x == v or x == w:
                        continue
                    if (min(w,x), max(w,x)) in current_edges or (w,x) in current_edges:
                        new_c4_count += 1
            
            # Each 4-cycle u-v-w-x-u is counted once here (for the specific ordering)
            # But wait, is it counted once or twice? For edge (u,v), the 4-cycle u-v-w-x-u 
            # is distinct from u-v-x-w-u. These are different 4-cycles (different vertex orderings)
            # but they represent the same subgraph. Hmm.
            # Actually, a 4-cycle as a subgraph {u,v,w,x} with edges {uv, vw, wx, xu} is the same as
            # {u,v,w,x} with edges {uv, vx, xw, wu} only if both sets of edges are present.
            # In our case, we're adding uv, and vw, wx, xu are already present.
            # The 4-cycle u-v-x-w-u would require edges uv, vx, xw, wu. 
            # vx is present (x in N(v)), wu is present (w in N(u)), xw is present.
            # So if both (w,x) and the other configuration work, we might double count.
            # 
            # Let me reconsider. For a fixed edge (u,v) being added, a new C4 is a set of 4 distinct
            # vertices {u,v,w,x} such that the 4 edges uv, vw, wx, xu are all present (with uv just added).
            # This is equivalent to: w in N(v)\{u}, x in N(u)\{v}, w != x, and (w,x) in E.
            # Each such pair (w,x) gives exactly one 4-cycle subgraph.
            # But (w,x) and (x,w) give the same subgraph! So we should count each unordered pair once.
            # In my loop above, I iterate w in N(v) and x in N(u), so (w,x) and (x,w) are different iterations
            # but give the same 4-cycle. So I should divide by... no wait.
            # If w in N(v) and x in N(u), the 4-cycle is u-v-w-x-u.
            # If x in N(v) and w in N(u), that's a different configuration: u-v-x-w-u.
            # These are the same subgraph {u,v,w,x} with edges {uv, vw, wx, xu} = {uv, vx, xw, wu}.
            # So yes, they're the same. I'm double counting.
            # 
            # Fix: only count when w < x or use a set.
            
            # Let me redo this more carefully.
            new_c4_count = 0
            seen = set()
            for w in adj[v]:
                if w == u:
                    continue
                for x in adj[u]:
                    if x == v or x == w:
                        continue
                    key = (min(w,x), max(w,x))
                    if key in seen:
                        continue
                    seen.add(key)
                    if key in current_edges:
                        new_c4_count += 1
            
            if new_c4_count < best_new_c4:
                best_new_c4 = new_c4_count
                best_edge = (u, v)
        
        if best_edge is None:
            break
        
        # Add the best edge
        current_edges.add(best_edge)
        adj[best_edge[0]].add(best_edge[1])
        adj[best_edge[1]].add(best_edge[0])
        
        # Remove from candidates
        all_possible.remove(best_edge)
        
        # Check deadline
        if time.time() > deadline - 5:
            break
    
    # If we didn't add enough edges, add arbitrary remaining ones
    while len(current_edges) < 91:
        if not all_possible:
            break
        # Pick an edge with minimum new C4s (or just first one if time is tight)
        if time.time() > deadline - 2:
            e = all_possible.pop(0)
        else:
            best_e = None
            best_count = float('inf')
            for e in all_possible:
                u, v = e
                new_c4_count = 0
                seen = set()
                for w in adj[v]:
                    if w == u:
                        continue
                    for x in adj[u]:
                        if x == v or x == w:
                            continue
                        key = (min(w,x), max(w,x))
                        if key in seen:
                            continue
                        seen.add(key)
                        if key in current_edges:
                            new_c4_count += 1
                if new_c4_count < best_count:
                    best_count = new_c4_count
                    best_e = e
            if best_e is not None:
                e = best_e
                all_possible.remove(e)
            else:
                e = all_possible.pop(0)
        
        current_edges.add(e)
        adj[e[0]].add(e[1])
        adj[e[1]].add(e[0])
    
    # Return exactly 91 edges
    result = [[u, v] for u, v in list(current_edges)[:91]]
    
    # Verify we have exactly 91 distinct edges
    assert len(result) == 91, f"Got {len(result)} edges instead of 91"
    edge_set = set()
    for e in result:
        u, v = e
        assert u < v, f"Edge {e} not in canonical form"
        assert 0 <= u < 31 and 0 <= v < 31, f"Edge {e} has out-of-range vertex"
        assert (u, v) not in edge_set, f"Duplicate edge {e}"
        edge_set.add((u, v))
    
    return result

# EVOLVE-BLOCK-END
