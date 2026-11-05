import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0
    
    n = 31
    target_edges = 91
    
    # Initialize with the polarity graph PG(2,5)
    # Points in PG(2,5) can be represented as (x, y, z) not all 0, scaled such that one coord is 1 or 0.
    # Standard representation for q=5:
    # (1, y, z) for y,z in 0..4 (25 points)
    # (0, 1, z) for z in 0..4 (5 points)
    # (0, 0, 1) (1 point)
    # Total 31 points.
    
    points = []
    for y in range(5):
        for z in range(5):
            points.append((1, y, z))
    for z in range(5):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    # Adjacency in polarity graph: x1*x2 + y1*y2 + z1*z2 == 0 (mod 5)
    adj = [set() for _ in range(n)]
    edges = set()
    
    for i in range(n):
        for j in range(i+1, n):
            x1, y1, z1 = points[i]
            x2, y2, z2 = points[j]
            if (x1*x2 + y1*y2 + z1*z2) % 5 == 0:
                adj[i].add(j)
                adj[j].add(i)
                edges.add((i, j))
    
    # We need 91 edges. Polarity graph has 56. We need to add 35.
    # We will use Simulated Annealing to find the best set of 91 edges.
    
    # Helper to calculate total C4 count
    # C4 count = sum_{u<v} C(|N(u) ∩ N(v)|, 2)
    def count_c4s(adj):
        total = 0
        for u in range(n):
            nu = adj[u]
            for v in nu:
                if u < v:
                    common = len(nu & adj[v])
                    if common >= 2:
                        total += common * (common - 1) // 2
        return total
    
    # Initialize edges to a random valid set of 91 edges
    # Start with polarity graph edges, then add random non-edges until we have 91.
    # This ensures we start in a "good" neighborhood (low C4s).
    all_possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    current_edges = set(edges)
    
    # Add random non-edges
    non_edges = [e for e in all_possible_edges if e not in current_edges]
    random.shuffle(non_edges)
    for e in non_edges:
        if len(current_edges) >= target_edges:
            break
        u, v = e
        current_edges.add((u, v))
        adj[u].add(v)
        adj[v].add(u)
        
    if len(current_edges) < target_edges:
        # Should not happen, but just in case
        for e in all_possible_edges:
            if len(current_edges) >= target_edges:
                break
            if e not in current_edges:
                current_edges.add(e)
                u, v = e
                adj[u].add(v)
                adj[v].add(u)

    current_c4s = count_c4s(adj)
    best_edges = set(current_edges)
    best_c4s = current_c4s
    
    # Simulated Annealing Parameters
    T = 50.0
    cooling_rate = 0.9995
    min_T = 0.1
    
    # Precompute list of all non-edges for fast sampling
    # We maintain a list of non-edges to sample from.
    # Since the set of non-edges changes, we can just sample from all_possible_edges and check if it's in current_edges.
    # n=31, total edges = 465. 91 edges -> 374 non-edges.
    # Sampling random edge and checking membership is O(1).
    
    moves = 0
    while time.time() < deadline - 5.0 and T > min_T:
        # Choose a random edge to remove and a random non-edge to add
        # To avoid bias, we can pick a random pair (e_remove, e_add)
        
        # Efficient move: pick random edge from current, pick random non-edge.
        # If they share vertices, it's a "local" move. If not, "global".
        # Swapping any edge with any non-edge is a valid move in the 91-edge space.
        
        # Pick random edge to remove
        e_remove = random.choice(list(current_edges))
        u, v = e_remove
        
        # Pick random non-edge to add
        # We need to ensure e_add is not in current_edges
        while True:
            e_add = random.choice(all_possible_edges)
            if e_add not in current_edges:
                break
        
        # Calculate delta in C4 count
        # Removing (u, v):
        # For every pair (a, b) where a in N(u)\{v} and b in N(v)\{u}:
        # The intersection N(a) ∩ N(b) changes.
        # Actually, it's easier to just recompute the C4 contribution of affected vertices?
        # No, C4s are global.
        
        # Let's use the property:
        # C4s containing (u,v) are destroyed.
        # C4s containing (a,b) [the new edge] are created.
        # But C4s can contain multiple edges.
        
        # Alternative:
        # Delta = C4s_created_by_adding(e_add) - C4s_destroyed_by_removing(e_remove)
        # This is only true if the sets of C4s are disjoint.
        # A C4 is a set of 4 vertices.
        # If a C4 contains both (u,v) and (a,b), it is both destroyed and created?
        # No, if we remove (u,v) and add (a,b), the C4 exists in the new graph if all 4 edges are present.
        # It existed in the old graph if all 4 edges were present.
        
        # Let's calculate the number of C4s through (u,v) in the current graph.
        # Let this be D.
        # Let's calculate the number of C4s through (a,b) in the graph where (u,v) is removed and (a,b) is added.
        # Let this be C.
        # The net change is C - D.
        
        # How to calculate D?
        # D = number of pairs (x, y) such that x in N(u), y in N(v), x != v, y != u, x != y, and (x, y) is an edge.
        # Wait, a C4 through (u,v) is u-w-x-v-u.
        # So w is neighbor of u, x is neighbor of v.
        # Edge (w, x) must exist.
        # Also w != x.
        
        def count_c4_through(u, v, adj):
            # u, v are adjacent
            nu = adj[u]
            nv = adj[v]
            count = 0
            # We need to count pairs (w, x) with w in nu\{v}, x in nv\{u}, w != x, and (w, x) in edges
            # To be fast, iterate over smaller set
            if len(nu) < len(nv):
                for w in nu:
                    if w == v: continue
                    for x in nv:
                        if x == u or x == w: continue
                        if x in adj[w]:
                            count += 1
            else:
                for x in nv:
                    if x == u: continue
                    for w in nu:
                        if w == v or w == x: continue
                        if w in adj[x]:
                            count += 1
            return count // 2 # Each C4 is counted twice? 
            # C4: u-w-x-v-u. 
            # Pairs (w,x) and (x,w)? No, w is from nu, x is from nv.
            # The cycle is u-w-x-v-u.
            # The pair (w,x) is unique for this C4 given the edge (u,v).
            # Wait. If we have u-w-x-v-u, the edges are (u,w), (w,x), (x,v), (v,u).
            # The pair is (w,x).
            # Is there another representation?
            # If we consider the other diagonal? No, we are counting C4s *containing* the edge (u,v).
            # Each such C4 corresponds to exactly one pair (w,x) where w~u, x~v, w~x.
            # So count is correct. No division by 2 needed for "through edge".
            # Let's re-verify.
            # C4 vertices: u, v, w, x.
            # Edges: (u,v), (u,w), (v,x), (w,x).
            # This is a C4.
            # The pair (w,x) is identified.
            # Could (w,x) be identified differently?
            # w is neighbor of u. x is neighbor of v.
            # If we swapped roles? No, u and v are fixed.
            # So count is the number of such pairs.
            return count

        # Calculate D: C4s destroyed by removing (u,v)
        D = count_c4_through(u, v, adj)
        
        # To calculate C: C4s created by adding (a,b) to graph G' = G - (u,v)
        # We need to count pairs (w, x) in G' such that w~a, x~b, w~x.
        # Note: G' is G with (u,v) removed.
        # So when checking w~x, we must ensure (w,x) is in G' (i.e., in G and not (u,v)).
        # Also, the neighbors of a and b in G' are their neighbors in G, except if u or v are involved.
        # But a,b are non-edges, so a!=u, a!=v, etc? No, a,b can be u,v.
        # If a==u and b==v, then we are adding back the same edge?
        # e_add is a non-edge, e_remove is an edge. So e_add != e_remove.
        # So (a,b) != (u,v).
        
        # Let's implement a fast check for C.
        # We need to count w in N'(a), x in N'(b) such that (w,x) in E'.
        
        # N'(a) = N(a) \ {v if a==u else empty} \ {u if a==v else empty} ?
        # No. N'(a) is N(a) in G, minus any neighbor connected to a via the removed edge?
        # The removed edge is (u,v).
        # So if a==u, N'(u) = N(u) \ {v}.
        # If a==v, N'(v) = N(v) \ {u}.
        # If a is not u or v, N'(a) = N(a).
        
        na = adj[a]
        nb = adj[b]
        
        if a == u:
            na = na - {v}
        if a == v:
            na = na - {u}
        if b == u:
            nb = nb - {v}
        if b == v:
            nb = nb - {u}
        
        # Now count pairs (w, x) with w in na, x in nb, w != x, and (w,x) in E'.
        # E' = E \ {(u,v)}.
        # So (w,x) in E' iff (w,x) in E and (w,x) != (u,v).
        
        C = 0
        # Optimize by iterating smaller set
        if len(na) < len(nb):
            for w in na:
                for x in nb:
                    if w == x: continue
                    # Check if (w,x) is in E'
                    # (w,x) is in E if x in adj[w]
                    # (w,x) != (u,v)
                    if x in adj[w]:
                        if (w == u and x == v) or (w == v and x == u):
                            continue
                        C += 1
        else:
            for x in nb:
                for w in na:
                    if w == x: continue
                    if w in adj[x]:
                        if (w == u and x == v) or (w == v and x == u):
                            continue
                        C += 1
                        
        delta = C - D
        
        # Simulated Annealing Acceptance
        if delta < 0 or random.random() < np.exp(-delta / T):
            # Accept move
            current_edges.discard((u, v))
            current_edges.add((a, b))
            
            adj[u].discard(v)
            adj[v].discard(u)
            adj[a].add(b)
            adj[b].add(a)
            
            current_c4s += delta
            moves += 1
            
            if current_c4s < best_c4s:
                best_c4s = current_c4s
                best_edges = set(current_edges)
                
        T *= cooling_rate
        
    # Final check and return
    # Ensure we return exactly 91 edges
    # best_edges should have 91 edges.
    
    return [[e[0], e[1]] for e in best_edges]
