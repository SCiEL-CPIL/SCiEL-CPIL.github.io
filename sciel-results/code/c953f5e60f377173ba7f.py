"""K3,3-free dense graph on 20 vertices via algebraic construction and fast local search."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 145
    n = 20

    # 1. Algebraic Construction: Polarity Graph of Projective Plane of Order 4
    # The projective plane of order 4 has q^2+q+1 = 21 points.
    # We use a subset of 20 vertices (indices 0..19) and define edges based on orthogonality.
    # Points in PG(2,4) can be represented by non-zero vectors in F_4^3 up to scaling.
    # F_4 = {0, 1, alpha, alpha^2} with alpha^2 = alpha + 1.
    # Multiplication: 1*1=1, 1*a=a, 1*a2=a2, a*a=a2, a*a2=1, a2*a2=a.
    # Addition: 0+x=x, 1+1=0, 1+a=a2, 1+a2=a, a+a=0, a+a2=1, a2+a2=0.
    
    # We generate the 21 points of PG(2,4).
    # Standard representative set:
    # (1,0,0), (0,1,0), (0,0,1)
    # (1,x,0) for x in F4\{0} -> (1,1,0), (1,a,0), (1,a2,0)
    # (1,0,y) for y in F4\{0} -> (1,0,1), (1,0,a), (1,0,a2)
    # (1,x,y) for x in F4\{0}, y in F4\{0} -> 9 points
    # Total 3 + 3 + 3 + 9 = 18? No.
    # Let's list all non-zero vectors in F4^3 and pick one from each 1D subspace.
    # Total non-zero vectors = 4^3 - 1 = 63.
    # Each line has 4-1=3 non-zero vectors.
    # Number of lines (points in dual) = 63/3 = 21.
    
    # We will explicitly construct the 21 points.
    # F4 elements: 0, 1, 2 (alpha), 3 (alpha^2)
    # Let's use a helper for F4 arithmetic
    def f4_add(a, b):
        return (a + b) % 4 if (a < 2 and b < 2) or (a >= 2 and b >= 2) else (a ^ b) 
        # Actually, F4 addition is XOR for the basis {1, alpha}. 
        # 0:00, 1:01, a:10, a2:11.
        # So 0,1,2,3 mapping: 0->0, 1->1, a->2, a2->3.
        # Addition is XOR.
        return a ^ b

    def f4_mul(a, b):
        # 0*x=0
        if a == 0 or b == 0: return 0
        # 1*x=x
        if a == 1: return b
        if b == 1: return a
        # a*a = a2 (2*2=3), a*a2 = 1 (2*3=1), a2*a2 = a (3*3=2)
        if a == 2 and b == 2: return 3
        if a == 2 and b == 3: return 1
        if a == 3 and b == 2: return 1
        if a == 3 and b == 3: return 2
        return 0

    # Generate all 63 non-zero vectors
    vecs = []
    for x in range(4):
        for y in range(4):
            for z in range(4):
                if x == 0 and y == 0 and z == 0: continue
                vecs.append((x,y,z))
    
    # Group into lines (1D subspaces)
    lines = []
    used = [False] * 63
    for i, v in enumerate(vecs):
        if used[i]: continue
        line = [v]
        used[i] = True
        # Add scalar multiples 2*v and 3*v
        for c in [2, 3]:
            w = (f4_mul(c, v[0]), f4_mul(c, v[1]), f4_mul(c, v[2]))
            idx = vecs.index(w)
            line.append(w)
            used[idx] = True
        lines.append(line)
        
    # Select one representative from each line
    points = [line[0] for line in lines]
    # We have 21 points. We need 20 vertices. We can drop one point, say points[20].
    # However, the polarity graph is defined on all 21 points.
    # The graph is self-complementary or close to it?
    # The number of edges in the polarity graph of PG(2,4) is known.
    # It is a (21, 8, 3, 3) strongly regular graph? Or similar.
    # Let's just build the graph on the 20 vertices (indices 0..19) using the first 20 points.
    
    # Polarity: u ~ v if u dot v = 0 in F4.
    # Note: In a projective plane, a point is orthogonal to its line.
    # The polarity graph has no loops if the polarity is null (no absolute points) or we handle it.
    # For PG(2,4), the standard polarity (Hermitian or similar) might have absolute points.
    # Let's use the standard dot product.
    
    adj = [0] * n
    # Map index 0..19 to points[0..19]
    for i in range(n):
        for j in range(i+1, n):
            u = points[i]
            v = points[j]
            dot = f4_add(f4_add(f4_mul(u[0], v[0]), f4_mul(u[1], v[1])), f4_mul(u[2], v[2]))
            if dot == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # Check if this is K3,3 free.
    # The polarity graph of PG(2,q) is K_{3,3}-free for q>=2?
    # Actually, for q=4, it is known to be triangle-free? No, it has triangles.
    # But it is K_{3,3}-free.
    # Let's verify with a fast check.
    
    def is_k33_free(adj_bits):
        for i in range(n):
            for j in range(i + 1, n):
                ij = adj_bits[i] & adj_bits[j]
                if ij.bit_count() < 3:
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if (ij & adj_bits[k]).bit_count() >= 3:
                        return False
        return True

    # If the algebraic construction fails (unlikely), fall back to random
    if not is_k33_free(adj):
         # Fallback: Random greedy
         adj = [0] * n
         all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
         random.shuffle(all_edges)
         for u, v in all_edges:
             adj[u] |= (1 << v)
             adj[v] |= (1 << u)
             if not is_k33_free(adj):
                 adj[u] &= ~(1 << v)
                 adj[v] &= ~(1 << u)
                 # Note: This fallback is slow, but should not trigger.

    # 2. Fast Local Search
    # We want to maximize edges.
    # Strategy: Try to add missing edges. If it creates a K3,3, try to remove an edge that breaks the K3,3 
    # and allows the new edge, or just skip.
    # Better: "Repair" strategy.
    # 1. List all missing edges.
    # 2. Shuffle them.
    # 3. For each missing edge (u,v):
    #    a. Add (u,v).
    #    b. Check for K3,3 violations.
    #    c. If none, keep it.
    #    d. If yes, find a "critical" edge to remove. 
    #       A K3,3 violation is a set of 3 vertices A and 3 vertices B such that all 9 edges between A and B exist.
    #       Actually, the definition in the task is "no three vertices have three common neighbours".
    #       This is equivalent to no K_{3,3}.
    #       If adding (u,v) creates a K_{3,3}, it must be part of some K_{3,3}.
    #       We can find one such K_{3,3} and remove an edge from it that is NOT (u,v).
    #       To maximize edges, we should remove an edge that is "least useful" or just any edge.
    #       Removing any edge from the K_{3,3} breaks it.
    #       To be efficient, we can identify the specific K_{3,3} created.
    
    def find_violation(adj_bits, u, v):
        # Returns a list of edges forming a K3,3 that includes (u,v) or is caused by it?
        # Actually, we just need to find ANY K3,3 in the graph.
        # Since we only added one edge, the violation must involve that edge or be pre-existing (which shouldn't happen).
        # Let's just find any K3,3.
        for i in range(n):
            for j in range(i + 1, n):
                ij = adj_bits[i] & adj_bits[j]
                if ij.bit_count() < 3:
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    common = ij & adj_bits[k]
                    if common.bit_count() >= 3:
                        # We have vertices {i, j, k} and at least 3 common neighbours.
                        # Let C be the set of common neighbours.
                        # We need to find 3 neighbours in C.
                        # The K3,3 is between {i, j, k} and any 3 vertices in C.
                        # Wait, the definition is "no three vertices have three common neighbours".
                        # This means if we pick 3 vertices A={i,j,k}, and they have 3 common neighbours B={x,y,z},
                        # then the bipartite subgraph between A and B is K3,3.
                        # So we can return the edges of this K3,3.
                        # Edges are (i,x), (i,y), (i,z), (j,x), (j,y), (j,z), (k,x), (k,y), (k,z).
                        # We can just return one of these edges to remove.
                        # To be smart, we should return an edge that is not (u,v) if possible.
                        # But any edge works to break the specific K3,3 instance.
                        # However, removing an edge might create other issues? No, removing edges preserves K3,3-freeness.
                        # So we just need to remove ONE edge from this K3,3.
                        # Which one? 
                        # Ideally, we want to keep (u,v). So we remove an edge that is not (u,v).
                        # Let's collect all edges in this K3,3.
                        common_list = [x for x in range(n) if (common >> x) & 1]
                        # Pick first 3
                        b1, b2, b3 = common_list[0], common_list[1], common_list[2]
                        edges = [
                            (i, b1), (i, b2), (i, b3),
                            (j, b1), (j, b2), (j, b3),
                            (k, b1), (k, b2), (k, b3)
                        ]
                        # Filter out (u,v) if it's in there
                        candidates = [e for e in edges if not (e[0] == u and e[1] == v) and not (e[0] == v and e[1] == u)]
                        if candidates:
                            return candidates[0]
                        else:
                            return edges[0]
        return None

    current_edges = sum(b.bit_count() for b in adj) // 2
    missing = [(i, j) for i in range(n) for j in range(i + 1, n) if not (adj[i] >> j) & 1]
    random.shuffle(missing)
    
    # Iterative improvement
    # We can do multiple passes or just one long pass.
    # Given the time budget, we can do many passes.
    
    max_time = deadline - 5
    while time.time() < max_time:
        improved = False
        # Shuffle missing again to avoid order bias
        random.shuffle(missing)
        for u, v in missing[:]: # Iterate over a copy
            if time.time() > max_time:
                break
            if (adj[u] >> v) & 1:
                # Already added? Remove from missing list
                missing.remove((u,v))
                continue
            
            # Try adding
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            # Check validity
            if not is_k33_free(adj):
                # Find a violation and remove an edge
                rem_edge = find_violation(adj, u, v)
                if rem_edge:
                    ra, rb = rem_edge
                    # Remove rem_edge
                    adj[ra] &= ~(1 << rb)
                    adj[rb] &= ~(1 << ra)
                    # Update missing list
                    if ra < rb:
                        missing.append((ra, rb))
                    else:
                        missing.append((rb, ra))
                    # We kept (u,v) and removed rem_edge. Net edge count change: 0.
                    # But we might have improved the structure for future additions.
                    # Is it always beneficial? 
                    # We traded one edge for another. 
                    # However, if rem_edge was "critical" for other potential additions, this might be bad.
                    # But generally, keeping the new edge is good.
                    improved = True
                else:
                    # Should not happen
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
            else:
                # Valid addition
                current_edges += 1
                missing.remove((u, v))
                improved = True
        
        if not improved:
            # No more improvements possible with simple addition/repair
            # We could try pair-exchanges, but the algebraic seed is likely very good.
            # Let's break to save time for final polish or just exit.
            break

    # Final polish: try to add any remaining missing edges without repair (just skip if invalid)
    # This is a fast pass
    for u, v in missing[:]:
        if time.time() > deadline - 1:
            break
        if (adj[u] >> v) & 1:
            continue
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        if not is_k33_free(adj):
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            missing.remove((u, v))

    # Convert to list of edges
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                result.append([i, j])

    return result

# EVOLVE-BLOCK-END
