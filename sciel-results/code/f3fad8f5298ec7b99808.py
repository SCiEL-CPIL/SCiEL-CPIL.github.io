"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import random
import time

def _has_path_of_length_2(adj, a, b, edge):
    """Check if there's a 2-path between a and b avoiding the specified edge (a,b)."""
    u, v = edge
    # For each neighbor of a (excluding b if b is the other endpoint), check if that neighbor is adjacent to b
    for m in adj[a]:
        if m == v:
            continue
        if b in adj[m]:
            return True
    return False

def _diameter_is_exactly_two(adj):
    """Check diameter is exactly 2 (no isolated pairs at distance > 2, and at least one pair at distance 2)."""
    n = len(adj)
    has_dist_2 = False
    for i in range(n):
        for j in range(i + 1, n):
            if i in adj[j]:
                continue
            # i and j are not adjacent, check if distance is 2
            found = False
            for m in adj[i]:
                if j in adj[m]:
                    found = True
                    has_dist_2 = True
                    break
            if not found:
                return False
    return has_dist_2

def _is_critical(adj, edge):
    """Check if removing this edge increases diameter above 2 or disconnects."""
    a, b = edge
    # After removing (a,b), check if a and b still have distance <= 2
    # They need a 2-path avoiding this edge
    if _has_path_of_length_2(adj, a, b, edge):
        # Distance a-b is still 2, so this edge might not be critical for a-b
        # But we need to check ALL pairs. However, for efficiency, if a-b still has dist 2,
        # the criticality must come from some other pair becoming disconnected or dist > 2.
        # This is more complex. Let's do a full check but optimized.
        pass
    # Full check: remove edge, check diameter
    # We'll simulate removal by checking each pair
    n = len(adj)
    # Create a modified adjacency where edge (a,b) is removed
    # For pairs (i,j), check if dist <= 2 in G - {(a,b)}
    # This is expensive. Let's think of a better way.
    
    # Actually, for a triangle-free graph with diameter 2, an edge (a,b) is critical 
    # iff there is no other 2-path between a and b. Why?
    # Because if every edge is on a unique 2-path for its endpoints, then removing it 
    # makes those endpoints have distance > 2 (or disconnected), increasing diameter.
    # And if an edge is NOT the only 2-path for its endpoints, then removing it doesn't 
    # affect a-b distance, but might affect other pairs? 
    # In a triangle-free graph, the only pairs affected by removing (a,b) are pairs 
    # whose all 2-paths go through (a,b). But since the graph is triangle-free, 
    # a 2-path from i to j through a-b would be i-a-b-j or i-b-a-j. 
    # For i-a-b-j to be a valid 2-path, i must be adjacent to a and b adjacent to j, 
    # and i != b, j != a. Also, since no triangles, i and b are not adjacent (otherwise 
    # i-a-b would be a triangle), and a and j are not adjacent.
    # So removing (a,b) breaks 2-paths of the form i-a-b-j where i ~ a, b ~ j, i != b, j != a.
    # These pairs (i,j) would need alternative 2-paths.
    
    # This is getting complex. Let's just do the full check but with early exit.
    
    # Actually, the key insight: in a triangle-free graph with diameter 2, 
    # an edge (a,b) is critical IF AND ONLY IF (a,b) is the UNIQUE 2-path between a and b.
    # Proof: If (a,b) is the unique 2-path for a-b, removing it makes dist(a,b) >= 3 or 
    # disconnected, so diameter > 2. 
    # If (a,b) is NOT the unique 2-path for a-b, then dist(a,b) = 2 still after removal. 
    # But could removing (a,b) make some OTHER pair have distance > 2? 
    # Suppose pair (i,j) had all its 2-paths going through (a,b). A 2-path through (a,b) 
    # means i ~ a and b ~ j (with i-a-b-j) or i ~ b and a ~ j (with i-b-a-j). 
    # Since graph is triangle-free, i is NOT adjacent to b (else i-a-b triangle), 
    # and a is NOT adjacent to j (else i-b-a-j... wait, a-j edge would form triangle i-b-a? No.)
    # Let's be careful: if i ~ a and b ~ j, and we consider path i-a-b-j, this is a 2-path 
    # from i to j. For this to be the ONLY 2-path, i and j must have no other common neighbor.
    # But i and b are not adjacent (triangle-free, since i~a and a~b), and a and j are not 
    # adjacent (triangle-free, since b~j and a~b... wait, a~b and b~j doesn't imply a~j is forbidden. 
    # Actually a~j IS possible as long as a and j aren't already connected through another edge forming triangle. 
    # Triangle would be a-b-j-a if a~j. So a~j is forbidden if a~b and b~j. Yes! So a and j are NOT adjacent.
    # Similarly i and b are NOT adjacent.
    # So the only 2-paths from i to j that use edge (a,b) are via i-a-b-j. Are there others? 
    # i-a-b-j uses edges (i,a), (a,b), (b,j). That's a 3-edge path, not 2! Wait, I'm confusing myself.
    # A 2-path from i to j is a path of length 2: i-m-j where m is a common neighbor. 
    # So a 2-path "through" edge (a,b) doesn't make sense for a 2-path. 
    # A 2-path from i to j is i-m-j. Removing edge (a,b) affects this only if m=a and b=j (so path i-a-j, 
    # but then we need i~a and a~j, and the path uses edges (i,a) and (a,j), not (a,b)), 
    # or m=b and a=j (path i-b-a, using edges (i,b) and (b,a)). 
    # So removing (a,b) affects 2-paths i-a-j (if a~j) or i-b-a (if i~b). 
    # But in a triangle-free graph, if i~a and a~b, then i NOT~ b. If a~j and a~b, then j NOT~ b.
    # So a 2-path i-a-j requires i~a and a~j. If we remove (a,b), this path is unaffected unless 
    # one of the edges IS (a,b), i.e., i=b or j=b. If i=b, path is b-a-j, using (b,a) and (a,j). 
    # Removing (a,b) breaks this. Similarly if j=b, path is i-a-b, using (i,a) and (a,b).
    
    # OK this is getting complicated. Let me just do the direct check for small n=16.
    
    # Direct check: remove edge (a,b), compute diameter.
    n = len(adj)
    # adj is a list of sets
    # Create adj2 without (a,b)
    # But doing this for each edge is O(n^3) per edge, and we have O(n^2) edges, so O(n^5) total.
    # For n=16, n^5 = 1,048,576, which is fine.
    
    # Actually, let's just check: after removing (a,b), is the diameter still 2?
    # We need to check all pairs (i,j) with i<j, i!=j:
    #   if i~j: ok
    #   else: check if there exists m such that i~m and m~j in G-{(a,b)}
    
    # For efficiency, let's precompute for each pair (i,j) the number of common neighbors.
    # If count >= 2, then removing one edge won't disconnect them at distance 2 (unless the edge is 
    # the only connection... no, if they have >= 2 common neighbors, removing one edge (a,b) 
    # only affects them if one of the common neighbors is a or b AND the other endpoint is the other of a,b.
    
    # Let me just do the brute force check. For n=16, it's fast enough.
    
    for i in range(n):
        for j in range(i + 1, n):
            if i == a and j == b:
                continue
            if i == b and j == a:
                continue
            if i in adj[j]:
                continue
            # Check if i and j have a 2-path in G - {(a,b)}
            found = False
            for m in adj[i]:
                if j in adj[m]:
                    # Path i-m-j. Check if this path uses edge (a,b).
                    # It uses (i,m) and (m,j). It uses (a,b) only if {(i,m),(m,j)} contains {a,b} as a set.
                    if (i == a and m == b) or (i == b and m == a) or (m == a and j == b) or (m == b and j == a):
                        # This 2-path uses the removed edge
                        continue
                    found = True
                    break
            if not found:
                return False
    return True

def construct():
    n = 16
    deadline = time.time() + 80  # 80 seconds, leave 10s safety margin
    
    # Start with K_{8,8}
    adj = [set() for _ in range(n)]
    edges = []
    for a in range(8):
        for b in range(8, 16):
            adj[a].add(b)
            adj[b].add(a)
            edges.append([a, b])
    
    # Verify K_{8,8} is valid
    # Diameter is 2 (triangle-free, bipartite, complete bipartite)
    # Every edge is critical: removing (a,b) makes dist(a,b) = 2 still? 
    # No! In K_{8,8}, removing (a,b) where a in left, b in right: 
    # dist(a,b) is still 2 via a-b' where b' is any other right vertex, and b'-b... wait.
    # a ~ b' (b' != b), b' ~ ? b. b' and b are both in right, not adjacent. 
    # So dist(a,b) after removing (a,b): a ~ b' (any b' != b), b' not ~ b. 
    # a ~ b'' ~ ? No, b'' not ~ b.
    # So dist(a,b) = ? a's neighbors are all right vertices except b. b's neighbors are all left vertices except a.
    # Common neighbors of a and b: none! Because a's neighbors are right\\{b}, b's neighbors are left\\{a}, disjoint.
    # So dist(a,b) = infinity after removing (a,b). K_{8,8} - (a,b) is disconnected!
    # So K_{8,8} IS valid: every edge is a bridge in some sense.
    
    # Can we add edges? Only within left or within right.
    # Try adding an edge within left: say (0,1).
    # Now check if all original edges are still critical and the new edge is critical.
    
    # Let's try to build up by adding intra-partition edges.
    best_edges = list(edges)
    best_adj = [s.copy() for s in adj]
    
    # Try adding each possible intra-partition edge
    candidates = []
    for i in range(8):
        for j in range(i+1, 8):
            candidates.append((i, j))
    for i in range(8, 16):
        for j in range(i+1, 16):
            candidates.append((i, j))
    
    random.shuffle(candidates)
    
    current_edges = list(edges)
    current_adj = [s.copy() for s in adj]
    
    for u, v in candidates:
        if time.time() > deadline - 10:
            break
        
        # Try adding (u,v)
        current_adj[u].add(v)
        current_adj[v].add(u)
        current_edges.append([u, v])
        
        # Check diameter is exactly 2
        valid = _diameter_is_exactly_two(current_adj)
        
        if valid:
            # Check all edges are critical
            all_critical = True
            for e in current_edges:
                if not _is_critical(current_adj, e):
                    all_critical = False
                    break
            if all_critical:
                best_edges = list(current_edges)
                best_adj = [s.copy() for s in current_adj]
            else:
                # Revert
                current_adj[u].remove(v)
                current_adj[v].remove(u)
                current_edges.pop()
        else:
            # Revert
            current_adj[u].remove(v)
            current_adj[v].remove(u)
            current_edges.pop()
    
    # Return the best
    out = []
    for e in best_edges:
        out.append([int(e[0]), int(e[1])])
    return out

# EVOLVE-BLOCK-END
