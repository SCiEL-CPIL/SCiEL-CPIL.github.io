"""Combined program: deterministic C4-free construction + fast verification."""
import time
import itertools

# EVOLVE-BLOCK-START

def _has_c4_bitmask(n, adj):
    """Check if graph has a C4 using bitmasks. Returns True if C4 found."""
    for u in range(n):
        for v in range(u + 1, n):
            common = adj[u] & adj[v]
            # Count bits in common
            # If at least 2 common neighbors, C4 exists
            if common & (common - 1):  # at least two bits set
                return True
    return False

def _independent_set_size_bitmask(n, adj, target=11):
    """Check if there exists an independent set of size >= target using branching with bitmasks."""
    
    def max_is(vertices_mask, remaining):
        if remaining <= 0:
            return 0
        if not vertices_mask:
            return 0
        # Upper bound: if number of vertices left < remaining, return count
        # Count bits
        count = 0
        temp = vertices_mask
        while temp:
            temp &= temp - 1
            count += 1
        if count < remaining:
            return count
        
        # Pick a vertex to branch on - use first set bit
        lsb = vertices_mask & -vertices_mask
        v = lsb.bit_length() - 1
        
        # Option 1: include v
        # Remove v and all its neighbors
        new_mask = vertices_mask & ~(1 << v) & ~adj[v]
        include_val = 1 + max_is(new_mask, remaining - 1)
        
        # Option 2: exclude v
        exclude_val = max_is(vertices_mask ^ (1 << v), remaining)
        
        return max(include_val, exclude_val)
    
    all_vertices = (1 << n) - 1
    return max_is(all_vertices, target)

def _verify_graph(n, edges):
    """Verify the graph is C4-free and has no independent set of size 11."""
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    if _has_c4_bitmask(n, adj):
        return False
    
    if _independent_set_size_bitmask(n, adj, 11) >= 11:
        return False
    
    return True

def _construct_39_vertex_graph():
    """
    Construct a known C4-free graph on 39 vertices with independence number < 11.
    
    We use a construction based on the projective plane of order 5, which has 31 points,
    but we need 39. Instead, let's use a known explicit construction.
    
    Actually, the best known R(C4, K11) > 39 comes from specific constructions.
    Let's build a graph using a greedy approach that we know works.
    
    For now, let's use a simpler approach: start with disjoint cliques and merge them
    carefully to maintain C4-freeness while reducing independence number.
    """
    # A simple construction: 13 disjoint triangles = 39 vertices, alpha = 13 (too high)
    # We need alpha < 11, so we need to add edges between triangles.
    
    # Let's use a different approach: use a known C4-free graph.
    # The Clebsch graph is C4-free with 16 vertices and alpha 5.
    # We can take disjoint copies and connect them.
    
    # For a practical solution within time limits, let's use a randomized construction
    # that we verify.
    
    n = 39
    edges = []
    adj = [0] * n
    
    # Start with 13 triangles
    for t in range(13):
        a, b, c = 3*t, 3*t+1, 3*t+2
        edges.append([a, b])
        edges.append([a, c])
        edges.append([b, c])
        adj[a] |= (1 << b) | (1 << c)
        adj[b] |= (1 << a) | (1 << c)
        adj[c] |= (1 << a) | (1 << b)
    
    # Now add edges between triangles to reduce independence number.
    # We need to be careful not to create C4s.
    
    # A vertex in triangle i can connect to vertices in triangle j only if
    # it doesn't create a C4. Since triangles are cliques, connecting two vertices
    # from different triangles is safe as long as we don't create a 4-cycle.
    
    # To avoid C4: if u in triangle i connects to v in triangle j, then no other
    # vertex in triangle i should connect to any vertex in triangle j that is
    # adjacent to v (but v is adjacent to all in its triangle).
    
    # Actually, if we connect u_i to v_j, and also u'_i to v'_j where u_i != u'_i
    # and v_j != v'_j, we get a C4: u_i - v_j - v'_j - u'_i - u_i (if u_i-u'_i and v_j-v'_j are edges,
    # which they are since they're in triangles).
    
    # So we can only connect at most one vertex from triangle i to at most one vertex from triangle j.
    # Or, we can connect multiple vertices but they must all go to the same vertex in the other triangle.
    
    # Let's connect triangle i to triangle j by connecting vertex 0 of triangle i to vertex 0 of triangle j.
    # This creates a "chain" of triangles.
    
    # But this might not reduce alpha enough. Let's try a more systematic approach.
    
    # Use a known construction: the graph from the 39-vertex C4-free graph with alpha 10.
    # This is a well-known result in extremal graph theory.
    
    # For simplicity and reliability, let's use a construction that we can verify quickly.
    
    # Alternative: use a random construction with many restarts.
    
    import random
    
    best_result = None
    deadline = time.time() + 100  # Leave margin for verification
    
    for seed in range(200):
        if time.time() > deadline:
            break
        
        random.seed(seed)
        
        # Start with 13 triangles
        adj = [0] * n
        edges = []
        for t in range(13):
            a, b, c = 3*t, 3*t+1, 3*t+2
            edges.append([a, b])
            edges.append([a, c])
            edges.append([b, c])
            adj[a] |= (1 << b) | (1 << c)
            adj[b] |= (1 << a) | (1 << c)
            adj[c] |= (1 << a) | (1 << b)
        
        # Add random edges between triangles while maintaining C4-freeness
        for _ in range(50):  # Try to add 50 more edges
            if time.time() > deadline:
                break
            
            # Pick two different triangles
            i, j = random.sample(range(13), 2)
            if i > j:
                i, j = j, i
            
            # Pick a vertex from triangle i and a vertex from triangle j
            ui = random.choice([3*i, 3*i+1, 3*i+2])
            vj = random.choice([3*j, 3*j+1, 3*j+2])
            
            # Check if adding this edge creates a C4
            # A C4 is created if there exist u' in triangle i (u' != ui) and v' in triangle j (v' != vj)
            # such that ui-vj, ui-u', u'-v', v'-vj form a cycle.
            # Since triangles are cliques, ui-u' and vj-v' are always edges.
            # So we need to check if there's any u' != ui in triangle i that is adjacent to any v' != vj in triangle j.
            
            # Actually, the C4 would be: ui - vj - v' - u' - ui
            # This requires: ui-vj (new edge), vj-v' (edge in triangle j), v'-u' (inter-triangle edge), u'-ui (edge in triangle i)
            # So we need to check if there's already an edge between triangle i and triangle j.
            
            # If there's already an edge between triangle i and triangle j, adding another edge might create a C4.
            # Specifically, if there's an edge u'-v' where u' in triangle i, v' in triangle j, u' != ui, v' != vj,
            # then we get a C4.
            
            # Check if adding (ui, vj) creates a C4
            creates_c4 = False
            for u_prime in range(3*i, 3*i+3):
                if u_prime == ui:
                    continue
                for v_prime in range(3*j, 3*j+3):
                    if v_prime == vj:
                        continue
                    # Check if u_prime-v_prime is an edge
                    if (adj[u_prime] >> v_prime) & 1:
                        creates_c4 = True
                        break
                if creates_c4:
                    break
            
            if creates_c4:
                continue
            
            # Also check the other direction: could this create a C4 with vertices from other triangles?
            # No, because C4 involves only 4 vertices, and the only edges between triangles i and j are the ones we've added.
            # A C4 involving triangles i and j would need 2 vertices from each, which we've checked.
            
            # Add the edge
            edges.append([ui, vj])
            adj[ui] |= (1 << vj)
            adj[vj] |= (1 << ui)
        
        # Check if this graph is valid
        if _has_c4_bitmask(n, adj):
            continue
        
        if _independent_set_size_bitmask(n, adj, 11) < 11:
            best_result = {'n': n, 'edges': edges.copy()}
            break
    
    if best_result:
        return best_result
    
    # Fallback: return a smaller valid graph
    # 10 disjoint triangles: 30 vertices, alpha = 10
    edges = []
    for t in range(10):
        a, b, c = 3*t, 3*t+1, 3*t+2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}

def construct():
    """Construct a C4-free graph with no independent set of size 11, maximizing n."""
    start_time = time.time()
    
    # Try to construct a 39-vertex graph
    result = _construct_39_vertex_graph()
    
    # Verify the result
    if _verify_graph(result['n'], result['edges']):
        return result
    
    # Fallback
    edges = []
    for t in range(10):
        a, b, c = 3*t, 3*t+1, 3*t+2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}

# EVOLVE-BLOCK-END
