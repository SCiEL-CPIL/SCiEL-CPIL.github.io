"""Combined program: deterministic C4-free construction + fast verification."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _popcount(x):
    """Count set bits in integer x."""
    count = 0
    while x:
        x &= x - 1
        count += 1
    return count

def _has_c4_bitmask(n, adj):
    """Check if graph has a C4 using bitmasks. Returns True if C4 found."""
    for u in range(n):
        for v in range(u + 1, n):
            common = adj[u] & adj[v]
            if _popcount(common) >= 2:
                return True
    return False

def _max_independent_set_branching(n, adj, target=11):
    """
    Check if there exists an independent set of size >= target.
    Uses branching with bitmasks and greedy upper bounds.
    Returns True if independent set of size >= target exists.
    """
    if target <= 0:
        return True
    
    def has_is_of_size(vertices_mask, remaining):
        if remaining <= 0:
            return True
        if not vertices_mask:
            return False
        
        count = _popcount(vertices_mask)
        if count < remaining:
            return False
        
        # Greedy upper bound: repeatedly pick vertex with minimum degree in remaining set
        # This gives a quick upper bound on independent set size
        # Use a simpler bound: count / (avg_degree + 1) is not tight, so just use count
        
        # Pick vertex to branch on - use vertex with highest degree in remaining set
        # to prune more effectively
        best_v = -1
        best_deg = -1
        temp = vertices_mask
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = _popcount(adj[v] & vertices_mask)
            if deg > best_deg:
                best_deg = deg
                best_v = v
            temp ^= lsb
        
        if best_v < 0:
            return False
        
        v = best_v
        
        # Option 1: include v
        # Remove v and all its neighbors
        new_mask = vertices_mask & ~(1 << v) & ~adj[v]
        if has_is_of_size(new_mask, remaining - 1):
            return True
        
        # Option 2: exclude v
        if has_is_of_size(vertices_mask ^ (1 << v), remaining):
            return True
        
        return False
    
    all_vertices = (1 << n) - 1
    return has_is_of_size(all_vertices, target)

def _verify_graph(n, edges):
    """Verify the graph is C4-free and has no independent set of size 11."""
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    if _has_c4_bitmask(n, adj):
        return False
    
    if _max_independent_set_branching(n, adj, 11):
        return False
    
    return True

def _construct_39_vertex_graph():
    """
    Construct a C4-free graph on 39 vertices with independence number < 11.
    Uses a systematic approach: start with 13 triangles, then add edges between
    triangles using a greedy strategy that maximizes inter-triangle edges while
    maintaining C4-freeness.
    """
    n = 39
    deadline = time.time() + 110  # Leave margin for verification
    
    best_result = None
    best_alpha = float('inf')
    
    for seed in range(500):
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
        
        # Greedy: try to add all possible inter-triangle edges that don't create C4
        # For each pair of triangles, determine which vertex pairs can be connected
        # without creating a C4.
        
        # Track which vertex in each triangle is "used" for inter-triangle connections
        # to avoid C4s. A C4 occurs if two vertices from triangle i connect to two
        # different vertices from triangle j.
        
        # Strategy: for each pair of triangles (i,j), allow connections only through
        # a "hub" vertex in each triangle. But this limits flexibility.
        
        # Better strategy: maintain for each triangle a set of "available" vertices
        # that can connect to other triangles without creating C4s.
        
        # Actually, the constraint is: if u1, u2 in triangle i and v1, v2 in triangle j
        # with u1-v1 and u2-v2 edges (u1!=u2, v1!=v2), then C4 exists.
        # So for each pair of triangles, the bipartite graph between them must be
        # a matching (at most one edge per vertex on each side) OR all edges must share
        # a common vertex on one side.
        
        # Let's use a simpler approach: randomly add edges and check C4-freeness.
        
        # Generate all possible inter-triangle edges
        inter_edges = []
        for i in range(13):
            for j in range(i+1, 13):
                for ui in range(3*i, 3*i+3):
                    for vj in range(3*j, 3*j+3):
                        inter_edges.append((ui, vj))
        
        random.shuffle(inter_edges)
        
        # Try to add as many edges as possible
        added_count = 0
        for ui, vj in inter_edges:
            if time.time() > deadline:
                break
            if added_count >= 60:  # Limit to keep verification fast
                break
            
            # Check if adding (ui, vj) creates a C4
            # A C4 exists if there are u' != ui in same triangle as ui, and v' != vj in same
            # triangle as vj, such that u'-v' is an edge.
            tri_i = ui // 3
            tri_j = vj // 3
            
            creates_c4 = False
            for u_prime in range(3*tri_i, 3*tri_i+3):
                if u_prime == ui:
                    continue
                for v_prime in range(3*tri_j, 3*tri_j+3):
                    if v_prime == vj:
                        continue
                    if (adj[u_prime] >> v_prime) & 1:
                        creates_c4 = True
                        break
                if creates_c4:
                    break
            
            if not creates_c4:
                # Also need to check C4s involving other triangles
                # A C4 could be: ui - vj - x - y - ui where x,y are in other triangles
                # But since ui and vj are only connected to their own triangles and each other,
                # the only possible C4 is the one we checked above.
                
                edges.append([ui, vj])
                adj[ui] |= (1 << vj)
                adj[vj] |= (1 << ui)
                added_count += 1
        
        # Verify C4-freeness
        if _has_c4_bitmask(n, adj):
            continue
        
        # Check independence number
        # Use a faster check: just verify no independent set of size 11
        if not _max_independent_set_branching(n, adj, 11):
            best_result = {'n': n, 'edges': edges.copy()}
            best_alpha = 10
            break
        else:
            # Estimate alpha for ranking
            # Simple greedy estimate
            remaining = (1 << n) - 1
            greedy_alpha = 0
            while remaining:
                # Pick vertex with minimum degree in remaining
                best_v = -1
                best_deg = float('inf')
                temp = remaining
                while temp:
                    lsb = temp & -temp
                    v = lsb.bit_length() - 1
                    deg = _popcount(adj[v] & remaining)
                    if deg < best_deg:
                        best_deg = deg
                        best_v = v
                    temp ^= lsb
                greedy_alpha += 1
                remaining &= ~(1 << best_v) & ~adj[best_v]
            
            if greedy_alpha < best_alpha:
                best_alpha = greedy_alpha
                best_result = {'n': n, 'edges': edges.copy()}
    
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
