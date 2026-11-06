"""Combined program: polarity graph base + local search extension."""
import time
import random
import itertools

# EVOLVE-BLOCK-START

def _build_polarity_graph_q5():
    """
    Construct the polarity graph of PG(2,5).
    n = 31.
    Vertices are points of PG(2,5). Two vertices are adjacent if their dot product is 0 mod 5.
    """
    q = 5
    # Points of PG(2, q)
    points = []
    # Affine points: (1, x, y)
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    # Points at infinity: (0, 1, x)
    for x in range(q):
        points.append((0, 1, x))
    # Point at infinity: (0, 0, 1)
    points.append((0, 0, 1))
    
    n = len(points)
    adj = [0] * n
    edges = []
    
    # Polarity: u . v = 0 mod q
    for i in range(n):
        u = points[i]
        for j in range(i + 1, n):
            v = points[j]
            dot = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % q
            if dot == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                edges.append([i, j])
                
    return n, adj, edges

def _has_c4(n, adj):
    """Check if graph has a C4 using bitmasks. Returns True if C4 found."""
    for u in range(n):
        for v in range(u + 1, n):
            common = adj[u] & adj[v]
            if common & (common - 1):  # at least two bits set
                return True
    return False

def _independent_set_size(n, adj, target=11):
    """
    Check if there exists an independent set of size >= target.
    Returns the size of the maximum independent set found (capped at target).
    Uses branch and bound with bitmasks.
    """
    def max_is(vertices_mask, remaining):
        if remaining <= 0:
            return 0
        if not vertices_mask:
            return 0
        
        # Upper bound: if number of vertices left < remaining, return count
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
        new_mask = vertices_mask & ~(1 << v) & ~adj[v]
        include_val = 1 + max_is(new_mask, remaining - 1)
        
        # Option 2: exclude v
        exclude_val = max_is(vertices_mask ^ (1 << v), remaining)
        
        return max(include_val, exclude_val)
    
    all_vertices = (1 << n) - 1
    return max_is(all_vertices, target)

def _verify_graph(n, edges):
    """Verify the graph is C4-free and has no independent set of size 11."""
    if n < 1:
        return False
    adj = [0] * n
    for u, v in edges:
        if u == v:
            return False
        if u < 0 or u >= n or v < 0 or v >= n:
            return False
        if adj[u] & (1 << v):
            return False # Duplicate edge
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    if _has_c4(n, adj):
        return False
    
    if _independent_set_size(n, adj, 11) >= 11:
        return False
    
    return True

def _try_add_edge(adj, n, u, v):
    """Check if adding edge (u,v) creates a C4. Returns True if safe to add."""
    common = adj[u] & adj[v]
    return not (common & (common - 1))

def _construct_large_graph():
    """
    Construct a graph with n as large as possible.
    Starts with PG(2,5) polarity graph (n=31) and tries to extend via local search.
    """
    deadline = time.time() + 110
    
    # Base graph
    n_base, adj_base, edges_base = _build_polarity_graph_q5()
    
    # Check if base is valid
    if not _verify_graph(n_base, edges_base):
        # Fallback to disjoint triangles
        edges = []
        for t in range(10):
            a, b, c = 3*t, 3*t+1, 3*t+2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}
    
    best_result = {'n': n_base, 'edges': edges_base}
    
    # Try to extend the graph by adding edges between existing vertices
    # This doesn't increase n, but may allow for a denser graph that could be extended
    # Actually, we want to increase n. Let's try adding new vertices.
    
    # Strategy: Try to add one vertex at a time, connecting it to a subset of existing vertices
    # such that no C4 is created and the independence number remains < 11.
    
    # For a new vertex v, its neighborhood S must be such that:
    # 1. For any u, w in existing, |N(u) ∩ N(w) ∩ S| <= 1
    # 2. The new graph has no independent set of size 11
    
    # We'll try random subsets S for the new vertex's neighborhood.
    
    for target_n in range(39, n_base, -1):
        if time.time() > deadline:
            break
        
        found = False
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            random.seed(attempt * 1000 + target_n)
            
            adj = adj_base.copy()
            edges = edges_base.copy()
            n = n_base
            
            success = True
            for _ in range(target_n - n_base):
                new_v = n
                n += 1
                adj.append(0)
                
                # Try to find a valid neighborhood S for new_v
                # S is a subset of {0, ..., new_v-1}
                # We'll try random subsets of varying sizes
                
                best_S = None
                for size in range(15, 0, -1):  # Try larger neighborhoods first
                    if time.time() > deadline:
                        break
                    
                    for trial in range(20):
                        # Pick random subset of size 'size'
                        candidates = list(range(new_v))
                        random.shuffle(candidates)
                        S = set(candidates[:size])
                        
                        # Check if S is valid (no C4)
                        valid = True
                        for u in S:
                            for w in S:
                                if u == w:
                                    continue
                                common = adj[u] & adj[w]
                                # If u and w share 2+ neighbors in existing graph, 
                                # and both are in S, then adding new_v connected to both
                                # creates C4: new_v - u - x - w - new_v
                                # where x is a common neighbor
                                if common & (common - 1):
                                    # Check if the common neighbors are in S
                                    # Actually, the C4 is new_v - u - x - w - new_v
                                    # This requires x to be a common neighbor of u and w
                                    # The condition is: for any u, w in S, N(u) ∩ N(w) must be empty?
                                    # No. If x is in N(u) ∩ N(w), then we have a C4.
                                    # So we need N(u) ∩ N(w) = empty for all u, w in S?
                                    # Wait, if x is in N(u) ∩ N(w), then u-x-w is a path of length 2.
                                    # Adding new_v connected to u and w creates new_v-u-x-w-new_v? No.
                                    # The C4 would be: new_v - u - x - w - new_v.
                                    # This is a 4-cycle if new_v is connected to u and w, and u-x and x-w are edges.
                                    # So yes, if u and w share a neighbor x, and both are in S, we have a C4.
                                    # Therefore, S must be an independent set in the "neighborhood overlap" graph?
                                    # Actually, S must be such that no two vertices in S share a neighbor.
                                    # This means S is an independent set in the graph where u~w if N(u)∩N(w)≠∅.
                                    # Or simpler: for any u, w in S, N(u) ∩ N(w) = ∅.
                                    valid = False
                                    break
                            if not valid:
                                break
                        
                        if valid:
                            # Check independence number
                            # Temporarily add new_v with neighborhood S
                            for s in S:
                                adj[new_v] |= (1 << s)
                                adj[s] |= (1 << new_v)
                            
                            if _independent_set_size(n, adj, 11) < 11:
                                best_S = S
                                break
                            else:
                                # Remove new_v
                                for s in S:
                                    adj[new_v] &= ~(1 << s)
                                    adj[s] &= ~(1 << new_v)
                            break
                        
                        if best_S is not None:
                            break
                    
                    if best_S is not None:
                        break
                
                if best_S is None:
                    success = False
                    break
                else:
                    # Add new_v with neighborhood best_S
                    for s in best_S:
                        adj[new_v] |= (1 << s)
                        adj[s] |= (1 << new_v)
                        edges.append([new_v, s])
            
            if success:
                found = True
                best_result = {'n': n, 'edges': edges}
                break
        
        if found:
            break
    
    return best_result

def construct():
    """
    Construct a C4-free graph with no independent set of size 11.
    Returns {'n': n, 'edges': [[u, v], ...]}.
    """
    start_time = time.time()
    deadline = start_time + 110
    
    try:
        result = _construct_large_graph()
        return result
    except Exception:
        # Fallback to disjoint triangles
        edges = []
        for t in range(10):
            a, b, c = 3*t, 3*t+1, 3*t+2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}

# EVOLVE-BLOCK-END
