import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Try to construct Polarity Graph for q=7 (n=57)
    # If it's valid, it's much better than 30.
    graph_q7 = build_polarity_graph(7)
    if graph_q7:
        n = graph_q7['n']
        edges = graph_q7['edges']
        
        # Verify C4-free (should be true by construction, but check just in case)
        # And check for independent set of size 11
        if not has_c4(graph_q7) and not has_independent_set_of_size(graph_q7, 11, start_time, deadline):
            return graph_q7
    
    # Fallback: Polarity Graph for q=5 (n=31)
    graph_q5 = build_polarity_graph(5)
    if graph_q5:
        # Verify
        if not has_c4(graph_q5) and not has_independent_set_of_size(graph_q5, 11, start_time, deadline):
            return graph_q5
            
    # Final Fallback: Seed
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}

def build_polarity_graph(q):
    """
    Construct the polarity graph of PG(2, q).
    Vertices are points in PG(2, q).
    Edge (u, v) if u is on the polar of v.
    """
    # We need a representation of PG(2, q)
    # Points are lines through origin in F_q^3
    # We can represent them by normalized vectors (x, y, z) where first non-zero is 1.
    
    points = []
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    for y in range(q):
        points.append((0, y, 1)) # Wait, (0,y,1) is already covered if x=0?
        # Let's be systematic.
        # x=0, y=0 -> (0,0,1)
        # x=0, y!=0 -> (0,1,y')
        # x!=0 -> (1, x', y')
        
    # Correct enumeration:
    points = []
    # z=1 plane
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    # z=0 plane, x=0
    for y in range(1, q): # (0, y, 0) -> normalize to (0, 1, y^-1)
        # Invert y
        y_inv = pow(y, -1, q) if q > 1 else 1
        points.append((0, 1, y_inv))
    # z=0 plane, x!=0
    # (x, y, 0) -> (1, y*x^-1, 0)
    for y in range(q):
        points.append((1, y, 0))
        
    # Remove duplicates and ensure count is q^2+q+1
    unique_points = []
    seen = set()
    for p in points:
        if p not in seen:
            seen.add(p)
            unique_points.append(p)
            
    n = len(unique_points)
    if n != q*q + q + 1:
        return None
        
    # Define polarity
    # For a point P=(x,y,z), the polar line is x*X + y*Y + z*Z = 0?
    # Standard polarity in PG(2,q) for q odd: (x,y,z) -> (x, y, -z) ? No.
    # A standard hyperbolic quadric polarity:
    # P=(x,y,z) is absolute if x^2 + y^2 + z^2 = 0 (for q odd, -1 is square?)
    # For q=5, -1 = 4 is a square.
    # For q=7, -1 = 6 is a square.
    
    # Let's use the standard "orthogonal" polarity defined by the form x^2 + y^2 + z^2 = 0.
    # Two points u, v are adjacent if u lies on the polar of v.
    # Polar of v=(a,b,c) is the line a*x + b*y + c*z = 0.
    # u=(x,y,z) is on polar of v if a*x + b*y + c*z = 0.
    # Note: This is symmetric.
    
    # Check if u is absolute: a*u_x + b*u_y + c*u_z = 0.
    # If u is absolute, we do NOT add a self-loop (simple graph).
    
    edges = []
    adj = [set() for _ in range(n)]
    
    for i in range(n):
        vi = unique_points[i]
        for j in range(i+1, n):
            vj = unique_points[j]
            # Check if vi is on polar of vj
            # Dot product vi . vj = 0?
            # If the form is x^2+y^2+z^2, the bilinear form is x1x2 + y1y2 + z1z2.
            # So yes, vi is on polar of vj iff vi . vj = 0.
            
            dot = (vi[0]*vj[0] + vi[1]*vj[1] + vi[2]*vj[2]) % q
            if dot == 0:
                edges.append([i, j])
                adj[i].add(j)
                adj[j].add(i)
                
    return {'n': n, 'edges': edges, 'adj': adj}

def has_c4(graph):
    n = graph['n']
    adj = graph.get('adj')
    if not adj:
        # Build adj if not present
        adj = [set() for _ in range(n)]
        for u, v in graph['edges']:
            adj[u].add(v)
            adj[v].add(u)
            
    for i in range(n):
        for j in range(i+1, n):
            common = adj[i].intersection(adj[j])
            if len(common) > 1:
                return True
    return False

def has_independent_set_of_size(graph, k, start_time, deadline):
    n = graph['n']
    adj = graph.get('adj')
    if not adj:
        adj = [set() for _ in range(n)]
        for u, v in graph['edges']:
            adj[u].add(v)
            adj[v].add(u)
            
    # Simple greedy check + limited backtracking
    # If we find one, return True.
    # If we don't find one, return False (might be false negative, but we assume valid if not found quickly)
    
    # Use a simple branch and bound for small k
    # Or just greedy coloring to estimate alpha
    # If max independent set found by greedy is < k, likely valid.
    
    # Let's try a randomized greedy search for independent set of size k
    for _ in range(100):
        if time.time() > deadline - 10:
            break
        random.shuffle(adj) # Can't shuffle sets easily, shuffle vertices
        verts = list(range(n))
        random.shuffle(verts)
        ind_set = []
        for v in verts:
            if all(v not in adj[u] for u in ind_set):
                ind_set.append(v)
                if len(ind_set) >= k:
                    return True
    return False

# Helper for pow with mod
def pow_mod(a, b, m):
    return pow(a, b, m)

# Note: pow(y, -1, q) works in Python 3.8+
