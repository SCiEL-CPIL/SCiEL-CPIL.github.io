"""Rewritten program: maximum clique via Tomita's algorithm with greedy coloring."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _greedy_color_sort(vertices, adj):
    """
    Greedy coloring to get an ordering and color classes.
    Returns (order, colors) where order is the vertex order and colors[i] is the color of order[i].
    The color of a vertex gives an upper bound on the clique size in its subgraph.
    """
    n = len(vertices)
    if n == 0:
        return [], []
    
    # Work with a list of vertices, assign colors greedily
    uncolored = list(vertices)
    order = []
    colors = []
    
    while uncolored:
        # Pick a vertex, try to assign the smallest possible color
        # For efficiency, process in some heuristic order
        # Simple approach: just take the first uncolored vertex each time
        v = uncolored[0]
        uncolored = uncolored[1:]
        
        # Find the smallest color not used by neighbors already in order
        used_colors = set()
        for u in adj[v]:
            if u in set(order):
                # Find the color of u in our ordering
                for i, ov in enumerate(order):
                    if ov == u:
                        used_colors.add(colors[i])
                        break
        
        # Assign smallest available color
        c = 0
        while c in used_colors:
            c += 1
        
        order.append(v)
        colors.append(c)
    
    return order, colors

def _max_clique_tomita(adj):
    """
    Tomita's algorithm for maximum clique.
    Uses greedy coloring for upper bounds.
    """
    n = P['n']
    deadline = time.time() + 50
    best_clique = []
    
    # Use all vertices as initial P set
    P_set = list(range(n))
    
    def _expand(R, P_list):
        """Recursively expand clique R using candidates P_list."""
        nonlocal best_clique
        
        if time.time() > deadline:
            return
        
        if not P_list:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        
        # Get greedy coloring of P_list
        order, colors = _greedy_color_sort(P_list, adj)
        
        # Process vertices in reverse color order (highest color first)
        # This allows early pruning: if current clique size + color <= best, prune
        for i in range(len(order) - 1, -1, -1):
            v = order[i]
            c = colors[i]
            
            # Upper bound: current clique size + color of v
            if len(R) + c <= len(best_clique):
                # All remaining vertices have color <= c, so no better clique possible
                break
            
            if time.time() > deadline:
                return
            
            # Add v to R and recurse
            new_R = R + [v]
            # New P: neighbors of v that are still in P_list (before v in order)
            new_P = [u for u in order[:i] if u in adj[v]]
            
            _expand(new_R, new_P)
    
    _expand([], P_set)
    
    if not best_clique:
        # Fallback: try single vertices
        for v in range(n):
            best_clique = [v]
            break
    
    return best_clique

def _find_simple_path(adj, u, v, forbidden):
    """Find a simple path from u to v avoiding forbidden vertices (except endpoints)."""
    if u == v:
        return [u]
    visited = {u}
    queue = deque([(u, [u])])
    while queue:
        node, path = queue.popleft()
        if node == v:
            return path
        for nb in adj[node]:
            if nb not in visited and nb not in forbidden:
                visited.add(nb)
                queue.append((nb, path + [nb]))
    return None

def construct():
    adj = _build_adj()
    
    # Find maximum clique
    clique = _max_clique_tomita(adj)
    
    # Ensure we have at least 2 branches
    if len(clique) < 2:
        u, v = P['edges'][0]
        clique = [u, v]
    
    clique = sorted(clique)
    branches = list(clique)
    
    # Build paths: for each pair of branches, find a simple path
    # Since they form a clique, each pair is directly adjacent, so path is just [u, v]
    paths = []
    for u, v in combinations(clique, 2):
        # Verify they are adjacent (they should be since it's a clique)
        if v in adj[u]:
            paths.append([u, v])
        else:
            # Find a path via BFS avoiding other branches
            forbidden = set(branches) - {u, v}
            found = _find_simple_path(adj, u, v, forbidden)
            if found:
                paths.append(found)
            else:
                # Last resort: direct edge or fallback
                paths.append([u, v])
    
    # Convert to proper types
    branches = [int(b) for b in branches]
    paths = [[int(p) for p in path] for path in paths]
    
    return {'branches': branches, 'paths': paths}
# EVOLVE-BLOCK-END
