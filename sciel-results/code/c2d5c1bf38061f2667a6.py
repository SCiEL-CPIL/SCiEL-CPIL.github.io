"""Q3-free dense graph on 16 vertices."""
import time
import random
import itertools

DEADLINE = 150.0
SAFETY = 10.0  # seconds margin

def _is_q3(adj, verts):
    """Check if induced subgraph on verts (8 vertices) is isomorphic to Q3."""
    n = len(verts)
    if n != 8:
        return False
    # Q3 has 12 edges, each vertex degree 3
    edges = 0
    for i in range(8):
        deg = 0
        for j in range(i+1, 8):
            if adj[verts[i]][verts[j]]:
                edges += 1
                deg += 1
        if deg != 3:
            return False
    if edges != 12:
        return False
    # Check bipartite with parts of size 4: try all partitions
    # Q3 is bipartite; check if graph is bipartite and has a 4-4 bipartition
    # Actually, a 3-regular bipartite graph on 8 vertices with 12 edges is Q3
    # Check bipartiteness
    color = {}
    for start in range(8):
        v = verts[start]
        if v in color:
            continue
        color[v] = 0
        stack = [v]
        while stack:
            u = stack.pop()
            for w in verts:
                if u == w:
                    continue
                if adj[u][w]:
                    if w in color:
                        if color[w] == color[u]:
                            return False
                    else:
                        color[w] = 1 - color[u]
                        stack.append(w)
    # Count parts
    part0 = sum(1 for v in verts if color.get(v, 0) == 0)
    part1 = 8 - part0
    if part0 != 4 or part1 != 4:
        return False
    return True

def _contains_q3(adj, n):
    """Return list of edges in a Q3 subgraph, or None."""
    for verts in itertools.combinations(range(n), 8):
        if _is_q3(adj, verts):
            # Collect edges in this Q3
            edges = []
            for i in range(8):
                for j in range(i+1, 8):
                    if adj[verts[i]][verts[j]]:
                        edges.append([verts[i], verts[j]])
            return edges
    return None

def _edge_key(e):
    return (min(e[0], e[1]), max(e[0], e[1]))

def construct():
    start_time = time.time()
    n = 16
    best_edges = None
    best_count = -1
    
    def elapsed():
        return time.time() - start_time
    
    def has_time():
        return elapsed() < DEADLINE - SAFETY
    
    # Precompute all 8-subsets for speed
    all_8subsets = list(itertools.combinations(range(n), 8))
    
    # Strategy: random dense graphs, repair, densify
    for attempt in range(200):
        if not has_time():
            break
        
        # Random dense graph
        p = 0.8 + 0.1 * random.random()
        adj = [[False] * n for _ in range(n)]
        edge_set = set()
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    adj[i][j] = adj[j][i] = True
                    edge_set.add((i, j))
        
        # Repair: remove edges to eliminate Q3s
        for _ in range(200):
            if not has_time():
                break
            q3_edges = _contains_q3(adj, n)
            if q3_edges is None:
                break
            # Remove a random edge from the Q3
            e = random.choice(q3_edges)
            ei, ej = e
            adj[ei][ej] = adj[ej][ei] = False
            edge_set.discard((min(ei, ej), max(ei, ej)))
        
        # Densify: try adding edges
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_possible)
        for i, j in all_possible:
            if not has_time():
                break
            if adj[i][j]:
                continue
            # Try adding
            adj[i][j] = adj[j][i] = True
            q3_edges = _contains_q3(adj, n)
            if q3_edges is not None:
                adj[i][j] = adj[j][i] = False
            else:
                edge_set.add((i, j))
        
        # Update best
        count = sum(1 for i in range(n) for j in range(i+1, n) if adj[i][j])
        if count > best_count:
            best_count = count
            best_edges = [[i, j] for i in range(n) for j in range(i+1, n) if adj[i][j]]
    
    if best_edges is None:
        # Fallback: K3,13
        best_edges = [[a, b] for a in range(3) for b in range(3, 16)]
    
    return best_edges
