"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import deque

P = {'n': 16}
_START_TIME = None

def _bfs_distances(adj, n):
    """Compute all-pairs shortest path distances using BFS."""
    dist = [[-1] * n for _ in range(n)]
    for s in range(n):
        dist[s][s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[s][v] == -1:
                    dist[s][v] = dist[s][u] + 1
                    q.append(v)
    return dist

def _has_diameter_exactly_2(adj, n):
    """Check if graph has diameter exactly 2."""
    dist = _bfs_distances(adj, n)
    max_d = 0
    for i in range(n):
        for j in range(i+1, n):
            if dist[i][j] == -1:
                return False  # disconnected
            if dist[i][j] > 2:
                return False
            max_d = max(max_d, dist[i][j])
    return max_d == 2

def _is_edge_essential(adj, n, u, v):
    """Check if removing edge (u,v) increases diameter above 2 or disconnects."""
    # Build adjacency without edge (u,v)
    adj2 = [list(neigh) for neigh in adj]
    adj2[u].remove(v)
    adj2[v].remove(u)
    return not _has_diameter_exactly_2(adj2, n)

def _validate(edges, n):
    """Full validation: diameter exactly 2, all edges essential."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    if not _has_diameter_exactly_2(adj, n):
        return False
    for u, v in edges:
        if not _is_edge_essential(adj, n, u, v):
            return False
    return True

def _count_edges(edges):
    return len(edges)

def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    n = P['n']
    deadline = _START_TIME + 80.0  # 10s safety margin
    
    # Strategy: search over graphs by adding edges to a base structure
    # Start with a known good structure and try to add edges while maintaining edge-essentiality
    
    # Base: try various initial graphs
    # A promising approach: start with a sparse diameter-2 graph and greedily add edges
    
    best_edges = None
    best_count = 0
    
    # Try different base structures
    # 1. Complete bipartite K(8,8) - but edges not essential
    # 2. Cycle with chords
    # 3. Random search
    
    # Let's try a systematic approach: 
    # For diameter 2 with all edges essential, the graph must be "thin"
    # in some sense. Let's try graphs where each vertex has moderate degree.
    
    # Approach: start from a path or cycle, add edges strategically
    # Key insight: if every edge is essential, the graph is likely to have
    # relatively low edge count. Let's search.
    
    # Try all subsets of a candidate edge set
    # First, generate candidate edges from a few structural templates
    
    # Template 1: 4x4 grid with diagonals
    # Template 2: Two K4's connected in a specific way
    # Template 3: Random graphs with controlled density
    
    import random
    random.seed(42)
    
    # Try random graphs with various densities
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Start with lower densities and increase
    for trial in range(2000):
        if time.time() > deadline:
            break
        
        # Random density between 0.15 and 0.35
        density = random.uniform(0.15, 0.35)
        num_edges = int(density * len(all_possible))
        
        edges = random.sample(all_possible, num_edges)
        
        # Quick check: must be connected and have diameter 2
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        if not _has_diameter_exactly_2(adj, n):
            continue
        
        # Check edge essentiality (expensive)
        valid = True
        for u, v in edges:
            if not _is_edge_essential(adj, n, u, v):
                valid = False
                break
        
        if valid:
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges[:]
    
    # Also try some structured constructions
    # Construction: take a 16-cycle and add specific chords
    # Cycle: 0-1-2-...-15-0
    # Add chords to make diameter 2
    
    # Try: cycle + all chords of length 4 (i.e., connect i to i+4 mod 16)
    edges_struct = []
    for i in range(n):
        edges_struct.append((i, (i+1) % n))
    for i in range(n):
        edges_struct.append((i, (i+4) % n))
    # Remove duplicates
    edges_struct = sorted(set(edges_struct))
    
    adj = [[] for _ in range(n)]
    for u, v in edges_struct:
        adj[u].append(v)
        adj[v].append(u)
    
    if _has_diameter_exactly_2(adj, n):
        valid = True
        for u, v in edges_struct:
            if not _is_edge_essential(adj, n, u, v):
                valid = False
                break
        if valid and len(edges_struct) > best_count:
            best_count = len(edges_struct)
            best_edges = edges_struct[:]
    
    # Try more structured: cycle + chords of length 3
    edges_struct2 = []
    for i in range(n):
        edges_struct2.append((i, (i+1) % n))
    for i in range(n):
        edges_struct2.append((i, (i+3) % n))
    edges_struct2 = sorted(set(edges_struct2))
    
    adj = [[] for _ in range(n)]
    for u, v in edges_struct2:
        adj[u].append(v)
        adj[v].append(u)
    
    if _has_diameter_exactly_2(adj, n):
        valid = True
        for u, v in edges_struct2:
            if not _is_edge_essential(adj, n, u, v):
                valid = False
                break
        if valid and len(edges_struct2) > best_count:
            best_count = len(edges_struct2)
            best_edges = edges_struct2[:]
    
    # If we found something, return it
    if best_edges is not None:
        # Convert to list of lists with int types
        result = [[int(u), int(v)] for u, v in best_edges]
        return result
    
    # Fallback: return a valid small graph (star has diameter 2 but edges not all essential)
    # Actually star: removing any edge from star disconnects, so all edges essential!
    # Star on 16 vertices: 15 edges, diameter 2
    fallback = [[0, v] for v in range(1, n)]
    return fallback

# EVOLVE-BLOCK-END
