"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
import sys
import time
from itertools import combinations, permutations

def _enumerate_simple_cycles(adj, n):
    """Enumerate all simple cycles in an undirected graph. Returns set of cycle lengths."""
    cycles = set()
    visited = [[False] * n for _ in range(n)]
    
    def dfs(start, current, length, path, visited):
        if length >= 4:  # girth >= 4
            # Check if current can connect back to start
            if start in adj[current]:
                cycles.add(length)
        # Try to extend
        for neighbor in adj[current]:
            if neighbor == start:
                continue
            if not visited[neighbor]:
                visited[neighbor] = True
                path.append(neighbor)
                dfs(start, neighbor, length + 1, path, visited)
                path.pop()
                visited[neighbor] = False
    
    for start in range(n):
        for first in adj[start]:
            if first > start:  # avoid duplicates
                visited[first] = True
                dfs(start, first, 2, [start, first], visited)
                visited[first] = False
    
    return cycles

def _count_distinct_cycle_lengths(edges, n=12):
    """Count distinct simple cycle lengths in the graph."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    cycle_lengths = _enumerate_simple_cycles(adj, n)
    return len(cycle_lengths) if cycle_lengths else 0

def _validate(edges, n=12):
    """Validate: min degree >= 3, triangle-free, has a cycle, no loops/duplicates."""
    if not edges:
        return False
    # Check no loops or duplicates
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        if u >= n or v >= n or u < 0 or v < 0:
            return False
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
    
    # Check min degree
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    if any(d < 3 for d in deg):
        return False
    
    # Check triangle-free
    adj = set()
    for u, v in edges:
        adj.add((min(u, v), max(u, v)))
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in adj:
                for w in range(v + 1, n):
                    if (u, w) in adj and (v, w) in adj:
                        return False
    
    # Check has a cycle
    cycle_lengths = _enumerate_simple_cycles(
        [[v for u2, v2 in edges for v in [u2, v2] if (u2 == u and v2 != u) or (v2 == u and u2 != u)] for u in range(n)], n
    )
    # Simpler: just check if edges > vertices (necessary but not sufficient for cycle)
    if len(edges) <= n:
        # Could still have a cycle in a component, but for safety
        pass
    # Actually, let's just check: if all components are trees, no cycle
    # For 12 vertices with min degree 3, min edges = 18, so there must be cycles
    # Since min degree 3 on 12 vertices means at least 18 edges, and a forest on 12 vertices has at most 11 edges,
    # any valid graph here MUST have a cycle.
    
    return True

def construct():
    """
    Constructs a graph on vertices 0..11 with minimum degree 3, triangle-free,
    minimizing the number of distinct simple cycle lengths.
    
    Strategy: Search for a graph with 1 distinct cycle length (all 4-cycles).
    Fallback to two disjoint K_3,3 (2 distinct lengths) if not found in time.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin
    
    n = 12
    
    # First, try the known good solution: two K_3,3
    k33_edges = []
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            k33_edges.append([int(u), int(v)])
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            k33_edges.append([int(u), int(v)])
    
    best_edges = k33_edges
    best_score = _count_distinct_cycle_lengths(k33_edges, n)
    
    # Try to find a graph with 1 distinct cycle length
    # Search space: 3-regular bipartite graphs on 12 vertices
    # Bipartition: {0,1,2,3,4,5} and {6,7,8,9,10,11}
    # Each vertex in left must connect to exactly 3 in right (for 3-regular)
    # Total edges = 18
    
    # We need to find a 3-regular bipartite graph where all cycles are the same length.
    # Let's try random search + local improvement
    
    import random
    random.seed(42)
    
    # Precompute: for a 3-regular bipartite graph with parts A={0..5}, B={6..11},
    # adjacency is a 6x6 0-1 matrix with row sums and column sums = 3.
    
    def build_graph_from_matrix(mat):
        """Build edge list from 6x6 binary matrix with row/col sums = 3."""
        edges = []
        for i in range(6):
            for j in range(6):
                if mat[i][j] == 1:
                    edges.append([int(i), int(6 + j)])
        return edges
    
    def is_valid_3reg_bipartite(mat):
        for i in range(6):
            if sum(mat[i]) != 3:
                return False
        for j in range(6):
            if sum(mat[i][j] for i in range(6)) != 3:
                return False
        return True
    
    def get_cycle_lengths(mat):
        edges = build_graph_from_matrix(mat)
        return _count_distinct_cycle_lengths(edges, n)
    
    # Random search over 3-regular bipartite graphs
    # Represent as list of 6 lists, each containing 3 column indices
    def random_3reg_bipartite():
        # Start with all rows having [0,1,2] and try to fix column sums
        # Use a random permutation approach
        # Each row picks 3 columns, each column must appear exactly 3 times total
        # This is equivalent to a 3-regular bipartite graph = a 6x6 0-1 matrix with row/col sums 3
        # Generate via: start with identity * 3, then random swaps
        # Simpler: random assignment
        cols = list(range(6))
        mat = [[0]*6 for _ in range(6)]
        # Each row needs 3 ones
        # Use a greedy + repair approach
        # Or: generate a random 6x6 matrix with row sums 3, then check col sums
        for _ in range(100):
            mat = [[0]*6 for _ in range(6)]
            for i in range(6):
                chosen = random.sample(range(6), 3)
                for j in chosen:
                    mat[i][j] = 1
            # Check column sums
            col_sums = [sum(mat[i][j] for i in range(6)) for j in range(6)]
            if all(cs == 3 for cs in col_sums):
                return mat
        return None
    
    # Try many random instances
    attempts = 0
    while time.time() < deadline and attempts < 5000:
        attempts += 1
        mat = random_3reg_bipartite()
        if mat is None:
            continue
        score = get_cycle_lengths(mat)
        if score == 1:
            edges = build_graph_from_matrix(mat)
            if _validate(edges, n):
                return edges
        if score < best_score:
            best_score = score
            best_edges = build_graph_from_matrix(mat)
    
    # Also try structured constructions
    # Try C4 x K3 type structures
    # C4 x K3: vertices (i,j) for i in 0..3, j in 0..2
    # Edges: (i,j)-(i+1,j) mod 4, and (i,j)-(i,j+1) mod 3
    # This is 3-regular, bipartite (since both C4 and K3... wait K3 has triangles)
    # C4 x K3 is NOT triangle-free because K3 has triangles.
    
    # Try C6 x K2 (ladder): 12 vertices, 3-regular, bipartite
    # Vertices: (i, 0) and (i, 1) for i in 0..5
    # Edges: (i,0)-(i+1,0) mod 6, (i,1)-(i+1,1) mod 6, (i,0)-(i,1)
    ladder_edges = []
    for i in range(6):
        u0 = 2 * i      # (i, 0)
        u1 = 2 * i + 1  # (i, 1)
        # Rail edges
        next_i = (i + 1) % 6
        ladder_edges.append([int(u0), int(2 * next_i)])
        ladder_edges.append([int(u1), int(2 * next_i + 1)])
        # Rung edge
        ladder_edges.append([int(u0), int(u1)])
    
    if _validate(ladder_edges, n):
        score = _count_distinct_cycle_lengths(ladder_edges, n)
        if score < best_score:
            best_score = score
            best_edges = ladder_edges
    
    # Return the best found
    return [[int(u), int(v)] for u, v in best_edges]

# EVOLVE-BLOCK-END
