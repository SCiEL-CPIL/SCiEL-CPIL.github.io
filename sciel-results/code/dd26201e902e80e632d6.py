"""Program for hexagon-free graphs on 40 vertices with many edges."""
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    n = 40
    # All possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m_total = len(all_edges)  # 780
    
    # Precompute all 6-cycles (sets of 6 edges)
    # A 6-cycle is a sequence of 6 distinct vertices v0-v1-v2-v3-v4-v5-v0
    # The edges are (v0,v1), (v1,v2), (v2,v3), (v3,v4), (v4,v5), (v5,v0)
    # We need to enumerate all such cycles up to rotation and reflection
    # To avoid duplicates, we fix v0=0 and require v1 < v5 (to break reflection symmetry)
    # Actually, let's just enumerate all simple 6-cycles and use a set of frozensets of edge indices
    
    # Map edge to index
    edge_to_idx = {}
    for idx, (u, v) in enumerate(all_edges):
        edge_to_idx[(u, v)] = idx
        edge_to_idx[(v, u)] = idx
    
    # Enumerate all 6-cycles
    # For each set of 6 vertices, there are (6-1)!/2 = 60 Hamiltonian cycles
    # But we only care about the specific 6-cycles in the complete graph K40
    # Total number of 6-cycles in K40 is C(40,6) * 60 = 3,838,380 * 60 = ~230 million
    # That's too many to precompute all. Let's use a different approach.
    
    # Instead, we'll use simulated annealing with on-the-fly cycle checking
    # Or we can use a smarter initialization and local search
    
    # Let's use a greedy construction followed by simulated annealing
    # Start with a random graph and optimize
    
    random.seed(42)
    
    # Initial solution: random graph with moderate edge density
    # Start with ~100 edges
    initial_density = 0.13  # ~101 edges
    graph = np.zeros((n, n), dtype=bool)
    
    # Create initial random graph
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < initial_density:
                graph[i, j] = True
                graph[j, i] = True
    
    def count_edges(g):
        return int(np.sum(g) // 2)
    
    def has_c6(g):
        """Check if graph g contains a 6-cycle. Returns True if any 6-cycle exists."""
        # For each pair of vertices at distance 2, check if they have a common neighbor
        # that forms a 6-cycle... This is complex.
        
        # Better: for each ordered pair (u, v) with u != v, compute the number of
        # length-3 paths from u to v. If any such path exists and there's also a direct
        # edge... no, that's for 4-cycles.
        
        # For 6-cycles: A 6-cycle exists iff there exist vertices v0,...,v5 such that
        # all consecutive pairs are edges and v5-v0 is an edge.
        
        # Efficient check: For each pair (u, w) of non-adjacent vertices, 
        # count paths of length 5 from u to w. If > 0, and u-w is not an edge, 
        # that doesn't directly give a 6-cycle.
        
        # Actually, a 6-cycle through edge (u,v) exists if there's a path of length 5
        # from v back to u not using edge (u,v).
        
        # Let's use a different approach: for each vertex v, consider its neighbors.
        # A 6-cycle containing v looks like v-a-b-c-d-e-v where a,b,c,d,e are distinct
        # and not v.
        
        # Simpler brute force for checking: iterate over all 6-tuples of distinct vertices
        # where v0 is fixed to the smallest to reduce symmetry.
        # This is still expensive but we can optimize.
        
        # Let's use the following: for each set of 6 vertices, check if they form a C6.
        # But C(40,6) = 3.8M, and for each we check 60 cycles. Too slow for repeated calls.
        
        # Alternative: use the fact that in a C6-free graph, for any two non-adjacent 
        # vertices u, w, the number of common neighbors at distance... 
        
        # Let me just do a direct check with some pruning.
        # For each vertex v, get its neighbors N(v).
        # For each pair (a, b) in N(v), a != b:
        #   For each pair (d, e) in N(v), d != e, d != a, e != b, d < e:
        #     Check if there's a path a-x-y-z-b or a-x-y-b where x,y,z are not in {v,a,b,d,e}
        #     and edges exist appropriately.
        
        # This is getting complex. Let me just use a simple but potentially slower check
        # that works for the sizes we're dealing with.
        
        # Actually, for 40 vertices, let's just check all 6-cycles by iterating
        # over combinations of 6 vertices.
        
        for verts in combinations(range(n), 6):
            v0, v1, v2, v3, v4, v5 = verts
            # Check all 60 Hamiltonian cycles on these 6 vertices
            # Fix v0, then permute the other 5, divide by 2 for direction
            for perm in permutations5(verts[1:]):
                u, a, b, c, d, e = v0, perm[0], perm[1], perm[2], perm[3], perm[4]
                # Check if u-a, a-b, b-c, c-d, d-e, e-u are all edges
                if (g[u, a] and g[a, b] and g[b, c] and g[c, d] and g[d, e] and g[e, u]):
                    return True
        return False
    
    # This is too slow. Let me use a different strategy.
    
    # Use a known good construction as starting point, then improve.
    # Start with 8 copies of K5 (104 edges) and try to add more.
    
    graph = np.zeros((n, n), dtype=bool)
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u = 5 * k + a
                v = 5 * k + b
                graph[u, v] = True
                graph[v, u] = True
    
    # Now try to add edges one by one, checking if it creates a C6
    # Use a fast C6 detection
    
    def creates_c6(g, u, v):
        """Check if adding edge (u,v) to graph g creates a 6-cycle."""
        # A new 6-cycle must use the new edge (u,v)
        # So we need a path of length 5 from u to v in g (without using edge u-v)
        # Let's find all paths of length 5 from u to v.
        
        # BFS/DFS for paths of length exactly 5 from u to v
        # State: (current_vertex, path_so_far)
        # We want paths of length 5 (5 edges, 6 vertices)
        
        # Use DFS with memoization on (vertex, depth, visited_set)
        # But visited_set makes it exponential. For 40 vertices and depth 5, it's manageable.
        
        # Better: use dynamic programming
        # dp[d][v] = set of vertices that can reach v in exactly d steps from u
        # But we need to avoid revisiting vertices.
        
        # Let's just do a BFS for paths of length up to 5, tracking the path.
        
        # Actually, let's use the following approach:
        # For each pair of vertices (x, y) that are both neighbors of u or v,
        # check if there's a path of length 3 from x to y not going through u or v.
        
        # Simpler: enumerate all length-5 paths from u to v.
        # A length-5 path from u to v: u - a - b - c - d - v
        # where a,b,c,d are distinct and not in {u,v}
        
        # For each a in N(u):
        #   For each b in N(a) \ {u}:
        #     For each c in N(b) \ {u, a}:
        #       For each d in N(c) \ {u, a, b}:
        #         If v in N(d) and d != v: found a path
        
        nu = [x for x in range(n) if g[u, x]]
        for a in nu:
            if a == v:
                continue
            na = [x for x in range(n) if g[a, x] and x != u]
            for b in na:
                if b == v:
                    continue
                nb = [x for x in range(n) if g[b, x] and x != u and x != a]
                for c in nb:
                    if c == v:
                        continue
                    nc = [x for x in range(n) if g[c, x] and x != u and x != a and x != b]
                    for d in nc:
                        if d == v:
                            continue
                        if g[d, v]:
                            return True
        return False
    
    # Greedy: try to add each missing edge if it doesn't create a C6
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                if not graph[i, j]:
                    if not creates_c6(graph, i, j):
                        graph[i, j] = True
                        graph[j, i] = True
                        improved = True
    
    # Now do simulated annealing to try to improve further
    # By swapping edges: remove one edge, add another
    
    def edge_count(g):
        return int(np.sum(g) // 2)
    
    current_edges = edge_count(graph)
    best_graph = graph.copy()
    best_edges = current_edges
    
    # Simulated annealing
    T = 100.0
    cooling = 0.999
    min_T = 0.1
    
    # Precompute adjacency lists for efficiency
    while T > min_T:
        # Pick a random edge to remove and a random non-edge to add
        # Or just flip a random edge
        for _ in range(100):  # iterations per temperature
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            
            if graph[i, j]:
                # Remove edge
                new_graph = graph.copy()
                new_graph[i, j] = False
                new_graph[j, i] = False
                # Check if this is valid (it always is, removing edges can't create C6)
                delta = -1
                new_edges = current_edges + delta
                if new_edges > best_edges or random.random() < np.exp(delta / T):
                    graph = new_graph
                    current_edges = new_edges
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_graph = graph.copy()
            else:
                # Add edge
                if creates_c6(graph, i, j):
                    continue
                new_graph = graph.copy()
                new_graph[i, j] = True
                new_graph[j, i] = True
                delta = 1
                new_edges = current_edges + delta
                if new_edges > best_edges or random.random() < np.exp(delta / T):
                    graph = new_graph
                    current_edges = new_edges
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_graph = graph.copy()
        T *= cooling
    
    # Use the best graph found
    final_graph = best_graph
    
    # Convert to edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if final_graph[i, j]:
                edges.append([i, j])
    
    return edges

def permutations5(lst):
    """Generate all permutations of a 5-element list."""
    if len(lst) == 0:
        yield ()
    else:
        for i in range(len(lst)):
            for p in permutations5(lst[:i] + lst[i+1:]):
                yield (lst[i],) + p

# EVOLVE-BLOCK-END
