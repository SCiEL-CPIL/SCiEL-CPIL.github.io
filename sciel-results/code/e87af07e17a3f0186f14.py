"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    n = 20
    # Candidate graph: Complete bipartite graph K10,10
    # Vertices 0-9 in part A, 10-19 in part B
    # Edges are all pairs (u, v) with u in [0,9], v in [10,19]
    # Total 100 edges.
    
    # Pre-compute all 4-cycles in K10,10
    # A 4-cycle in bipartite graph is u1-v1-u2-v2-u1
    # u1, u2 in A (distinct), v1, v2 in B (distinct)
    # Edges: (u1, v1), (u1, v2), (u2, v1), (u2, v2)
    
    A = list(range(10))
    B = list(range(10, 20))
    
    # Map edge to index
    # Edge (u, v) where u < 10, v >= 10
    # Index = u * 10 + (v - 10)
    edge_index = {}
    for u in A:
        for v in B:
            edge_index[(u, v)] = u * 10 + (v - 10)
            
    # List of 4-cycles: each is a tuple of 4 edge indices
    cycles = []
    for i1 in range(10):
        for i2 in range(i1 + 1, 10):
            u1 = A[i1]
            u2 = A[i2]
            for j1 in range(10):
                for j2 in range(j1 + 1, 10):
                    v1 = B[j1]
                    v2 = B[j2]
                    e1 = edge_index[(u1, v1)]
                    e2 = edge_index[(u1, v2)]
                    e3 = edge_index[(u2, v1)]
                    e4 = edge_index[(u2, v2)]
                    cycles.append((e1, e2, e3, e4))
                    
    num_cycles = len(cycles)
    # Convert to numpy array for fast indexing
    cycles_arr = np.array(cycles, dtype=np.int32)
    
    # Initial state: all edges present
    # We want to maximize edges, so start with all 1s and remove edges that cause violations.
    # However, starting with all 1s means many violations.
    # Better: start with a random subset or a known valid structure?
    # Let's start with all 1s and use local search to remove edges.
    # Objective: Maximize sum(mask) - Penalty(mask)
    # Penalty is large if any cycle is non-rainbow.
    # Actually, simpler: Maximize edges subject to all cycles being rainbow.
    # If a cycle is not rainbow, it's invalid.
    # We can use a penalty function:
    # Score = num_edges - M * num_violations
    # If num_violations > 0, score is negative.
    # We want to find a state with num_violations == 0 and max num_edges.
    
    # Let's use a different approach:
    # Start with a sparse valid graph (e.g., star or matching) and add edges?
    # Or start with dense and remove?
    # Removing edges from K10,10 is easier to check?
    # If we remove an edge, we might fix violations but lose an edge.
    # If we add an edge, we might create violations.
    
    # Let's try starting with a random valid configuration.
    # A matching is always valid.
    # Let's start with a random matching of size 10.
    # Then try to add edges.
    
    # Better strategy: Simulated Annealing on the space of subsets of K10,10.
    # State: mask of 100 bits.
    # Energy: -num_edges + M * num_violations.
    # We want to minimize Energy.
    
    # Initialize with a random valid graph.
    # Let's use a random permutation matching.
    mask = np.zeros(100, dtype=np.uint8)
    perm = np.random.permutation(10)
    for i in range(10):
        mask[i * 10 + perm[i]] = 1
        
    def get_violations(mask):
        # Check all cycles
        # For each cycle, get the 4 edge colors?
        # Wait, the problem says "edge-coloured with 20 colours".
        # The colors are part of the solution!
        # I missed this in the quick thought.
        # The task: "construct() -> coloured edges [[u, v, c], ...]"
        # So I need to assign colors too.
        
        # This changes the problem significantly.
        # Variables: edge presence (binary) and edge color (int 0-19).
        # Constraint: For every 4-cycle, the 4 edges must have distinct colors.
        
        # This is a much larger search space.
        # 100 edges * 20 colors = 2000 possibilities per edge? No, 20 choices.
        # Total states: 2^100 * 20^100.
        
        # How to search?
        # 1. Fix a graph structure (e.g., K10,10 or a subgraph).
        # 2. Assign colors to satisfy the constraint.
        
        # If the graph is bipartite, 4-cycles are the main constraint.
        # Can we always color a bipartite graph such that 4-cycles are rainbow?
        # Not necessarily.
        
        # Let's re-read inspirations.
        # Inspiration 1: Simulated Annealing.
        # Inspiration 2: Constructive greedy.
        
        # Let's try a hybrid:
        # 1. Start with a sparse graph (e.g., a tree or matching) which is always colorable.
        # 2. Greedily add edges. For each new edge, try to assign a color that doesn't violate any 4-cycle involving it.
        # 3. If no color works, skip the edge.
        # 4. Use local search to swap colors or remove/add edges to improve.
        
        # This is complex to implement robustly in 120s.
        
        # Alternative:
        # Use a known structure that is easy to color.
        # If the graph has no 4-cycles, it's trivially valid (any coloring works).
        # Max edges in C4-free graph on 20 vertices?
        # Zarankiewicz number z(20, 20; 2, 2).
        # Approx sqrt(20) * 20 = 20 * 4.47 = 89.
        # So we can have ~90 edges with no 4-cycles.
        # If we have no 4-cycles, the condition "every 4-cycle is rainbow" is vacuously true.
        # So the task reduces to: Find a C4-free graph on 20 vertices with max edges.
        # Then assign arbitrary colors (e.g., all 0).
        
        # Is this allowed? "Every 4-cycle ... must receive four distinct colours".
        # If there are no 4-cycles, the condition holds for all 4-cycles (vacuously).
        # Yes, this is a valid strategy.
        
        # So, problem: Maximize edges in a C4-free graph on 20 vertices.
        # This is a well-studied problem.
        # We can use a greedy algorithm or local search to find a dense C4-free graph.
        
        # Let's implement a local search for C4-free graphs.
        # State: adjacency matrix (20x20).
        # Objective: Maximize edges, subject to no 4-cycles.
        # Penalty: -edges + M * num_4_cycles.
        
        # This is simpler than the coloring problem!
        
        pass

    # Let's restart the logic with this insight.
    # 1. Find a dense C4-free graph on 20 vertices.
    # 2. Assign color 0 to all edges.
    # 3. Return the edges.
    
    # Algorithm for C4-free graph:
    # Start with empty graph.
    # Greedy add edges?
    # Or start with complete graph and remove edges?
    # Or use known constructions (e.g., incidence of finite geometry)?
    # For n=20, we can use local search.
    
    # Let's use Simulated Annealing to find a C4-free graph with max edges.
    
    # State: 20x20 boolean matrix (upper triangle).
    # 190 possible edges.
    
    def count_c4(adj):
        # Count 4-cycles
        # A 4-cycle is u-v-w-x-u.
        # We can count by looking at pairs of vertices and their common neighbors.
        # For each pair (u, w), let k = number of common neighbors.
        # The number of 4-cycles involving u and w as opposite vertices is C(k, 2).
        # Total 4-cycles = sum over all pairs (u, w) of C(common_neighbors(u, w), 2).
        # Note: each 4-cycle is counted once (as two pairs of opposite vertices? No).
        # In a 4-cycle u-v-w-x-u, the pairs of opposite vertices are (u, w) and (v, x).
        # So each 4-cycle is counted twice?
        # Let's check.
        # Cycle: 0-1-2-3-0.
        # Opposite pairs: (0,2) and (1,3).
        # Common neighbors of 0 and 2: {1, 3}. C(2, 2) = 1.
        # Common neighbors of 1 and 3: {0, 2}. C(2, 2) = 1.
        # Sum = 2.
        # So total count is 2 * (number of 4-cycles).
        # So num_4_cycles = 0.5 * sum.
        
        # Efficient calculation:
        # adj is 20x20 boolean matrix.
        # Common neighbors = adj @ adj (matrix multiplication).
        # diag(common_neighbors) is degree, so we ignore it.
        # off-diagonal entries M[i, j] = number of common neighbors.
        # Sum of C(M[i, j], 2) for i < j.
        
        common = adj.astype(np.int32) @ adj.astype(np.int32)
        # Zero out diagonal
        np.fill_diagonal(common, 0)
        # We only need upper triangle
        iu = np.triu_indices(20, k=1)
        vals = common[iu]
        # C(k, 2) = k * (k-1) / 2
        # Sum of k*(k-1)/2
        # Since we want to minimize 4-cycles, we can just sum k*(k-1)
        # and divide by 2 later or just use the sum as penalty.
        penalty = np.sum(vals * (vals - 1))
        return penalty // 2
        
    # Local Search
    # Start with a random graph or empty graph?
    # Start with empty graph and add edges?
    # Greedy addition:
    # For each edge, if adding it doesn't create a 4-cycle, add it.
    # This is O(E * E) roughly.
    
    # Let's try greedy first, then local search to improve.
    
    adj = np.zeros((20, 20), dtype=np.uint8)
    
    # Greedy phase
    # Order edges randomly?
    edges_to_try = [(i, j) for i in range(20) for j in range(i+1, 20)]
    random.shuffle(edges_to_try)
    
    for u, v in edges_to_try:
        # Check if adding (u, v) creates a 4-cycle
        # A 4-cycle involving (u, v) must have form u-x-v-y-u or u-x-y-v-u?
        # No, if we add (u, v), the new 4-cycles must use this edge.
        # So they are of form u-v-w-x-u.
        # This requires w to be a neighbor of v and x to be a neighbor of u,
        # AND (w, x) must be an edge.
        # So we check:
        # For each w in neighbors(v):
        #   For each x in neighbors(u):
        #     if adj[w, x] == 1:
        #       Then u-v-w-x-u is a 4-cycle.
        
        # Efficient check:
        # Let N_u = neighbors of u
        # Let N_v = neighbors of v
        # We need to check if there is an edge between any w in N_v and x in N_u.
        # i.e., is (N_v x N_u) intersecting with the edge set?
        
        # Using matrix:
        # row_u = adj[u]
        # row_v = adj[v]
        # We want to know if there exists w, x such that row_v[w]=1, row_u[x]=1, and adj[w,x]=1.
        # This is equivalent to checking if (row_v @ adj @ row_u) > 0?
        # row_v @ adj gives a vector where entry x is number of common neighbors of v and x? No.
        # row_v @ adj is a vector y where y[x] = sum_w row_v[w] * adj[w, x].
        # This is the number of paths of length 2 from v to x.
        # We want to know if any x in N_u has y[x] > 0.
        # So dot product of y and row_u.
        
        # paths_from_v = adj[v] @ adj  (vector of length 20)
        # Check if dot(paths_from_v, adj[u]) > 0
        
        # Note: adj[u] and adj[v] are rows.
        
        # Let's use this.
        
        # Optimization: Precompute neighbors lists?
        # Or just use the vector operation.
        
        # 20x20 matrix mult is very fast.
        
        # Check
        # paths_v = adj[v] @ adj  # (20,)
        # conflict = np.dot(paths_v, adj[u])
        
        # If conflict > 0, adding (u, v) creates a 4-cycle.
        
        # Wait, does it?
        # paths_v[x] > 0 means there is a path v-w-x.
        # If x is a neighbor of u (adj[u, x] == 1), then u-x-w-v-u is a 4-cycle.
        # Yes.
        
        # So:
        # if np.dot(adj[v] @ adj, adj[u]) > 0:
        #     continue
        # else:
        #     adj[u, v] = 1
        #     adj[v, u] = 1
        
        # Let's implement this.
        
        # To speed up, we can maintain a list of neighbors.
        # But 20 is small.
        
        if time.time() > deadline:
            break
            
        # Check conflict
        # Using NumPy
        # Note: adj is uint8. Cast to int for multiplication to avoid overflow?
        # Max degree 19. 19*19 = 361. Fits in uint16 or int32.
        # adj @ adj is safe.
        
        # paths_v = adj[v] @ adj
        # conflict = np.dot(paths_v, adj[u])
        
        # Let's do it.
        
        # Small optimization: if degree(u) == 0 or degree(v) == 0, no conflict.
        
        if np.dot(adj[v] @ adj, adj[u]) == 0:
            adj[u, v] = 1
            adj[v, u] = 1
            
    # Now we have a C4-free graph.
    # Can we improve it?
    # Local search: try to add an edge that was skipped, or swap edges?
    # Adding an edge is only possible if it doesn't create a C4.
    # The greedy algorithm might have skipped an edge because of current neighbors.
    # If we remove some edges, we might be able to add more.
    
    # Let's do a few iterations of local search.
    # Move: Remove a random edge, then try to add random edges (including the removed one) greedily.
    # If the new graph has more edges, keep it.
    
    best_adj = adj.copy()
    best_edges = np.sum(best_adj) // 2
    
    # Time budget for local search
    ls_start = time.time()
    ls_deadline = start_time + 115.0
    
    while time.time() < ls_deadline:
        # Create a perturbation
        curr_adj = best_adj.copy()
        
        # Remove 1-3 random edges
        num_remove = random.randint(1, 3)
        for _ in range(num_remove):
            # Pick random edge
            while True:
                u = random.randint(0, 19)
                v = random.randint(0, 19)
                if u < v and curr_adj[u, v] == 1:
                    curr_adj[u, v] = 0
                    curr_adj[v, u] = 0
                    break
        
        # Try to add edges greedily
        # Random order
        edges_to_try = [(i, j) for i in range(20) for j in range(i+1, 20) if curr_adj[i, j] == 0]
        random.shuffle(edges_to_try)
        
        for u, v in edges_to_try:
            if time.time() > ls_deadline:
                break
            if np.dot(curr_adj[v] @ curr_adj, curr_adj[u]) == 0:
                curr_adj[u, v] = 1
                curr_adj[v, u] = 1
                
        curr_edges = np.sum(curr_adj) // 2
        if curr_edges > best_edges:
            best_adj = curr_adj
            best_edges = curr_edges
            
    # Construct output
    # Assign color 0 to all edges
    result = []
    for u in range(20):
        for v in range(u + 1, 20):
            if best_adj[u, v] == 1:
                result.append([u, v, 0])
                
    return result

# EVOLVE-BLOCK-END
