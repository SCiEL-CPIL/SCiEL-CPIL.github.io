<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # K3,13: any Q3 would need four vertices on each side.
    return [[a, b] for a in range(3) for b in range(3, 16)]
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n = 16
    
    # Q3 has 8 vertices, 12 edges. Canonical embedding on 0..7:
    # Vertices are 3-bit numbers 0..7
    # Edges connect vertices differing in exactly one bit
    q3_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            diff = i ^ j
            if diff & (diff - 1) == 0:  # power of 2 = differ in one bit
                q3_edges.append((i, j))
    
    # Precompute all permutations of 8 vertices from 16? Too many.
    # Instead: check Q3 by trying all 8-vertex subsets and all bijections.
    # Better: use backtracking on vertex assignments.
    
    def has_q3(adj_matrix):
        """Check if graph contains Q3 as subgraph using backtracking."""
        # Try to embed Q3 vertices 0..7 into graph vertices 0..15
        # Use the fact that Q3 has a specific structure
        
        # For each assignment of Q3 vertex 0 to a graph vertex,
        # try to build the rest
        for v0 in range(n):
            assignment = [-1] * 8
            assignment[0] = v0
            if _backtrack_q3(assignment, adj_matrix, 1, 0):
                return True
        return False
    
    def _backtrack_q3(assignment, adj_matrix, q3_idx, prev_q3):
        """Try to assign Q3 vertex q3_idx to a graph vertex."""
        if q3_idx >= 8:
            # All assigned, verify all Q3 edges exist
            for (a, b) in q3_edges:
                if not adj_matrix[assignment[a]][assignment[b]]:
                    return False
            return True
        
        # Get constraints from already-assigned neighbors in Q3
        valid_vertices = set(range(n))
        for (a, b) in q3_edges:
            if a == q3_idx and b < q3_idx:
                if assignment[b] == -1:
                    continue
                # Must be adjacent to assignment[b]
                valid_vertices = {v for v in valid_vertices if adj_matrix[assignment[b]][v]}
            elif b == q3_idx and a < q3_idx:
                if assignment[a] == -1:
                    continue
                valid_vertices = {v for v in valid_vertices if adj_matrix[assignment[a]][v]}
        
        # Also ensure not already assigned
        used = set(assignment[:q3_idx])
        valid_vertices -= used
        
        if not valid_vertices:
            return False
        
        for v in sorted(valid_vertices):
            assignment[q3_idx] = v
            if _backtrack_q3(assignment, adj_matrix, q3_idx + 1, q3_idx):
                return True
        assignment[q3_idx] = -1
        return False
    
    # Build adjacency matrix
    adj = [[False] * n for _ in range(n)]
    edge_set = set()
    
    # Start with a random dense graph and remove edges that create Q3
    # Strategy: start with complete graph, remove edges greedily
    # Better: start with empty, add edges greedily
    
    random.seed(42)
    
    # Generate all possible edges
    all_edges = list(itertools.combinations(range(n), 2))
    random.shuffle(all_edges)
    
    best_edge_count = 0
    best_edges = []
    
    for u, v in all_edges:
        if time.time() > deadline:
            break
        
        # Try adding edge (u, v)
        adj[u][v] = True
        adj[v][u] = True
        edge_set.add((min(u,v), max(u,v)))
        
        if has_q3(adj):
            # Remove this edge
            adj[u][v] = False
            adj[v][u] = False
            edge_set.remove((min(u,v), max(u,v)))
        else:
            if len(edge_set) > best_edge_count:
                best_edge_count = len(edge_set)
                best_edges = [[u, v] for (u, v) in edge_set]
    
    # Final check
    if time.time() > deadline:
        if best_edges:
            return best_edges
        # Fallback to seed
        return [[a, b] for a in range(3) for b in range(3, 16)]
    
    # Verify final graph is Q3-free
    if not has_q3(adj):
        return [[u, v] for (u, v) in edge_set]
    
    # If something went wrong, return best known valid
    if best_edges:
        return best_edges
    
    return [[a, b] for a in range(3) for b in range(3, 16)]
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, the backtracking Q3 checker might be too slow. Let me think about a faster approach.

The key issue: checking if adding one edge creates a Q3 can be done more efficiently. When we add edge (u,v), a Q3 containing this edge must have u and v mapped to adjacent vertices in Q3. There are 12 edges in Q3, so we need to check if there's an embedding where u,v map to one of these 12 edge pairs.

Let me rewrite with a more efficient incremental check:

