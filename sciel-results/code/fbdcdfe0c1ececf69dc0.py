"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    """Construct a K4-free graph on 18 vertices minimizing the largest triangle-free induced subgraph."""
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    N = 18
    all_vertices = list(range(N))
    
    # Precompute all 4-subsets for K4 check
    all_4subsets = list(combinations(range(N), 4))
    
    def is_k4_free(adj):
        """Check if graph with adjacency matrix adj is K4-free."""
        for quad in all_4subsets:
            a, b, c, d = quad
            if adj[a][b] and adj[a][c] and adj[a][d] and adj[b][c] and adj[b][d] and adj[c][d]:
                return False
        return True
    
    def count_k4s(adj):
        """Count number of K4 subgraphs."""
        count = 0
        for quad in all_4subsets:
            a, b, c, d = quad
            if adj[a][b] and adj[a][c] and adj[a][d] and adj[b][c] and adj[b][d] and adj[c][d]:
                count += 1
        return count
    
    def max_triangle_free_size(adj):
        """Find the largest triangle-free induced subgraph size using branch and bound."""
        n = len(adj)
        best = [0]
        
        # Use bitmask for efficiency
        # adj_mask[i] is a bitmask of neighbors of i
        adj_mask = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
        
        def has_triangle(vertices):
            """Check if induced subgraph on vertices has a triangle."""
            for i in range(len(vertices)):
                for j in range(i+1, len(vertices)):
                    for k in range(j+1, len(vertices)):
                        a, b, c = vertices[i], vertices[j], vertices[k]
                        if adj_mask[a] & (1 << b) and adj_mask[a] & (1 << c) and adj_mask[b] & (1 << c):
                            return True
            return False
        
        # Branch and bound: try to find largest triangle-free set
        # Start with all vertices, remove vertices that form triangles
        # Use recursive approach
        
        def solve(candidate_set, current_size):
            """Recursively find max triangle-free subset of candidate_set."""
            if time.time() > deadline - 2:
                return current_size
            
            if current_size + len(candidate_set) <= best[0]:
                return current_size
            
            if not candidate_set:
                if current_size > best[0]:
                    best[0] = current_size
                return current_size
            
            # If no triangle in candidate_set, we can take all
            if not has_triangle(candidate_set):
                if current_size + len(candidate_set) > best[0]:
                    best[0] = current_size + len(candidate_set)
                return current_size + len(candidate_set)
            
            # Find a triangle in candidate_set
            tri = None
            verts = list(candidate_set)
            for i in range(len(verts)):
                for j in range(i+1, len(verts)):
                    for k in range(j+1, len(verts)):
                        a, b, c = verts[i], verts[j], verts[k]
                        if adj_mask[a] & (1 << b) and adj_mask[a] & (1 << c) and adj_mask[b] & (1 << c):
                            tri = (a, b, c)
                            break
                    if tri:
                        break
                if tri:
                    break
            
            if tri is None:
                # No triangle found (shouldn't happen if has_triangle returned True)
                if current_size + len(candidate_set) > best[0]:
                    best[0] = current_size + len(candidate_set)
                return current_size + len(candidate_set)
            
            a, b, c = tri
            # We must remove at least one of a, b, c
            # Try removing each one
            max_result = current_size
            for v in [a, b, c]:
                new_candidate = candidate_set - {v}
                result = solve(new_candidate, current_size + 1)
                if result > max_result:
                    max_result = result
            
            return max_result
        
        # Initial candidate is all vertices
        full_set = frozenset(range(n))
        solve(full_set, 0)
        
        return best[0]
    
    def random_k4_free():
        """Generate a random K4-free graph."""
        # Start with empty graph
        adj = [[False] * N for _ in range(N)]
        
        # Add edges in random order, checking K4-freeness
        edge_order = [(i, j) for i in range(N) for j in range(i+1, N)]
        random.shuffle(edge_order)
        
        for i, j in edge_order:
            # Try adding edge (i, j)
            adj[i][j] = True
            adj[j][i] = True
            if not is_k4_free(adj):
                adj[i][j] = False
                adj[j][i] = False
        
        return adj
    
    def local_search(adj, max_time=5.0):
        """Local search: try to improve by removing edges that create K4s or by strategic edge swaps."""
        start = time.time()
        current_score = max_triangle_free_size(adj)
        
        while time.time() - start < max_time and time.time() < deadline - 5:
            improved = False
            # Try removing random edges
            for _ in range(100):
                if time.time() - start > max_time or time.time() > deadline - 5:
                    break
                i, j = random.randint(0, N-1), random.randint(0, N-1)
                if i == j or not adj[i][j]:
                    continue
                
                # Try removing edge
                adj[i][j] = False
                adj[j][i] = False
                new_score = max_triangle_free_size(adj)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    # Restore
                    adj[i][j] = True
                    adj[j][i] = True
            
            if not improved:
                break
        
        return adj, current_score
    
    # Start with complete tripartite K6,6,6 as baseline
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    base_adj = [[False] * N for _ in range(N)]
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    base_adj[a][b] = True
                    base_adj[b][a] = True
    
    base_score = max_triangle_free_size(base_adj)
    best_adj = [row[:] for row in base_adj]
    best_score = base_score
    
    # Simulated annealing / random restarts
    rng = random.Random(42)
    
    for restart in range(20):
        if time.time() > deadline - 10:
            break
        
        # Generate random K4-free graph
        adj = random_k4_free()
        score = max_triangle_free_size(adj)
        
        if score < best_score:
            best_adj = [row[:] for row in adj]
            best_score = score
        
        # Local search
        adj, score = local_search(adj, max_time=3.0)
        
        if score < best_score:
            best_adj = [row[:] for row in adj]
            best_score = score
    
    # Additional local search on best solution
    if time.time() < deadline - 10:
        adj, score = local_search(best_adj, max_time=30.0)
        if score < best_score:
            best_adj = [row[:] for row in adj]
            best_score = score
    
    # Convert adjacency matrix to edge list
    edges = []
    for i in range(N):
        for j in range(i+1, N):
            if best_adj[i][j]:
                edges.append([i, j])
    
    return edges
# EVOLVE-BLOCK-END
