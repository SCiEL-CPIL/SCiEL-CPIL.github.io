"""Construct a graph with 31 vertices and 91 edges minimizing 4-cycles."""
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    n = 31
    m = 91
    
    # Generate polarity graph on PG(2,5)
    # Points of PG(2,5) are 1D subspaces of F_5^3
    # There are (5^3 - 1)/(5 - 1) = 31 points
    
    # Generate all 31 points in PG(2,5)
    points = []
    seen = set()
    for a in range(5):
        for b in range(5):
            for c in range(5):
                if a == 0 and b == 0 and c == 0:
                    continue
                # Normalize: make first non-zero coordinate 1
                vec = [a, b, c]
                first_nonzero = -1
                for i in range(3):
                    if vec[i] != 0:
                        first_nonzero = i
                        break
                if first_nonzero == -1:
                    continue
                # Scale so that first non-zero element is 1
                scale = vec[first_nonzero]
                normalized = tuple([x * pow(scale, 4, 5) % 5 for x in vec])
                if normalized not in seen:
                    seen.add(normalized)
                    points.append(list(normalized))
    
    # Should have exactly 31 points
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    # Define polarity: symmetric bilinear form
    # Use the standard polarity: <u, v> = u[0]*v[0] + u[1]*v[1] + u[2]*v[2] mod 5
    # But this gives a regular graph. Let's use a different one.
    # Actually, for a polarity graph, we connect u and v if <u,v> = 0
    # where <,> is a symmetric bilinear form.
    
    # Let's use the form: <u,v> = u[0]*v[1] + u[1]*v[0] + u[1]*v[2] + u[2]*v[1] + u[2]*v[2] mod 5
    # This is a non-degenerate symmetric bilinear form.
    
    def dot_product(u, v):
        # Symmetric bilinear form
        return (u[0]*v[1] + u[1]*v[0] + u[1]*v[2] + u[2]*v[1] + u[2]*v[2]) % 5
    
    # Build adjacency matrix
    adj = np.zeros((n, n), dtype=np.int8)
    edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            if dot_product(points[i], points[j]) == 0:
                adj[i, j] = 1
                adj[j, i] = 1
                edges.append([i, j])
    
    # Count how many edges we have
    base_edge_count = len(edges)
    
    # The polarity graph should have 90 edges (ex(31;C4) = 90)
    # We need 91 edges, so we add one more edge
    # Find the edge that adds the fewest 4-cycles
    
    def count_c4s(adj_matrix):
        """Count the number of 4-cycles in the graph."""
        # A 4-cycle is a-b-c-d-a where all edges exist
        # Count using adjacency matrix
        # Number of 4-cycles = (1/8) * sum over all pairs (i,j) of (A^2)[i,j] * (A^2)[i,j] - ...
        # Actually, the number of 4-cycles can be computed as:
        # For each pair of vertices (u,v), let N(u,v) be the number of common neighbors
        # The number of 4-cycles containing edge (u,v) is N(u,v)
        # But each 4-cycle is counted 4 times (once for each edge)
        # Wait, no. Let me think again.
        
        # The number of 4-cycles (as subgraphs) is:
        # (1/8) * sum_{i,j} (A^2)[i,j] * (A^2)[i,j] - (1/2) * sum_i deg(i)^2 + ...
        
        # Simpler: count all sequences a-b-c-d-a where a<b and the cycle is valid
        # Use the formula: C4 count = (1/8) * (sum_{i,j} (A^2)_{ij}^2 - 4*sum_i deg(i)^2 + 2*sum_i deg(i))
        # Wait, that's not right either.
        
        # Let's use a direct approach:
        # For each pair of vertices (i, j) with i < j, count the number of common neighbors
        # A 4-cycle i-a-j-b-i requires i and j to have at least 2 common neighbors a, b
        # The number of 4-cycles where i and j are opposite vertices is C(common_neighbors(i,j), 2)
        # But each 4-cycle has 2 pairs of opposite vertices, so we divide by 2
        
        A2 = adj_matrix @ adj_matrix
        c4_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                common = A2[i, j]
                if common >= 2:
                    c4_count += common * (common - 1) // 2
        return c4_count // 2  # Each 4-cycle counted twice (once for each pair of opposite vertices)
    
    # First, count C4s in the base graph
    base_c4_count = count_c4s(adj)
    
    # Find the best edge to add (one that minimizes C4 increase)
    best_edge = None
    best_c4_increase = float('inf')
    
    # Try all non-edges
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 0:
                # Temporarily add edge
                adj[i, j] = 1
                adj[j, i] = 1
                new_c4 = count_c4s(adj)
                increase = new_c4 - base_c4_count
                # Remove edge
                adj[i, j] = 0
                adj[j, i] = 0
                
                if increase < best_c4_increase:
                    best_c4_increase = increase
                    best_edge = [i, j]
    
    # Add the best edge
    if best_edge is not None:
        i, j = best_edge
        adj[i, j] = 1
        adj[j, i] = 1
        edges.append([i, j])
    
    # Now we have 91 edges. Let's do local search to minimize C4 count.
    # We'll try swapping edges: remove one edge and add another, keeping 91 edges total.
    
    def try_improve(adj, edges, max_iterations=10000):
        """Try to improve the C4 count by edge swaps."""
        current_c4 = count_c4s(adj)
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            improved = False
            # Try random swaps
            for _ in range(50):
                # Pick a random edge to remove
                idx = random.randint(0, len(edges) - 1)
                u, v = edges[idx]
                
                # Pick a random non-edge to add
                non_edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i, j] == 0 and not (i == u and j == v) and not (i == v and j == u):
                            non_edges.append([i, j])
                
                if not non_edges:
                    continue
                
                new_u, new_v = random.choice(non_edges)
                
                # Check if this creates a valid swap (we're replacing one edge with another)
                # Remove old edge
                adj[u, v] = 0
                adj[v, u] = 0
                # Add new edge
                adj[new_u, new_v] = 1
                adj[new_v, new_u] = 1
                
                new_c4 = count_c4s(adj)
                
                if new_c4 < current_c4:
                    # Accept the swap
                    current_c4 = new_c4
                    edges[idx] = [new_u, new_v]
                    improved = True
                    break
                else:
                    # Revert
                    adj[u, v] = 1
                    adj[v, u] = 1
                    adj[new_u, new_v] = 0
                    adj[new_v, new_u] = 0
            
            if not improved:
                break
        
        return current_c4
    
    # Run local search
    improved_c4 = try_improve(adj, edges)
    
    # Extract final edges
    final_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 1:
                final_edges.append([i, j])
    
    # Verify we have exactly 91 edges
    assert len(final_edges) == 91, f"Expected 91 edges, got {len(final_edges)}"
    
    return final_edges
