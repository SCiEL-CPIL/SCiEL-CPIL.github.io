"""Program for fewest four-cycles in graphs with 31 vertices and 91 edges."""
import numpy as np
from scipy.sparse import coo_matrix
import random
import time

def construct():
    """
    Construct a graph with 31 vertices and exactly 91 edges minimizing the number of C4 subgraphs.
    
    Approach:
    1. Build a C4-free base graph with 90 edges using a polarity graph construction on PG(2,5).
    2. Add one edge carefully to minimize the number of new C4s created.
    3. Perform local search (edge swaps) to further minimize C4 count with incremental updates.
    """
    n = 31
    target_edges = 91
    
    # Step 1: Build a C4-free base graph with 90 edges
    # Use a polarity graph construction for q=5
    # The vertices are the 31 points of PG(2,5)
    # We represent points as homogeneous coordinates [x,y,z] over GF(5)
    
    # Generate all points in PG(2,5)
    points = []
    seen = set()
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: make first non-zero coordinate 1
                pt = [x, y, z]
                for i in range(3):
                    if pt[i] != 0:
                        inv = pow(pt[i], 3, 5)  # inverse in GF(5)
                        pt = [(pt[j] * inv) % 5 for j in range(3)]
                        break
                if tuple(pt) not in seen:
                    seen.add(tuple(pt))
                    points.append(pt)
    
    # Should have 31 points
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    # Define a polarity: A point p = (x,y,z) is incident to point q = (u,v,w) 
    # if x*w + y*u + z*v = 0 in GF(5) (symmetric bilinear form)
    # This defines a symmetric relation, so we need to handle self-incidence carefully
    
    # Build adjacency matrix
    adj = np.zeros((n, n), dtype=np.int8)
    
    for i in range(n):
        for j in range(i + 1, n):
            p = points[i]
            q = points[j]
            # Check polarity condition: p[0]*q[2] + p[1]*q[0] + p[2]*q[1] == 0 mod 5
            val = (p[0] * q[2] + p[1] * q[0] + p[2] * q[1]) % 5
            if val == 0:
                adj[i, j] = 1
                adj[j, i] = 1
    
    # Count edges in base graph
    edge_count = sum(adj[i, j] for i in range(n) for j in range(i + 1, n))
    
    # If we don't have exactly 90 edges, adjust
    if edge_count < 90:
        # Add non-edges that minimize C4 increase
        random.seed(42)
        added = 0
        while added < 90 - edge_count:
            best_c4_increase = float('inf')
            best_pair = None
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i, j] == 0:
                        common = np.sum(adj[i] & adj[j])
                        c4_increase = common * (common - 1) // 2
                        if c4_increase < best_c4_increase:
                            best_c4_increase = c4_increase
                            best_pair = (i, j)
            if best_pair is None:
                break
            i, j = best_pair
            adj[i, j] = 1
            adj[j, i] = 1
            added += 1
    elif edge_count > 90:
        # Remove edges that minimize C4 decrease (i.e., keep edges that contribute most to C4)
        random.seed(42)
        removed = 0
        while removed < edge_count - 90:
            best_c4_decrease = -1
            best_pair = None
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i, j] == 1:
                        common = np.sum(adj[i] & adj[j])
                        c4_decrease = common * (common - 1) // 2
                        if c4_decrease > best_c4_decrease:
                            best_c4_decrease = c4_decrease
                            best_pair = (i, j)
            if best_pair is None:
                break
            i, j = best_pair
            adj[i, j] = 0
            adj[j, i] = 0
            removed += 1
    
    # Now we have 90 edges. Add one more edge to get 91.
    # To minimize C4s, add an edge between two vertices with minimum common neighbors
    min_common = float('inf')
    best_pair = None
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 0:
                common = np.sum(adj[i] & adj[j])
                if common < min_common:
                    min_common = common
                    best_pair = (i, j)
    
    if best_pair is not None:
        i, j = best_pair
        adj[i, j] = 1
        adj[j, i] = 1
    
    # Now we have 91 edges. Perform local search to minimize C4 count.
    # Use incremental C4 update for efficiency
    
    def count_c4_full(adj):
        c4_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                common = np.sum(adj[i] & adj[j])
                c4_count += common * (common - 1) // 2
        return c4_count
    
    def get_c4_delta_remove(adj, a, b):
        """Calculate C4 change when removing edge (a,b)"""
        delta = 0
        # For each common neighbor pair, count how many C4s are broken
        common = adj[a] & adj[b]
        common_idx = np.where(common)[0]
        for idx in common_idx:
            # Each common neighbor contributes to C4s with other common neighbors
            pass
        # Actually, removing edge (a,b) affects C4s that use this edge
        # A C4 containing edge (a,b) has form a-b-x-y-a where x,y are common neighbors
        # Number of such C4s = C(common_count, 2)
        common_count = np.sum(common)
        return common_count * (common_count - 1) // 2
    
    def get_c4_delta_add(adj, c, d):
        """Calculate C4 change when adding edge (c,d)"""
        common = adj[c] & adj[d]
        common_count = np.sum(common)
        return common_count * (common_count - 1) // 2
    
    current_c4 = count_c4_full(adj)
    
    # Local search with incremental updates
    random.seed(42)
    no_improvement = 0
    max_iterations = 20000
    iteration = 0
    
    # Precompute edge list for faster access
    def get_edges(adj):
        return [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j] == 1]
    
    def get_non_edges(adj):
        return [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j] == 0]
    
    while no_improvement < 3000 and iteration < max_iterations:
        iteration += 1
        
        # Pick a random edge to remove
        edges = get_edges(adj)
        if len(edges) != 91:
            break
        a, b = random.choice(edges)
        
        # Pick a random non-edge to add
        non_edges = get_non_edges(adj)
        c, d = random.choice(non_edges)
        
        # Ensure we're not just swapping the same edge
        if (a == c and b == d) or (a == d and b == c):
            continue
        
        # Calculate delta: C4_change = -delta_remove + delta_add
        delta_remove = get_c4_delta_remove(adj, a, b)
        delta_add = get_c4_delta_add(adj, c, d)
        c4_change = -delta_remove + delta_add
        
        if c4_change < 0:
            # Accept the move
            adj[a, b] = 0
            adj[b, a] = 0
            adj[c, d] = 1
            adj[d, c] = 1
            current_c4 += c4_change
            no_improvement = 0
        else:
            no_improvement += 1
    
    # Convert to list of edges
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 1:
                result.append([i, j])
    
    assert len(result) == 91, f"Expected 91 edges, got {len(result)}"
    
    return result
