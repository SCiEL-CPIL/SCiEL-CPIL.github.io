import time
import random
from itertools import combinations

def construct():
    """Construct a dense triangle-free graph on 16 vertices that is not 3-colorable."""
    deadline = time.time() + 115  # Safety margin before 150s limit
    
    n = 16
    all_pairs = list(combinations(range(n), 2))
    
    def is_triangle_free(adj_bits):
        """Check triangle-free using bitmasks. adj_bits[i] is a bitmask of neighbors of i."""
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_bits[u] >> v) & 1:
                    # Check if u and v share a common neighbor
                    common = adj_bits[u] & adj_bits[v]
                    if common:
                        return False
        return True
    
    def is_3_colorable(adj_bits):
        """Check 3-colorability using backtracking with bitmasks."""
        color = [-1] * n
        color_masks = [0, 0, 0]  # bitmask of vertices with each color
        
        # Order vertices by degree (descending) for better pruning
        degrees = [bin(adj_bits[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda i: -degrees[i])
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            # Determine which colors are forbidden by neighbors
            forbidden = 0
            for c in range(3):
                if color_masks[c] & adj_bits[v]:
                    forbidden |= (1 << c)
            # Try each available color
            for c in range(3):
                if not (forbidden & (1 << c)):
                    color[v] = c
                    color_masks[c] |= (1 << v)
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
                    color_masks[c] &= ~(1 << v)
            return False
        
        return backtrack(0)
    
    def random_triangle_free_graph(density):
        """Generate a random triangle-free graph with given edge probability."""
        adj_bits = [0] * n
        edges = []
        
        pairs = all_pairs[:]
        random.shuffle(pairs)
        
        for u, v in pairs:
            if random.random() < density:
                # Check if adding (u,v) creates a triangle
                common = adj_bits[u] & adj_bits[v]
                if not common:
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                    edges.append((u, v))
        
        return adj_bits, edges
    
    def add_edge_greedy(adj_bits, edges, u, v):
        """Try to add edge (u,v) if it doesn't create a triangle."""
        if (adj_bits[u] >> v) & 1:
            return False
        common = adj_bits[u] & adj_bits[v]
        if common:
            return False
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)
        edges.append((u, v))
        return True
    
    def remove_edge(adj_bits, edges, u, v):
        """Remove edge (u,v) from the graph."""
        adj_bits[u] &= ~(1 << v)
        adj_bits[v] &= ~(1 << u)
        edges.remove((u, v))
    
    best_edges = []
    best_edge_count = 0
    
    random.seed(42)
    
    # Phase 1: Try various random triangle-free graphs
    for trial in range(5000):
        if time.time() > deadline - 30:
            break
        
        # Try different densities
        density = 0.2 + 0.6 * (trial % 10) / 10
        
        adj_bits, edges = random_triangle_free_graph(density)
        edge_count = len(edges)
        
        if edge_count <= best_edge_count:
            continue
        
        # Verify triangle-free
        if not is_triangle_free(adj_bits):
            continue
        
        # Check non-3-colorability
        if not is_3_colorable(adj_bits):
            best_edges = edges[:]
            best_edge_count = edge_count
    
    # Phase 2: Local search - try to add edges to improve
    if best_edges:
        adj_bits = [0] * n
        for u, v in best_edges:
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
        
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for u, v in all_pairs:
                if not ((adj_bits[u] >> v) & 1):
                    if add_edge_greedy(adj_bits, best_edges, u, v):
                        # Check if still not 3-colorable
                        if not is_3_colorable(adj_bits):
                            improved = True
                        else:
                            # Revert
                            adj_bits[u] &= ~(1 << v)
                            adj_bits[v] &= ~(1 << u)
                            best_edges.pop()
    
    # Phase 3: Try to find denser non-3-colorable subgraphs by edge removal
    if best_edges and time.time() < deadline - 5:
        # Start from a very dense triangle-free graph and remove edges to maintain non-3-colorability
        for density in [0.8, 0.7, 0.6]:
            if time.time() > deadline - 5:
                break
            
            adj_bits, edges = random_triangle_free_graph(density)
            edge_count = len(edges)
            
            if edge_count <= best_edge_count:
                continue
            
            if not is_triangle_free(adj_bits):
                continue
            
            # If it's 3-colorable, try removing edges to make it non-3-colorable
            if is_3_colorable(adj_bits):
                # Try removing random edges until it becomes non-3-colorable
                removed = 0
                while is_3_colorable(adj_bits) and removed < 50 and time.time() < deadline - 5:
                    if not edges:
                        break
                    idx = random.randint(0, len(edges) - 1)
                    u, v = edges[idx]
                    remove_edge(adj_bits, edges, u, v)
                    removed += 1
                
                if not is_3_colorable(adj_bits) and len(edges) > best_edge_count:
                    best_edges = edges[:]
                    best_edge_count = len(edges)
            else:
                # Already non-3-colorable
                if len(edges) > best_edge_count:
                    best_edges = edges[:]
                    best_edge_count = len(edges)
    
    return [[u, v] for u, v in best_edges]
