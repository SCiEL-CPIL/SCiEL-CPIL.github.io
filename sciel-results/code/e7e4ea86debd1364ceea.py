"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import itertools
import random
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Greedy sequential vertex addition algorithm.
    Starts with an empty graph and adds vertices one by one.
    For each new vertex, we try all possible edge sets to existing vertices,
    selecting the one that minimizes the number of new regular induced subgraphs
    of size 6+ created.
    
    A regular induced subgraph is one where every vertex in the induced subgraph
    has the same degree within that subgraph.
    """
    
    def is_regular(subset, adj_matrix, n_current):
        """Check if the induced subgraph on 'subset' is regular."""
        k = len(subset)
        if k < 6:
            return False
        # Compute degrees within the subset
        idx_map = {v: i for i, v in enumerate(subset)}
        degrees = [0] * k
        for i, u in enumerate(subset):
            for j, v in enumerate(subset):
                if i != j and adj_matrix[u][v]:
                    degrees[i] += 1
        # Check if all degrees are equal
        if len(set(degrees)) == 1:
            return True
        return False
    
    def count_new_regular_subgraphs(adj_matrix, n_current, new_vertex, new_edges):
        """
        Count the number of regular induced subgraphs of size >= 6 that include
        the new vertex, given the current adjacency matrix and the new vertex's
        connections.
        
        We only need to check subsets that include the new vertex, since those
        are the only new subgraphs created by adding the new vertex.
        """
        count = 0
        existing_vertices = list(range(n_current))
        
        # Try all subsets of existing vertices of size 0 to n_current-1
        # such that total subset size (including new_vertex) is >= 6
        min_existing = 5  # so that 5 + 1 = 6
        
        for size_existing in range(min_existing, n_current + 1):
            for subset in itertools.combinations(existing_vertices, size_existing):
                # Build the full subset including new_vertex
                full_subset = list(subset) + [new_vertex]
                
                # Check if this induced subgraph is regular
                k = len(full_subset)
                # Compute degrees
                degrees = [0] * k
                idx_map = {v: i for i, v in enumerate(full_subset)}
                
                for i, u in enumerate(full_subset):
                    for j, v in enumerate(full_subset):
                        if i == j:
                            continue
                        if u == new_vertex:
                            # Use new_edges
                            if new_edges.get(v, False):
                                degrees[i] += 1
                        elif v == new_vertex:
                            if new_edges.get(u, False):
                                degrees[i] += 1
                        else:
                            if adj_matrix[u][v]:
                                degrees[i] += 1
                
                if len(set(degrees)) == 1:
                    count += 1
        
        return count
    
    # Start with a small graph and build up
    # We'll try to build up to 22 vertices
    max_n = 22
    
    # Initialize with vertex 0
    adj = [[False] * max_n for _ in range(max_n)]
    n_current = 1
    
    # For each new vertex, try all possible edge configurations to existing vertices
    for new_vertex in range(1, max_n):
        # Try all 2^(n_current) possible edge sets
        best_edges = None
        best_count = float('inf')
        best_config = None
        
        # For efficiency, we can use symmetry breaking or random sampling
        # But let's try a few random configurations plus some systematic ones
        # Actually, for n_current up to 21, 2^21 = 2 million, which might be too slow
        # Let's use a smarter approach: try random configurations and keep the best
        
        # We'll try a mix: some random, some based on patterns
        candidates = []
        
        # Random candidates
        random.seed(42 + new_vertex)
        for _ in range(500):
            edges = {v: random.random() < 0.5 for v in range(n_current)}
            candidates.append(edges)
        
        # Some systematic candidates
        # No edges
        candidates.append({v: False for v in range(n_current)})
        # All edges
        candidates.append({v: True for v in range(n_current)})
        # Alternating
        candidates.append({v: (v % 2 == 0) for v in range(n_current)})
        # Half edges
        half = n_current // 2
        candidates.append({v: (v < half) for v in range(n_current)})
        
        for edges in candidates:
            count = count_new_regular_subgraphs(adj, n_current, new_vertex, edges)
            if count < best_count:
                best_count = count
                best_edges = edges
                best_config = edges
        
        # Apply the best configuration
        if best_edges is not None:
            for v in range(n_current):
                adj[new_vertex][v] = best_edges[v]
                adj[v][new_vertex] = best_edges[v]
        
        n_current += 1
        
        # Early termination: if we've added too many regular subgraphs, 
        # it might be better to stop
        # But let's continue and see
    
    # Now extract the actual edges
    edges = []
    for i in range(n_current):
        for j in range(i + 1, n_current):
            if adj[i][j]:
                edges.append([i, j])
    
    # Verify: check that no subset of size >= 6 induces a regular subgraph
    # This is a sanity check
    # For efficiency, we'll do a partial check
    
    # Actually, let's just return what we have
    return {'n': n_current, 'edges': edges}

# EVOLVE-BLOCK-END
