"""Program for constructing a graph with 40 vertices and 401 edges minimizing the largest book size."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    """
    Construct a graph with 40 vertices and 401 edges that minimizes the largest book size.
    
    Strategy: Start with K_{20,20} and use local search to find a configuration
    with multiple intra-part edges that distributes the triangle load more evenly.
    
    The key insight is that a single intra-part edge in K_{20,20} creates book size 20,
    but by having multiple intra-part edges and removing some bipartite edges, we can
    reduce the maximum book size.
    """
    n = 40
    target_edges = 401
    A = list(range(20))  # Part A: vertices 0-19
    B = list(range(20, 40))  # Part B: vertices 20-39
    
    def build_adj(edge_set):
        """Build adjacency list from edge set."""
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def compute_max_book(adj, edge_set):
        """Compute the maximum book size."""
        max_book = 0
        for u, v in edge_set:
            common = len(adj[u] & adj[v])
            if common > max_book:
                max_book = common
        return max_book
    
    def get_all_non_edges(edge_set):
        """Get all non-edges in the graph."""
        non_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (u, v) not in edge_set and (v, u) not in edge_set:
                    non_edges.append((u, v))
        return non_edges
    
    # Initialize with K_{20,20} plus one intra-part edge
    edges = set()
    for a in A:
        for b in B:
            edges.add((a, b))
    edges.add((0, 1))
    
    adj = build_adj(edges)
    current_score = compute_max_book(adj, edges)
    best_score = current_score
    best_edges = set(edges)
    
    # Local search with edge swaps
    random.seed(42)
    start_time = time.time()
    time_limit = 115  # Leave 5s buffer
    
    # Try different strategies for edge swaps
    while time.time() - start_time < time_limit:
        # Strategy 1: Swap an intra-part edge with a bipartite edge
        intra_edges = [(u, v) for u, v in edges if (u < 20 and v < 20) or (u >= 20 and v >= 20)]
        bipartite_edges = [(u, v) for u, v in edges if (u < 20 and v >= 20) or (u >= 20 and v < 20)]
        
        if not intra_edges:
            break
        
        random.shuffle(intra_edges)
        random.shuffle(bipartite_edges)
        
        # Try several swaps per iteration
        for _ in range(100):
            if time.time() - start_time > time_limit:
                break
            
            if not intra_edges or not bipartite_edges:
                break
            
            remove_edge = random.choice(intra_edges)
            
            # Get non-edges to add
            non_edges = get_all_non_edges(edges)
            if not non_edges:
                break
            
            add_edge = random.choice(non_edges)
            
            new_edges = set(edges)
            new_edges.discard(remove_edge)
            new_edges.add(add_edge)
            
            if len(new_edges) != target_edges:
                continue
            
            adj_new = build_adj(new_edges)
            new_score = compute_max_book(adj_new, new_edges)
            
            if new_score <= current_score:
                edges = new_edges
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_edges = set(edges)
                
                # Update intra and bipartite edge lists
                intra_edges = [(u, v) for u, v in edges if (u < 20 and v < 20) or (u >= 20 and v >= 20)]
                bipartite_edges = [(u, v) for u, v in edges if (u < 20 and v >= 20) or (u >= 20 and v < 20)]
            
            # If we've reached a good score, do more aggressive search
            if best_score <= 8:
                # Try more aggressive swaps
                for _ in range(50):
                    if time.time() - start_time > time_limit:
                        break
                    
                    intra_edges = [(u, v) for u, v in edges if (u < 20 and v < 20) or (u >= 20 and v >= 20)]
                    if not intra_edges:
                        break
                    
                    random.shuffle(intra_edges)
                    remove_edge = random.choice(intra_edges)
                    
                    non_edges = get_all_non_edges(edges)
                    if not non_edges:
                        break
                    
                    add_edge = random.choice(non_edges)
                    
                    new_edges = set(edges)
                    new_edges.discard(remove_edge)
                    new_edges.add(add_edge)
                    
                    if len(new_edges) != target_edges:
                        continue
                    
                    adj_new = build_adj(new_edges)
                    new_score = compute_max_book(adj_new, new_edges)
                    
                    if new_score <= current_score:
                        edges = new_edges
                        current_score = new_score
                        if current_score < best_score:
                            best_score = current_score
                            best_edges = set(edges)
    
    # Return the best result
    result_edges = best_edges
    return [[u, v] for u, v in result_edges]

# EVOLVE-BLOCK-END
