import random
import numpy as np

def construct():
    n = 40
    m = 401
    max_book_allowed = 7  # Target minimum
    
    # We will use a greedy approach with randomization to avoid getting stuck.
    # We try to build the graph such that no edge has more than max_book_allowed common neighbors.
    
    # To increase the chance of success, we can try multiple random seeds/orderings if the first fails.
    # But let's try a single robust greedy pass first.
    
    def try_construct(seed):
        random.seed(seed)
        adj = [[False] * n for _ in range(n)]
        # book_size[u][v] stores the number of common neighbors for edge (u,v)
        # Only valid if adj[u][v] is True
        book_size = [[0] * n for _ in range(n)]
        edges = []
        
        # Pre-compute all possible edges
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        # We will iteratively pick an edge to add.
        # To make it efficient, we can just scan all non-edges each time.
        # n=40, so 780 edges. 401 iterations. 401 * 780 checks. ~300k checks. Fine.
        
        for _ in range(m):
            valid_edges = []
            
            for u, v in all_edges:
                if adj[u][v]:
                    continue
                    
                # Calculate common neighbors
                # Using list comprehension for simplicity, n is small
                common = [w for w in range(n) if adj[u][w] and adj[v][w]]
                c_len = len(common)
                
                if c_len > max_book_allowed:
                    continue
                
                # Check if adding this edge would violate the book size limit for existing edges
                # Adding (u,v) increases book_size[u][w] and book_size[v][w] by 1 for w in common
                # We need book_size[u][w] + 1 <= max_book_allowed
                # i.e., book_size[u][w] < max_book_allowed
                valid = True
                for w in common:
                    if book_size[u][w] >= max_book_allowed or book_size[v][w] >= max_book_allowed:
                        valid = False
                        break
                        
                if valid:
                    valid_edges.append((u, v, c_len))
            
            if not valid_edges:
                return None # Failed to construct
            
            # Pick a random valid edge, or the one with smallest common neighbors?
            # Smallest common neighbors is usually better for future flexibility.
            # Let's pick randomly from those with minimal common neighbors to add variety.
            min_c = min(e[2] for e in valid_edges)
            candidates = [e for e in valid_edges if e[2] == min_c]
            u, v, _ = random.choice(candidates)
            
            # Add edge (u,v)
            adj[u][v] = True
            adj[v][u] = True
            edges.append((u, v))
            
            # Update book sizes
            common = [w for w in range(n) if adj[u][w] and adj[v][w]]
            book_size[u][v] = len(common)
            book_size[v][u] = len(common)
            
            for w in common:
                book_size[u][w] += 1
                book_size[w][u] += 1
                book_size[v][w] += 1
                book_size[w][v] += 1
                
        return edges

    # Try with several seeds to ensure success
    for seed in range(100):
        result = try_construct(seed)
        if result is not None:
            # Convert to list of lists as required
            return [list(e) for e in result]
            
    # Fallback: Return a random graph if greedy fails (should not happen)
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_possible)
    return [list(e) for e in all_possible[:m]]
