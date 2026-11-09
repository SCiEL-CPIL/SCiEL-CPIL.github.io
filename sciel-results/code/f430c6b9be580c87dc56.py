import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    n = 40
    target_edges = 401
    
    # 1. Initialize Random Graph
    # Use a set for edges
    vertices = list(range(n))
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_pairs)
    edges = set(all_pairs[:target_edges])
    
    # Adjacency sets for fast common neighbor calculation
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    def get_book_size(u, v):
        if v not in adj[u]:
            return 0
        return len(adj[u].intersection(adj[v]))
        
    def get_max_book():
        max_b = 0
        # Iterate over all edges
        for u, v in edges:
            b = get_book_size(u, v)
            if b > max_b:
                max_b = b
        return max_b

    current_max_book = get_max_book()
    
    # 2. Local Search
    # We want to minimize current_max_book
    # Strategy: Find edges with high book size, try to swap them with random non-edges
    
    # Precompute non-edges for fast sampling? 
    # Or just sample random pairs and check if in edges.
    
    # To speed up, we can identify "bad" edges
    # Bad edges are those contributing to the max book
    
    while time.time() < deadline:
        # Find an edge with high book size
        # To avoid O(m) scan every time, we can just pick a random edge and check if it's "bad"
        # Or maintain a list of edges sorted by book size? 
        # Given small n, O(m) is 401 ops. 10000 swaps = 4M ops. Fast enough.
        
        # Let's just pick a random edge from the current set
        u, v = random.choice(list(edges))
        
        # Calculate its book size
        b_uv = get_book_size(u, v)
        
        # If it's not the max, we might still want to move it if it's high
        # But let's focus on the max. 
        # Better: Find the actual max edge.
        max_edge = None
        max_b_val = 0
        for e in edges:
            eu, ev = e
            b = get_book_size(eu, ev)
            if b > max_b_val:
                max_b_val = b
                max_edge = e
                
        if max_edge is None:
            break
            
        u, v = max_edge
        
        # Try to swap (u,v) with a random non-edge (x,y)
        # We need to pick x,y such that (x,y) is not in edges.
        # And ideally, the new edge (x,y) has low book size.
        
        swapped = False
        for _ in range(50): # Try 50 random swaps for this bad edge
            x = random.randint(0, n-1)
            y = random.randint(0, n-1)
            if x == y:
                continue
            if x > y:
                x, y = y, x
            
            if (x, y) in edges:
                continue
            if (u, v) == (x, y): # Should not happen as (x,y) not in edges
                continue
                
            # Check if (x,y) would create a worse book?
            # We don't know yet. Let's simulate.
            
            # Remove (u,v)
            adj[u].remove(v)
            adj[v].remove(u)
            edges.remove((u,v))
            
            # Add (x,y)
            adj[x].add(y)
            adj[y].add(x)
            edges.add((x,y))
            
            # Check new max book
            # Only books involving u, v, x, y changed
            # We need to check if any of these now have a book > current_max_book
            
            # Actually, we just need to check if the NEW max is <= OLD max
            # And if it's strictly less, we definitely accept.
            # If equal, we might accept to explore.
            
            # Efficient check:
            # The maximum book size can only increase if one of the new/changed edges has a high book.
            # Or if we removed an edge that was "hiding" a high book? No, removing edges reduces books.
            # Adding edges increases books.
            
            # So, the new max book is max(
            #   old max book excluding edges incident to u,v,x,y,
            #   book(u,v) -- removed, so 0
            #   book(x,y) -- new
            #   book(u,w) for w in neighbors(u)
            #   book(v,w) for w in neighbors(v)
            #   book(x,w) for w in neighbors(x)
            #   book(y,w) for w in neighbors(y)
            # )
            
            # This is complex to track incrementally without a full re-scan or careful bookkeeping.
            # Given n=40, a full re-scan of 401 edges is very fast (~1ms).
            # Let's just do a full re-scan for the new state.
            
            new_max = get_max_book()
            
            if new_max < current_max_book:
                current_max_book = new_max
                swapped = True
                break
            elif new_max == current_max_book:
                # Accept with some probability to escape local minima?
                # Or just accept if we are stuck?
                # For now, let's only accept if it improves or is equal but we are "stuck" (no improvement in last 1000 steps).
                # Simpler: Accept if equal, to keep exploring.
                swapped = True
                break
            else:
                # Revert
                edges.remove((x,y))
                edges.add((u,v))
                adj[x].remove(y)
                adj[y].remove(x)
                adj[u].add(v)
                adj[v].add(u)
                # current_max_book remains the same (we reverted to old state)
        
        if not swapped:
            # Try a different bad edge? 
            # The loop above picks the max edge every time. 
            # If we can't improve the max edge, we are stuck.
            # We should try to improve edges that are *close* to max.
            pass

    result = [[u, v] for u, v in edges]
    return result
