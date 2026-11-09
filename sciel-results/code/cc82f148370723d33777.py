import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    n = 40
    target_edges = 401
    
    # 1. Structural Initialization: Replicate Inspiration 1 (Score 2.22)
    # This structure likely has a book size of 7 or 8.
    A = list(range(9))
    B = list(range(9, 40))
    
    edges = set()
    
    # 1. Add all edges between A and B
    for a in A:
        for b in B:
            edges.add((a, b))
            
    # 2. Add 122 edges within B
    B1 = list(range(9, 24))
    B2 = list(range(24, 40))
    
    needed = 122
    added = 0
    
    deg_b1 = {v: 0 for v in B1}
    deg_b2 = {v: 0 for v in B2}
    
    pairs = [(u, v) for u in B1 for v in B2]
    random.shuffle(pairs)
    
    for u, v in pairs:
        if added == needed:
            break
        if deg_b1[u] < 9 and deg_b2[v] < 9:
            edges.add((u, v))
            deg_b1[u] += 1
            deg_b2[v] += 1
            added += 1
            
    if added < needed:
        # Fill remaining if necessary, ignoring degree constraint
        for u, v in pairs:
            if added == needed:
                break
            if (u, v) not in edges:
                edges.add((u, v))
                added += 1

    # Adjacency sets for fast common neighbor calculation
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    def get_book_size(u, v):
        if v not in adj[u]:
            return 0
        return len(adj[u].intersection(adj[v]))
        
    def get_max_book_and_edges():
        max_b = 0
        max_edges = []
        for u, v in edges:
            b = get_book_size(u, v)
            if b > max_b:
                max_b = b
                max_edges = [(u, v)]
            elif b == max_b:
                max_edges.append((u, v))
        return max_b, max_edges

    current_max_book, current_max_edges = get_max_book_and_edges()
    
    # 2. Local Search Refinement with Greedy Swaps
    # Strategy: 
    # 1. Pick a max-book edge to remove.
    # 2. Find a non-edge with minimum book size to add.
    # 3. Accept if it reduces max book, or keeps it same but reduces count of max edges.
    
    no_improve_count = 0
    max_no_improve = 500 # Restart or accept if stuck
    
    while time.time() < deadline:
        if current_max_book <= 7:
            # We hit the theoretical lower bound (integer). 
            # We can try to polish further but 7 is likely the floor.
            break
        
        # Pick a random edge from the max-book set to remove
        u, v = random.choice(current_max_edges)
        
        # Find the best non-edge to add
        # We want to minimize the book size of the new edge, and ideally not increase others.
        # A good heuristic: find non-edge (x,y) with min common neighbors.
        min_book_new = float('inf')
        best_x, best_y = -1, -1
        
        # Iterate over all non-edges
        # To speed up, we can just check a sample, but N=40 is small.
        # 40*39/2 = 780 pairs.
        for x in range(n):
            for y in range(x+1, n):
                if (x, y) in edges or (y, x) in edges:
                    continue
                # Check if it's a valid non-edge
                # Calculate book size
                b = len(adj[x].intersection(adj[y]))
                if b < min_book_new:
                    min_book_new = b
                    best_x, best_y = x, y
        
        if best_x == -1:
            # No non-edges? Impossible with 401 edges out of 780.
            break
            
        # Perform the swap: remove (u,v), add (best_x, best_y)
        # We need to check if this swap improves the global max.
        
        # Save state for rollback
        old_max = current_max_book
        old_max_count = len(current_max_edges)
        
        # Remove (u,v)
        edges.remove((u, v))
        adj[u].remove(v)
        adj[v].remove(u)
        
        # Add (best_x, best_y)
        edges.add((best_x, best_y))
        adj[best_x].add(best_y)
        adj[best_y].add(best_x)
        
        # Recalculate max book
        new_max, new_max_edges = get_max_book_and_edges()
        
        # Acceptance Criteria:
        # 1. new_max < old_max: Always accept.
        # 2. new_max == old_max:
        #    - If len(new_max_edges) < old_max_count: Accept (reduces "bottleneck" edges).
        #    - Else: Accept with probability 0.1 to escape local minima? Or reject.
        # 3. new_max > old_max: Reject.
        
        accepted = False
        if new_max < old_max:
            accepted = True
            no_improve_count = 0
        elif new_max == old_max:
            if len(new_max_edges) < old_max_count:
                accepted = True
                no_improve_count = 0
            else:
                # Small chance to accept to move around
                if random.random() < 0.05:
                    accepted = True
                else:
                    # Rollback
                    pass # Rollback happens if not accepted
        else:
            # Rollback
            pass

        if accepted:
            current_max_book = new_max
            current_max_edges = new_max_edges
        else:
            # Rollback
            edges.add((u, v))
            adj[u].add(v)
            adj[v].add(u)
            
            edges.remove((best_x, best_y))
            adj[best_x].remove(best_y)
            adj[best_y].remove(best_x)
            
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Random restart or perturb
                # Just break and rely on the fact that we tried our best?
                # Or do a few random swaps to jump out of basin.
                for _ in range(10):
                    # Random swap
                    e1 = random.choice(list(edges))
                    u1, v1 = e1
                    # Find random non-edge
                    while True:
                        x = random.randint(0, n-1)
                        y = random.randint(0, n-1)
                        if x == y: continue
                        if x > y: x, y = y, x
                        if (x, y) not in edges: break
                    
                    edges.remove(e1)
                    adj[u1].remove(v1)
                    adj[v1].remove(u1)
                    
                    edges.add((x, y))
                    adj[x].add(y)
                    adj[y].add(x)
                    
                    new_max, new_max_edges = get_max_book_and_edges()
                    if new_max <= current_max_book:
                        current_max_book = new_max
                        current_max_edges = new_max_edges
                        no_improve_count = 0
                        break
                    else:
                        # Rollback
                        edges.remove((x, y))
                        adj[x].remove(y)
                        adj[y].remove(x)
                        edges.add(e1)
                        adj[u1].add(v1)
                        adj[v1].add(u1)
                
                no_improve_count = 0
                max_no_improve += 100 # Allow more patience

    result = [[u, v] for u, v in edges]
    return result
