import time
import random
import heapq

def construct():
    n = 40
    m = 401
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Helper to create K_20,20 + 1 edge
    def init_turan_plus_one():
        edges = set()
        # Partitions 0..19 and 20..39
        for u in range(20):
            for v in range(20, 40):
                edges.add((u, v))
        # Add one edge within partition 0
        # Pick random distinct u1, u2 from 0..19
        u1 = random.randint(0, 19)
        u2 = random.randint(0, 19)
        while u2 == u1:
            u2 = random.randint(0, 19)
        edges.add((u1, u2))
        return edges

    # Helper to create random graph
    def init_random():
        edges = set()
        while len(edges) < m:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v:
                edges.add((min(u, v), max(u, v)))
        return edges

    def setup_graph(edges):
        masks = [0] * n
        for u, v in edges:
            masks[u] |= 1 << v
            masks[v] |= 1 << u
        
        # Calculate book size for each edge
        # Store in a dict: edge -> book_size
        book_sizes = {}
        max_book = 0
        for u, v in edges:
            b = (masks[u] & masks[v]).bit_count()
            book_sizes[(u, v)] = b
            if b > max_book:
                max_book = b
        return masks, book_sizes, max_book

    def get_max_book_from_dict(book_sizes):
        max_b = 0
        for b in book_sizes.values():
            if b > max_b:
                max_b = b
        return max_b

    best_score = float('inf')
    best_edges = None
    
    # Try multiple strategies/restarts
    # Strategy 1: Turan + 1
    # Strategy 2: Random
    
    for attempt in range(20): # Try a few restarts if time permits
        if time.time() > deadline:
            break
            
        if attempt % 2 == 0:
            edges = init_turan_plus_one()
        else:
            edges = init_random()
            
        masks, book_sizes, current_max_book = setup_graph(edges)
        
        if current_max_book < best_score:
            best_score = current_max_book
            best_edges = set(edges)
            
        # Local Search
        no_improve = 0
        while time.time() < deadline - 5.0: # 5s margin for final return
            # Pick random edge to remove
            u, v = random.choice(list(edges))
            
            # Pick random non-edge to add
            # We need a non-edge.
            # Total pairs 780. Edges 401. Non-edges 379.
            # Probability of hit is ~50%.
            while True:
                x = random.randint(0, n - 1)
                y = random.randint(0, n - 1)
                if x == y:
                    continue
                x, y = min(x, y), max(x, y)
                if (x, y) not in edges:
                    break
            
            # Perform Swap
            # 1. Remove (u, v)
            edges.remove((u, v))
            masks[u] &= ~(1 << v)
            masks[v] &= ~(1 << u)
            
            # Update book sizes for edges affected by removal of (u, v)
            # Edges (u, w) for w in N(u) and (v, z) for z in N(v)
            # Note: N(u) and N(v) are derived from masks AFTER removal?
            # No, we need the neighbors BEFORE removal to know which edges' books changed.
            # But we already removed it from masks. Let's restore logic or use pre-computed neighbors.
            # Actually, the book size of an edge (a, b) is |N(a) \cap N(b)|.
            # If we remove (u, v):
            # - The edge (u, v) is gone.
            # - For any edge (u, w) where w != v, the set N(u) loses v.
            #   So the book size of (u, w) changes only if v was in N(w) as well?
            #   i.e., if (u, v, w) was a triangle.
            #   If (u, v, w) was a triangle, then v was in N(u) \cap N(w).
            #   Removing (u, v) removes v from N(u). So v is no longer in N(u) \cap N(w).
            #   So book size of (u, w) decreases by 1 if (u, v, w) was a triangle.
            #   Similarly for (v, w).
            
            # Let's identify affected edges.
            # Neighbors of u before removal: N_u_old
            # Neighbors of v before removal: N_v_old
            # We need N_u_old and N_v_old. We can get them from masks before we cleared the bits?
            # We cleared them. Let's re-add them temporarily or just use the fact that we know u and v.
            # Actually, it's easier to just recalculate book sizes for all edges incident to u or v.
            # Degree is ~20. So 40 edges.
            
            # Restore masks temporarily to find neighbors? No, we can just look at the edges set.
            # But iterating edges set is O(M).
            # Let's use the masks. We need the OLD masks.
            # Let's save old masks? No, too much memory/time.
            # Let's just recalculate book sizes for all edges incident to u, v, x, y.
            # This is safe and simple.
            
            # 2. Add (x, y)
            edges.add((x, y))
            masks[x] |= 1 << y
            masks[y] |= 1 << x
            
            # Identify affected edges:
            # All edges incident to u, v, x, y.
            affected_nodes = {u, v, x, y}
            affected_edges = set()
            for node in affected_nodes:
                # Find all edges incident to node
                # We can iterate through masks[node] bits
                mask = masks[node]
                while mask:
                    lsb = mask & -mask
                    neighbor = lsb.bit_length() - 1
                    edge = (min(node, neighbor), max(node, neighbor))
                    affected_edges.add(edge)
                    mask ^= lsb
            
            # Recalculate book sizes for affected edges
            # Also, the new edge (x, y) is in affected_edges.
            # The removed edge (u, v) is NOT in edges anymore, so we should delete it from book_sizes.
            
            for e in affected_edges:
                if e in edges:
                    a, b = e
                    b_size = (masks[a] & masks[b]).bit_count()
                    book_sizes[e] = b_size
                else:
                    # This edge was removed (u, v) or was not in affected set?
                    # If e is (u, v), it is not in edges.
                    if e in book_sizes:
                        del book_sizes[e]
            
            # What if (u, v) was not in affected_edges?
            # It is incident to u and v, so it IS in affected_edges.
            # So it will be deleted.
            
            # Calculate new max book
            # We can just scan all book_sizes.
            new_max_book = 0
            for b in book_sizes.values():
                if b > new_max_book:
                    new_max_book = b
            
            if new_max_book < current_max_book:
                current_max_book = new_max_book
                no_improve = 0
                if current_max_book < best_score:
                    best_score = current_max_book
                    best_edges = set(edges)
            elif new_max_book == current_max_book:
                # Accept with some probability to allow random walk
                if random.random() < 0.5:
                    no_improve = 0
                else:
                    # Revert
                    edges.remove((x, y))
                    edges.add((u, v))
                    masks[x] &= ~(1 << y)
                    masks[y] &= ~(1 << x)
                    masks[u] |= 1 << v
                    masks[v] |= 1 << u
                    no_improve += 1
            else:
                # Revert
                edges.remove((x, y))
                edges.add((u, v))
                masks[x] &= ~(1 << y)
                masks[y] &= ~(1 << x)
                masks[u] |= 1 << v
                masks[v] |= 1 << u
                no_improve += 1
            
            # If stuck for too long, restart this attempt
            if no_improve > 5000:
                break

    # Ensure we return a valid graph
    if best_edges is None:
        best_edges = init_random()
        
    return [[u, v] for u, v in best_edges]
