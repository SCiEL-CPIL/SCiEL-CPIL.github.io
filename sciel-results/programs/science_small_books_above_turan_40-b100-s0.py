import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n = 40
    m = 401
    random.seed(42)
    
    # Bitmask representation for fast intersection
    # adj_mask[i] is a 40-bit integer where bit j is set if edge (i,j) exists
    
    def compute_books(adj_mask):
        """Compute book size for each edge using bitmask intersection."""
        books = {}
        for u in range(n):
            # Iterate over set bits in adj_mask[u]
            mask = adj_mask[u]
            while mask:
                v = (mask & -mask).bit_length() - 1
                mask &= mask - 1
                if u < v:
                    common = adj_mask[u] & adj_mask[v]
                    books[(u, v)] = bin(common).count('1')
        return books

    def get_max_book(books):
        return max(books.values()) if books else 0

    def get_sum_sq_books(books):
        return sum(b*b for b in books.values())

    # 1. Initialization: Create a graph close to regular
    # Average degree is 2*m/n = 802/40 = 20.05
    # So we want degrees around 20 or 21.
    # Start with a random regular graph of degree 20 (40*20/2 = 400 edges)
    # Then add 1 edge.
    
    # Construct a random 20-regular graph on 40 vertices
    # This is possible since n*d is even (40*20=800).
    # We use a simple greedy approach or a known algorithm.
    # For simplicity and speed, we can start with a random graph and balance degrees.
    
    def random_regular_graph(n, d):
        """Generate a random d-regular graph."""
        # This implementation is a heuristic for small n
        adj_mask = [0] * n
        edges = set()
        
        # List of vertices sorted by current degree
        # We need to add d edges per vertex.
        # Total stubs = n*d
        
        # Simple approach: Generate random edges and fix degrees? 
        # Better: Use a pairing model or specific construction.
        # For n=40, d=20, we can just take a random graph with 400 edges 
        # and try to balance degrees via swaps if they are very uneven.
        
        # Let's just generate a random graph with 400 edges first.
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_possible)
        chosen = all_possible[:400]
        
        for u, v in chosen:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            edges.add((u, v))
            
        # Check degrees
        degrees = [bin(adj_mask[i]).count('1') for i in range(n)]
        
        # If not regular, try to balance via swaps.
        # This might take time, so we do a few rounds.
        for _ in range(100):
            # Find a vertex with high degree and one with low degree
            max_idx = max(range(n), key=lambda i: degrees[i])
            min_idx = min(range(n), key=lambda i: degrees[i])
            
            if degrees[max_idx] - degrees[min_idx] < 2:
                break
                
            # Find a neighbor of max_idx that is not connected to min_idx
            # and a non-neighbor of max_idx that is connected to min_idx
            # Swap them.
            
            # Neighbors of max_idx
            neigh_max = [j for j in range(n) if (adj_mask[max_idx] >> j) & 1]
            # Non-neighbors of max_idx (excluding self)
            non_neigh_max = [j for j in range(n) if j != max_idx and not ((adj_mask[max_idx] >> j) & 1)]
            
            # Neighbors of min_idx
            neigh_min = [j for j in range(n) if (adj_mask[min_idx] >> j) & 1]
            # Non-neighbors of min_idx (excluding self)
            non_neigh_min = [j for j in range(n) if j != min_idx and not ((adj_mask[min_idx] >> j) & 1)]
            
            # We want to remove (max_idx, u) where u is in neigh_max
            # and add (max_idx, v) where v is in non_neigh_max
            # AND remove (min_idx, v) ... wait, this changes degrees.
            # Swap: Remove (max_idx, u) and (min_idx, v)? No.
            # To balance degrees, we need to move an edge from max_idx to min_idx.
            # Find u such that (max_idx, u) is an edge.
            # Find v such that (min_idx, v) is an edge? No, we want to ADD to min_idx.
            # So we need to remove an edge from max_idx and add it to min_idx.
            # This requires a common neighbor? No.
            # We need to find an edge (max_idx, u) and an edge (min_idx, w) such that
            # we can swap them? No, that keeps degrees same.
            
            # To change degrees, we must add an edge to min_idx and remove from max_idx.
            # This changes total edges? No, we must swap an edge.
            # If we remove (max_idx, u) and add (min_idx, u), degrees change:
            # max_idx -1, min_idx +1. But (min_idx, u) must not exist.
            # And (max_idx, u) must exist.
            # This is a valid move if u is not connected to min_idx.
            
            moved = False
            for u in neigh_max:
                if u == min_idx:
                    continue
                if not ((adj_mask[min_idx] >> u) & 1):
                    # Remove (max_idx, u), Add (min_idx, u)
                    adj_mask[max_idx] &= ~(1 << u)
                    adj_mask[u] &= ~(1 << max_idx)
                    edges.discard((min(max_idx, u), max(max_idx, u)))
                    
                    adj_mask[min_idx] |= (1 << u)
                    adj_mask[u] |= (1 << min_idx)
                    edges.add((min(min_idx, u), max(min_idx, u)))
                    
                    degrees[max_idx] -= 1
                    degrees[min_idx] += 1
                    degrees[u] = degrees[u] # Degree of u unchanged? 
                    # Wait, u lost connection to max_idx, gained to min_idx. Degree of u is same.
                    moved = True
                    break
            
            if not moved:
                break
                
        return edges, adj_mask

    # Initialize
    edge_set, adj_mask = random_regular_graph(n, 20)
    
    # We have 400 edges. We need 401.
    # Add one random edge that doesn't already exist.
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    missing = [e for e in all_possible if e not in edge_set]
    random.shuffle(missing)
    if missing:
        e_add = missing[0]
        u, v = e_add
        edge_set.add(e_add)
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    books = compute_books(adj_mask)
    current_max_book = get_max_book(books)
    current_sum_sq = get_sum_sq_books(books)
    
    best_max_book = current_max_book
    best_sum_sq = current_sum_sq
    best_edge_set = set(edge_set)
    best_adj_mask = list(adj_mask)
    
    T = 5.0
    T_min = 0.1
    cooling = 0.9999
    
    # Precompute list of edges for random selection
    # We need to be careful with the list of edges in edge_set
    
    while time.time() < deadline and T > T_min:
        # Select an edge to remove
        # Strategy: Prefer edges with high book size
        if random.random() < 0.7:
            # Find top 10% edges by book size
            sorted_books = sorted(books.items(), key=lambda x: -x[1])
            top_k = max(1, len(sorted_books) // 10)
            e_remove = random.choice(sorted_books[:top_k])[0]
        else:
            e_remove = random.choice(list(edge_set))
            
        u, v = e_remove
        
        # Select an edge to add
        # Strategy: Prefer edges that connect vertices with low common neighbors
        # or random
        if random.random() < 0.5:
            e_add = random.choice([e for e in all_possible if e not in edge_set])
        else:
            # Find candidates that reduce book size
            # We want to add (a,b) such that common neighbors are low
            # This is expensive to sort all candidates.
            # Just pick a few random candidates and pick the best
            candidates = random.sample([e for e in all_possible if e not in edge_set], min(50, len(missing)))
            best_cand = None
            best_score = 100
            for e in candidates:
                a, b = e
                score = bin(adj_mask[a] & adj_mask[b]).count('1')
                if score < best_score:
                    best_score = score
                    best_cand = e
            if best_cand:
                e_add = best_cand
            else:
                continue
                
        if e_add == e_remove or e_add in edge_set:
            T *= cooling
            continue
            
        a, b = e_add
        
        # Calculate delta
        # Affected edges are those incident to u, v, a, b
        affected_verts = {u, v, a, b}
        
        # Store old books for affected edges
        old_books = {}
        for e in edge_set:
            x, y = e
            if x in affected_verts or y in affected_verts:
                old_books[e] = books[e]
                
        # Also store old book for e_add if it existed (it doesn't)
        
        # Apply move
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        adj_mask[a] |= (1 << b)
        adj_mask[b] |= (1 << a)
        
        edge_set.remove(e_remove)
        edge_set.add(e_add)
        
        # Recompute books for affected edges
        new_books = {}
        for e in edge_set:
            x, y = e
            if x in affected_verts or y in affected_verts:
                common = adj_mask[x] & adj_mask[y]
                new_books[e] = bin(common).count('1')
                
        # Update global books dict
        for e in old_books:
            if e in new_books:
                books[e] = new_books[e]
            else:
                del books[e]
        for e in new_books:
            if e not in old_books:
                books[e] = new_books[e]
                
        new_max_book = get_max_book(books)
        new_sum_sq = get_sum_sq_books(books)
        
        # Acceptance
        # Primary: Max book
        # Secondary: Sum of squares (spectral proxy)
        
        accept = False
        if new_max_book < current_max_book:
            accept = True
        elif new_max_book == current_max_book:
            # Spectral improvement: lower variance/sum of squares
            if new_sum_sq < current_sum_sq:
                accept = True
            else:
                # SA step
                # Delta is effectively 0 for max book, but we consider the "energy" drop
                # We can just use random acceptance with T
                if random.random() < math.exp(-(new_sum_sq - current_sum_sq) / (T * 1000)):
                    accept = True
        else:
            # Max book increased
            if random.random() < math.exp(-(new_max_book - current_max_book) / T):
                accept = True
                
        if accept:
            current_max_book = new_max_book
            current_sum_sq = new_sum_sq
            
            if current_max_book < best_max_book or (current_max_book == best_max_book and current_sum_sq < best_sum_sq):
                best_max_book = current_max_book
                best_sum_sq = current_sum_sq
                best_edge_set = set(edge_set)
                best_adj_mask = list(adj_mask)
        else:
            # Revert
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            adj_mask[a] &= ~(1 << b)
            adj_mask[b] &= ~(1 << a)
            
            edge_set.remove(e_add)
            edge_set.add(e_remove)
            
            # Revert books
            for e in new_books:
                if e in old_books:
                    books[e] = old_books[e]
                else:
                    del books[e]
            for e in old_books:
                if e not in new_books:
                    books[e] = old_books[e]
                    
        T *= cooling
        
    return [list(e) for e in best_edge_set]
