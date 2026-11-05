import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    n = 40
    m = 401
    random.seed(42)
    
    # Bitmask representation for fast intersection
    adj_mask = [0] * n
    
    def compute_books(adj_mask):
        """Compute book size for each edge using bitmask intersection."""
        books = {}
        for u in range(n):
            mask = adj_mask[u]
            while mask:
                v = (mask & -mask).bit_length() - 1
                mask &= mask - 1
                if u < v:
                    common = adj_mask[u] & adj_mask[v]
                    books[(u, v)] = common.bit_count()
        return books

    def update_books_incremental(adj_mask, edge_set, removed_edge, added_edge, books):
        """
        Update the books dictionary efficiently after removing 'removed_edge' and adding 'added_edge'.
        Only edges incident to the endpoints of these two edges are affected.
        """
        u, v = removed_edge
        a, b = added_edge
        
        # Vertices whose adjacency changed
        changed_verts = {u, v, a, b}
        
        # Identify all edges that might have changed book size
        affected_edges = set()
        
        for vert in changed_verts:
            mask = adj_mask[vert]
            while mask:
                neighbor = (mask & -mask).bit_length() - 1
                mask &= mask - 1
                e = (min(vert, neighbor), max(vert, neighbor))
                affected_edges.add(e)
                
        # Remove the old edge from books
        if removed_edge in books:
            del books[removed_edge]
            
        # Add the new edge to books
        common_ab = (adj_mask[a] & adj_mask[b]).bit_count()
        books[(a, b)] = common_ab
        
        # Update books for other affected edges
        for e in affected_edges:
            if e == removed_edge:
                continue
            if e == (a, b):
                continue
                
            x, y = e
            if e in edge_set:
                common = (adj_mask[x] & adj_mask[y]).bit_count()
                books[e] = common

    def get_max_book(books):
        return max(books.values()) if books else 0

    def get_sum_sq_books(books):
        return sum(b*b for b in books.values())

    def get_sum_books(books):
        return sum(books.values())

    def balanced_bipartite_init(n, m):
        """
        Initialize with a graph close to the complete bipartite graph K20,20 (400 edges),
        plus one random edge. This is the extremal structure for minimizing books.
        """
        adj_mask = [0] * n
        edge_set = set()
        
        # Split into two sets of 20
        set_a = list(range(20))
        set_b = list(range(20, 40))
        
        # Add all edges between set_a and set_b (400 edges)
        for u in set_a:
            for v in set_b:
                edge_set.add((u, v))
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
        
        # Add 1 random edge within a set
        # Pick a random pair from set_a or set_b
        if random.random() < 0.5:
            u, v = random.sample(set_a, 2)
        else:
            u, v = random.sample(set_b, 2)
        
        u, v = min(u, v), max(u, v)
        edge_set.add((u, v))
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
        return edge_set, adj_mask

    def random_regular_init(n, m):
        """
        Alternative: Random graph with 401 edges, balanced to be near-regular.
        """
        adj_mask = [0] * n
        edge_set = set()
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_possible)
        
        for u, v in all_possible[:m]:
            edge_set.add((u, v))
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        
        # Balance degrees via swaps
        degrees = [adj_mask[i].bit_count() for i in range(n)]
        
        for _ in range(200):
            max_idx = max(range(n), key=lambda i: degrees[i])
            min_idx = min(range(n), key=lambda i: degrees[i])
            
            if degrees[max_idx] - degrees[min_idx] < 2:
                break
            
            # Try to move an edge from max_idx to min_idx
            moved = False
            neigh_max = [j for j in range(n) if (adj_mask[max_idx] >> j) & 1]
            random.shuffle(neigh_max)
            
            for u in neigh_max:
                if u == min_idx:
                    continue
                if not ((adj_mask[min_idx] >> u) & 1):
                    # Remove (max_idx, u), Add (min_idx, u)
                    adj_mask[max_idx] &= ~(1 << u)
                    adj_mask[u] &= ~(1 << max_idx)
                    edge_set.discard((min(max_idx, u), max(max_idx, u)))
                    
                    adj_mask[min_idx] |= (1 << u)
                    adj_mask[u] |= (1 << min_idx)
                    edge_set.add((min(min_idx, u), max(min_idx, u)))
                    
                    degrees[max_idx] -= 1
                    degrees[min_idx] += 1
                    moved = True
                    break
            
            if not moved:
                break
                
        return edge_set, adj_mask

    def greedy_randomized_init(n, m):
        """
        GRASP: Greedy Randomized Adaptive Search Procedure.
        Build the graph edge by edge, each time choosing from a restricted
        candidate list that minimizes the resulting book size.
        """
        adj_mask = [0] * n
        edge_set = set()
        all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        # Build a bipartition-like structure first for good base
        set_a = list(range(20))
        set_b = list(range(20, 40))
        
        # Start with a random subset of bipartite edges
        bipartite_edges = [(u, v) for u in set_a for v in set_b]
        random.shuffle(bipartite_edges)
        
        # Add a random fraction of bipartite edges to start
        num_bipartite = random.randint(350, 390)
        for u, v in bipartite_edges[:num_bipartite]:
            edge_set.add((u, v))
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        
        # Now add remaining edges greedily
        remaining_needed = m - len(edge_set)
        
        # Generate candidate edges not yet in the graph
        all_candidates = [e for e in all_possible if e not in edge_set]
        random.shuffle(all_candidates)
        
        for _ in range(remaining_needed):
            if not all_candidates:
                break
            
            # Create a restricted candidate list (RCL)
            # Score each candidate by the book size it would create
            scored_candidates = []
            for e in all_candidates[:min(50, len(all_candidates))]:
                u, v = e
                common = (adj_mask[u] & adj_mask[v]).bit_count()
                scored_candidates.append((common, e))
            
            if not scored_candidates:
                # Fallback: just pick randomly
                e = random.choice(all_candidates)
                edge_set.add(e)
                adj_mask[e[0]] |= (1 << e[1])
                adj_mask[e[1]] |= (1 << e[0])
                all_candidates.remove(e)
                continue
            
            # Sort by book size and pick from top fraction
            scored_candidates.sort(key=lambda x: x[0])
            rcl_size = max(1, len(scored_candidates) // 3)
            rcl = scored_candidates[:rcl_size]
            
            # Randomly pick from RCL
            chosen_book, chosen_edge = random.choice(rcl)
            
            # Add the edge
            u, v = chosen_edge
            edge_set.add(chosen_edge)
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            
            # Remove from candidates
            if chosen_edge in all_candidates:
                all_candidates.remove(chosen_edge)
        
        return edge_set, adj_mask

    # Try multiple initializations and keep the best
    best_edge_set = None
    best_adj_mask = None
    best_max_book = float('inf')
    best_sum_sq = float('inf')
    
    init_methods = [balanced_bipartite_init, random_regular_init, greedy_randomized_init]
    
    for init_method in init_methods:
        edge_set, adj_mask = init_method(n, m)
        books = compute_books(adj_mask)
        current_max_book = get_max_book(books)
        current_sum_sq = get_sum_sq_books(books)
        
        if current_max_book < best_max_book or (current_max_book == best_max_book and current_sum_sq < best_sum_sq):
            best_max_book = current_max_book
            best_sum_sq = current_sum_sq
            best_edge_set = set(edge_set)
            best_adj_mask = list(adj_mask)
    
    edge_set = best_edge_set
    adj_mask = best_adj_mask
    books = compute_books(adj_mask)
    current_max_book = best_max_book
    current_sum_sq = best_sum_sq
    
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # If we already hit 7, we can't do better.
    if current_max_book == 7:
        return [list(e) for e in edge_set]
    
    # Additional local search: try to reduce max book by targeted edge swaps
    # Focus on edges with high book sizes
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        books = compute_books(adj_mask)
        current_max_book = get_max_book(books)
        current_sum_sq = get_sum_sq_books(books)
        
        if current_max_book <= 7:
            break
        
        # Find edges with book size >= current_max_book
        high_book_edges = [(e, b) for e, b in books.items() if b >= current_max_book]
        
        if not high_book_edges:
            break
        
        # Try to improve by swapping edges
        improved = False
        for e_high, b_high in random.sample(high_book_edges, min(5, len(high_book_edges))):
            u, v = e_high
            
            # Try to find a better replacement: remove (u,v) and add an edge
            # that doesn't create a large book
            candidates = []
            for _ in range(20):
                a, b = random.choice(all_possible)
                a, b = min(a, b), max(a, b)
                if (a, b) not in edge_set and (a, b) != e_high:
                    common = (adj_mask[a] & adj_mask[b]).bit_count()
                    if common < b_high:
                        candidates.append((common, (a, b)))
            
            if candidates:
                candidates.sort(key=lambda x: x[0])
                best_common, best_edge = candidates[0]
                
                # Test the swap
                # We need to be careful with incremental updates.
                # Let's stick to full recompute for this "targeted" phase to ensure correctness,
                # as it's only 200 iterations.
                # But for the main SA loop, we will use incremental.
                
                # Actually, let's just use full recompute here for simplicity and safety,
                # as the number of iterations is small.
                old_mask_u = adj_mask[u]
                old_mask_v = adj_mask[v]
                old_mask_a = adj_mask[best_edge[0]]
                old_mask_b = adj_mask[best_edge[1]]
                
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
                adj_mask[best_edge[0]] |= (1 << best_edge[1])
                adj_mask[best_edge[1]] |= (1 << best_edge[0])
                
                edge_set.remove(e_high)
                edge_set.add(best_edge)
                
                new_books = compute_books(adj_mask)
                new_max_book = get_max_book(new_books)
                new_sum_sq = get_sum_sq_books(new_books)
                
                if new_max_book < current_max_book or (new_max_book == current_max_book and new_sum_sq < current_sum_sq):
                    improved = True
                    current_max_book = new_max_book
                    current_sum_sq = new_sum_sq
                    books = new_books
                    
                    if current_max_book < best_max_book or (current_max_book == best_max_book and current_sum_sq < best_sum_sq):
                        best_max_book = current_max_book
                        best_sum_sq = current_sum_sq
                        best_edge_set = set(edge_set)
                        best_adj_mask = list(adj_mask)
                        
                        if best_max_book == 7:
                            return [list(e) for e in best_edge_set]
                else:
                    # Revert
                    adj_mask[u] = old_mask_u
                    adj_mask[v] = old_mask_v
                    adj_mask[best_edge[0]] = old_mask_a
                    adj_mask[best_edge[1]] = old_mask_b
                    
                    edge_set.remove(best_edge)
                    edge_set.add(e_high)
        
        if not improved:
            break
    
    # Simulated annealing with spectral-inspired moves
    # The key insight: minimize the variance of book sizes (flatten the distribution)
    # while keeping the max low.
    
    T = 2.0
    T_min = 0.05
    cooling = 0.9998
    
    while time.time() < deadline and T > T_min:
        # Strategy: Focus on reducing the maximum book and flattening distribution
        sorted_books = sorted(books.items(), key=lambda x: -x[1])
        
        # Select an edge to remove: prefer high-book edges
        if random.random() < 0.85:
            # Pick from top 15% of edges by book size
            top_k = max(1, len(sorted_books) // 7)
            e_remove = random.choice(sorted_books[:top_k])[0]
        else:
            e_remove = random.choice(list(edge_set))
            
        u, v = e_remove
        u, v = min(u, v), max(u, v)
        
        # Select an edge to add: prefer edges that would have low book size
        # Sample more candidates and pick the one with minimum common neighbors
        candidates = []
        for _ in range(50):
            a, b = random.choice(all_possible)
            a, b = min(a, b), max(a, b)
            if (a, b) not in edge_set and (a, b) != e_remove:
                candidates.append((a, b))
        
        if not candidates:
            T *= cooling
            continue
            
        # Pick candidate with minimum common neighbors
        best_cand = None
        best_common = 100
        for a, b in candidates:
            common = (adj_mask[a] & adj_mask[b]).bit_count()
            if common < best_common:
                best_common = common
                best_cand = (a, b)
        
        if best_cand is None:
            T *= cooling
            continue
            
        a, b = best_cand
        
        # Apply the move
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        adj_mask[a] |= (1 << b)
        adj_mask[b] |= (1 << a)
        
        edge_set.remove(e_remove)
        edge_set.add((a, b))
        
        # Use incremental update
        update_books_incremental(adj_mask, edge_set, e_remove, (a, b), books)
        
        new_max_book = get_max_book(books)
        new_sum_sq = get_sum_sq_books(books)
        
        # Acceptance criterion: prioritize max book, then variance
        accept = False
        if new_max_book < current_max_book:
            accept = True
        elif new_max_book == current_max_book:
            if new_sum_sq < current_sum_sq:
                accept = True
            else:
                # Allow small increases in sum_sq with temperature
                if random.random() < math.exp(-(new_sum_sq - current_sum_sq) / (T * 300)):
                    accept = True
        else:
            # Max book increased - only accept with high temperature
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
                
                if best_max_book == 7:
                    # We hit the theoretical minimum
                    return [list(e) for e in best_edge_set]
        else:
            # Revert
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            adj_mask[a] &= ~(1 << b)
            adj_mask[b] &= ~(1 << a)
            
            edge_set.remove((a, b))
            edge_set.add(e_remove)
            
            # Revert books using incremental update logic in reverse
            # Remove (a, b) and add (u, v)
            update_books_incremental(adj_mask, edge_set, (a, b), e_remove, books)
                    
        T *= cooling
        
    return [list(e) for e in best_edge_set]
