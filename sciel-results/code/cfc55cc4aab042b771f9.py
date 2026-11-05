import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n = 40
    m = 401
    random.seed(42)
    
    # Helper to compute books
    def compute_books(adj_mask):
        books = {}
        for u in range(n):
            mask = adj_mask[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                mask ^= lsb
                if u < v:
                    books[(u, v)] = (adj_mask[u] & adj_mask[v]).bit_count()
        return books

    # Helper to get max and sum of squares
    def get_metrics(books):
        if not books:
            return 0, 0
        max_book = max(books.values())
        sum_sq = sum(b * b for b in books.values())
        return max_book, sum_sq

    # Generate a random graph with m edges
    def generate_random_graph(seed):
        rng = random.Random(seed)
        adj_mask = [0] * n
        edge_set = set()
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        # Shuffle and pick m
        rng.shuffle(all_edges)
        for e in all_edges[:m]:
            u, v = e
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            edge_set.add(e)
        return adj_mask, edge_set

    # Local search
    def local_search(adj_mask, edge_set, T_init, T_min, cooling, max_steps):
        books = compute_books(adj_mask)
        current_max_book, current_sum_sq = get_metrics(books)
        best_max_book = current_max_book
        best_sum_sq = current_sum_sq
        best_adj_mask = list(adj_mask)
        best_edge_set = set(edge_set)
        
        T = T_init
        non_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if (i, j) not in edge_set]
        
        step_count = 0
        while T > T_min and step_count < max_steps:
            step_count += 1
            if time.time() > deadline - 5:
                break
            
            # Select edge to remove: bias towards high book sizes
            if random.random() < 0.8:
                # Pick from top 20% book sizes
                sorted_books = sorted(books.items(), key=lambda x: -x[1])
                top_k = max(1, len(sorted_books) // 5)
                top_edges = [e for e, b in sorted_books[:top_k]]
                e_remove = random.choice(top_edges)
            else:
                e_remove = random.choice(list(edge_set))
            
            u, v = e_remove
            
            # Select edge to add
            if not non_edges:
                break
                
            # Sample candidates
            sample_size = min(100, len(non_edges))
            candidates = random.sample(non_edges, sample_size)
            
            best_add = None
            best_add_score = float('inf')
            
            # Evaluate candidates
            for e in candidates:
                a, b = e
                # Simulate addition
                # Common neighbors of a and b in current graph
                common = (adj_mask[a] & adj_mask[b]).bit_count()
                # This is the book size if we add (a,b)
                # We want to minimize the new max book size
                # Approximate new max book size:
                # 1. The new edge (a,b) has book size 'common'
                # 2. Other edges might decrease their book size
                # Use 'common' as a proxy
                if common < best_add_score:
                    best_add_score = common
                    best_add = e
                    
            if not best_add:
                T *= cooling
                continue
                
            a, b = best_add
            
            # Apply move
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
            adj_mask[a] |= (1 << b)
            adj_mask[b] |= (1 << a)
            
            edge_set.remove(e_remove)
            edge_set.add(best_add)
            
            non_edges.remove(best_add)
            non_edges.append(e_remove)
            
            # Recompute books
            new_books = compute_books(adj_mask)
            new_max_book, new_sum_sq = get_metrics(new_books)
            
            # Acceptance
            accept = False
            if new_max_book < current_max_book:
                accept = True
            elif new_max_book == current_max_book:
                if new_sum_sq < current_sum_sq:
                    accept = True
                else:
                    delta_sq = new_sum_sq - current_sum_sq
                    if delta_sq > 0:
                        prob = math.exp(-delta_sq / (T * 100))
                        if random.random() < prob:
                            accept = True
                    else:
                        accept = True
            else:
                delta_max = new_max_book - current_max_book
                prob = math.exp(-delta_max / T)
                if random.random() < prob:
                    accept = True
                    
            if accept:
                current_max_book = new_max_book
                current_sum_sq = new_sum_sq
                books = new_books
                
                if current_max_book < best_max_book or (current_max_book == best_max_book and current_sum_sq < best_sum_sq):
                    best_max_book = current_max_book
                    best_sum_sq = current_sum_sq
                    best_adj_mask = list(adj_mask)
                    best_edge_set = set(edge_set)
            else:
                # Revert
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                adj_mask[a] &= ~(1 << b)
                adj_mask[b] &= ~(1 << a)
                
                edge_set.remove(best_add)
                edge_set.add(e_remove)
                
                non_edges.remove(e_remove)
                non_edges.append(best_add)
                
            T *= cooling
            
        return best_adj_mask, best_edge_set, best_max_book, best_sum_sq

    best_global_max = float('inf')
    best_global_sum_sq = float('inf')
    best_global_adj = None
    best_global_edges = None
    
    # Multiple restarts
    restarts = 0
    while time.time() < deadline - 10:
        restarts += 1
        # Generate initial graph
        adj_mask, edge_set = generate_random_graph(42 + restarts)
        
        # Run local search
        T_init = 5.0
        T_min = 0.01
        cooling = 0.999
        max_steps = 50000
        
        res_adj, res_edges, res_max, res_sq = local_search(adj_mask, edge_set, T_init, T_min, cooling, max_steps)
        
        if res_max < best_global_max or (res_max == best_global_max and res_sq < best_global_sum_sq):
            best_global_max = res_max
            best_global_sum_sq = res_sq
            best_global_adj = res_adj
            best_global_edges = res_edges
            
        # If we found 7, we can stop early?
        # The theoretical lower bound is 7.
        if best_global_max == 7:
            break
            
    if best_global_edges is None:
        # Fallback
        adj_mask, edge_set = generate_random_graph(42)
        best_global_edges = edge_set
        
    return [list(e) for e in best_global_edges]
