import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 148.0  # Safety margin for 150s limit
    
    n = 40
    target_edges = 401
    
    def run_sa(seed, init_strategy, time_budget):
        random.seed(seed)
        
        adj = [0] * n
        edges = set()
        book_sizes = {}
        
        def add_edge(u, v):
            if u == v: return False
            if u > v: u, v = v, u
            if (u, v) in edges: return False
            edges.add((u, v))
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            return True
        
        def remove_edge(u, v):
            if u == v: return False
            if u > v: u, v = v, u
            if (u, v) not in edges: return False
            edges.remove((u, v))
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            # Clean up book_sizes to avoid ghost edges
            if (u, v) in book_sizes:
                del book_sizes[(u, v)]
            return True
        
        def calc_book(u, v):
            common = adj[u] & adj[v]
            return common.bit_count()
        
        def update_books_for_vertex(v):
            for w in range(n):
                if (1 << w) & adj[v]:
                    u, x = (v, w) if v < w else (w, v)
                    book_sizes[(u, x)] = calc_book(u, x)
        
        def get_max_book():
            if not book_sizes: return 0
            return max(book_sizes.values())
        
        def get_max_edges():
            m = get_max_book()
            # Only consider edges that are actually in the graph
            return [(u, v) for (u, v), s in book_sizes.items() if s == m and (u, v) in edges]
        
        # Initialize based on strategy
        if init_strategy == 0:
            # K20,20 + 1 edge
            for i in range(20):
                for j in range(20, 40):
                    add_edge(i, j)
            add_edge(0, 1)
        elif init_strategy == 1:
            # K20,20 - 5 cross + 6 intra
            for i in range(20):
                for j in range(20, 40):
                    add_edge(i, j)
            # Remove 5 random cross edges
            cross_edges = [(i, j) for i in range(20) for j in range(20, 40)]
            random.shuffle(cross_edges)
            for k in range(5):
                u, v = cross_edges[k]
                remove_edge(u, v)
            # Add 6 random intra edges
            intra_edges = [(i, j) for i in range(20) for j in range(i+1, 20)] + [(i, j) for i in range(20, 40) for j in range(i+1, 40)]
            random.shuffle(intra_edges)
            for k in range(6):
                u, v = intra_edges[k]
                add_edge(u, v)
        elif init_strategy == 2:
            # K20,20 - 10 cross + 11 intra
            for i in range(20):
                for j in range(20, 40):
                    add_edge(i, j)
            cross_edges = [(i, j) for i in range(20) for j in range(20, 40)]
            random.shuffle(cross_edges)
            for k in range(10):
                u, v = cross_edges[k]
                remove_edge(u, v)
            intra_edges = [(i, j) for i in range(20) for j in range(i+1, 20)] + [(i, j) for i in range(20, 40) for j in range(i+1, 40)]
            random.shuffle(intra_edges)
            for k in range(11):
                u, v = intra_edges[k]
                add_edge(u, v)
        elif init_strategy == 3:
            # Random graph with 401 edges
            all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(all_pairs)
            for i in range(target_edges):
                add_edge(all_pairs[i][0], all_pairs[i][1])
        else:
            # K20,20 + 1 edge (default)
            for i in range(20):
                for j in range(20, 40):
                    add_edge(i, j)
            add_edge(0, 1)
        
        # Initialize book sizes
        for u, v in edges:
            book_sizes[(u, v)] = calc_book(u, v)
        
        best_max = get_max_book()
        best_edges = set(edges)
        
        T = 5.0
        T_min = 0.1
        
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        while time.time() - start_time < deadline and time.time() < start_time + time_budget:
            if T < T_min:
                T = 5.0  # Restart temperature
                
            current_max = get_max_book()
            
            max_edges = get_max_edges()
            if not max_edges:
                break
                
            u, v = random.choice(max_edges)
            
            candidate = None
            candidate_book = 0
            
            # Try more candidates to find a better swap
            for _ in range(50):
                x, y = random.choice(all_pairs)
                if (x, y) in edges:
                    continue
                
                pot_book = calc_book(x, y)
                
                if pot_book > current_max:
                    continue
                
                if candidate is None or pot_book < candidate_book:
                    candidate = (x, y)
                    candidate_book = pot_book
                    
            if candidate is None:
                continue
                
            x, y = candidate
            
            # Perform swap
            if not remove_edge(u, v):
                continue
            update_books_for_vertex(u)
            update_books_for_vertex(v)
            
            if not add_edge(x, y):
                # Rollback
                add_edge(u, v)
                update_books_for_vertex(u)
                update_books_for_vertex(v)
                continue
            update_books_for_vertex(x)
            update_books_for_vertex(y)
            
            new_max = get_max_book()
            
            if new_max < current_max:
                best_max = new_max
                best_edges = set(edges)
                T = max(T_min, T * 0.9)
            elif new_max == current_max:
                T = max(T_min, T * 0.99)
            else:
                delta = new_max - current_max
                if random.random() < pow(2.718281828, -delta / T):
                    T = max(T_min, T * 0.99)
                else:
                    # Rollback
                    remove_edge(x, y)
                    update_books_for_vertex(x)
                    update_books_for_vertex(y)
                    
                    add_edge(u, v)
                    update_books_for_vertex(u)
                    update_books_for_vertex(v)
                    
                    T = max(T_min, T * 0.95)
        
        return best_max, best_edges
    
    # Run multiple SA instances with different seeds and strategies
    best_overall_max = float('inf')
    best_overall_edges = None
    
    # Allocate time: try several diverse starts
    strategies = [
        (0, 0), (1, 1), (2, 2), (3, 3), (0, 4), (1, 5), (2, 6), (3, 7),
        (0, 8), (1, 9), (2, 10), (3, 11), (0, 12), (1, 13), (2, 14), (3, 15)
    ]
    
    num_runs = min(len(strategies), 10)  # Do up to 10 runs
    time_per_run = 14.0  # ~10 * 14 = 140s, leaving buffer
    
    for i in range(num_runs):
        if time.time() - start_time > deadline - 5:
            break
            
        strategy = strategies[i][0]
        seed = strategies[i][1]
        
        cur_max, cur_edges = run_sa(seed, strategy, time_per_run)
        
        if cur_max < best_overall_max:
            best_overall_max = cur_max
            best_overall_edges = cur_edges
            
        # If we found book size 6 or less, we can stop early
        if best_overall_max <= 6:
            break
    
    if best_overall_edges is None:
        # Fallback: return K20,20 + 1
        return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
    
    return [[u, v] for u, v in best_overall_edges]
