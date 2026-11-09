import time
import random
import math
from collections import Counter

def construct():
    start_time = time.time()
    time_limit = 118.0  # Safety margin for 120s limit (evaluator kills at 150s)
    
    n = 40
    target_edges = 401
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    total_pairs = len(all_pairs)  # 780
    non_edge_count = total_pairs - target_edges  # 379
    
    def build_graph(seed):
        random.seed(seed)
        adj = [0] * n
        in_edges = [set() for _ in range(n)]
        books = {}
        book_counter = Counter()
        edges = set()
        
        def add_edge(u, v):
            if u > v:
                u, v = v, u
            if (u, v) in edges:
                return
            edges.add((u, v))
            in_edges[u].add(v)
            in_edges[v].add(u)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            common = adj[u] & adj[v]
            b = common.bit_count()
            books[(u, v)] = b
            book_counter[b] += 1
        
        def remove_edge(u, v):
            if u > v:
                u, v = v, u
            if (u, v) not in edges:
                return
            b = books[(u, v)]
            book_counter[b] -= 1
            if book_counter[b] == 0:
                del book_counter[b]
            del books[(u, v)]
            edges.remove((u, v))
            in_edges[u].discard(v)
            in_edges[v].discard(u)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        
        # Build K20,20
        for u in range(20):
            for v in range(20, 40):
                add_edge(u, v)
        
        # Add 1 intra-partition edge
        u = random.randint(0, 19)
        v = random.randint(0, 19)
        while u == v:
            v = random.randint(0, 19)
        add_edge(u, v)
        
        return adj, in_edges, books, book_counter, edges
    
    best_result = None
    best_max_book = float('inf')
    best_sum_sq = float('inf')
    
    # Multiple restarts
    for attempt in range(10):
        if time.time() - start_time > time_limit - 5.0:
            break
        
        seed = 42 + attempt * 137
        adj, in_edges, books, book_counter, edges = build_graph(seed)
        
        # Track current best for this run
        cur_max = book_counter.most_common(1)[0][0]
        cur_sum_sq = sum(b * b for b in books.values())
        run_best_max = cur_max
        run_best_sum_sq = cur_sum_sq
        run_best_books = books.copy()
        run_best_edges = edges.copy()
        run_best_adj = adj[:]
        run_best_in_edges = [s.copy() for s in in_edges]
        
        # Simulated annealing parameters
        T = 2.0
        cooling = 0.99995
        min_T = 0.001
        
        # Pre-compute non-edges for faster selection
        non_edges = [p for p in all_pairs if p not in edges]
        
        # Main optimization loop
        while time.time() - start_time < time_limit - 2.0:
            # Get current max book
            if not book_counter:
                break
            cur_max = max(book_counter.keys())
            
            # Find edges with high book sizes
            high_book_edges = [e for e, b in books.items() if b >= cur_max]
            
            # Pick edge to remove: bias towards high book edges
            if high_book_edges and random.random() < 0.7:
                u_rem, v_rem = random.choice(high_book_edges)
            else:
                u_rem, v_rem = random.choice(list(edges))
            
            # Pick non-edge to add: bias towards low book potential
            # Estimate book size if we add edge (u, v)
            # The book size would be popcount(adj[u] & adj[v])
            # We want to add edges with low potential book size
            
            if non_edges:
                # Sample a few candidates and pick the one with lowest potential
                candidates = random.sample(non_edges, min(5, len(non_edges)))
                best_candidate = None
                best_potential = float('inf')
                for (u_c, v_c) in candidates:
                    potential = (adj[u_c] & adj[v_c]).bit_count()
                    if potential < best_potential:
                        best_potential = potential
                        best_candidate = (u_c, v_c)
                u_add, v_add = best_candidate
            else:
                break
            
            # Perform swap
            # Collect affected edges
            affected = set()
            for w in in_edges[u_rem]:
                a, b = min(u_rem, w), max(u_rem, w)
                affected.add((a, b))
            for w in in_edges[v_rem]:
                a, b = min(v_rem, w), max(v_rem, w)
                affected.add((a, b))
            for w in in_edges[u_add]:
                a, b = min(u_add, w), max(u_add, w)
                affected.add((a, b))
            for w in in_edges[v_add]:
                a, b = min(v_add, w), max(v_add, w)
                affected.add((a, b))
            
            # Save old book values
            old_books = {}
            for e in affected:
                if e in books:
                    old_books[e] = books[e]
            
            # Remove edge
            remove_edge(u_rem, v_rem)
            # Add edge
            add_edge(u_add, v_add)
            
            # Update books for affected edges
            for e in affected:
                u, v = e
                if e in edges:
                    new_b = (adj[u] & adj[v]).bit_count()
                    if e in books:
                        old_b = books[e]
                        if old_b != new_b:
                            book_counter[old_b] -= 1
                            if book_counter[old_b] == 0:
                                del book_counter[old_b]
                            books[e] = new_b
                            book_counter[new_b] += 1
                    else:
                        books[e] = new_b
                        book_counter[new_b] += 1
            
            # Compute new stats
            if book_counter:
                new_max = max(book_counter.keys())
                new_sum_sq = sum(b * b for b in books.values())
            else:
                new_max = 0
                new_sum_sq = 0
            
            # Acceptance criteria
            accepted = False
            if new_max < run_best_max:
                accepted = True
            elif new_max == run_best_max and new_sum_sq < run_best_sum_sq:
                accepted = True
            elif new_max <= run_best_max:
                delta_sq = new_sum_sq - run_best_sum_sq
                if delta_sq < 0:
                    accepted = True
                else:
                    if T > min_T:
                        prob = math.exp(-delta_sq / (T * 100))
                        if random.random() < prob:
                            accepted = True
            else:
                # new_max > run_best_max
                if T > 0.5 and new_max == run_best_max + 1:
                    if random.random() < T * 0.05:
                        accepted = True
            
            if accepted:
                run_best_max = new_max
                run_best_sum_sq = new_sum_sq
                run_best_books = books.copy()
                run_best_edges = edges.copy()
                run_best_adj = adj[:]
                run_best_in_edges = [s.copy() for s in in_edges]
            else:
                # Rollback
                remove_edge(u_add, v_add)
                add_edge(u_rem, v_rem)
                # Restore books
                for e, val in old_books.items():
                    if e in books:
                        cur_b = books[e]
                        if cur_b != val:
                            book_counter[cur_b] -= 1
                            if book_counter[cur_b] == 0:
                                del book_counter[cur_b]
                        books[e] = val
                        book_counter[val] += 1
                    else:
                        books[e] = val
                        book_counter[val] += 1
                # Fix the added edge's book
                if (u_add, v_add) in edges:
                    pass  # already handled
                elif (u_add, v_add) in books:
                    b = books[(u_add, v_add)]
                    book_counter[b] -= 1
                    if book_counter[b] == 0:
                        del book_counter[b]
                    del books[(u_add, v_add)]
            
            T *= cooling
            if T < min_T:
                T = min_T
        
        # Update global best
        if run_best_max < best_max_book or (run_best_max == best_max_book and run_best_sum_sq < best_sum_sq):
            best_max_book = run_best_max
            best_sum_sq = run_best_sum_sq
            best_result = run_best_edges
    
    # If we found a result, return it
    if best_result is not None:
        return [[u, v] for u, v in best_result]
    
    # Fallback: return K20,20 + 1 edge
    adj, in_edges, books, book_counter, edges = build_graph(42)
    return [[u, v] for u, v in edges]
