import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    n = 40
    m = 401

    def build_bitmasks(edge_set):
        adj = [0] * n
        for (u, v) in edge_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def book(u, v, adj):
        return bin(adj[u] & adj[v]).count('1')

    def max_book(books):
        return max(books.values()) if books else 0

    def all_books(edge_set, adj):
        return {(u, v): book(u, v, adj) for (u, v) in edge_set}

    def grasp_init():
        """
        Greedy Randomized Adaptive Search Procedure (GRASP) for initialization.
        Adds edges one by one, choosing from a candidate list of edges with low
        immediate book size to minimize the max book early on.
        """
        edge_set = set()
        adj = [0] * n
        deg = [0] * n
        
        # Pre-compute all possible edges
        all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        
        for _ in range(m):
            # Generate candidate list: edges that haven't been added
            # We want to pick edges with low book size.
            # To speed up, we sample a subset of candidates if the pool is large
            # but since m is small (401) and total edges is 780, we can evaluate
            # a random subset or all if time permits.
            
            # Strategy: Pick a random subset of size K from available edges,
            # evaluate their book sizes, and pick the best (or best random).
            
            available = [e for e in all_edges if e not in edge_set]
            if not available:
                break
            
            # If available is small, evaluate all. Else sample.
            if len(available) < 100:
                candidates = available
            else:
                candidates = random.sample(available, 100)
            
            best_edge = None
            best_book = float('inf')
            
            # Evaluate book size for candidates
            # Note: book size depends on current adj.
            for u, v in candidates:
                b = book(u, v, adj)
                if b < best_book:
                    best_book = b
                    best_edge = (u, v)
                # Randomization: with some probability, pick a non-best edge
                # to diversify. Or just pick best to keep it greedy.
                # Let's use a "best of random" approach:
                # Actually, pure greedy might get stuck in local minima of construction.
                # Let's pick the one with minimum book, tie-break randomly.
                # To add randomness, we can pick from the top 5% or similar.
                # For simplicity, let's pick the absolute best book size.
                # If there are ties, pick random.
            
            if best_edge is None:
                best_edge = random.choice(available)
            
            u, v = best_edge
            edge_set.add(best_edge)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
            
        return edge_set

    def local_search(edge_set, adj, books, current_score, max_steps=5000, stuck_limit=80):
        """
        Local search via edge swaps.
        Tries to reduce the maximum book size.
        """
        stuck = 0
        # Track worst edges more efficiently?
        # For now, iterate all books to find max. 401 items is fast.
        
        for step in range(max_steps):
            if time.time() > deadline - 3:
                break

            # Find worst edge
            worst_edge = None
            worst_val = -1
            for e, val in books.items():
                if val > worst_val:
                    worst_val = val
                    worst_edge = e

            if worst_val <= 7:
                break

            u, v = worst_edge
            swapped = False

            # Generate candidate replacement edges
            # We want to remove (u,v) and add (x,y) such that
            # the new max book is lower.
            # Heuristic: (x,y) should have low book size in the graph
            # after removing (u,v).
            
            # Remove (u,v) temporarily to evaluate candidates
            edge_set.remove((u, v))
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            
            # Calculate book sizes for potential new edges
            # We can't check all 780 edges every step, so sample.
            # Sample 200 random non-edges.
            cands = []
            seen = set()
            # Pre-compute non-edges? No, just check on the fly.
            # To avoid checking too many, we sample vertices.
            
            for _ in range(300):
                ru = random.randint(0, n - 1)
                rv = random.randint(0, n - 1)
                if ru == rv:
                    continue
                if ru > rv:
                    ru, rv = rv, ru
                if (ru, rv) in edge_set:
                    continue
                if (ru, rv) in seen:
                    continue
                seen.add((ru, rv))
                # Book size of (ru, rv) in the current graph (with u-v removed)
                tc = book(ru, rv, adj)
                cands.append((tc, ru, rv))
            
            cands.sort(key=lambda x: x[0])
            
            # Try top candidates
            for tc, ru, rv in cands[:30]:
                # Apply swap: add (ru, rv)
                edge_set.add((ru, rv))
                adj[ru] |= (1 << rv)
                adj[rv] |= (1 << ru)
                
                # Update books for affected edges
                # Affected edges are those incident to u, v, ru, rv
                affected = {u, v, ru, rv}
                new_books = {}
                for e, val in books.items():
                    eu, ev = e
                    if eu in affected or ev in affected:
                        new_books[e] = book(eu, ev, adj)
                    else:
                        new_books[e] = val
                # Add book for the new edge
                new_books[(ru, rv)] = book(ru, rv, adj)
                
                new_max = max_book(new_books)
                
                if new_max < current_score:
                    current_score = new_max
                    books = new_books
                    swapped = True
                    stuck = 0
                    break
                else:
                    # Revert addition of (ru, rv)
                    edge_set.remove((ru, rv))
                    adj[ru] &= ~(1 << rv)
                    adj[rv] &= ~(1 << ru)
            
            # Re-add (u, v) if no swap was made
            if not swapped:
                edge_set.add((u, v))
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                stuck += 1
                if stuck > stuck_limit:
                    break
            else:
                # (u,v) is already removed, (ru,rv) added.
                # Books are updated.
                pass

        return books, current_score

    def two_opt_search(edge_set, adj, books, current_score, max_steps=2000):
        """
        Alternative local search: 2-opt style swap.
        Remove two edges, add two edges.
        This allows escaping local minima of 1-swap.
        """
        stuck = 0
        for step in range(max_steps):
            if time.time() > deadline - 3:
                break
            
            # Pick two worst edges
            # Sort books by value desc
            sorted_edges = sorted(books.items(), key=lambda x: x[1], reverse=True)
            e1 = sorted_edges[0][0]
            e2 = sorted_edges[1][0]
            
            u1, v1 = e1
            u2, v2 = e2
            
            # Remove both
            edge_set.remove(e1)
            edge_set.remove(e2)
            adj[u1] &= ~(1 << v1)
            adj[v1] &= ~(1 << u1)
            adj[u2] &= ~(1 << v2)
            adj[v2] &= ~(1 << u2)
            
            # Find two new edges to add
            # Sample pairs of non-edges
            # This is more expensive.
            # Let's just do simple random restarts instead for simplicity and speed.
            
            # Revert
            edge_set.add(e1)
            edge_set.add(e2)
            adj[u1] |= (1 << v1)
            adj[v1] |= (1 << u1)
            adj[u2] |= (1 << v2)
            adj[v2] |= (1 << u2)
            
            break # Skip 2-opt for now to save time, rely on restarts
            
        return books, current_score

    best_score = 999
    best_edge_set = None

    # Multiple restarts with GRASP init and local search
    num_restarts = 50 # Adjust based on time
    
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # GRASP initialization
        edge_set = grasp_init()
        adj = build_bitmasks(edge_set)
        books = all_books(edge_set, adj)
        current_score = max_book(books)
        
        if current_score < best_score:
            best_score = current_score
            best_edge_set = set(edge_set)
            
        if current_score <= 7:
            break
        
        # Local search
        books, current_score = local_search(edge_set, adj, books, current_score)
        
        if current_score < best_score:
            best_score = current_score
            best_edge_set = set(edge_set)
            
        if current_score <= 7:
            break

    if best_edge_set is None:
        # Fallback: random init
        edge_set = set()
        while len(edge_set) < m:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v:
                if u > v: u, v = v, u
                edge_set.add((u, v))
        best_edge_set = edge_set

    return [[u, v] for (u, v) in best_edge_set]
