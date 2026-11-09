import numpy as np
import random
import time

def construct():
    n = 40
    m = 401
    t_start = time.time()
    deadline = t_start + 118.0  # safety margin before 120s kill

    # ---------- Fast book-size computation using bitmasks ----------
    # For each vertex, store its neighbors as a 40-bit integer
    # book(u,v) = popcount(neigh[u] & neigh[v])
    
    def popcount(x):
        return bin(x).count('1')
    
    # Precompute popcount for 16-bit chunks for speed
    pc16 = [0] * 65536
    for i in range(65536):
        pc16[i] = bin(i).count('1')
    
    def fast_popcount(x):
        # Split 40-bit into three 16-bit chunks (last is 8-bit)
        return pc16[x & 0xFFFF] + pc16[(x >> 16) & 0xFFFF] + pc16[(x >> 32) & 0xFFFF]
    
    def compute_books(neigh):
        """Compute book size for every edge. Returns dict (u,v)->count for u<v."""
        books = {}
        for u in range(n):
            for v in range(u + 1, n):
                if (neigh[u] >> v) & 1:
                    c = fast_popcount(neigh[u] & neigh[v])
                    books[(u, v)] = c
        return books
    
    def max_book(books):
        return max(books.values()) if books else 0
    
    # ---------- Build initial K20,20 + 1 intra edge ----------
    def initial_neigh():
        neigh = [0] * n
        edges = set()
        # K20,20: vertices 0..19 vs 20..39
        for u in range(20):
            for v in range(20, 40):
                neigh[u] |= (1 << v)
                neigh[v] |= (1 << u)
                edges.add((u, v))
        # Add 1 intra-partition edge to get 401
        neigh[0] |= (1 << 1)
        neigh[1] |= (1 << 0)
        edges.add((0, 1))
        return neigh, edges
    
    neigh, edges = initial_neigh()
    books = compute_books(neigh)
    cur_max = max_book(books)
    best_max = cur_max
    best_edges = set(edges)
    best_neigh = neigh[:]
    
    # Precompute all pairs and non-edges
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def get_non_edges(neigh):
        non_edges = []
        for (i, j) in all_pairs:
            if not ((neigh[i] >> j) & 1):
                non_edges.append((i, j))
        return non_edges
    
    non_edges = get_non_edges(neigh)
    
    # ---------- Phase 1: Local search with bitmask speedup ----------
    stagnation_counter = 0
    max_stagnation = 3000
    
    while time.time() < deadline - 15.0:
        if best_max <= 5:
            break
        
        # Pick a random edge with the highest book
        max_book_val = max_book(books)
        max_edges = [e for e, b in books.items() if b == max_book_val]
        out_e = random.choice(max_edges)
        
        # Pick a random non-edge
        if not non_edges:
            break
        in_e = random.choice(non_edges)
        
        u1, v1 = out_e
        u2, v2 = in_e
        
        # Apply swap
        neigh[u1] &= ~(1 << v1)
        neigh[v1] &= ~(1 << u1)
        neigh[u2] |= (1 << v2)
        neigh[v2] |= (1 << u2)
        
        # Recompute books
        new_books = compute_books(neigh)
        new_max = max_book(new_books)
        
        if new_max <= max_book_val:
            books = new_books
            # Update edge sets
            edges.discard(out_e)
            edges.add(in_e)
            non_edges.remove(in_e)
            non_edges.append(out_e)
            
            if new_max < best_max:
                best_max = new_max
                best_edges = set(edges)
                best_neigh = neigh[:]
                stagnation_counter = 0
            else:
                stagnation_counter += 1
        else:
            # Revert
            neigh[u2] &= ~(1 << v2)
            neigh[v2] &= ~(1 << u2)
            neigh[u1] |= (1 << v1)
            neigh[v1] |= (1 << u1)
            
            edges.discard(in_e)
            edges.add(out_e)
            non_edges.remove(out_e)
            non_edges.append(in_e)
            
            stagnation_counter += 1
        
        if stagnation_counter > max_stagnation:
            # Restart from best
            neigh = best_neigh[:]
            edges = set(best_edges)
            books = compute_books(neigh)
            non_edges = get_non_edges(neigh)
            stagnation_counter = 0
            if best_max <= 5:
                break
    
    # ---------- Phase 2: Simulated Annealing to escape local minima ----------
    if best_max > 5 and time.time() < deadline - 5.0:
        sa_temp = 2.0
        sa_cool = 0.95
        
        neigh = best_neigh[:]
        edges = set(best_edges)
        books = compute_books(neigh)
        non_edges = get_non_edges(neigh)
        cur_max = max_book(books)
        
        for step in range(50000):
            if time.time() > deadline - 5.0:
                break
            if cur_max <= 5:
                best_max = 5
                best_edges = set(edges)
                best_neigh = neigh[:]
                break
            
            max_book_val = max_book(books)
            max_edges = [e for e, b in books.items() if b == max_book_val]
            out_e = random.choice(max_edges)
            
            if not non_edges:
                break
            in_e = random.choice(non_edges)
            
            u1, v1 = out_e
            u2, v2 = in_e
            
            # Apply swap
            neigh[u1] &= ~(1 << v1)
            neigh[v1] &= ~(1 << u1)
            neigh[u2] |= (1 << v2)
            neigh[v2] |= (1 << u2)
            
            new_books = compute_books(neigh)
            new_max = max_book(new_books)
            
            # Acceptance: always accept improvement, sometimes accept worse
            if new_max < cur_max:
                accept = True
            elif new_max == cur_max:
                accept = random.random() < 0.5
            else:
                # Accept worse with probability based on temperature
                delta = new_max - cur_max
                accept = random.random() < np.exp(-delta / sa_temp)
            
            if accept:
                books = new_books
                cur_max = new_max
                edges.discard(out_e)
                edges.add(in_e)
                non_edges.remove(in_e)
                non_edges.append(out_e)
                
                if cur_max < best_max:
                    best_max = cur_max
                    best_edges = set(edges)
                    best_neigh = neigh[:]
            else:
                # Revert
                neigh[u2] &= ~(1 << v2)
                neigh[v2] &= ~(1 << u2)
                neigh[u1] |= (1 << v1)
                neigh[v1] |= (1 << u1)
                edges.discard(in_e)
                edges.add(out_e)
                non_edges.remove(out_e)
                non_edges.append(in_e)
            
            sa_temp *= sa_cool
    
    # ---------- Return best ----------
    result = [[u, v] for u, v in best_edges]
    return result
