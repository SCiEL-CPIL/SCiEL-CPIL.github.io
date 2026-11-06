import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n = 40
    target_edges = 401

    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]

    def init_graph():
        """Initialize with a near-regular bipartite structure plus strategic edges."""
        A = np.zeros((n, n), dtype=np.int8)
        # Split vertices into two groups of 20
        left = list(range(20))
        right = list(range(20, 40))
        # Start with complete bipartite K_{20,20} = 400 edges
        for u in left:
            for v in right:
                A[u, v] = 1
                A[v, u] = 1
        # Add 1 more edge within one side (this creates the minimum book)
        # Choose an edge that minimizes the maximum book
        # Adding edge (0,1) in left side: books for edges from 0 or 1 to right get +1
        # All right-side vertices are connected to both 0 and 1, so each edge (0,v) and (1,v) for v in right gets book size 19+1=20
        # Actually let's think: in K_{20,20}, every edge has book size 19 (all other 19 vertices on the opposite side)
        # If we add edge (0,1) within left, then for edge (0,v) where v in right: common neighbors are all 19 other right vertices + vertex 1 = 20
        # For edge (1,v): common neighbors are all 19 other right vertices + vertex 0 = 20
        # For edge (0,1): common neighbors are all 20 right vertices = 20
        # So max book becomes 20. That's the seed.
        # 
        # Better: don't use complete bipartite. Use a near-regular structure.
        # Let's try: start with a 20-regular graph on 40 vertices (if it exists)
        # A 20-regular graph on 40 vertices has 40*20/2 = 400 edges. We need 401.
        # 
        # Alternative: use a random regular-ish graph
        # Let's just do random sampling but with a bias toward bipartite structure
        edges = set()
        # Add bipartite edges randomly (most of them)
        bipartite_pairs = [(u, v) for u in left for v in right]
        random.shuffle(bipartite_pairs)
        # Add 390 bipartite edges
        for u, v in bipartite_pairs[:390]:
            edges.add((u, v))
        # Add 11 more edges (mix of within-side and cross-side)
        remaining_bipartite = [(u, v) for u, v in bipartite_pairs[390:] if (u, v) not in edges]
        within_left = [(i, j) for i in range(20) for j in range(i+1, 20)]
        within_right = [(i+20, j+20) for i in range(20) for j in range(i+1, 20)]
        random.shuffle(remaining_bipartite)
        random.shuffle(within_left)
        random.shuffle(within_right)
        
        extra_pool = remaining_bipartite + within_left[:5] + within_right[:5]
        random.shuffle(extra_pool)
        for u, v in extra_pool:
            if len(edges) >= target_edges:
                break
            edges.add((u, v))
        
        # Fill if needed
        if len(edges) < target_edges:
            for u, v in all_pairs:
                if len(edges) >= target_edges:
                    break
                if (u, v) not in edges and (v, u) not in edges:
                    edges.add((u, v))
        
        for u, v in edges:
            A[u, v] = 1
            A[v, u] = 1
        return A, list(edges)

    def calc_books_fast(A):
        """Calculate book sizes using matrix operations. Returns dict and max."""
        # Book size for edge (u,v) = number of common neighbors = sum(A[u,:] * A[v,:])
        # Use matrix multiplication: A @ A.T gives common neighbor counts for all pairs
        common = A.astype(np.int32) @ A.astype(np.int32).T
        books = {}
        max_b = 0
        for i in range(n):
            for j in range(i+1, n):
                if A[i, j] == 1:
                    b = int(common[i, j])
                    books[(i, j)] = b
                    if b > max_b:
                        max_b = b
                else:
                    books[(i, j)] = 0
        return books, max_b

    def get_max_edges(books, max_b):
        return [(u, v) for (u, v), b in books.items() if b == max_b and b > 0]

    def get_non_edges(A):
        return [(u, v) for u, v in all_pairs if A[u, v] == 0]

    def evaluate_swap(cur_A, cur_books, cur_max_b, u, v, x, y):
        """Evaluate removing edge (u,v) and adding edge (x,y). Returns (new_max_b, new_books) or None if invalid."""
        if cur_A[u, v] == 0 or cur_A[x, y] == 1:
            return None
        # Check if x,y are already adjacent to each other (would create duplicate)
        # Also check that removing (u,v) and adding (x,y) doesn't violate constraints
        
        new_A = cur_A.copy()
        new_A[u, v] = 0
        new_A[v, u] = 0
        new_A[x, y] = 1
        new_A[y, x] = 1
        
        # Only need to recompute books for edges incident to u, v, x, y
        affected = set()
        affected.add((min(u,v), max(u,v)))
        affected.add((min(x,y), max(x,y)))
        for w in range(n):
            if w != u and new_A[u, w] == 1:
                affected.add((min(u,w), max(u,w)))
            if w != v and new_A[v, w] == 1:
                affected.add((min(v,w), max(v,w)))
            if w != x and new_A[x, w] == 1:
                affected.add((min(x,w), max(x,w)))
            if w != y and new_A[y, w] == 1:
                affected.add((min(y,w), max(y,w)))
        
        new_books = cur_books.copy()
        new_max_b = 0
        for e in affected:
            a, b = e
            if new_A[a, b] == 1:
                cnt = int(np.sum(new_A[a, :] * new_A[b, :]))
                new_books[e] = cnt
            else:
                new_books[e] = 0
        
        for e, b in new_books.items():
            if b > new_max_b:
                new_max_b = b
        
        return new_max_b, new_books

    best_score = None
    best_A = None
    best_books = None
    best_max_b = None

    # Try multiple random restarts with different initialization strategies
    for trial in range(30):
        if time.time() > deadline - 25:
            break

        A, edges = init_graph()
        books, max_b = calc_books_fast(A)

        if best_score is None or max_b < best_score:
            best_score = max_b
            best_A = A.copy()
            best_books = books.copy()
            best_max_b = max_b

        if max_b <= 7:
            break

        cur_books = books
        cur_max_b = max_b
        cur_A = A

        stuck = 0
        iterations = 0
        while time.time() < deadline - 10:
            iterations += 1
            max_edges = get_max_edges(cur_books, cur_max_b)
            if not max_edges:
                break

            # Pick a random max edge to remove
            u, v = random.choice(max_edges)

            non_edges = get_non_edges(cur_A)
            if not non_edges:
                break

            # Try multiple random swaps, pick the best one
            best_swap = None
            best_swap_max = cur_max_b + 1

            # Sample up to 30 candidate edges to add
            sample_size = min(30, len(non_edges))
            candidates = random.sample(non_edges, sample_size)

            for x, y in candidates:
                result = evaluate_swap(cur_A, cur_books, cur_max_b, u, v, x, y)
                if result is None:
                    continue
                new_max_b, new_books = result
                if new_max_b < best_swap_max:
                    best_swap_max = new_max_b
                    best_swap = (x, y, new_max_b, new_books)

            if best_swap is not None:
                x, y, new_max_b, new_books = best_swap
                cur_A = cur_A.copy()
                cur_A[u, v] = 0
                cur_A[v, u] = 0
                cur_A[x, y] = 1
                cur_A[y, x] = 1
                cur_books = new_books
                cur_max_b = new_max_b
                stuck = 0

                if cur_max_b < best_score:
                    best_score = cur_max_b
                    best_A = cur_A.copy()
                    best_books = cur_books.copy()
                    best_max_b = cur_max_b

                if cur_max_b <= 7:
                    break
            else:
                stuck += 1
                if stuck > 50:
                    break

    # Extract edges from best_A
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_A[i, j] == 1:
                edges.append([i, j])

    if len(edges) != target_edges:
        if len(edges) > target_edges:
            edges = edges[:target_edges]
        else:
            existing = set(tuple(e) for e in edges)
            for u, v in all_pairs:
                if len(edges) == target_edges:
                    break
                if (u, v) not in existing and (v, u) not in existing:
                    edges.append([u, v])
                    existing.add((u, v))
                    existing.add((v, u))

    return edges
