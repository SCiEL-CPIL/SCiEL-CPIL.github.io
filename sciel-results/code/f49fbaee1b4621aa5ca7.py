import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 118.0
    n = 40
    target_edges = 401
    
    def build_books(A):
        """Compute full book size matrix from adjacency matrix."""
        B = A.astype(np.int32) @ A.astype(np.int32)
        # Mask non-edges to 0
        return np.where(A > 0, B, 0)
    
    def get_max_book(books):
        return int(books.max())
    
    def score_tuple(books):
        max_b = int(books.max())
        count_max = int(np.sum(books == max_b))
        sum_books = int(books.sum())
        return (max_b, count_max, sum_books)
    
    def update_books_on_add(A, books, u, v):
        """Incrementally update books when adding edge (u,v). Returns new books."""
        new_books = books.copy()
        new_books[u, :] = A[u, :].astype(np.int32) @ A.astype(np.int32)
        new_books[v, :] = A[v, :].astype(np.int32) @ A.astype(np.int32)
        new_books[:, u] = new_books[u, :]
        new_books[:, v] = new_books[v, :]
        new_books = np.where(A > 0, new_books, 0)
        return new_books
    
    def update_books_on_remove(A, books, u, v):
        """Incrementally update books when removing edge (u,v)."""
        new_books = books.copy()
        new_books[u, :] = A[u, :].astype(np.int32) @ A.astype(np.int32)
        new_books[v, :] = A[v, :].astype(np.int32) @ A.astype(np.int32)
        new_books[:, u] = new_books[u, :]
        new_books[:, v] = new_books[v, :]
        new_books = np.where(A > 0, new_books, 0)
        return new_books

    def generate_circulant(n, k):
        """Generates a k-regular circulant graph."""
        A = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(1, k+1):
                u = i
                v = (i + j) % n
                A[u, v] = 1
                A[v, u] = 1
        return A

    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    best_score = None
    best_A = None
    best_books = None
    
    for trial in range(30):
        if time.time() > deadline - 10:
            break
        
        # Initialize with circulant graph (20-regular)
        A = generate_circulant(n, 20)
        
        # Add 1 random edge to reach 401
        edge_count = np.sum(A) // 2
        needed = target_edges - edge_count
        
        # Find non-edges
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if A[i, j] == 0]
        random.shuffle(non_edges)
        
        for u, v in non_edges[:needed]:
            A[u, v] = 1
            A[v, u] = 1
            
        books = build_books(A)
        cur_score = score_tuple(books)
        cur_max = cur_score[0]
        
        if best_score is None or cur_score < best_score:
            best_score = cur_score
            best_A = A.copy()
            best_books = books.copy()
            
        if cur_max <= 7:
            break
            
        # Local Search with incremental updates
        stuck = 0
        max_iterations = 2000
        for iteration in range(max_iterations):
            if time.time() > deadline - 10:
                break
            
            # Find edges with maximum book size
            max_val = int(books.max())
            if max_val <= 7:
                break
            
            # Get all max-book edges
            max_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if A[i, j] == 1 and books[i, j] == max_val:
                        max_edges.append((i, j))
                        
            if not max_edges:
                break
                
            u, v = random.choice(max_edges)
            
            # Strategy: try to remove (u,v) and add a non-edge (x,y)
            # Also try 2-opt swaps: remove (u,v) and (x,y), add (u,x) and (v,y)
            
            # Get common neighbors of u and v
            common_neighbors = []
            for k in range(n):
                if k != u and k != v and A[u, k] == 1 and A[v, k] == 1:
                    common_neighbors.append(k)
            
            # Build candidate non-edges for simple swap
            candidates = []
            
            # Priority 1: edges between common neighbors
            for i_idx in range(len(common_neighbors)):
                for j_idx in range(i_idx+1, len(common_neighbors)):
                    x, y = common_neighbors[i_idx], common_neighbors[j_idx]
                    if A[x, y] == 0:
                        candidates.append((x, y))
            
            # Priority 2: random non-edges
            for _ in range(50):
                x = random.randint(0, n-1)
                y = random.randint(0, n-1)
                if x != y and A[x, y] == 0:
                    x, y = min(x, y), max(x, y)
                    if x != u or y != v:
                        candidates.append((x, y))
            
            random.shuffle(candidates)
            
            # Build 2-opt candidates
            # We want to swap (u,v) and (x,y) with (u,x) and (v,y)
            # Conditions:
            # 1. (x,y) is an edge
            # 2. x != u, x != v, y != u, y != v
            # 3. (u,x) is a non-edge
            # 4. (v,y) is a non-edge
            # 5. (u,y) and (v,x) are NOT the edges being removed (to avoid self loops or duplicate logic, though distinctness handles it)
            
            two_opt_candidates = []
            # Sample some edges to try 2-opt swaps
            current_edges = [(i, j) for i in range(n) for j in range(i+1, n) if A[i, j] == 1]
            random.shuffle(current_edges)
            
            for x, y in current_edges[:20]:
                if x == u or x == v or y == u or y == v:
                    continue
                # Check if (u,x) and (v,y) are non-edges
                if A[u, x] == 0 and A[v, y] == 0:
                    # Also ensure we don't create duplicate edges or self loops
                    # (u,x) and (v,y) must be distinct from each other and from (u,v) and (x,y)
                    # Since x,y distinct from u,v, (u,x) != (v,y) unless u=v or x=y (false) or u=y and v=x.
                    # If u=y and v=x, then we are swapping (u,v) and (x,y) with (u,y) and (v,x) which is (u,x) and (v,y).
                    # This is a valid 2-opt.
                    two_opt_candidates.append((x, y))
            
            random.shuffle(two_opt_candidates)
            
            improved = False
            
            # Try simple swaps first
            for x, y in candidates[:30]:
                if A[x, y] == 1:
                    continue
                    
                # Perform swap: remove (u,v), add (x,y)
                A[u, v] = 0
                A[v, u] = 0
                books = update_books_on_remove(A, books, u, v)
                
                A[x, y] = 1
                A[y, x] = 1
                books = update_books_on_add(A, books, x, y)
                
                new_score = score_tuple(books)
                
                if new_score < cur_score:
                    cur_score = new_score
                    cur_max = new_score[0]
                    improved = True
                    if best_score is None or cur_score < best_score:
                        best_score = cur_score
                        best_A = A.copy()
                        best_books = books.copy()
                    break
                else:
                    # Revert
                    A[x, y] = 0
                    A[y, x] = 0
                    books = update_books_on_remove(A, books, x, y)
                    
                    A[u, v] = 1
                    A[v, u] = 1
                    books = update_books_on_add(A, books, u, v)
            
            if improved:
                stuck = 0
                continue
            
            # Try 2-opt swaps
            for x, y in two_opt_candidates:
                # Perform 2-opt swap: remove (u,v) and (x,y), add (u,x) and (v,y)
                # Order of operations matters for incremental updates?
                # It's safer to do:
                # 1. Remove (u,v)
                # 2. Remove (x,y)
                # 3. Add (u,x)
                # 4. Add (v,y)
                
                A[u, v] = 0
                A[v, u] = 0
                books = update_books_on_remove(A, books, u, v)
                
                A[x, y] = 0
                A[y, x] = 0
                books = update_books_on_remove(A, books, x, y)
                
                A[u, x] = 1
                A[x, u] = 1
                books = update_books_on_add(A, books, u, x)
                
                A[v, y] = 1
                A[y, v] = 1
                books = update_books_on_add(A, books, v, y)
                
                new_score = score_tuple(books)
                
                if new_score < cur_score:
                    cur_score = new_score
                    cur_max = new_score[0]
                    improved = True
                    if best_score is None or cur_score < best_score:
                        best_score = cur_score
                        best_A = A.copy()
                        best_books = books.copy()
                    break
                else:
                    # Revert
                    A[v, y] = 0
                    A[y, v] = 0
                    books = update_books_on_remove(A, books, v, y)
                    
                    A[u, x] = 0
                    A[x, u] = 0
                    books = update_books_on_remove(A, books, u, x)
                    
                    A[x, y] = 1
                    A[y, x] = 1
                    books = update_books_on_add(A, books, x, y)
                    
                    A[u, v] = 1
                    A[v, u] = 1
                    books = update_books_on_add(A, books, u, v)
            
            if improved:
                stuck = 0
            else:
                stuck += 1
                if stuck > 30:
                    break
    
    # Extract edges from best_A
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_A[i, j] == 1:
                edges.append([i, j])
                
    if len(edges) != target_edges:
        existing = set(tuple(e) for e in edges)
        if len(edges) > target_edges:
            while len(edges) > target_edges:
                idx = random.randint(0, len(edges)-1)
                e = edges.pop(idx)
                existing.discard(tuple(e))
                existing.discard(tuple(reversed(e)))
        else:
            for u, v in all_pairs:
                if len(edges) == target_edges:
                    break
                if (u, v) not in existing and (v, u) not in existing:
                    edges.append([u, v])
                    existing.add((u, v))
                    existing.add((v, u))
                    
    return edges
