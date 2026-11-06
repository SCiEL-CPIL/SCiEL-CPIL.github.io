"""Minimize the largest book in a 40-vertex, 401-edge graph."""
import time
import numpy as np

def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    n = 40
    m = 401
    
    def compute_books(A):
        """Compute book size for each edge using matrix multiplication."""
        # Triangle count per edge: for edge (i,j), book = sum_k A[i,k]*A[j,k]
        # This is (A @ A.T)[i,j] - 1 (subtract self if needed, but A[i,i]=0)
        # Actually: number of triangles containing edge (i,j) = (A @ A.T)[i,j]
        # since A[i,k]*A[j,k] = 1 iff both edges (i,k) and (j,k) exist
        AA = A @ A.T
        # For each edge (i,j), book size = AA[i,j] (number of common neighbors)
        # We need the max over all edges
        books = np.zeros((n, n), dtype=np.int32)
        np.fill_diagonal(books, 0)
        # Only consider actual edges
        books = AA * A  # zero out non-edges
        return books
    
    def max_book(A):
        books = compute_books(A)
        return int(books.max())
    
    def random_graph(n, m):
        """Generate a random graph with exactly m edges."""
        A = np.zeros((n, n), dtype=np.int32)
        edges = []
        while len(edges) < m:
            i, j = np.random.randint(0, n, 2)
            if i != j and A[i, j] == 0:
                A[i, j] = 1
                A[j, i] = 1
                edges.append((i, j))
        return A, edges
    
    def greedy_local_search(A, edges, max_iter=5000):
        """Try edge swaps to reduce max book."""
        current_best = max_book(A)
        best_A = A.copy()
        best_edges = list(edges)
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            # Pick a random edge to remove
            idx = np.random.randint(len(edges))
            i, j = edges[idx]
            
            # Try adding a random non-edge
            attempts = 0
            while attempts < 20:
                a, b = np.random.randint(0, n, 2)
                if a != b and A[a, b] == 0:
                    break
                attempts += 1
            else:
                continue
            
            # Apply swap
            A[i, j] = 0
            A[j, i] = 0
            A[a, b] = 1
            A[b, a] = 1
            edges[idx] = (a, b)
            
            new_best = max_book(A)
            
            if new_best < current_best:
                current_best = new_best
                best_A = A.copy()
                best_edges = list(edges)
            else:
                # Revert
                A[a, b] = 0
                A[b, a] = 0
                A[i, j] = 1
                A[j, i] = 1
                edges[idx] = (i, j)
        
        return best_A, best_edges, current_best
    
    def targeted_swap(A, edges):
        """More targeted: find the edge with max book, try to reduce it."""
        books = compute_books(A)
        max_val = int(books.max())
        
        # Find edges with book size close to max
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            # Pick a high-book edge
            flat_idx = np.argmax(books)
            i, j = np.unravel_index(flat_idx, (n, n))
            if A[i, j] == 0:
                continue
            
            # Remove this edge
            A[i, j] = 0
            A[j, i] = 0
            edge_idx = None
            for k, (u, v) in enumerate(edges):
                if (u == i and v == j) or (u == j and v == i):
                    edge_idx = k
                    break
            
            # Try adding a non-edge that doesn't create large books
            best_new = max_val + 1
            best_add = None
            for _ in range(50):
                a, b = np.random.randint(0, n, 2)
                if a != b and A[a, b] == 0:
                    A[a, b] = 1
                    A[b, a] = 1
                    nb = max_book(A)
                    A[a, b] = 0
                    A[b, a] = 0
                    if nb < best_new:
                        best_new = nb
                        best_add = (a, b)
            
            if best_add is not None:
                a, b = best_add
                A[a, b] = 1
                A[b, a] = 1
                if edge_idx is not None:
                    edges[edge_idx] = best_add
                # Check if improved
                new_max = max_book(A)
                if new_max < max_val:
                    max_val = new_max
            else:
                # Revert removal
                A[i, j] = 1
                A[j, i] = 1
        
        return A, edges, max_val
    
    best_overall = float('inf')
    best_result_A = None
    
    np.random.seed(42)
    
    # Phase 1: Multiple random starts with local search
    n_restarts = 15
    for r in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        A, edges = random_graph(n, m)
        A, edges, score = greedy_local_search(A, edges, max_iter=3000)
        
        if score < best_overall:
            best_overall = score
            best_result_A = A.copy()
            best_result_edges = list(edges)
    
    # Phase 2: More targeted refinement on the best solution
    if time.time() < deadline - 15 and best_result_A is not None:
        A = best_result_A.copy()
        edges = list(best_result_edges)
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            A, edges, score = targeted_swap(A, edges)
            if score < best_overall:
                best_overall = score
                best_result_A = A.copy()
                best_result_edges = list(edges)
    
    # Final: one more round of greedy on best
    if time.time() < deadline - 5 and best_result_A is not None:
        A = best_result_A.copy()
        edges = list(best_result_edges)
        A, edges, score = greedy_local_search(A, edges, max_iter=2000)
        if score < best_overall:
            best_overall = score
            best_result_A = A.copy()
            best_result_edges = list(edges)
    
    # Convert back to edge list
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_result_A[i, j]:
                result.append([i, j])
    
    # Verify exactly 401 edges
    assert len(result) == 401, f"Expected 401 edges, got {len(result)}"
    # Verify all vertices 0..39 and distinct
    seen = set()
    for e in result:
        u, v = e
        assert 0 <= u <= 39 and 0 <= v <= 39
        key = (min(u,v), max(u,v))
        assert key not in seen
        seen.add(key)
    
    return result
