"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 40
    target_edges = 401
    
    def get_max_book(A):
        """Compute max book size efficiently using matrix multiplication."""
        # Book size of edge (i,j) = number of common neighbors = (A @ A)[i,j]
        # But we need to be careful: A @ A gives number of paths of length 2
        # For undirected graph, common neighbors of i and j = sum_k A[i,k] * A[k,j]
        # This is (A @ A)[i,j] but we need to exclude the direct edge contribution
        # Actually, (A @ A)[i,j] counts all common neighbors including if i~j
        # For our purpose, we want common neighbors, which is exactly (A @ A)[i,j]
        # when A[i,j] = 1, since the diagonal doesn't contribute to off-diagonal
        A_int = A.astype(np.int32)
        common_neighbors = A_int @ A_int
        # Get max over all edges
        max_b = 0
        for i in range(n):
            for j in range(i+1, n):
                if A[i, j]:
                    b = common_neighbors[i, j]
                    if b > max_b:
                        max_b = b
        return max_b
    
    def compute_all_books(A):
        """Compute book sizes for all edges."""
        A_int = A.astype(np.int33) if hasattr(np, 'int33') else A.astype(np.int32)
        common_neighbors = A_int @ A_int
        books = {}
        for i in range(n):
            for j in range(i+1, n):
                if A[i, j]:
                    books[(i, j)] = common_neighbors[i, j]
        return books
    
    def spectral_score(A):
        """
        Compute spectral score: sum of squared Laplacian eigenvalues.
        This correlates with total number of triangles.
        L = D - A, where D is degree matrix.
        trace(L^2) = sum of squared eigenvalues = 2 * sum of squared degrees - 4 * number_of_edges
        But we want to minimize triangles, so we can use this as a proxy.
        """
        A_int = A.astype(np.int32)
        degrees = A_int.sum(axis=1)
        # L = D - A
        # trace(L^2) = trace((D-A)^2) = trace(D^2) - 2*trace(D*A) + trace(A^2)
        # trace(D^2) = sum(d_i^2)
        # trace(D*A) = sum_i d_i * A[i,i] = 0 (since no self-loops)
        # trace(A^2) = sum_{i,j} A[i,j]*A[j,i] = 2 * number_of_edges
        trace_D2 = np.sum(degrees ** 2)
        trace_A2 = 2 * target_edges
        return trace_D2 - 2 * 0 + trace_A2
    
    def spectral_score_direct(A):
        """Direct computation of sum of squared Laplacian eigenvalues."""
        A_int = A.astype(np.float64)
        degrees = A_int.sum(axis=1)
        D = np.diag(degrees)
        L = D - A_int
        # Sum of squared eigenvalues = trace(L^2)
        L2 = L @ L
        return np.trace(L2)
    
    # Initialize with a balanced bipartite-like structure
    # Split vertices into two groups of 20
    # Start with complete bipartite K_{20,20} which has 400 edges
    # Then add 1 more edge within one partition
    A = np.zeros((n, n), dtype=bool)
    
    # Partition: 0-19 and 20-39
    for i in range(20):
        for j in range(20, 40):
            A[i, j] = True
            A[j, i] = True
    
    # We have 400 edges, need 1 more
    # Add an edge within the first partition that minimizes book size
    # In K_{20,20}, any edge within a partition has book size 20 (all vertices in other partition)
    # So all are equivalent. Just add (0, 1)
    A[0, 1] = True
    A[1, 0] = True
    
    current_max = get_max_book(A)
    best_A = A.copy()
    best_max = current_max
    
    # Spectral optimization: use gradient-based approach
    # We'll do simulated annealing on edge swaps, but guided by spectral score
    # Key insight: minimizing max book is hard, but minimizing total triangles (via spectral)
    # often helps reduce max book too
    
    # Precompute list of all possible edges
    all_possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Simulated annealing
    T = 1.0
    T_min = 0.01
    cooling = 0.9995
    
    for iteration in range(50000):
        if time.time() > deadline:
            break
        
        # Current state
        A_int = A.astype(np.int32)
        current_books = {}
        common_neighbors = A_int @ A_int
        for i in range(n):
            for j in range(i+1, n):
                if A[i, j]:
                    current_books[(i, j)] = common_neighbors[i, j]
        
        current_max = max(current_books.values()) if current_books else 0
        current_spectral = spectral_score_direct(A)
        
        # Pick a random edge to remove and a random non-edge to add
        current_edges = [e for e in all_possible_edges if A[e[0], e[1]]]
        if not current_edges:
            break
        
        # Choose an edge to remove - bias towards high-book edges
        if random.random() < 0.7:
            # Pick from top 20% book sizes
            sorted_edges = sorted(current_edges, key=lambda e: current_books[e], reverse=True)
            top_count = max(1, len(sorted_edges) // 5)
            remove_edge = random.choice(sorted_edges[:top_count])
        else:
            remove_edge = random.choice(current_edges)
        
        u, v = remove_edge
        A[u, v] = False
        A[v, u] = False
        
        # Choose a non-edge to add
        non_edges = [e for e in all_possible_edges if not A[e[0], e[1]]]
        if not non_edges:
            A[u, v] = True
            A[v, u] = True
            continue
        
        # Bias towards low-book non-edges
        if random.random() < 0.7:
            # Compute book sizes for non-edges
            non_edge_books = []
            for a, b in non_edges:
                b_size = common_neighbors[a, b]
                non_edge_books.append((b_size, (a, b)))
            non_edge_books.sort()
            bottom_count = max(1, len(non_edge_books) // 5)
            add_edge = non_edge_books[random.randint(0, bottom_count-1)][1]
        else:
            add_edge = random.choice(non_edges)
        
        a, b = add_edge
        A[a, b] = True
        A[b, a] = True
        
        # Evaluate new state
        A_int_new = A.astype(np.int32)
        common_neighbors_new = A_int_new @ A_int_new
        new_books = {}
        for i in range(n):
            for j in range(i+1, n):
                if A[i, j]:
                    new_books[(i, j)] = common_neighbors_new[i, j]
        
        new_max = max(new_books.values()) if new_books else 0
        new_spectral = spectral_score_direct(A)
        
        # Acceptance criterion: primarily based on max book, secondarily on spectral
        if new_max < current_max:
            # Always accept if max book improves
            if new_max < best_max:
                best_max = new_max
                best_A = A.copy()
        elif new_max == current_max:
            # Accept if spectral improves (fewer total triangles)
            if new_spectral < current_spectral:
                pass  # Accept
            else:
                # Revert
                A[u, v] = True
                A[v, u] = True
                A[a, b] = False
                A[b, a] = False
        else:
            # new_max > current_max, accept with probability based on temperature
            delta = new_max - current_max
            prob = np.exp(-delta / T)
            if random.random() < prob:
                pass  # Accept worsening move
            else:
                # Revert
                A[u, v] = True
                A[v, u] = True
                A[a, b] = False
                A[b, a] = False
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Final local search polish
    A = best_A.copy()
    for polish_iter in range(1000):
        if time.time() > deadline:
            break
        
        A_int = A.astype(np.int32)
        common_neighbors = A_int @ A_int
        current_books = {}
        for i in range(n):
            for j in range(i+1, n):
                if A[i, j]:
                    current_books[(i, j)] = common_neighbors[i, j]
        
        current_max = max(current_books.values()) if current_books else 0
        
        # Find edges with max book
        max_edges = [e for e, b in current_books.items() if b == current_max]
        if not max_edges:
            break
        
        improved = False
        for u, v in random.sample(max_edges, min(len(max_edges), 10)):
            A[u, v] = False
            A[v, u] = False
            
            # Try to add a non-edge that doesn't create a book as large
            non_edges = [(a, b) for a in range(n) for b in range(a+1, n) if not A[a, b]]
            if not non_edges:
                A[u, v] = True
                A[v, u] = True
                continue
            
            # Sort non-edges by their current book size
            non_edge_books = []
            for a, b in non_edges:
                b_size = common_neighbors[a, b]
                non_edge_books.append((b_size, (a, b)))
            non_edge_books.sort()
            
            # Try the best few
            for _, (a, b) in non_edge_books[:20]:
                A[a, b] = True
                A[b, a] = True
                
                A_int_check = A.astype(np.int32)
                cn_check = A_int_check @ A_int_check
                new_max = 0
                for i in range(n):
                    for j in range(i+1, n):
                        if A[i, j]:
                            b_val = cn_check[i, j]
                            if b_val > new_max:
                                new_max = b_val
                
                if new_max < current_max:
                    improved = True
                    break
                else:
                    A[a, b] = False
                    A[b, a] = False
            
            if improved:
                break
            
            A[u, v] = True
            A[v, u] = True
        
        if not improved:
            break
    
    # Convert best_A to list of edges
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_A[i, j]:
                result.append([i, j])
    
    # Verify count
    if len(result) != target_edges:
        # Fallback: fix the count
        if len(result) > target_edges:
            # Remove excess edges
            random.shuffle(result)
            result = result[:target_edges]
        elif len(result) < target_edges:
            # Add missing edges
            existing = set(tuple(sorted(e)) for e in result)
            all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            missing = [e for e in all_edges if tuple(e) not in existing]
            random.shuffle(missing)
            for e in missing:
                if len(result) >= target_edges:
                    break
                result.append(list(e))
    
    return result
# EVOLVE-BLOCK-END
