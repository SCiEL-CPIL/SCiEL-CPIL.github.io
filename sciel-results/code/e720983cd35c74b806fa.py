import numpy as np
import time
import random

def construct():
    n = 40
    m = 401
    deadline = time.time() + 115.0

    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    num_edges_total = len(all_edges)

    def compute_books(A):
        A2 = A @ A
        np.fill_diagonal(A2, 0)
        return A2

    def get_metrics(A):
        A2 = compute_books(A)
        max_b = int(np.max(A2))
        # Count edges achieving max book (each edge appears twice in A2)
        count_max = int(np.sum(A2 == max_b)) // 2
        # Sum of all book sizes (each triangle contributes 3 to sum, each edge in triangle gets +1)
        # Actually, sum of A2 over edges = 3 * number of triangles
        sum_books = int(np.sum(A2)) // 2
        return max_b, count_max, sum_books

    def edges_from_matrix(A):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j] == 1:
                    edges.append([i, j])
        return edges

    # Initialize with a bipartite graph K_20,20 (400 edges) + 1 extra edge
    # This starts with 0 triangles, so books are small
    A = np.zeros((n, n), dtype=np.int8)
    left = list(range(20))
    right = list(range(20, 40))
    
    for u in left:
        for v in right:
            A[u, v] = 1
            A[v, u] = 1
            
    # Add 1 random edge within left partition
    u = random.choice(left)
    v = random.choice(left)
    while u == v:
        v = random.choice(left)
    A[u, v] = 1
    A[v, u] = 1

    # Verify we have exactly 401 edges
    edge_count = int(np.sum(A)) // 2
    if edge_count != m:
        # Fallback: random graph
        A = np.zeros((n, n), dtype=np.int8)
        random.shuffle(all_edges)
        for u, v in all_edges[:m]:
            A[u, v] = 1
            A[v, u] = 1

    current_metrics = get_metrics(A)
    best_metrics = current_metrics
    best_A = A.copy()

    # Multi-objective local search with simulated annealing
    # Objective: minimize (max_book, count_max, sum_books)
    T = 5.0
    T_min = 0.01
    cooling = 0.9998
    
    # Maintain list of non-edges for efficient sampling
    non_edges = [e for e in all_edges if A[e[0], e[1]] == 0]
    current_edges = [e for e in all_edges if A[e[0], e[1]] == 1]
    
    no_improve = 0
    max_no_improve = 500

    while time.time() < deadline - 3:
        # Pick a random edge to remove
        rem_idx = random.randint(0, len(current_edges) - 1)
        rem_u, rem_v = current_edges[rem_idx]
        
        # Pick a random non-edge to add
        if not non_edges:
            break
        add_idx = random.randint(0, len(non_edges) - 1)
        add_u, add_v = non_edges[add_idx]
        
        # Perform swap
        A[rem_u, rem_v] = 0
        A[rem_v, rem_u] = 0
        A[add_u, add_v] = 1
        A[add_v, add_u] = 1
        
        new_metrics = get_metrics(A)
        
        # Compare metrics: better if lower
        old = current_metrics
        new = new_metrics
        
        # Determine if new is better, worse, or equal
        if (new[0] < old[0]) or \
           (new[0] == old[0] and new[1] < old[1]) or \
           (new[0] == old[0] and new[1] == old[1] and new[2] < old[2]):
            # New is better
            is_better = True
            is_worse = False
        elif (new[0] > old[0]) or \
             (new[0] == old[0] and new[1] > old[1]) or \
             (new[0] == old[0] and new[1] == old[1] and new[2] > old[2]):
            # New is worse
            is_better = False
            is_worse = True
            # Calculate delta for annealing
            delta = 0
            if new[0] > old[0]:
                delta += (new[0] - old[0]) * 10.0
            if new[0] == old[0] and new[1] > old[1]:
                delta += (new[1] - old[1]) * 1.0
            if new[0] == old[0] and new[1] == old[1] and new[2] > old[2]:
                delta += (new[2] - old[2]) * 0.1
        else:
            # Equal
            is_better = False
            is_worse = False
            delta = 0

        if is_better:
            # Accept
            current_edges.remove((rem_u, rem_v))
            current_edges.append((add_u, add_v))
            non_edges.remove((add_u, add_v))
            non_edges.append((rem_u, rem_v))
            current_metrics = new_metrics
            no_improve = 0
            T = max(T_min, T * 0.95)  # Cool faster on improvement
            
            # Update best
            if (new_metrics[0] < best_metrics[0]) or \
               (new_metrics[0] == best_metrics[0] and new_metrics[1] < best_metrics[1]) or \
               (new_metrics[0] == best_metrics[0] and new_metrics[1] == best_metrics[1] and new_metrics[2] < best_metrics[2]):
                best_metrics = new_metrics
                best_A = A.copy()
                
                if best_metrics[0] <= 6:
                    break
        elif is_worse:
            # Accept with probability exp(-delta/T)
            if T > T_min and random.random() < np.exp(-delta / T):
                current_edges.remove((rem_u, rem_v))
                current_edges.append((add_u, add_v))
                non_edges.remove((add_u, add_v))
                non_edges.append((rem_u, rem_v))
                current_metrics = new_metrics
                no_improve += 1
                T = max(T_min, T * cooling)
            else:
                # Revert
                A[rem_u, rem_v] = 1
                A[rem_v, rem_u] = 1
                A[add_u, add_v] = 0
                A[add_v, add_u] = 0
                T = max(T_min, T * cooling)
                no_improve += 1
        else:
            # Equal, accept with some probability
            if random.random() < 0.5:
                current_edges.remove((rem_u, rem_v))
                current_edges.append((add_u, add_v))
                non_edges.remove((add_u, add_v))
                non_edges.append((rem_u, rem_v))
                current_metrics = new_metrics
                no_improve += 1
            else:
                # Revert
                A[rem_u, rem_v] = 1
                A[rem_v, rem_u] = 1
                A[add_u, add_v] = 0
                A[add_v, add_u] = 0
                T = max(T_min, T * cooling)
                no_improve += 1

        # Random restart if stuck
        if no_improve > max_no_improve:
            no_improve = 0
            T = 5.0
            # Do a few random swaps
            for _ in range(10):
                if not non_edges:
                    break
                rem_idx = random.randint(0, len(current_edges) - 1)
                rem_u, rem_v = current_edges[rem_idx]
                add_idx = random.randint(0, len(non_edges) - 1)
                add_u, add_v = non_edges[add_idx]
                
                A[rem_u, rem_v] = 0
                A[rem_v, rem_u] = 0
                A[add_u, add_v] = 1
                A[add_v, add_u] = 1
                
                current_edges.remove((rem_u, rem_v))
                current_edges.append((add_u, add_v))
                non_edges.remove((add_u, add_v))
                non_edges.append((rem_u, rem_v))
                
                current_metrics = get_metrics(A)
                if (current_metrics[0] < best_metrics[0]) or \
                   (current_metrics[0] == best_metrics[0] and current_metrics[1] < best_metrics[1]) or \
                   (current_metrics[0] == best_metrics[0] and current_metrics[1] == best_metrics[1] and current_metrics[2] < best_metrics[2]):
                    best_metrics = current_metrics
                    best_A = A.copy()

    return edges_from_matrix(best_A)
