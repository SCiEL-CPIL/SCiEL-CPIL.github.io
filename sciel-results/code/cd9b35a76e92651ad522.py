import numpy as np
import random
import time

def construct():
    n = 40
    target_edges = 401
    time_limit = 110.0
    start_time = time.time()
    
    best_score = float('inf')
    best_edges = None
    
    def get_edges_from_matrix(A):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if A[i][j]:
                    edges.append([i, j])
        return edges

    def compute_max_book(A):
        # A is n x n boolean/integer matrix
        # Number of triangles containing edge (i,j) is sum(A[i] & A[j])
        # We want the max over all edges.
        # A^2 gives number of paths of length 2.
        # For undirected graph, (A^2)[i][j] is the number of common neighbors.
        # This is exactly the book size for edge (i,j).
        A2 = np.dot(A, A)
        # Mask out diagonal and non-edges? 
        # We only care about edges.
        # Create a mask of edges
        mask = A > 0
        np.fill_diagonal(mask, False)
        
        # Get values at edge positions
        # We can just take the max of A2 where mask is True
        # But A2[i][j] is symmetric.
        vals = np.where(mask, A2, 0)
        return int(np.max(vals))

    def random_graph():
        # Generate random graph with exactly target_edges edges
        A = np.zeros((n, n), dtype=int)
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_pairs)
        for i in range(target_edges):
            u, v = all_pairs[i]
            A[u][v] = 1
            A[v][u] = 1
        return A

    def local_search(A, current_max):
        # Returns improved A and score
        no_improve_count = 0
        max_no_improve = 5000
        while time.time() - start_time < time_limit and no_improve_count < max_no_improve:
            # Find an edge with max book size
            A2 = np.dot(A, A)
            mask = A > 0
            np.fill_diagonal(mask, False)
            
            # Find indices of edges with max book
            max_val = int(np.max(np.where(mask, A2, 0)))
            if max_val == current_max:
                 # Check if we can improve
                 pass
                 
            # Get list of edges with max book size
            # We need to find (i,j) such that A[i][j]==1 and A2[i][j]==max_val
            # For efficiency, just pick one random edge with high book size
            # Let's get all edges with book size >= max_val - 1
            candidates = []
            for i in range(n):
                for j in range(i+1, n):
                    if A[i][j]:
                        if A2[i][j] >= max_val - 1:
                            candidates.append((i, j))
            
            if not candidates:
                break
                
            u, v = random.choice(candidates)
            
            # Try to remove (u,v) and add a random non-edge
            # Or remove a neighbor edge
            # Let's try random swap first
            
            # Find non-edges
            # To speed up, we can precompute non-edges or just sample
            # Since n=40, total pairs 780. 401 edges. ~379 non-edges.
            # Sampling is fine.
            
            improved = False
            for _ in range(20): # Try 20 random swaps
                # Pick random non-edge
                while True:
                    x = random.randint(0, n-1)
                    y = random.randint(0, n-1)
                    if x != y and A[x][y] == 0:
                        break
                
                # Apply swap
                A[u][v] = 0
                A[v][u] = 0
                A[x][y] = 1
                A[y][x] = 1
                
                new_max = compute_max_book(A)
                
                if new_max < current_max:
                    current_max = new_max
                    improved = True
                    no_improve_count = 0
                    break
                else:
                    # Revert
                    A[u][v] = 1
                    A[v][u] = 1
                    A[x][y] = 0
                    A[y][x] = 0
                    
            if not improved:
                no_improve_count += 1
                # If stuck, maybe try a different strategy or just break
                if no_improve_count > 100:
                    break
                    
        return A, current_max

    # Main loop: Random restarts
    while time.time() - start_time < time_limit:
        A = random_graph()
        curr_max = compute_max_book(A)
        
        if curr_max < best_score:
            best_score = curr_max
            best_edges = get_edges_from_matrix(A)
            
        # Local search
        A, curr_max = local_search(A, curr_max)
        
        if curr_max < best_score:
            best_score = curr_max
            best_edges = get_edges_from_matrix(A)
            
        # If we hit the theoretical lower bound (7), we can stop early?
        # Theorem says >= 40/6 = 6.66, so 7 is possible.
        if best_score <= 7:
            break

    return best_edges

# EVOLVE-BLOCK-START
def construct():
    n = 40
    target_edges = 401
    time_limit = 110.0
    start_time = time.time()
    
    best_score = float('inf')
    best_edges = None
    
    import numpy as np
    import random
    import time

    def get_edges_from_matrix(A):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if A[i][j]:
                    edges.append([i, j])
        return edges

    def compute_max_book(A):
        A2 = np.dot(A, A)
        mask = A > 0
        np.fill_diagonal(mask, False)
        vals = np.where(mask, A2, 0)
        return int(np.max(vals))

    def random_graph():
        A = np.zeros((n, n), dtype=int)
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_pairs)
        for i in range(target_edges):
            u, v = all_pairs[i]
            A[u][v] = 1
            A[v][u] = 1
        return A

    def local_search(A, current_max):
        no_improve_count = 0
        max_no_improve = 2000
        while time.time() - start_time < time_limit and no_improve_count < max_no_improve:
            A2 = np.dot(A, A)
            mask = A > 0
            np.fill_diagonal(mask, False)
            max_val = int(np.max(np.where(mask, A2, 0)))
            
            if max_val < current_max:
                current_max = max_val
                no_improve_count = 0
                continue

            # Find edges with book size == max_val
            candidates = []
            for i in range(n):
                for j in range(i+1, n):
                    if A[i][j] and A2[i][j] == max_val:
                        candidates.append((i, j))
            
            if not candidates:
                break
                
            u, v = random.choice(candidates)
            improved = False
            
            # Try swapping (u,v) with a random non-edge
            for _ in range(10):
                while True:
                    x = random.randint(0, n-1)
                    y = random.randint(0, n-1)
                    if x != y and A[x][y] == 0:
                        break
                
                A[u][v] = 0
                A[v][u] = 0
                A[x][y] = 1
                A[y][x] = 1
                
                new_max = compute_max_book(A)
                
                if new_max < current_max:
                    current_max = new_max
                    improved = True
                    no_improve_count = 0
                    break
                else:
                    A[u][v] = 1
                    A[v][u] = 1
                    A[x][y] = 0
                    A[y][x] = 0
            
            if not improved:
                no_improve_count += 1
                if no_improve_count > 100:
                    break
                    
        return A, current_max

    while time.time() - start_time < time_limit:
        A = random_graph()
        curr_max = compute_max_book(A)
        
        if curr_max < best_score:
            best_score = curr_max
            best_edges = get_edges_from_matrix(A)
            
        A, curr_max = local_search(A, curr_max)
        
        if curr_max < best_score:
            best_score = curr_max
            best_edges = get_edges_from_matrix(A)
            
        if best_score <= 7:
            break

    return best_edges
# EVOLVE-BLOCK-END
