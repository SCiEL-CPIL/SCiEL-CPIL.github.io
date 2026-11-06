import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    n = 40
    target_edges = 401
    
    def calc_books(A):
        # A is n x n binary adjacency matrix
        # Book size for edge (u,v) = number of common neighbors = (A @ A)[u,v]
        M = A @ A
        masked = np.where(A > 0, M, 0)
        max_b = int(masked.max())
        count_max = int(np.sum(masked == max_b)) // 2
        sum_books = int(np.sum(masked)) // 2
        return max_b, count_max, sum_books, masked
    
    def spectral_init():
        """
        Initialize with a regularized complete bipartite structure.
        For n=40, split into two sets of 20. Complete bipartite K_{20,20} has 400 edges.
        We need 401 edges, so add 1 more edge within one partition.
        This structure minimizes triangles (books) since K_{20,20} is triangle-free.
        Adding 1 edge creates at most 19 triangles for that edge.
        But we can do better by not using pure bipartite.
        """
        A = np.zeros((n, n), dtype=np.int8)
        
        # Split into two halves
        half = n // 2
        left = list(range(half))
        right = list(range(half, n))
        
        # Complete bipartite: all edges between left and right
        for u in left:
            for v in right:
                A[u, v] = 1
                A[v, u] = 1
        
        # Currently have 400 edges. Need 1 more.
        # Add an edge within left partition. To minimize triangles, pick an edge
        # where both endpoints have few neighbors in right... but all have 20.
        # Any edge within left creates 20 triangles (common neighbors in right).
        # Actually, edge (u,v) in left: common neighbors are vertices w such that
        # A[u,w]=1 and A[v,w]=1. In bipartite, w must be in right, and both u,w and v,w
        # are edges, so all 20 vertices in right are common neighbors.
        # So book size = 20. That's bad.
        
        # Better approach: instead of pure bipartite + 1 edge, let's use a different structure.
        # Start with K_{20,20} but remove some edges and add others to balance.
        
        # Alternative: use a quasi-regular graph approach.
        # Let's just do random initialization but biased toward bipartite structure.
        
        # Actually, let's try a different init: random graph with 401 edges
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        init_edges = all_edges[:target_edges]
        
        A = np.zeros((n, n), dtype=np.int8)
        for u, v in init_edges:
            A[u, v] = 1
            A[v, u] = 1
            
        return A
    
    def bipartite_plus_init():
        """
        Start with K_{20,20} (400 edges) + 1 intra-partition edge.
        Then we'll optimize from there.
        """
        A = np.zeros((n, n), dtype=np.int8)
        half = n // 2
        left = list(range(half))
        right = list(range(half, n))
        
        for u in left:
            for v in right:
                A[u, v] = 1
                A[v, u] = 1
        
        # Add one edge within left
        u, v = random.sample(left, 2)
        A[u, v] = 1
        A[v, u] = 1
        
        return A
    
    def better_init():
        """
        Try to find a good initial configuration using a greedy approach.
        Start empty, add edges one by one choosing the edge that minimizes
        the maximum book size increase.
        """
        A = np.zeros((n, n), dtype=np.int8)
        edge_count = 0
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        # We'll use a priority approach: for each candidate edge, compute
        # the book size it would create, and pick the one with smallest book.
        
        while edge_count < target_edges:
            best_edge = None
            best_book = float('inf')
            
            # Sample a subset of candidate edges for speed
            candidates = random.sample(all_edges, min(len(all_edges), 200))
            
            for u, v in candidates:
                if A[u, v] == 1:
                    continue
                # Book size for (u,v) = number of common neighbors
                common = np.sum(A[u, :] & A[v, :])
                if common < best_book:
                    best_book = common
                    best_edge = (u, v)
            
            if best_edge is None:
                # All sampled edges exist, try all
                for u, v in all_edges:
                    if A[u, v] == 0:
                        common = np.sum(A[u, :] & A[v, :])
                        if common < best_book:
                            best_book = common
                            best_edge = (u, v)
            
            if best_edge is None:
                break
                
            u, v = best_edge
            A[u, v] = 1
            A[v, u] = 1
            edge_count += 1
            
            if time.time() > deadline - 5:
                break
        
        # If we didn't reach target, fill randomly
        if edge_count < target_edges:
            existing = set()
            for i in range(n):
                for j in range(i+1, n):
                    if A[i, j] == 1:
                        existing.add((i, j))
            
            for u, v in random.sample(all_edges, len(all_edges)):
                if edge_count >= target_edges:
                    break
                if (u, v) not in existing:
                    A[u, v] = 1
                    A[v, u] = 1
                    existing.add((u, v))
                    edge_count += 1
        
        return A
    
    # Try multiple initialization strategies
    best_score = None
    best_A = None
    
    # Strategy 1: Random
    A = spectral_init()
    cur_max, cur_cnt, cur_sum, masked = calc_books(A)
    cur_score = (cur_max, cur_cnt, cur_sum)
    if best_score is None or cur_score < best_score:
        best_score = cur_score
        best_A = A.copy()
    
    # Strategy 2: Bipartite + 1
    if time.time() < deadline - 10:
        A = bipartite_plus_init()
        cur_max, cur_cnt, cur_sum, masked = calc_books(A)
        cur_score = (cur_max, cur_cnt, cur_sum)
        if best_score is None or cur_score < best_score:
            best_score = cur_score
            best_A = A.copy()
    
    # Strategy 3: Greedy construction
    if time.time() < deadline - 10:
        A = better_init()
        cur_max, cur_cnt, cur_sum, masked = calc_books(A)
        cur_score = (cur_max, cur_cnt, cur_sum)
        if best_score is None or cur_score < best_score:
            best_score = cur_score
            best_A = A.copy()
    
    # Now do local search from the best initialization
    A = best_A.copy()
    cur_max, cur_cnt, cur_sum, masked = calc_books(A)
    current_score = (cur_max, cur_cnt, cur_sum)
    
    stuck_counter = 0
    no_improve_counter = 0
    
    while time.time() < deadline:
        # Evaluate current
        cur_max, cur_cnt, cur_sum, masked = calc_books(A)
        current_score = (cur_max, cur_cnt, cur_sum)
        
        # Update best
        if (cur_max < best_score[0]) or \
           (cur_max == best_score[0] and cur_cnt < best_score[1]) or \
           (cur_max == best_score[0] and cur_cnt == best_score[1] and cur_sum < best_score[2]):
            best_score = current_score
            best_A = A.copy()
            stuck_counter = 0
            no_improve_counter = 0
            
            if best_score[0] <= 7:
                break
        else:
            stuck_counter += 1
            no_improve_counter += 1
        
        # If stuck for a long time, perform a random kick (accept worse move)
        if stuck_counter > 80:
            edges_list = [(i, j) for i in range(n) for j in range(i+1, n) if A[i, j] == 1]
            non_edges_list = [(i, j) for i in range(n) for j in range(i+1, n) if A[i, j] == 0]
            if edges_list and non_edges_list:
                u, v = random.choice(edges_list)
                x, y = random.choice(non_edges_list)
                A[u, v] = 0
                A[v, u] = 0
                A[x, y] = 1
                A[y, x] = 1
                stuck_counter = 0
                continue
        
        # Generate candidates: focus on edges with high book size
        max_idxs = np.argwhere(masked == cur_max)
        max_edges = [(u, v) for u, v in max_idxs if u < v]
        
        if not max_edges:
            # No edges at max book size? Shouldn't happen if cur_max > 0
            break
        
        # Pick a random edge from those with max book size
        u, v = random.choice(max_edges)
        
        best_swap_score = None
        best_swap = None
        
        # Try swapping: remove edge (u,v), add edge (x,y)
        # Sample non-edges, prioritizing those that might reduce book sizes
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if A[i, j] == 0]
        random.shuffle(non_edges)
        
        # Try up to 50 swaps
        for x, y in non_edges[:50]:
            if (x, y) == (u, v) or (x, y) == (v, u):
                continue
            
            # Temporarily swap
            A[u, v] = 0
            A[v, u] = 0
            A[x, y] = 1
            A[y, x] = 1
            
            new_max, new_cnt, new_sum, _ = calc_books(A)
            new_score = (new_max, new_cnt, new_sum)
            
            if best_swap_score is None or \
               (new_score[0] < best_swap_score[0]) or \
               (new_score[0] == best_swap_score[0] and new_score[1] < best_swap_score[1]) or \
               (new_score[0] == best_swap_score[0] and new_score[1] == best_swap_score[1] and new_score[2] < best_swap_score[2]):
                best_swap_score = new_score
                best_swap = (x, y)
            
            # Revert
            A[u, v] = 1
            A[v, u] = 1
            A[x, y] = 0
            A[y, x] = 0
            
            if time.time() > deadline - 5:
                break
        
        if best_swap is not None:
            x, y = best_swap
            if (best_swap_score[0] < cur_max) or \
               (best_swap_score[0] == cur_max and best_swap_score[1] < cur_cnt) or \
               (best_swap_score[0] == cur_max and best_swap_score[1] == cur_cnt and best_swap_score[2] < cur_sum):
                A[u, v] = 0
                A[v, u] = 0
                A[x, y] = 1
                A[y, x] = 1
            else:
                stuck_counter += 1
        
        # If no improvement for many iterations, try a restart with perturbation
        if no_improve_counter > 300:
            # Perturb current best
            A = best_A.copy()
            for _ in range(5):
                edges_list = [(i, j) for i in range(n) for j in range(i+1, n) if A[i, j] == 1]
                non_edges_list = [(i, j) for i in range(n) for j in range(i+1, n) if A[i, j] == 0]
                if edges_list and non_edges_list:
                    u, v = random.choice(edges_list)
                    x, y = random.choice(non_edges_list)
                    A[u, v] = 0
                    A[v, u] = 0
                    A[x, y] = 1
                    A[y, x] = 1
            no_improve_counter = 0
            stuck_counter = 0
    
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
            all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(all_edges)
            for u, v in all_edges:
                if len(edges) == target_edges:
                    break
                if (u, v) not in existing and (v, u) not in existing:
                    edges.append([u, v])
                    existing.add((u, v))
                    existing.add((v, u))

    return edges
