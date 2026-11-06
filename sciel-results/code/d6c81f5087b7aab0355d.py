import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n = 40
    target_edges = 401
    
    def calc_scores(A):
        M = A @ A
        masked = np.where(A > 0, M, 0)
        max_b = int(masked.max())
        count_max = int(np.sum(masked == max_b)) // 2
        sum_books = int(np.sum(masked)) // 2
        return max_b, count_max, sum_books, M, masked

    def local_search(A, deadline):
        best_A = A.copy()
        best_score = calc_scores(A)[:3]
        
        stuck_counter = 0
        current_score = best_score
        
        # Precompute all possible edges for random kicks
        all_possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        while time.time() < deadline:
            cur_max, cur_cnt, cur_sum, M, masked = calc_scores(A)
            current_score = (cur_max, cur_cnt, cur_sum)
            
            # Update best
            if (cur_max < best_score[0]) or \
               (cur_max == best_score[0] and cur_cnt < best_score[1]) or \
               (cur_max == best_score[0] and cur_cnt == best_score[1] and cur_sum < best_score[2]):
                best_score = current_score
                best_A = A.copy()
                stuck_counter = 0
                
                # Theoretical min is 7. If we hit 7, we can stop early to save time
                # or continue to verify stability. Stopping is safer for time.
                if best_score[0] <= 7:
                    break
            else:
                stuck_counter += 1

            # If stuck, perform a random kick
            if stuck_counter > 20:
                # Find a random edge and a random non-edge to swap
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

            # Targeted improvement: Pick a max-book edge and try to replace it
            max_idxs = np.argwhere(masked == cur_max)
            max_edges = [(u, v) for u, v in max_idxs if u < v]
            
            if not max_edges:
                break
                
            u, v = random.choice(max_edges)
            
            best_swap_score = None
            best_swap = None
            
            # Try 50 random non-edges to swap with (u, v)
            for _ in range(50):
                x = random.randint(0, n-1)
                y = random.randint(0, n-1)
                if x == y or A[x, y] == 1:
                    continue
                if (x, y) == (u, v) or (x, y) == (v, u):
                    continue
                    
                A[u, v] = 0
                A[v, u] = 0
                A[x, y] = 1
                A[y, x] = 1
                
                new_max, new_cnt, new_sum, _, _ = calc_scores(A)
                new_score = (new_max, new_cnt, new_sum)
                
                if best_swap_score is None or \
                   (new_score[0] < best_swap_score[0]) or \
                   (new_score[0] == best_swap_score[0] and new_score[1] < best_swap_score[1]) or \
                   (new_score[0] == best_swap_score[0] and new_score[1] == best_swap_score[1] and new_score[2] < best_swap_score[2]):
                    best_swap_score = new_score
                    best_swap = (x, y)
                    
                A[u, v] = 1
                A[v, u] = 1
                A[x, y] = 0
                A[y, x] = 0
                
            if best_swap is not None:
                x, y = best_swap
                if (best_swap_score[0] < cur_max) or \
                   (best_swap_score[0] == cur_max and best_swap_score[1] < cur_cnt) or \
                   (best_swap_score[0] == cur_max and best_swap_score[1] == cur_cnt and best_swap_score[2] < cur_sum):
                    A[u, v] = 0
                    A[v, u] = 0
                    A[x, y] = 1
                    A[y, x] = 1
                    stuck_counter = 0

        return best_A, best_score

    def greedy_build():
        A = np.zeros((n, n), dtype=np.int8)
        remaining = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(remaining)
        
        for _ in range(target_edges):
            if time.time() > deadline - 20:
                break
                
            # Sample candidates
            sample_size = min(50, len(remaining))
            candidates = random.sample(remaining, sample_size)
            
            best_edge = None
            best_max = float('inf')
            
            for u, v in candidates:
                A[u, v] = 1
                A[v, u] = 1
                # Quick check: max book
                M = A @ A
                masked = np.where(A > 0, M, 0)
                cur_max = int(masked.max())
                A[u, v] = 0
                A[v, u] = 0
                
                if cur_max < best_max:
                    best_max = cur_max
                    best_edge = (u, v)
                    
            if best_edge:
                u, v = best_edge
                A[u, v] = 1
                A[v, u] = 1
                remaining.remove(best_edge)
                
        # Fill if time ran out
        while len([e for e in remaining if A[e[0], e[1]] == 0]) > 0 and time.time() < deadline - 5:
             # This loop is inefficient, just add random remaining
             if not remaining: break
             e = remaining.pop()
             if A[e[0], e[1]] == 0:
                 A[e[0], e[1]] = 1
                 A[e[1], e[0]] = 1
                 
        return A

    def random_build():
        A = np.zeros((n, n), dtype=np.int8)
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        for u, v in all_edges[:target_edges]:
            A[u, v] = 1
            A[v, u] = 1
        return A

    # Run Greedy
    A_greedy = greedy_build()
    best_A, best_score = local_search(A_greedy, deadline - 10)
    
    # Run Random if time permits
    if time.time() < deadline - 15:
        A_random = random_build()
        A_rand, score_rand = local_search(A_random, deadline)
        if score_rand < best_score:
            best_A = A_rand
            best_score = score_rand

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
            for u, v in [(i, j) for i in range(n) for j in range(i+1, n)]:
                if len(edges) == target_edges:
                    break
                if (u, v) not in existing and (v, u) not in existing:
                    edges.append([u, v])
                    existing.add((u, v))
                    existing.add((v, u))

    return edges
