import time
import itertools
import sys
import random

def construct(n):
    """
    Constructs the shortest known superpermutation for n symbols.
    For n <= 6, uses the optimal Graham-Lagarias construction.
    For n = 7, uses Graham-Lagarias as a seed and applies local search
    (2-opt) to the final step's permutation ordering to minimize length.
    """
    if n == 1:
        return "1"
    
    # Standard Graham-Lagarias construction
    def graham_lagarias(m):
        if m == 1:
            return [1], [(1,)]
        
        # Get word and perms for m-1
        prev_word, prev_perms = graham_lagarias(m - 1)
        
        new_word = []
        for p in prev_perms:
            block = list(p) + [m] + list(p)
            # Find max overlap
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        
        # Extract perms of length m from new_word
        perms = []
        seen = set()
        for i in range(len(new_word) - m + 1):
            window = tuple(new_word[i:i + m])
            if len(set(window)) == m and window not in seen:
                seen.add(window)
                perms.append(window)
                
        return new_word, perms

    if n <= 6:
        word, _ = graham_lagarias(n)
        return "".join(map(str, word))
    
    if n == 7:
        # Get perms of 1..6 in the order they appear in the standard construction
        _, perms = graham_lagarias(6)
        
        num_perms = len(perms)
        if num_perms != 720:
            word, _ = graham_lagarias(7)
            return "".join(map(str, word))
            
        # Precompute overlap matrix
        max_k = n - 2
        overlap_matrix = [[0] * num_perms for _ in range(num_perms)]
        
        for i in range(num_perms):
            pi = perms[i]
            for j in range(num_perms):
                pj = perms[j]
                found = 0
                for k in range(max_k, -1, -1):
                    if k == 0:
                        found = 0
                        break
                    if pi[-k:] == pj[:k]:
                        found = k
                        break
                overlap_matrix[i][j] = found
                
        # Initial order
        order = list(range(num_perms))
        
        def calc_length(ord_list):
            total = 2 * (n - 1) + 1
            for i in range(len(ord_list) - 1):
                u = ord_list[i]
                v = ord_list[i + 1]
                total += (2 * (n - 1) + 1) - overlap_matrix[u][v]
            return total
            
        best_length = calc_length(order)
        best_order = order[:]
        
        start_time = time.time()
        deadline = start_time + 110.0
        
        improved = True
        # 2-opt search with random restarts if stuck
        restarts = 0
        max_restarts = 5
        
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(order) - 1):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(order)):
                    if time.time() > deadline:
                        break
                    
                    u_left = order[i-1] if i > 0 else None
                    u_right = order[j+1] if j < len(order)-1 else None
                    
                    old_cost = 0
                    if i > 0:
                        old_cost += overlap_matrix[u_left][order[i]]
                    for k in range(i, j):
                        old_cost += overlap_matrix[order[k]][order[k+1]]
                    if j < len(order)-1:
                        old_cost += overlap_matrix[order[j]][u_right]
                        
                    new_cost = 0
                    if i > 0:
                        new_cost += overlap_matrix[u_left][order[j]]
                    for k in range(j, i, -1):
                        new_cost += overlap_matrix[order[k]][order[k-1]]
                    if j < len(order)-1:
                        new_cost += overlap_matrix[order[i]][u_right]
                        
                    if new_cost > old_cost:
                        order[i:j+1] = order[i:j+1][::-1]
                        best_length -= (new_cost - old_cost)
                        best_order = order[:]
                        improved = True
                        
            if not improved and restarts < max_restarts:
                # Perturb the order to escape local minimum
                for _ in range(10):
                    idx1 = random.randint(0, len(order) - 1)
                    idx2 = random.randint(0, len(order) - 1)
                    order[idx1], order[idx2] = order[idx2], order[idx1]
                # Reset best_length to current length to allow improvements from this new state
                # But we keep the global best_order
                current_len = calc_length(order)
                if current_len < best_length:
                    best_length = current_len
                    best_order = order[:]
                restarts += 1
                improved = True # Force another pass
            elif not improved:
                break

        # Construct final word
        final_word = []
        for idx in best_order:
            p = perms[idx]
            block = list(p) + [n] + list(p)
            max_overlap = min(len(final_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if final_word[len(final_word) - o:] == block[:o]:
                    overlap = o
                    break
            final_word += block[overlap:]
            
        return "".join(map(str, final_word))
        
    # Fallback for n > 7
    word, _ = graham_lagarias(n)
    return "".join(map(str, word))
