import time
import random
import sys

def construct(n):
    if n <= 1:
        return "1"
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # 1. Generate standard superpermutation for n-1
    word = [1]
    for m in range(2, n):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find max overlap
            max_overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if new[len(new) - o:] == block[:o]:
                    max_overlap = o
                    break
            new += block[max_overlap:]
        word = new
    
    # word is S_{n-1}
    # Extract the sequence of permutations of length n-1
    perms_seq = []
    seen = set()
    perm_len = n - 1
    for i in range(len(word) - perm_len + 1):
        window = tuple(word[i:i + perm_len])
        if len(set(window)) == perm_len and window not in seen:
            seen.add(window)
            perms_seq.append(window)
            
    current_order = perms_seq
    num_perms = len(current_order)
    N = num_perms
    
    # Precompute overlap matrix M[i][j] = max k such that p_i[-k:] == p_j[:k]
    M = [[0] * num_perms for _ in range(num_perms)]
    
    for i in range(num_perms):
        p_i = current_order[i]
        for j in range(num_perms):
            p_j = current_order[j]
            max_ov = 0
            for k in range(n - 1, -1, -1):
                if p_i[-k:] == p_j[:k]:
                    max_ov = k
                    break
            M[i][j] = max_ov
    
    def calc_score(order_indices):
        """Returns total overlap sum for a given ordering."""
        total = 0
        for k in range(len(order_indices) - 1):
            total += M[order_indices[k]][order_indices[k+1]]
        return total
    
    def local_search(order, start_time):
        """Perform adjacent swaps and 2-opt moves until no improvement or deadline."""
        best_indices = list(order)
        best_score = calc_score(best_indices)
        
        while time.time() < start_time + deadline:
            improved = False
            
            # Pass 1: Adjacent swaps (left to right, multiple passes until stable)
            while True:
                pass_improved = False
                for i in range(N - 1):
                    if time.time() >= start_time + deadline:
                        return best_indices, best_score
                    
                    idx_i = best_indices[i]
                    idx_j = best_indices[i + 1]
                    
                    if i > 0:
                        idx_l = best_indices[i - 1]
                        old_left = M[idx_l][idx_i]
                        new_left = M[idx_l][idx_j]
                    else:
                        old_left = 0
                        new_left = 0
                        
                    old_mid = M[idx_i][idx_j]
                    new_mid = M[idx_j][idx_i]
                    
                    if i + 2 < N:
                        idx_r = best_indices[i + 2]
                        old_right = M[idx_j][idx_r]
                        new_right = M[idx_i][idx_r]
                    else:
                        old_right = 0
                        new_right = 0
                        
                    delta = (new_left + new_mid + new_right) - (old_left + old_mid + old_right)
                    
                    if delta > 0:
                        best_indices[i], best_indices[i + 1] = best_indices[i + 1], best_indices[i]
                        best_score += delta
                        pass_improved = True
                        improved = True
                
                if not pass_improved:
                    break
            
            # Pass 2: 2-opt moves (reverse segments)
            while True:
                pass_improved = False
                for i in range(N - 2):
                    if time.time() >= start_time + deadline:
                        return best_indices, best_score
                    
                    # Try reversing best_indices[i..j] for various j
                    max_j = min(N - 1, i + 30)  # Limit segment length for speed
                    
                    for j in range(i + 1, max_j + 1):
                        if time.time() >= start_time + deadline:
                            return best_indices, best_score
                        
                        # Compute delta for reversing [i..j]
                        old_score = 0
                        if i > 0:
                            old_score += M[best_indices[i - 1]][best_indices[i]]
                        if j + 1 < N:
                            old_score += M[best_indices[j]][best_indices[j + 1]]
                        
                        for k in range(i, j):
                            old_score += M[best_indices[k]][best_indices[k + 1]]
                        
                        new_score = 0
                        if i > 0:
                            new_score += M[best_indices[i - 1]][best_indices[j]]
                        if i > 0:
                            new_score += M[best_indices[i]][best_indices[i - 1]]
                        
                        for k in range(i, j):
                            new_score += M[best_indices[j - (k - i)]][best_indices[j - (k - i) - 1]]
                        
                        if j + 1 < N:
                            new_score += M[best_indices[i]][best_indices[j + 1]]
                        
                        delta = new_score - old_score
                        
                        if delta > 0:
                            best_indices[i:j + 1] = best_indices[i:j + 1][::-1]
                            best_score += delta
                            pass_improved = True
                            improved = True
                            break
                
                if not pass_improved:
                    break
            
            if not improved:
                break
        
        return best_indices, best_score
    
    # Initial ordering: standard construction order
    init_indices = list(range(num_perms))
    
    # Run local search from initial order
    best_indices, best_score = local_search(init_indices, start_time)
    
    # If time remains, try a few random restarts
    rng = random.Random(42)
    while time.time() < start_time + deadline - 5:
        perturbed = list(best_indices)
        
        for _ in range(5):
            i = rng.randint(0, N - 2)
            j = rng.randint(i + 1, min(N - 1, i + 20))
            perturbed[i:j + 1] = perturbed[i:j + 1][::-1]
        
        candidate_indices, candidate_score = local_search(perturbed, start_time)
        
        if candidate_score > best_score:
            best_indices = candidate_indices
            best_score = candidate_score
    
    # Reconstruct the word
    final_order = [current_order[idx] for idx in best_indices]
    
    if not final_order:
        return ""
        
    first_block = list(final_order[0]) + [n] + list(final_order[0])
    result = first_block
    
    for p in final_order[1:]:
        block = list(p) + [n] + list(p)
        max_ov = 0
        for o in range(n - 1, -1, -1):
            if result[-o:] == block[:o]:
                max_ov = o
                break
        result += block[max_ov:]
        
    return "".join(map(str, result))
