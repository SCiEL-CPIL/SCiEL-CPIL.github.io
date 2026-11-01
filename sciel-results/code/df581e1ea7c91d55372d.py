"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    if n == 1:
        return "1"
    
    # Step 1: Build the classic superpermutation for n-1
    def build_superperm_recursive(n_val):
        if n_val == 1:
            return [1]
        word = [1]
        for m in range(2, n_val + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                max_overlap = min(len(new), len(block))
                overlap = 0
                for o in range(max_overlap, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return word

    base_word = build_superperm_recursive(n - 1)
    
    # Step 2: Extract all unique permutations of 1..n-1 from base_word
    perms, seen = [], set()
    for i in range(len(base_word) - (n - 2)):
        window = tuple(base_word[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
            
    m = len(perms)
    
    # Step 3: Create blocks for adding symbol n
    blocks = [list(p) + [n] + list(p) for p in perms]
    block_len = 2 * (n - 1) + 1
    
    # Step 4: Precompute overlap matrix
    overlap_mat = [[0] * m for _ in range(m)]
    for i in range(m):
        bi_str = "".join(map(str, blocks[i]))
        for j in range(m):
            bj_str = "".join(map(str, blocks[j]))
            max_ov = block_len - 1
            ov = 0
            for o in range(max_ov, -1, -1):
                if bi_str[-o:] == bj_str[:o]:
                    ov = o
                    break
            overlap_mat[i][j] = ov

    # Also compute overlap with the end of base_word
    base_str = "".join(map(str, base_word))
    first_ov = [0] * m
    for i in range(m):
        bi_str = "".join(map(str, blocks[i]))
        max_ov = min(len(base_str), block_len - 1)
        for o in range(max_ov, -1, -1):
            if base_str[-o:] == bi_str[:o]:
                first_ov[i] = o
                break

    # Step 5: Local Search to maximize total overlap
    order = list(range(m))
    
    def calc_total_overlap(ord):
        total = first_ov[ord[0]]
        for k in range(1, len(ord)):
            total += overlap_mat[ord[k-1]][ord[k]]
        return total

    current_overlap = calc_total_overlap(order)
    
    # 2-Opt Local Search
    improved = True
    while improved and (time.time() < deadline - 10):
        improved = False
        for i in range(m - 1):
            if time.time() > deadline - 10:
                break
            for j in range(i + 1, m):
                idx_a = i - 1
                idx_b = i
                idx_c = j
                idx_d = j + 1
                
                old_ov = 0
                new_ov = 0
                
                if idx_a >= 0:
                    old_ov += overlap_mat[order[idx_a]][order[idx_b]]
                    new_ov += overlap_mat[order[idx_a]][order[idx_c]]
                else:
                    old_ov += first_ov[order[idx_b]]
                    new_ov += first_ov[order[idx_c]]
                    
                if i < j:
                    for k in range(i, j):
                        old_ov += overlap_mat[order[k]][order[k+1]]
                    for k in range(i, j):
                        new_ov += overlap_mat[order[k+1]][order[k]]
                        
                if idx_d < m:
                    old_ov += overlap_mat[order[idx_c]][order[idx_d]]
                    new_ov += overlap_mat[order[idx_b]][order[idx_d]]
                    
                if new_ov > old_ov:
                    order[i:j+1] = reversed(order[i:j+1])
                    current_overlap = new_ov + (current_overlap - old_ov)
                    improved = True

    # Step 6: Construct the final word
    final_word = list(base_word)
    first_block_idx = order[0]
    first_block = blocks[first_block_idx]
    ov = first_ov[first_block_idx]
    final_word += first_block[ov:]
    
    for k in range(1, m):
        prev_idx = order[k-1]
        curr_idx = order[k]
        curr_block = blocks[curr_idx]
        ov = overlap_mat[prev_idx][curr_idx]
        final_word += curr_block[ov:]
        
    # Step 7: Fast validity check and greedy deletion
    all_perms = set(itertools.permutations(range(1, n+1)))
    
    def is_valid(w):
        if len(w) < n:
            return False
        present = set()
        for i in range(len(w) - n + 1):
            present.add(tuple(w[i:i+n]))
        return all_perms.issubset(present)
        
    if not is_valid(final_word):
        return "".join(map(str, final_word))
        
    # Greedy deletion from end to start
    changed = True
    while changed and (time.time() < deadline - 2):
        changed = False
        for i in range(len(final_word) - 1, -1, -1):
            if time.time() > deadline - 2:
                break
            temp_word = final_word[:i] + final_word[i+1:]
            if is_valid(temp_word):
                final_word = temp_word
                changed = True

    return "".join(map(str, final_word))

# EVOLVE-BLOCK-END
