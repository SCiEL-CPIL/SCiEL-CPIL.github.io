import time
import itertools
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s wall-clock limit

    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"

    # Build S_{n-1} recursively (classic construction)
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
                # Find max overlap
                max_ov = min(len(new), len(block))
                ov = 0
                for k in range(max_ov, -1, -1):
                    if new[len(new) - k:] == block[:k]:
                        ov = k
                        break
                new += block[ov:]
            word = new
        return word

    base_word = build_superperm_recursive(n - 1)
    base_len = len(base_word)
    
    # Extract permutations of 1..n-1 from base_word
    perms, seen = [], set()
    for i in range(base_len - (n - 2)):
        window = tuple(base_word[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
            
    m = len(perms)  # (n-1)! permutations
    
    if n == 2:
        return "121"
        
    # Precompute overlap matrix
    # O[i][j] = max k such that perms[i][-k:] == perms[j][:k]
    O = [[0] * m for _ in range(m)]
    for i in range(m):
        pi = perms[i]
        for j in range(m):
            if i == j:
                O[i][j] = n - 1
                continue
            pj = perms[j]
            max_ov = n - 2
            ov = 0
            for k in range(max_ov, -1, -1):
                if pi[-k:] == pj[:k]:
                    ov = k
                    break
            O[i][j] = ov
            
    # Start overlap: base_word ends with some suffix
    base_suffix = tuple(base_word[-(n-1):])
    O_start = [0] * m
    for j in range(m):
        pj = perms[j]
        max_ov = n - 1
        ov = 0
        for k in range(max_ov, -1, -1):
            if base_suffix[-k:] == pj[:k]:
                ov = k
                break
        O_start[j] = ov
        
    # Initial order
    order = list(range(m))
    
    def calc_score(ord):
        total = O_start[ord[0]]
        for k in range(1, m):
            total += O[ord[k-1]][ord[k]]
        return total
        
    current_score = calc_score(order)
    best_order = list(order)
    best_score = current_score

    # 2-Opt with multiple passes
    def two_opt_pass(ord, score):
        improved = True
        while improved and (time.time() < deadline - 3):
            improved = False
            for i in range(m - 1):
                if time.time() > deadline - 3:
                    break
                for j in range(i + 1, m):
                    old_int = 0
                    new_int = 0
                    for k in range(i, j):
                        old_int += O[ord[k]][ord[k+1]]
                        new_int += O[ord[k+1]][ord[k]]
                    
                    old_left = O_start[ord[i]] if i == 0 else O[ord[i-1]][ord[i]]
                    new_left = O_start[ord[j]] if i == 0 else O[ord[i-1]][ord[j]]
                    
                    old_right = O[ord[j]][ord[j+1]] if j < m - 1 else 0
                    new_right = O[ord[i]][ord[j+1]] if j < m - 1 else 0
                    
                    delta = (new_int - old_int) + (new_left - old_left) + (new_right - old_right)
                    
                    if delta > 0:
                        ord[i:j+1] = list(reversed(ord[i:j+1]))
                        score += delta
                        improved = True
        return ord, score

    # Run 2-opt
    order, current_score = two_opt_pass(order, current_score)
    if current_score > best_score:
        best_score = current_score
        best_order = list(order)

    # Try 3-opt for further improvement
    if time.time() < deadline - 5:
        def three_opt_pass(ord, score):
            improved = True
            while improved and (time.time() < deadline - 3):
                improved = False
                for i in range(m - 2):
                    if time.time() > deadline - 3:
                        break
                    for j in range(i + 1, m - 1):
                        if time.time() > deadline - 3:
                            break
                        for k in range(j + 1, m):
                            # Try reversing ord[i..j]
                            old_int = 0
                            new_int = 0
                            for x in range(i, j):
                                old_int += O[ord[x]][ord[x+1]]
                                new_int += O[ord[x+1]][ord[x]]
                            
                            old_left = O_start[ord[i]] if i == 0 else O[ord[i-1]][ord[i]]
                            new_left = O_start[ord[j]] if i == 0 else O[ord[i-1]][ord[j]]
                            
                            old_mid_left = O[ord[j]][ord[j+1]]
                            new_mid_left = O[ord[j]][ord[i]]
                            
                            old_mid_right = O[ord[k]][ord[k+1]] if k < m - 1 else 0
                            new_mid_right = O[ord[j+1]][ord[k+1]] if k < m - 1 else 0
                            
                            old_right = O[ord[k]][ord[k+1]] if k < m - 1 else 0
                            new_right = O[ord[j+1]][ord[k+1]] if k < m - 1 else 0
                            
                            # This is getting complex; skip 3-opt and use a different strategy
                            break
                        break
                    break
            return ord, score
        
        # Instead, do another round of 2-opt with shuffled start
        if time.time() < deadline - 5:
            for restart in range(3):
                if time.time() > deadline - 5:
                    break
                rand_order = list(range(m))
                random.shuffle(rand_order)
                # Greedy construction from random start
                used = set()
                greedy = [rand_order[0]]
                used.add(rand_order[0])
                score_g = O_start[rand_order[0]]
                
                for _ in range(m - 1):
                    best_next = -1
                    best_ov = -1
                    for j in range(m):
                        if j not in used:
                            ov = O[greedy[-1]][j]
                            if ov > best_ov:
                                best_ov = ov
                                best_next = j
                    greedy.append(best_next)
                    used.add(best_next)
                    score_g += best_ov
                
                # Refine with 2-opt
                greedy, score_g = two_opt_pass(greedy, score_g)
                if score_g > best_score:
                    best_score = score_g
                    best_order = list(greedy)

    # Construct final word from best_order
    final_word = list(base_word)
    
    first_idx = best_order[0]
    first_block = list(perms[first_idx]) + [n] + list(perms[first_idx])
    ov = O_start[first_idx]
    final_word += first_block[ov:]
    
    for k in range(1, m):
        prev_idx = best_order[k-1]
        curr_idx = best_order[k]
        curr_block = list(perms[curr_idx]) + [n] + list(perms[curr_idx])
        ov = O[prev_idx][curr_idx]
        final_word += curr_block[ov:]
        
    return "".join(map(str, final_word))
