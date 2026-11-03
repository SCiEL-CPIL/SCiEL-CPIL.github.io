import time
import sys
import random
import numpy as np

def construct(n):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "12341231423124312132413214321"
    
    # Generate standard S_{n-1} word
    # For n=7, this generates S_6 word
    word = [1]
    for m in range(2, n):
        perms, seen = [], set()
        w_size = m - 1
        for i in range(len(word) - w_size + 1):
            window = tuple(word[i:i + w_size])
            if len(set(window)) == w_size:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            # Optimize overlap check
            # Since we are appending blocks of form P, m, P
            # The overlap is determined by suffix of P and prefix of P
            # But here we are building the S_{m} word from S_{m-1} perms.
            # The current code does a generic overlap check.
            # For speed, we can stick to generic but it's O(L^2) per block.
            # L is small (m).
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        word = new_word
    
    # Extract perms of size n-1
    perms, seen = [], set()
    w_size = n - 1
    for i in range(len(word) - w_size + 1):
        window = tuple(word[i:i + w_size])
        if len(set(window)) == w_size:
            if window not in seen:
                seen.add(window)
                perms.append(window)
                
    N = len(perms)
    
    if n >= 5:
        # Precompute overlap matrix W
        # W[i][j] = overlap(perms[i], perms[j])
        # This is the length of the longest suffix of perms[i] that is a prefix of perms[j]
        W = np.zeros((N, N), dtype=np.int8)
        for i in range(N):
            pi = perms[i]
            for j in range(N):
                if i == j: 
                    continue
                pj = perms[j]
                # Max possible overlap is n-2
                max_len = n - 2
                ov = 0
                for k in range(1, max_len + 1):
                    if pi[-k:] == pj[:k]:
                        ov = k
                    else:
                        break
                W[i][j] = ov

        # Objective: Maximize sum of W[order[i]][order[i+1]]
        
        # Initial order: Greedy Nearest Neighbor
        # Start with a random permutation
        order = list(range(N))
        random.shuffle(order)
        
        # Greedy improvement: repeatedly find the best swap? No, that's expensive.
        # Let's use a simple greedy construction:
        # Start with 0. Repeatedly add the unused perm with max overlap to current last.
        # This is O(N^2).
        greedy_order = [0]
        used = [False] * N
        used[0] = True
        for _ in range(N - 1):
            last = greedy_order[-1]
            best_next = -1
            best_ov = -1
            for j in range(N):
                if not used[j]:
                    ov = W[last][j]
                    if ov > best_ov:
                        best_ov = ov
                        best_next = j
            if best_next != -1:
                greedy_order.append(best_next)
                used[best_next] = True
        
        # Calculate score for greedy_order
        def calc_score(ord):
            s = 0
            for i in range(N - 1):
                s += W[ord[i]][ord[i+1]]
            return s
        
        best_order = greedy_order
        best_score = calc_score(best_order)
        
        # Simulated Annealing
        # We have ~100 seconds left.
        # N=720.
        # Move: Or-opt (move one element to another position) or Swap.
        # Or-opt is more powerful for TSP-like problems.
        
        current_order = best_order[:]
        current_score = best_score
        
        T = 10.0
        T_min = 0.01
        cooling = 0.999
        
        # Precompute indices for faster access? 
        # Using numpy for W is fast.
        
        while time.time() < deadline - 10:
            # Pick a move
            # 80% Or-opt, 20% Swap
            if random.random() < 0.8:
                i = random.randint(0, N - 1)
                j = random.randint(0, N - 1)
                if i == j or i == j + 1:
                    continue
                
                elem = current_order[i]
                
                # Calculate delta
                # Remove i, insert at j
                # Affected edges:
                # Old: (i-1, i), (i, i+1)
                # New: (i-1, i+1) [if valid], (j-1, elem), (elem, j) [adjust indices]
                
                old_ov = 0
                new_ov = 0
                
                # Old edges
                if i > 0:
                    old_ov += W[current_order[i-1]][elem]
                if i < N - 1:
                    old_ov += W[elem][current_order[i+1]]
                
                # New edges
                # If i < j:
                #   Remove i. j becomes j-1.
                #   Left neighbor of new pos: current_order[j-1] (if j-1 > i) else current_order[i-1]
                #   Right neighbor: current_order[j] (if j < N)
                # If i > j:
                #   Remove i. j stays j.
                #   Left neighbor: current_order[j-1] (if j > 0)
                #   Right neighbor: current_order[j]
                
                if i < j:
                    # Insert at j-1 in the conceptual list after removal?
                    # Actually, let's just construct the new local neighborhood
                    # Left index in current_order:
                    if j - 1 > i:
                        left_idx = j - 1
                    else:
                        left_idx = i - 1
                    
                    right_idx = j
                    
                    if left_idx >= 0:
                        new_ov += W[current_order[left_idx]][elem]
                    if right_idx < N:
                        new_ov += W[elem][current_order[right_idx]]
                    
                    # Edge connecting the gap left by i
                    if i > 0 and i < N - 1:
                        new_ov += W[current_order[i-1]][current_order[i+1]]
                    elif i == 0:
                        # No left edge to remove, but we connected i-1 (none) to i+1? No.
                        # If i=0, we remove 0.
                        # New left neighbor is none (if inserted at start) or current_order[0] (old 1).
                        # Wait, if i=0, left_idx = -1.
                        # If j=1, left_idx = -1.
                        # This logic is tricky.
                        pass
                    elif i == N - 1:
                        pass
                
                else: # i > j
                    left_idx = j - 1
                    right_idx = j
                    
                    if left_idx >= 0:
                        new_ov += W[current_order[left_idx]][elem]
                    if right_idx < N:
                        new_ov += W[elem][current_order[right_idx]]
                    
                    if i > 0 and i < N - 1:
                        new_ov += W[current_order[i-1]][current_order[i+1]]
                
                delta = new_ov - old_ov
                
                # Perform move if accepted
                if delta > 0 or random.random() < np.exp(delta / T):
                    # Apply move
                    current_order.pop(i)
                    if i < j:
                        j -= 1
                    current_order.insert(j, elem)
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_order = current_order[:]
                        
            else:
                # Swap
                i = random.randint(0, N - 2)
                j = i + 1
                # Swap i and i+1
                a = current_order[i]
                b = current_order[i+1]
                
                old_ov = W[a][b]
                new_ov = W[b][a]
                
                # Check neighbors
                if i > 0:
                    old_ov += W[current_order[i-1]][a]
                    new_ov += W[current_order[i-1]][b]
                if i + 1 < N - 1:
                    old_ov += W[b][current_order[i+2]]
                    new_ov += W[a][current_order[i+2]]
                
                delta = new_ov - old_ov
                
                if delta > 0 or random.random() < np.exp(delta / T):
                    current_order[i], current_order[i+1] = b, a
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_order = current_order[:]
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        order = best_order

    # Construct final word
    final_word = []
    for idx in order:
        p = perms[idx]
        block = list(p) + [n] + list(p)
        max_overlap = min(len(final_word), len(block))
        overlap = 0
        for o in range(max_overlap, -1, -1):
            if o == 0:
                overlap = 0
                break
            if final_word[len(final_word) - o:] == block[:o]:
                overlap = o
                break
        final_word += block[overlap:]
        
    return "".join(map(str, final_word))
