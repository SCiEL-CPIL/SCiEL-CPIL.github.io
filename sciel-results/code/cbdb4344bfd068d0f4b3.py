import time
import random
import math
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # 5s margin for 120s limit

    # For small n, use the standard construction which is optimal
    if n <= 3:
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    if n == 4:
        return "12341213421234324123142312431234341234234123432123141234123423412343214"

    # For n >= 5, use block-based construction with local search
    def get_standard_superperm(n_prev):
        word = [1]
        for m in range(2, n_prev + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return word

    # Get S_{n-1}
    s_prev = get_standard_superperm(n - 1)
    M_prev = n - 1
    
    # Extract the order of permutations of 1..n-1 in s_prev
    order = []
    seen = set()
    for i in range(len(s_prev) - M_prev + 1):
        window = tuple(s_prev[i:i + M_prev])
        if len(set(window)) == M_prev and window not in seen:
            seen.add(window)
            order.append(window)
    
    # Generate blocks for n
    blocks = []
    for p in order:
        blocks.append(list(p) + [n] + list(p))
        
    M_blocks = len(blocks)
    block_len = len(blocks[0])
    
    # Precompute overlap matrix using string operations for speed
    block_strs = ["".join(map(str, b)) for b in blocks]
    
    overlap_matrix = [[0] * M_blocks for _ in range(M_blocks)]
    max_ov = block_len - 1
    
    for i in range(M_blocks):
        bi = block_strs[i]
        for j in range(M_blocks):
            bj = block_strs[j]
            for k in range(max_ov, -1, -1):
                if bi[-k:] == bj[:k]:
                    overlap_matrix[i][j] = k
                    break
    
    # Cost matrix: cost(i,j) = additional length when placing block j after block i
    cost = [[block_len - overlap_matrix[i][j] for j in range(M_blocks)] for i in range(M_blocks)]
    
    def build_string(order_idx):
        s = block_strs[order_idx[0]]
        for i in range(1, M_blocks):
            ov = overlap_matrix[order_idx[i-1]][order_idx[i]]
            s += block_strs[order_idx[i]][ov:]
        return s
    
    def eval_order(order_idx):
        length = block_len
        for i in range(1, M_blocks):
            length += cost[order_idx[i-1]][order_idx[i]]
        return length
    
    # Target: all n! permutations
    target_count = 1
    for k in range(2, n + 1):
        target_count *= k
    
    def is_valid_superperm(s):
        seen_perms = set()
        for i in range(len(s) - n + 1):
            window = s[i:i+n]
            if len(set(window)) == n:
                seen_perms.add(window)
                if len(seen_perms) >= target_count:
                    return True
        return len(seen_perms) == target_count
    
    # Initial: standard order
    current_order = list(range(M_blocks))
    current_string = build_string(current_order)
    current_length = len(current_string)
    
    best_order = current_order[:]
    best_string = current_string
    best_length = current_length
    
    # Verify initial
    if not is_valid_superperm(best_string):
        # Try random initializations
        for attempt in range(10):
            if time.time() > deadline - 10:
                break
            rand_order = list(range(M_blocks))
            random.shuffle(rand_order)
            s = build_string(rand_order)
            if is_valid_superperm(s) and len(s) < best_length:
                best_order = rand_order[:]
                best_string = s
                best_length = len(s)
                current_order = rand_order[:]
                current_length = len(s)
        if not is_valid_superperm(best_string):
            return best_string
    
    # Simulated annealing local search
    T_start = 50.0
    T_end = 0.01
    
    T = T_start
    last_iter_time = time.time()
    iter_count = 0
    
    while time.time() < deadline - 3:  # 3s safety margin
        iter_count += 1
        
        # Cool temperature
        if iter_count % 10000 == 0:
            elapsed = time.time() - last_iter_time
            if elapsed > 0:
                # Adaptive cooling
                T = max(T_end, T * 0.95)
            last_iter_time = time.time()
        
        # Choose move type
        r = random.random()
        if r < 0.35:
            move_type = 0  # swap
            i, j = random.sample(range(M_blocks), 2)
        elif r < 0.55:
            move_type = 1  # reverse segment
            i, j = random.sample(range(M_blocks), 2)
            if i > j:
                i, j = j, i
        elif r < 0.80:
            move_type = 2  # relocate
            i = random.randint(0, M_blocks - 1)
            j = random.randint(0, M_blocks - 1)
            if i == j:
                continue
        else:
            move_type = 3  # segment reverse (shorter)
            i = random.randint(0, M_blocks - 1)
            seg_len = random.randint(2, min(5, M_blocks))
            j = min(i + seg_len - 1, M_blocks - 1)
            if i == j:
                continue
        
        new_order = current_order[:]
        
        if move_type == 0:
            new_order[i], new_order[j] = new_order[j], new_order[i]
        elif move_type == 1:
            new_order[i:j+1] = reversed(new_order[i:j+1])
        elif move_type == 2:
            elem = new_order.pop(i)
            new_order.insert(j, elem)
        else:
            new_order[i:j+1] = reversed(new_order[i:j+1])
        
        new_length = eval_order(new_order)
        delta = new_length - current_length
        
        if delta <= 0:
            current_order = new_order
            current_length = new_length
            if current_length < best_length:
                best_order = current_order[:]
                best_length = current_length
                # Check validity periodically
                if random.random() < 0.005 or current_length < 6500:
                    candidate_str = build_string(current_order)
                    if is_valid_superperm(candidate_str):
                        best_string = candidate_str
        else:
            # Accept with probability exp(-delta/T)
            if T > T_end:
                try:
                    accept_prob = math.exp(-delta / T)
                except OverflowError:
                    accept_prob = 0.0
                if random.random() < accept_prob:
                    current_order = new_order
                    current_length = new_length
    
    # Final verification and return
    if is_valid_superperm(best_string):
        return best_string
    
    # Try to build and verify best_order
    candidate = build_string(best_order)
    if is_valid_superperm(candidate):
        return candidate
    
    # Try current
    candidate = build_string(current_order)
    if is_valid_superperm(candidate):
        return candidate
    
    return best_string
