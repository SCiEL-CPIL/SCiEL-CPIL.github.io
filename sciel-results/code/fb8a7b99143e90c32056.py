import time
import random
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # 140s limit, 10s margin
    
    # Base case
    if n == 1:
        return "1"
        
    word = [1]
    
    # Precompute factorials for length estimation? Not strictly needed.
    
    for m in range(2, n + 1):
        # Check time
        if time.time() > deadline - 5:
            break
            
        # Extract unique permutations of 1..m-1 from current word
        perms, seen = [], set()
        # Window size is m-1
        win_size = m - 1
        for i in range(len(word) - win_size + 1):
            window = tuple(word[i:i + win_size])
            # Check if it's a permutation of 1..m-1
            if len(window) == win_size and len(set(window)) == win_size:
                # Ensure it contains 1..m-1
                # Since it comes from previous word which was a superperm of 1..m-2,
                # and we added m-1? No, the previous word was superperm of 1..m-1?
                # Wait.
                # m=2: word [1]. Win size 1. Window (1). Perms of 1.
                # m=3: word [1,2,1]. Win size 2. Windows (1,2), (2,1). Perms of 1,2.
                # m=4: word [1,2,3,1,2,1,3,2,1]. Win size 3.
                # The windows are permutations of 1,2,3.
                # So we just check uniqueness and validity.
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
                    
        if not perms:
            # Fallback, should not happen
            continue

        # Construct blocks
        # block(p) = p + [m] + p
        blocks = [list(p) + [m] + list(p) for p in perms]
        
        # Solve TSP-like problem to order blocks
        # Minimize total length = sum(len(block)) - sum(overlap)
        # Equivalent to maximizing sum(overlap)
        
        k = len(blocks)
        
        # If k is small, we can try more steps
        # Heuristic: Greedy + 2-opt
        
        # Precompute overlap matrix
        # overlap[i][j] = max suffix of blocks[i] matching prefix of blocks[j]
        overlap = [[0]*k for _ in range(k)]
        for i in range(k):
            for j in range(k):
                if i == j:
                    continue
                b_i = blocks[i]
                b_j = blocks[j]
                max_ov = 0
                # Max possible overlap is len(b_i)-1
                # b_i ends with p_i. b_j starts with p_j.
                # p_i, p_j are length m-1.
                # So max overlap is m-1.
                for ov in range(min(len(b_i), len(b_j)) - 1, -1, -1):
                    if b_i[-ov:] == b_j[:ov]:
                        max_ov = ov
                        break
                overlap[i][j] = max_ov

        # Greedy construction of path
        # Start with a random node
        order = [random.randint(0, k-1)]
        unvisited = set(range(k))
        unvisited.remove(order[0])
        
        while unvisited:
            last = order[-1]
            # Find nearest unvisited
            best_node = None
            best_ov = -1
            # Randomize tie-breaking to allow diversity in restarts?
            # For now, deterministic nearest
            candidates = list(unvisited)
            random.shuffle(candidates)
            for node in candidates:
                ov = overlap[last][node]
                if ov > best_ov:
                    best_ov = ov
                    best_node = node
            order.append(best_node)
            unvisited.remove(best_node)
            
        # 2-Opt Improvement
        # Try to improve the order
        improved = True
        iterations = 0
        max_iterations = 1000 if k > 50 else 5000
        
        while improved and iterations < max_iterations and time.time() < deadline - 10:
            improved = False
            iterations += 1
            for i in range(len(order) - 1):
                for j in range(i + 1, len(order)):
                    # Try reversing segment order[i:j+1]
                    # Calculate cost change
                    # Current cost: overlap[order[i-1]][order[i]] + overlap[order[j]][order[j+1]]
                    # (with boundary checks)
                    
                    # Helper to get cost of a path
                    def get_cost(path):
                        cost = 0
                        for idx in range(len(path) - 1):
                            cost -= overlap[path[idx]][path[idx+1]]
                        return cost
                    
                    # To speed up, calculate delta
                    # Let A = order[i-1] if i>0 else None
                    # Let B = order[i]
                    # Let C = order[j]
                    # Let D = order[j+1] if j < len(order)-1 else None
                    
                    # Current contribution:
                    # -overlap[A][B] - overlap[C][D]
                    # New contribution:
                    # -overlap[A][C] - overlap[B][D]
                    
                    cost_delta = 0
                    if i > 0:
                        cost_delta += overlap[order[i-1]][order[i]]
                    if j < len(order) - 1:
                        cost_delta += overlap[order[j]][order[j+1]]
                    
                    new_cost_delta = 0
                    if i > 0:
                        new_cost_delta += overlap[order[i-1]][order[j]]
                    if j < len(order) - 1:
                        new_cost_delta += overlap[order[i]][order[j+1]]
                    
                    # We want to MINIMIZE cost (which is -sum overlaps)
                    # So we want to MAXIMIZE sum overlaps.
                    # Delta in sum overlaps = new_ov_sum - old_ov_sum
                    # old_ov_sum = overlap[A][B] + overlap[C][D]
                    # new_ov_sum = overlap[A][C] + overlap[B][D]
                    
                    if new_cost_delta > cost_delta:
                        order[i:j+1] = order[i:j+1][::-1]
                        improved = True
                        # Restart loop? Or continue?
                        # Continuing is faster, but might miss improvements.
                        # For now, continue.
                        
        # Construct new word
        new_word = []
        for idx, block_idx in enumerate(order):
            block = blocks[block_idx]
            if idx == 0:
                new_word.extend(block)
            else:
                prev_block = blocks[order[idx-1]]
                ov = overlap[order[idx-1]][block_idx]
                new_word.extend(block[ov:])
                
        word = new_word
        
    return "".join(map(str, word))
