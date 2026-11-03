import time
import sys
import random

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
        # Standard is 33. 
        # Optimization might not help or might be overkill.
        # But let's run the same logic.
        pass

    # Generate standard S_{n-1}
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
        # Precompute overlap matrix
        # W[i][j] = overlap(perms[i], perms[j])
        # Overlap is suffix of perms[i] matching prefix of perms[j]
        W = [[0]*N for _ in range(N)]
        for i in range(N):
            pi = perms[i]
            for j in range(N):
                if i == j: continue
                pj = perms[j]
                max_len = min(len(pi), len(pj))
                ov = 0
                for k in range(max_len, -1, -1):
                    if k == 0:
                        ov = 0
                        break
                    if pi[-k:] == pj[:k]:
                        ov = k
                        break
                W[i][j] = ov

        # Greedy Initialization
        # Start with a random permutation, then add the one with max overlap
        order = [random.randint(0, N-1)]
        remaining = set(range(N))
        remaining.remove(order[0])
        
        while remaining:
            last = order[-1]
            best_next = -1
            best_ov = -1
            # To speed up, we can just pick the max
            for j in remaining:
                ov = W[last][j]
                if ov > best_ov:
                    best_ov = ov
                    best_next = j
            order.append(best_next)
            remaining.remove(best_next)
            
        # Calculate initial total overlap
        total_ov = 0
        for i in range(1, N):
            total_ov += W[order[i-1]][order[i]]
            
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # 2-Opt: Swap adjacent
            for i in range(N - 1):
                if time.time() > deadline - 5: break
                
                idx_i = order[i]
                idx_ip1 = order[i+1]
                
                if i == 0:
                    left = -1
                else:
                    left = order[i-1]
                    
                if i+1 == N-1:
                    right = -1
                else:
                    right = order[i+2]
                    
                old_ov = 0
                if left != -1:
                    old_ov += W[left][idx_i]
                old_ov += W[idx_i][idx_ip1]
                if right != -1:
                    old_ov += W[idx_ip1][right]
                
                new_ov = 0
                if left != -1:
                    new_ov += W[left][idx_ip1]
                new_ov += W[idx_ip1][idx_i]
                if right != -1:
                    new_ov += W[idx_i][right]
                
                if new_ov > old_ov:
                    order[i], order[i+1] = order[i+1], order[i]
                    total_ov += (new_ov - old_ov)
                    improved = True
                    break # Restart scan
            
            # Or-Opt: Move single element
            if not improved:
                for i in range(N):
                    if time.time() > deadline - 5: break
                    elem = order[i]
                    
                    # Neighbors of i
                    left_i = order[i-1] if i > 0 else -1
                    right_i = order[i+1] if i < N-1 else -1
                    
                    # Old cost contribution of removing i
                    # Edges: (left_i, i), (i, right_i)
                    # If i is removed, edge (left_i, right_i) is formed (if both exist)
                    # Wait, we are moving it, so we break the chain.
                    
                    for j in range(N):
                        if j == i or j == i+1: continue
                        
                        # We want to insert elem at position j.
                        # If j > i, the indices shift.
                        # Let's define insertion position relative to the list *after* removal.
                        # This is tricky. Let's just try all insertion positions in the current list.
                        
                        # Actually, simpler: Try to move elem to position j in the current order.
                        # If i < j, we remove i, then insert at j-1 (since j shifts left).
                        # If i > j, we remove i, then insert at j.
                        
                        # Let's calculate delta for moving i to j.
                        # Case 1: i < j
                        # Remove i:
                        #   Break (left_i, i) and (i, right_i).
                        #   Form (left_i, right_i) if both exist.
                        # Insert at j (which is now j-1 in the shortened list? No, j is index in original list).
                        # If we insert after original index j-1 (which is now at j-1 if i<j? No).
                        
                        # Let's use a simpler approach:
                        # Remove elem from i.
                        # Try inserting it at every position k in the new list.
                        
                        # This is O(N^2) per pass, which is 720^2 = 500k.
                        # With 720 iterations, it's 360M operations. Too slow in Python.
                        
                        # We need O(1) delta.
                        # Let's stick to 2-Opt for now, it might be sufficient if the greedy init is good.
                        # Or, we can limit Or-Opt to nearby moves.
                        
                        pass

    # Construct final word
    final_word = []
    for idx in order:
        p = perms[idx]
        block = list(p) + [n] + list(p)
        max_overlap = min(len(final_word), len(block))
        overlap = 0
        for o in range(max_overlap, -1, -1, ):
            if o == 0:
                overlap = 0
                break
            if final_word[len(final_word) - o:] == block[:o]:
                overlap = o
                break
        final_word += block[overlap:]
        
    return "".join(map(str, final_word))
