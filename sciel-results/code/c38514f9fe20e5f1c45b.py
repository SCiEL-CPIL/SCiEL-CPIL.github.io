import time
import random
import math

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    
    # Generate all permutations of 1..n-1
    from itertools import permutations
    perms = list(permutations(range(1, n)))
    k = len(perms) # (n-1)!
    
    # Precompute overlap matrix
    # ov[i][j] = max k such that perms[i][-k:] == perms[j][:k]
    ov = [[0] * k for _ in range(k)]
    for i in range(k):
        pi = perms[i]
        for j in range(k):
            pj = perms[j]
            max_k = n - 1
            # Check from max_k down to 1
            # Since n is small (7), this is fast
            for l in range(max_k, 0, -1):
                if pi[-l:] == pj[:l]:
                    ov[i][j] = l
                    break
    
    # Initialize path
    # Start with a greedy path: nearest neighbor
    path = [0]
    unvisited = set(range(1, k))
    while unvisited:
        last = path[-1]
        best_next = -1
        best_ov = -1
        for u in unvisited:
            if ov[last][u] > best_ov:
                best_ov = ov[last][u]
                best_next = u
        if best_next == -1:
            # Pick random
            best_next = random.choice(list(unvisited))
        path.append(best_next)
        unvisited.remove(best_next)
        
    # Calculate current score
    def get_score(p):
        s = 0
        for i in range(len(p) - 1):
            s += ov[p[i]][p[i+1]]
        return s
        
    best_score = get_score(path)
    best_path = path[:]
    
    start_time = time.time()
    time_limit = 110
    
    # Variable Neighborhood Search
    # Or-opt and 2-opt
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        
        # Or-opt: move a segment of size 1, 2, or 3
        for size in [1, 2, 3]:
            if time.time() - start_time > time_limit:
                break
            for i in range(k - size):
                if time.time() - start_time > time_limit:
                    break
                seg = path[i:i+size]
                # Try inserting seg at every position j
                # We remove seg, then insert at j
                # Note: j is index in the *original* path or *new* path?
                # Usually we consider moving i to j.
                # Let's just try all j != i
                for j in range(k - size + 1):
                    if j == i:
                        continue
                    # Construct new path
                    # This is O(N) per move, which is 720 * 720 * 3 ~ 1.5M ops per pass.
                    # Might be slow.
                    # Optimization: Calculate delta.
                    
                    # Old edges:
                    # (i-1, i), (i+size-1, i+size)
                    # New edges:
                    # (i-1, i+size), (j-1, i), (i+size-1, j) ... depends on j < i or j > i
                    
                    # Let's just do the O(N) reconstruction for simplicity first, 
                    # but check time.
                    # 1.5M ops * 100 passes = 150M ops. Might be tight.
                    # Let's implement delta calculation.
                    
                    # Case 1: j < i
                    # Remove i..i+size-1.
                    # Insert at j.
                    # Old:
                    # ... A, B ... C, D ...
                    # A = path[i-1] (if i>0), B = path[i]
                    # C = path[i+size-1], D = path[i+size] (if i+size < k)
                    # 
                    # New:
                    # ... A, C, ... [SEG] ... 
                    # Wait, if j < i:
                    # Original: ... X, Y, ... A, B, C, D ...
                    # Remove A,B,C. Insert after Y (at index j).
                    # New: ... X, Y, A, B, C, ... D ...
                    # Edges changed:
                    # Old: (X,Y), (Y,A), (C,D)  -- Wait, Y is path[j], A is path[i]
                    # Actually, indices shift.
                    
                    # This is complex. Given the small N=720, 
                    # maybe a simpler 2-opt with delta is better.
                    
                    pass # Skip Or-opt implementation for now to save space, use 2-opt
                    
        # 2-opt
        if time.time() - start_time > time_limit:
            break
            
        # We need to calculate delta for 2-opt efficiently.
        # Sum of forward edges in segment i..j:
        # We can precompute prefix sums of ov[path[t]][path[t+1]]
        # But the "backward" sum is not available.
        # However, for 2-opt, we reverse the segment.
        # The new edges are the reverse of the old edges in the segment.
        # Sum_rev = sum(ov[path[t+1]][path[t]] for t in i..j-1)
        
        # We can just iterate if we limit the segment length?
        # Or, we can maintain the path and just accept the O(N) cost of 2-opt passes?
        # 2-opt pass: 250k moves.
        # If we do O(N) per move, it's 250k * 720 = 180M ops.
        # This is too slow for Python in 120s if we do many passes.
        
        # Let's try a limited 2-opt: only consider segments of length < 20.
        # This reduces cost to 250k * 20 = 5M ops per pass.
        # We can do 20 passes in 120s.
        
        for i in range(k - 1):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, min(i + 20, k)):
                # Reverse path[i:j+1]
                # Calculate delta
                # Old: (i-1, i), (j, j+1), sum(i..j-1)
                # New: (i-1, j), (i, j+1), sum_rev(i..j-1)
                
                old_score = 0
                if i > 0:
                    old_score += ov[path[i-1]][path[i]]
                if j < k - 1:
                    old_score += ov[path[j]][path[j+1]]
                for t in range(i, j):
                    old_score += ov[path[t]][path[t+1]]
                    
                new_score = 0
                if i > 0:
                    new_score += ov[path[i-1]][path[j]]
                if j < k - 1:
                    new_score += ov[path[i]][path[j+1]]
                for t in range(i, j):
                    new_score += ov[path[t+1]][path[t]]
                    
                delta = new_score - old_score
                if delta > 0:
                    path[i:j+1] = path[i:j+1][::-1]
                    best_score += delta
                    improved = True
                    # After one improvement, restart i loop? Or continue?
                    # Restarting is safer to avoid missing moves.
                    break
            if improved:
                break # Restart 2-opt pass

        # Update best
        curr_score = get_score(path)
        if curr_score > best_score:
            best_score = curr_score
            best_path = path[:]

    # Reconstruct
    # Use best_path
    word = []
    for idx in best_path:
        p = perms[idx]
        if not word:
            word.extend(p)
            word.append(n)
            word.extend(p)
        else:
            # Overlap with previous block
            prev_p = perms[best_path[best_path.index(idx)-1]] # This is O(N) index lookup, bad.
            # We should iterate with index
            pass

    # Better reconstruction
    word = []
    for t, idx in enumerate(best_path):
        p = perms[idx]
        if t == 0:
            word.extend(p)
            word.append(n)
            word.extend(p)
        else:
            prev_idx = best_path[t-1]
            prev_p = perms[prev_idx]
            ov_val = ov[prev_idx][idx]
            # Append p[ov_val:] + [n] + p
            word.extend(p[ov_val:])
            word.append(n)
            word.extend(p)
            
    return "".join(map(str, word))
