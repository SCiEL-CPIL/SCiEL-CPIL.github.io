import time
import random
import math
from itertools import permutations

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    
    # Generate all permutations of 1..n-1
    perms = list(permutations(range(1, n)))
    k = len(perms)  # (n-1)!
    
    # Precompute overlap matrix
    # ov[i][j] = max l such that perms[i][-l:] == perms[j][:l]
    ov = [[0] * k for _ in range(k)]
    for i in range(k):
        pi = perms[i]
        for j in range(k):
            pj = perms[j]
            max_l = n - 1
            for l in range(max_l, 0, -1):
                if pi[-l:] == pj[:l]:
                    ov[i][j] = l
                    break
    
    # Greedy initial path: nearest neighbor
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
            best_next = random.choice(list(unvisited))
        path.append(best_next)
        unvisited.remove(best_next)
    
    def get_score(p):
        s = 0
        for i in range(len(p) - 1):
            s += ov[p[i]][p[i+1]]
        return s
    
    best_score = get_score(path)
    best_path = path[:]
    
    start_time = time.time()
    time_limit = 110
    
    # Or-opt: move a segment of size 1, 2, or 3 to a new position
    def or_opt_move(path, i, size, j):
        """Move segment path[i:i+size] to position j (before path[j]).
        Returns (new_path, delta) or (None, 0) if invalid."""
        if j == i:
            return None, 0
        if j < i:
            # Remove i..i+size-1, insert before j
            # Old edges affected: (i-1,i), (i+size-1,i+size), (j-1,j) if j>0
            # New edges: (i-1, i+size) if i>0 and i+size<k, (j-1, i) if j>0, (i+size-1, j)
            seg = path[i:i+size]
            new_path = path[:j] + seg + path[j:i] + path[i+size:]
        else:
            # Remove i..i+size-1, insert before j (j > i)
            # After removal, indices shift. Insert before original j.
            seg = path[i:i+size]
            new_path = path[:i] + path[i+size:j] + seg + path[j:]
        # Calculate delta
        old_score = 0
        new_score = 0
        # Old edges
        if i > 0:
            old_score += ov[path[i-1]][path[i]]
        if i + size < k:
            old_score += ov[path[i+size-1]][path[i+size]]
        if j > 0 and j != i:
            old_score += ov[path[j-1]][path[j]]
        # New edges
        if i > 0:
            new_score += ov[path[i-1]][path[i+size]] if i + size < k else 0
        if i + size < k:
            new_score += ov[path[i+size-1]][path[i]] if i > 0 else 0
        if j > 0:
            new_score += ov[path[j-1]][path[i]]
        if i + size < k and j > 0:
            new_score += ov[path[i+size-1]][path[j]]
        # This is getting complex. Let's just compute directly.
        return None, 0  # Placeholder
    
    # Simpler approach: just do 2-opt with proper delta and Or-opt with full recomputation for small segments
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        
        # 2-opt: reverse segment i..j
        for i in range(k - 1):
            if time.time() - start_time > time_limit:
                break
            # Limit j to i+1..min(i+50, k-1) for efficiency
            for j in range(i + 1, min(i + 50, k)):
                # Calculate delta for reversing path[i:j+1]
                # Old edges: (i-1,i), (j,j+1), and internal edges i..j-1
                # New edges: (i-1,j), (i,j+1), and reversed internal edges
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
                    break
            if improved:
                break
        
        # Or-opt: move single element
        if time.time() - start_time > time_limit:
            break
        for i in range(k):
            if time.time() - start_time > time_limit:
                break
            if i == 0:
                continue
            # Try moving path[i] to position j
            elem = path[i]
            # Remove elem from position i
            path_without = path[:i] + path[i+1:]
            # Try inserting at each position j in path_without
            best_j = -1
            best_delta = 0
            for j in range(len(path_without) + 1):
                if j == i:
                    continue  # Same position
                # Calculate delta
                old_score = 0
                if i > 0:
                    old_score += ov[path[i-1]][path[i]]
                if i < k - 1:
                    old_score += ov[path[i]][path[i+1]]
                
                new_score = 0
                # After removal, indices shift
                # If j <= i-1: insert before path_without[j]
                # If j > i-1: insert before path_without[j] which is original path[j+1]
                
                # Let's compute directly
                if j == 0:
                    new_score += ov[elem][path_without[0]] if path_without else 0
                    if i < k - 1:
                        new_score += ov[path[i-1]][path[i+1]] if i > 0 else 0
                elif j == len(path_without):
                    new_score += ov[path_without[-1]][elem] if path_without else 0
                    if i < k - 1:
                        new_score += ov[path[i-1]][path[i+1]] if i > 0 else 0
                else:
                    new_score += ov[path_without[j-1]][elem]
                    new_score += ov[elem][path_without[j]]
                    if i < k - 1:
                        new_score += ov[path[i-1]][path[i+1]] if i > 0 else 0
                
                delta = new_score - old_score
                if delta > best_delta:
                    best_delta = delta
                    best_j = j
            
            if best_j >= 0 and best_delta > 0:
                # Perform the move
                path = path_without[:best_j] + [elem] + path_without[best_j:]
                best_score += best_delta
                improved = True
                break
        
        # Update best
        curr_score = get_score(path)
        if curr_score > best_score:
            best_score = curr_score
            best_path = path[:]
    
    # Reconstruct the word from best_path
    word = []
    for t, idx in enumerate(best_path):
        p = perms[idx]
        if t == 0:
            word.extend(p)
            word.append(n)
            word.extend(p)
        else:
            prev_idx = best_path[t-1]
            ov_val = ov[prev_idx][idx]
            # Append p[ov_val:] + [n] + p
            word.extend(p[ov_val:])
            word.append(n)
            word.extend(p)
    
    return "".join(map(str, word))
