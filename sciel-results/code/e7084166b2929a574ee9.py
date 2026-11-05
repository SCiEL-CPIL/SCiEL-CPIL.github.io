import itertools
import time
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin before 120s limit
    
    # 1. Generate standard superpermutation for n < 7
    # This is optimal for n <= 4 and good for n=5,6
    if n <= 6:
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
        
    # 2. For n=7, use TSP heuristic on S6 permutations
    # First, generate S6 standard superpermutation to get the list of S6 permutations
    word6 = [1]
    for m in range(2, 7):
        perms, seen = [], set()
        for i in range(len(word6) - m + 2):
            window = tuple(word6[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word6 = new
        
    # Extract all 720 permutations of 1..6 in order of appearance in word6
    perms6, seen = [], set()
    for i in range(len(word6) - 5):
        window = tuple(word6[i:i + 6])
        if len(set(window)) == 6 and window not in seen:
            seen.add(window)
            perms6.append(window)
            
    N = len(perms6) # Should be 720
    
    # Precompute overlap matrix: overlap[i][j] = longest suffix of perms6[i] that is prefix of perms6[j]
    # Max overlap is 5
    overlap_mat = [[0] * N for _ in range(N)]
    for i in range(N):
        pi = perms6[i]
        for j in range(N):
            if i == j:
                overlap_mat[i][j] = 6 # Self overlap, though we won't use it in path
                continue
            pj = perms6[j]
            ov = 0
            # Check k from 5 down to 0
            # pi[-k:] == pj[:k]
            for k in range(5, -1, -1):
                if pi[-k:] == pj[:k]:
                    ov = k
                    break
            overlap_mat[i][j] = ov
            
    # Helper to calculate total overlap of a path
    def calc_total_overlap(path):
        total = 0
        for i in range(len(path) - 1):
            total += overlap_mat[path[i]][path[i + 1]]
        return total
        
    # Helper to build word from path
    def build_word_from_path(path):
        w = []
        for idx in path:
            perm = perms6[idx]
            block = list(perm) + [7] + list(perm)
            overlap = 0
            min_len = min(len(w), len(block))
            for o in range(min_len, -1, -1):
                if w[len(w) - o:] == block[:o]:
                    overlap = o
                    break
            w += block[overlap:]
        return w
        
    # 1. Generate Naive Word (standard order)
    naive_path = list(range(N))
    best_path = naive_path[:]
    best_overlap = calc_total_overlap(naive_path)
    
    # 2. Heuristic Optimization: Nearest Neighbor
    random.seed(42)
    unvisited = list(range(N))
    random.shuffle(unvisited)
    path = [unvisited.pop()]
    
    while unvisited:
        last = path[-1]
        best_idx = -1
        best_ov = -1
        for idx in unvisited:
            ov = overlap_mat[last][idx]
            if ov > best_ov:
                best_ov = ov
                best_idx = idx
        path.append(best_idx)
        unvisited.remove(best_idx)
        
    current_overlap = calc_total_overlap(path)
    if current_overlap > best_overlap:
        best_overlap = current_overlap
        best_path = path[:]
    else:
        best_path = naive_path[:]
        path = naive_path[:]
        current_overlap = best_overlap
        
    # 3. 2-opt improvement
    improved = True
    while improved and time.time() < deadline - 15:
        improved = False
        
        # 2-Opt
        for i in range(len(path) - 2):
            if time.time() > deadline - 15:
                break
            for j in range(i + 1, len(path) - 1):
                if j == len(path) - 1:
                    old_ov = overlap_mat[path[i]][path[i + 1]]
                    new_ov = overlap_mat[path[i]][path[j]]
                    if new_ov > old_ov:
                        path[i + 1:j + 1] = reversed(path[i + 1:j + 1])
                        current_overlap += new_ov - old_ov
                        improved = True
                        break
                else:
                    old_ov1 = overlap_mat[path[i]][path[i + 1]]
                    old_ov2 = overlap_mat[path[j]][path[j + 1]]
                    new_ov1 = overlap_mat[path[i]][path[j]]
                    new_ov2 = overlap_mat[path[i + 1]][path[j + 1]]
                    if new_ov1 + new_ov2 > old_ov1 + old_ov2:
                        path[i + 1:j + 1] = reversed(path[i + 1:j + 1])
                        current_overlap += (new_ov1 + new_ov2) - (old_ov1 + old_ov2)
                        improved = True
                        break
            if improved:
                break
                
        # Or-Opt (Move segment of length 1 or 2)
        if not improved and time.time() < deadline - 15:
            for L in [1, 2]:
                for i in range(len(path) - L):
                    if time.time() > deadline - 15:
                        break
                    # Segment to move: path[i : i+L]
                    # Remove it
                    seg = path[i : i+L]
                    left_idx = i-1 if i > 0 else None
                    right_idx = i+L if i+L < N else None
                    
                    # Try inserting at every position k
                    # We can optimize by only checking nearby or random, but full scan is N=720, so 720*720*2 ~ 1M ops per L
                    # This might be too slow if done repeatedly. Let's limit Or-Opt or make it smarter.
                    # Actually, 2-opt is O(N^2) per pass. N=720. 720^2 = 518k.
                    # Or-Opt is O(N^2) per pass as well.
                    # Let's stick to 2-opt and maybe Or-Opt(1) only if time permits.
                    
                    # For simplicity and speed, let's just do 2-opt in the loop above.
                    # If we have time, we can do Or-Opt.
                    
                    # Let's try a limited Or-Opt(1)
                    if L == 1:
                        node = path[i]
                        # Remove node
                        if i > 0 and i+1 < N:
                            old_cost = overlap_mat[path[i-1]][node] + overlap_mat[node][path[i+1]]
                        elif i == 0 and i+1 < N:
                            old_cost = overlap_mat[node][path[i+1]]
                        elif i > 0 and i+1 == N:
                            old_cost = overlap_mat[path[i-1]][node]
                        else:
                            continue
                        
                        # Try inserting at k
                        best_k = -1
                        best_gain = 0
                        for k in range(N):
                            if k == i:
                                continue
                            # Calculate new cost
                            # Neighbors of k in the modified path
                            # This is complex to calculate on the fly without updating path.
                            # Skip Or-Opt for now to ensure we finish in time. 2-opt is usually sufficient for this scale.
                            pass
                            
    # Update best if improved
    if current_overlap > best_overlap:
        best_overlap = current_overlap
        best_path = path[:]
        
    # Build the word from the best path
    w = build_word_from_path(best_path)
    
    # 4. Greedy removal polish (from current program)
    # This might find a few more characters to remove if the structure allows
    all_perms = set(itertools.permutations(range(1, n+1)))
    total_perms = len(all_perms)
    
    # Count occurrences
    perm_counts = {}
    for i in range(len(w) - n + 1):
        p = tuple(w[i:i+n])
        if p in all_perms:
            perm_counts[p] = perm_counts.get(p, 0) + 1
            
    if len(perm_counts) < total_perms:
        # Should not happen, but return what we have
        return "".join(map(str, w))
        
    def is_removable(w, i, perm_counts, n, all_perms):
        for start in range(max(0, i-n+1), min(i+1, len(w)-n+1)):
            p = tuple(w[start:start+n])
            if p in all_perms:
                if perm_counts.get(p, 0) == 1:
                    return False
        return True
        
    def remove_char(w, i, perm_counts, n, all_perms):
        for start in range(max(0, i-n+1), min(i+1, len(w)-n+1)):
            p = tuple(w[start:start+n])
            if p in all_perms:
                perm_counts[p] -= 1
                if perm_counts[p] == 0:
                    del perm_counts[p]
        
        w.pop(i)
        
        for start in range(max(0, i-n+1), i):
            if start + n <= len(w):
                p = tuple(w[start:start+n])
                if p in all_perms:
                    perm_counts[p] = perm_counts.get(p, 0) + 1

    # Run a few passes of greedy removal
    for pass_num in range(5):
        if time.time() > deadline - 5:
            break
        improved = False
        i = 0
        while i < len(w) - n + 1:
            if time.time() > deadline - 5:
                break
            if is_removable(w, i, perm_counts, n, all_perms):
                remove_char(w, i, perm_counts, n, all_perms)
                improved = True
            else:
                i += 1
        if not improved:
            break
            
    return "".join(map(str, w))
