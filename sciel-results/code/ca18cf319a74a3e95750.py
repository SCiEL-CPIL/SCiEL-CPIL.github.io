"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

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

    if n == 7:
        # Build S_6 standard construction
        word = [1]
        for m in range(2, 7):
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

        # Extract all 720 permutations of 1..6 in order of appearance
        perms, seen = [], set()
        for i in range(len(word) - 5):
            window = tuple(word[i:i + 6])
            if len(set(window)) == 6 and window not in seen:
                seen.add(window)
                perms.append(window)

        N = len(perms)

        # Precompute overlap matrix using numpy for speed
        import numpy as np
        # Convert perms to list of tuples for quick access
        # overlap[i][j] = longest suffix of perms[i] that is prefix of perms[j]
        # We can compute this efficiently
        overlap_mat = np.zeros((N, N), dtype=np.int8)
        for i in range(N):
            pi = perms[i]
            for j in range(N):
                if i == j:
                    overlap_mat[i][j] = 6
                    continue
                pj = perms[j]
                ov = 0
                for k in range(5, -1, -1):
                    if pi[-k:] == pj[:k]:
                        ov = k
                        break
                overlap_mat[i][j] = ov

        # Helper to calculate total overlap
        def calc_total_overlap(p):
            total = 0
            for i in range(len(p) - 1):
                total += overlap_mat[p[i]][p[i + 1]]
            return total

        # Helper to build word from path
        def build_word_from_path(p):
            w = []
            for idx in p:
                perm = perms[idx]
                block = list(perm) + [7] + list(perm)
                overlap = 0
                min_len = min(len(w), len(block))
                for o in range(min_len, -1, -1):
                    if w[len(w) - o:] == block[:o]:
                        overlap = o
                        break
                w += block[overlap:]
            return "".join(map(str, w))

        # 1. Generate Naive Word (standard order)
        naive_path = list(range(N))
        naive_word = build_word_from_path(naive_path)
        best_word = naive_word
        best_overlap = calc_total_overlap(naive_path)
        best_path = naive_path[:]

        # 2. Greedy construction (nearest neighbor) with multiple random starts
        random.seed(42)
        for start_seed in range(10):
            random.seed(start_seed)
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

        # 3. Local improvement: 2-opt + 3-opt + Or-Opt
        path = best_path[:]
        current_overlap = best_overlap
        time_limit = deadline - 10

        def try_2opt(path, current_overlap, max_seg=100):
            """Try 2-opt moves, return (path, overlap, improved)"""
            N = len(path)
            for i in range(N - 1):
                if time.time() > time_limit:
                    return path, current_overlap, False
                for j in range(i + 1, min(i + max_seg, N)):
                    if j == N - 1:
                        old_ov = overlap_mat[path[i]][path[i + 1]]
                        new_ov = overlap_mat[path[i]][path[j]]
                        if new_ov > old_ov:
                            path[i + 1:j + 1] = reversed(path[i + 1:j + 1])
                            current_overlap += new_ov - old_ov
                            return path, current_overlap, True
                    else:
                        old_ov1 = overlap_mat[path[i]][path[i + 1]]
                        old_ov2 = overlap_mat[path[j]][path[j + 1]]
                        new_ov1 = overlap_mat[path[i]][path[j]]
                        new_ov2 = overlap_mat[path[i + 1]][path[j + 1]]
                        if new_ov1 + new_ov2 > old_ov1 + old_ov2:
                            path[i + 1:j + 1] = reversed(path[i + 1:j + 1])
                            current_overlap += (new_ov1 + new_ov2) - (old_ov1 + old_ov2)
                            return path, current_overlap, True
            return path, current_overlap, False

        def try_3opt(path, current_overlap, max_seg=50):
            """Try 3-opt moves (reverse two segments), return (path, overlap, improved)"""
            N = len(path)
            for i in range(N - 2):
                if time.time() > time_limit:
                    return path, current_overlap, False
                for j in range(i + 1, min(i + max_seg, N - 1)):
                    for k in range(j + 1, min(j + max_seg // 2, N)):
                        # Three segments: [0..i], [i+1..j], [j+1..k], [k+1..N]
                        # Option 1: reverse [i+1..j] and [j+1..k]
                        # Option 2: reverse [i+1..k] (this is 2-opt)
                        # Option 3: reverse [i+1..j], keep [j+1..k]
                        # Let's try: reverse segment [i+1..j], then reverse segment [j+1..k]
                        # This is equivalent to reversing [i+1..k] and then swapping the two parts
                        # Actually, let's try a different 3-opt: take two adjacent segments and reverse them
                        # Segment A = path[i+1:j+1], Segment B = path[j+1:k+1]
                        # New order: ... path[i] + B_reversed + A_reversed + path[k+1] ...
                        
                        # Old connections:
                        old_ov1 = overlap_mat[path[i]][path[i + 1]]
                        old_ov2 = overlap_mat[path[j]][path[j + 1]]
                        old_ov3 = overlap_mat[path[k]][path[k + 1]] if k < N - 1 else 0
                        
                        # New connections after reversing both segments:
                        # path[i] connects to reversed(B)[0] = path[k]
                        # reversed(B)[-1] = path[j+1] connects to reversed(A)[0] = path[j]
                        # reversed(A)[-1] = path[i+1] connects to path[k+1]
                        
                        new_ov1 = overlap_mat[path[i]][path[k]]
                        new_ov2 = overlap_mat[path[j + 1]][path[j]]
                        new_ov3 = overlap_mat[path[i + 1]][path[k + 1]] if k < N - 1 else 0
                        
                        if new_ov1 + new_ov2 + new_ov3 > old_ov1 + old_ov2 + old_ov3:
                            # Perform the move
                            segA = path[i + 1:j + 1]
                            segB = path[j + 1:k + 1]
                            path[i + 1:k + 1] = list(reversed(segB)) + list(reversed(segA))
                            current_overlap += (new_ov1 + new_ov2 + new_ov3) - (old_ov1 + old_ov2 + old_ov3)
                            return path, current_overlap, True
            return path, current_overlap, False

        def try_or_opt(path, current_overlap, max_dist=200):
            """Try Or-Opt moves (move single element), return (path, overlap, improved)"""
            N = len(path)
            for i in range(N):
                if time.time() > time_limit:
                    return path, current_overlap, False
                elem = path[i]
                
                # Old connections
                old_left = overlap_mat[path[i - 1]][elem] if i > 0 else 0
                old_right = overlap_mat[elem][path[i + 1]] if i < N - 1 else 0
                old_bridge = 0
                if i > 0 and i < N - 1:
                    old_bridge = overlap_mat[path[i - 1]][path[i + 1]]
                
                # Try inserting at various positions
                for offset in range(1, max_dist + 1):
                    # Insert before position i + offset (if i + offset < N)
                    j = i + offset
                    if j >= N:
                        break
                    
                    # New connections
                    new_left = overlap_mat[path[j - 1]][elem] if j > 0 else 0
                    new_right = overlap_mat[elem][path[j]]
                    new_bridge = old_bridge  # Bridge stays the same
                    
                    # But we need to account for the fact that after removal, positions shift
                    # Let's just compute the delta properly
                    # After removing elem from i, the element at i+1 shifts to i
                    # Then we insert elem at position j (which is now at j-1 if j > i)
                    
                    # Simplified: just try the move and check if it improves
                    # For speed, only consider promising moves
                    if new_left + new_right > old_left + old_right:
                        # Perform the move
                        path.pop(i)
                        if j > i:
                            j -= 1
                        path.insert(j, elem)
                        # Recompute overlap for accuracy
                        # But for speed, approximate the delta
                        current_overlap += (new_left + new_right) - (old_left + old_right)
                        return path, current_overlap, True
                    
                    # Also try inserting before position i - offset (if i - offset >= 0)
                    j = i - offset
                    if j < 0:
                        break
                    
                    new_left = overlap_mat[path[j - 1]][elem] if j > 0 else 0
                    new_right = overlap_mat[elem][path[j]]
                    
                    if new_left + new_right > old_left + old_right:
                        path.pop(i)
                        path.insert(j, elem)
                        current_overlap += (new_left + new_right) - (old_left + old_right)
                        return path, current_overlap, True
            return path, current_overlap, False

        # Iterative improvement
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Try 2-opt
            path, current_overlap, imp = try_2opt(path, current_overlap)
            if imp:
                improved = True
                if current_overlap > best_overlap:
                    best_overlap = current_overlap
                    best_path = path[:]
                continue
            
            # Try 3-opt
            path, current_overlap, imp = try_3opt(path, current_overlap)
            if imp:
                improved = True
                if current_overlap > best_overlap:
                    best_overlap = current_overlap
                    best_path = path[:]
                continue
            
            # Try Or-Opt
            path, current_overlap, imp = try_or_opt(path, current_overlap)
            if imp:
                improved = True
                if current_overlap > best_overlap:
                    best_overlap = current_overlap
                    best_path = path[:]

        # Build final word from best_path
        final_word = build_word_from_path(best_path)
        
        # Compare with naive
        if len(final_word) < len(naive_word):
            return final_word
        else:
            return naive_word

    # Fallback for n > 7
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

    # Fallback for n > 7
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
# EVOLVE-BLOCK-END
