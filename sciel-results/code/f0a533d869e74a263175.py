# EVOLVE-BLOCK-START
import time
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    if n <= 6:
        # Standard construction is optimal for n <= 6
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
        # Generate S_6 standard
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
        
        # Extract perms of 1..6 in the order they appear in S_6
        perms, seen = [], set()
        for i in range(len(word) - 5):
            window = tuple(word[i:i + 6])
            if len(set(window)) == 6 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        N = len(perms)
        
        # Precompute overlap matrix
        # ov[i][j] = overlap(perms[i], perms[j])
        ov = [[0]*N for _ in range(N)]
        for i in range(N):
            p = perms[i]
            for j in range(N):
                if i == j: continue
                q = perms[j]
                # Max overlap 5
                best = 0
                for k in range(5, 0, -1):
                    if p[-k:] == q[:k]:
                        best = k
                        break
                ov[i][j] = best
        
        # Start with standard order (identity)
        path = list(range(N))
        
        # Calculate initial total overlap
        def calc_total_overlap(path):
            total = 0
            for i in range(len(path) - 1):
                total += ov[path[i]][path[i+1]]
            return total
        
        current_overlap = calc_total_overlap(path)
        
        # Local Search: 2-opt and Or-Opt (move 1 or 2 elements)
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            
            # 2-opt
            for i in range(N - 1):
                if time.time() > deadline - 10: break
                for j in range(i + 1, N):
                    # Reverse path[i+1..j]
                    # Old edges: (i, i+1) and (j, j+1) [if j < N-1]
                    # New edges: (i, j) and (i+1, j+1) [if j < N-1]
                    
                    if j == N - 1:
                        old_ov = ov[path[i]][path[i+1]]
                        new_ov = ov[path[i]][path[j]]
                        if new_ov > old_ov:
                            path[i+1:j+1] = reversed(path[i+1:j+1])
                            current_overlap += new_ov - old_ov
                            improved = True
                            break
                    else:
                        old_ov = ov[path[i]][path[i+1]] + ov[path[j]][path[j+1]]
                        new_ov = ov[path[i]][path[j]] + ov[path[i+1]][path[j+1]]
                        if new_ov > old_ov:
                            path[i+1:j+1] = reversed(path[i+1:j+1])
                            current_overlap += new_ov - old_ov
                            improved = True
                            break
                if improved:
                    break
            
            # Or-Opt (move 1 or 2 elements)
            if not improved:
                for i in range(N):
                    if time.time() > deadline - 10: break
                    for length in [1, 2]:
                        if i + length > N: continue
                        block = path[i:i+length]
                        # Remove block
                        path_no_block = path[:i] + path[i+length:]
                        # Try inserting at all positions
                        best_pos = -1
                        best_gain = 0
                        for k in range(len(path_no_block) - 1):
                            # Insert between k and k+1
                            # Old edges around gap: (k-1, k) and (k, k+1) ? No, we removed block.
                            # Let's just calculate gain directly.
                            # This is complex to do with indices if we modify list.
                            # Simpler: just try all insertions and check gain.
                            # To avoid O(N^2) list manipulations, we can be smarter.
                            # But N=720, so O(N^2) is ~500k.
                            # If we do this for every i, it's O(N^3) which is too slow.
                            # We will skip Or-Opt if it's too slow, or optimize it.
                            # For now, let's stick to 2-opt which is O(N^2) per pass.
                            # 2-opt might be sufficient to get close to 5905.
                            pass
                        
        # Construct S_7 using the optimized path
        new_word = []
        for idx in path:
            p = perms[idx]
            block = list(p) + [7] + list(p)
            overlap = 0
            min_len = min(len(new_word), len(block))
            for o in range(min_len, -1, -1):
                if new_word[len(new_word)-o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
            
        return "".join(map(str, new_word))
    
    # Fallback for n > 7 (should not happen per task, but good practice)
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
