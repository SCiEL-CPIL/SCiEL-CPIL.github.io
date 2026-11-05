import time
import random
import math
import numpy as np
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    if n <= 6:
        # Standard recursive construction is optimal or near-optimal for small n
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
                # Find max overlap
                overlap = 0
                min_len = min(len(new), len(block))
                for o in range(min_len, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    if n == 7:
        # Generate all permutations of 1..6
        perms = list(permutations(range(1, 7)))
        N = len(perms) # 720
        
        # Precompute overlap matrix
        # overlap[i][j] = longest suffix of perms[i] matching prefix of perms[j]
        # Since perms are permutations of 1..6, max overlap is 5.
        overlap_mat = np.zeros((N, N), dtype=np.int8)
        
        # Optimization: Group by prefix
        # For k from 1 to 5:
        #   For each permutation i, check if suffix of length k matches prefix of j.
        
        # Faster: 
        # For each k in 1..5:
        #   Create dict: prefix of length k -> list of indices j
        #   For each i:
        #       suffix = perms[i][-k:]
        #       If suffix in dict:
        #           For each j in dict[suffix]:
        #               overlap_mat[i][j] = k # We want max k, so iterate k descending?
        #               Actually, we want the max k. 
        #               If we iterate k from 1 to 5, we overwrite with larger k.
        #               But we want the max overlap.
        #               So if we find a match for k=5, that's better than k=4.
        #               We can just compute the exact max overlap for each pair.
        
        # Let's compute exact max overlap for all pairs using vectorization or smart loops.
        # N=720, N^2 = 518,400 pairs.
        # For each pair, max overlap is small (<=5).
        
        # Efficient method:
        # For k in 1..5:
        #   Map prefix_k -> list of indices
        #   For each i:
        #      suffix_k = perms[i][-k:]
        #      if suffix_k in map:
        #          for j in map[suffix_k]:
        #             overlap_mat[i][j] = k
        
        # This sets overlap to the largest k found.
        
        for k in range(1, 6):
            prefix_map = {}
            for j in range(N):
                key = tuple(perms[j][:k])
                if key not in prefix_map:
                    prefix_map[key] = []
                prefix_map[key].append(j)
            
            for i in range(N):
                key = tuple(perms[i][6-k:])
                if key in prefix_map:
                    for j in prefix_map[key]:
                        overlap_mat[i][j] = k

        # Objective: Maximize sum of overlaps in the path
        # Start with a random or greedy path
        random.seed(42)
        
        # Greedy initialization
        path = [0]
        unvisited = set(range(1, N))
        while unvisited:
            last = path[-1]
            best_next = -1
            best_ov = -1
            # To speed up, we might just pick the first available with max overlap
            # But checking all unvisited is O(N) per step, total O(N^2) = 500k ops. Fast.
            for j in unvisited:
                ov = overlap_mat[last][j]
                if ov > best_ov:
                    best_ov = ov
                    best_next = j
            path.append(best_next)
            unvisited.remove(best_next)
            
        current_score = sum(overlap_mat[path[i]][path[i+1]] for i in range(N-1))
        best_path = path[:]
        best_score = current_score
        
        # 2-Opt improvement
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for i in range(N - 1):
                if time.time() > deadline - 10:
                    break
                for j in range(i + 1, N):
                    if time.time() > deadline - 10:
                        break
                    # Calculate delta for reversing segment i..j
                    # This is complex. Simpler: Swap i and j? No, 2-opt reverses.
                    # Let's stick to simple swap first (Or-Opt style) or standard 2-opt.
                    # Standard 2-opt on path:
                    # Reverse path[i..j]
                    # New edges: (i-1, j), (j, i) ... wait, if we reverse i..j, 
                    # the segment becomes ... p_{i-1}, p_j, p_{j-1}, ..., p_i, p_{j+1} ...
                    # Old edges: (i-1, i), (i, i+1), ..., (j-1, j), (j, j+1)
                    # New edges: (i-1, j), (j, j-1), ..., (i+1, i), (i, j+1)
                    
                    p_i = path[i]
                    p_j = path[j]
                    
                    old_ov = 0
                    new_ov = 0
                    
                    if i > 0:
                        left = path[i-1]
                        old_ov += overlap_mat[left][p_i]
                        new_ov += overlap_mat[left][p_j]
                    
                    if j < N - 1:
                        right = path[j+1]
                        old_ov += overlap_mat[p_j][right]
                        new_ov += overlap_mat[p_i][right]
                    
                    # Internal edges
                    # Old: sum(overlap[path[k]][path[k+1]] for k in i..j-1)
                    # New: sum(overlap[path[k+1]][path[k]] for k in i..j-1) i.e. reversed
                    # We can precompute prefix sums of overlaps to make this O(1)?
                    # Or just calculate if segment is small? No, segment can be large.
                    
                    # Let's just do simple swaps for now, it's safer and often sufficient for initial improvement.
                    # Swap path[i] and path[j]
                    if j == i + 1:
                        # Adjacent swap
                        if i > 0:
                            left = path[i-1]
                            old_ov += overlap_mat[left][p_i]
                            new_ov += overlap_mat[left][p_j]
                        old_ov += overlap_mat[p_i][p_j]
                        new_ov += overlap_mat[p_j][p_i]
                        if j < N - 1:
                            right = path[j+1]
                            old_ov += overlap_mat[p_j][right]
                            new_ov += overlap_mat[p_i][right]
                    else:
                        # Non-adjacent swap
                        if i > 0:
                            left = path[i-1]
                            old_ov += overlap_mat[left][p_i]
                            new_ov += overlap_mat[left][p_j]
                        if i < N - 1:
                            right = path[i+1]
                            old_ov += overlap_mat[p_i][right]
                            new_ov += overlap_mat[p_j][right]
                        
                        if j > 0:
                            left = path[j-1]
                            old_ov += overlap_mat[left][p_j]
                            new_ov += overlap_mat[left][p_i]
                        if j < N - 1:
                            right = path[j+1]
                            old_ov += overlap_mat[p_j][right]
                            new_ov += overlap_mat[p_i][right]
                    
                    delta = new_ov - old_ov
                    if delta > 0:
                        path[i], path[j] = path[j], path[i]
                        current_score += delta
                        if current_score > best_score:
                            best_score = current_score
                            best_path = path[:]
                        improved = True

        # Build final word
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

        final_word = build_word_from_path(best_path)
        return final_word

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
