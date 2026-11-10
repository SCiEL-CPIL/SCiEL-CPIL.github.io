"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import permutations

def _overlap(a, b, max_len=None):
    """Maximum overlap of suffix of a with prefix of b."""
    if max_len is None:
        max_len = min(len(a), len(b))
    for o in range(max_len, 0, -1):
        if a[-o:] == b[:o]:
            return o
    return 0

def _greedy_tsp(blocks, start_idx, time_limit=None):
    """Nearest-neighbor TSP heuristic on blocks with precomputed overlaps."""
    n = len(blocks)
    if n == 0:
        return ""
    
    # Precompute overlap matrix
    ov_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                ov_matrix[i][j] = _overlap(blocks[i], blocks[j])
    
    visited = [False] * n
    current = start_idx
    visited[current] = True
    path = [current]
    
    for _ in range(n - 1):
        best_next = -1
        best_ov = -1
        for j in range(n):
            if not visited[j] and ov_matrix[current][j] > best_ov:
                best_ov = ov_matrix[current][j]
                best_next = j
        if best_next == -1:
            break
        visited[best_next] = True
        current = best_next
        path.append(best_next)
    
    # Build word
    word = blocks[path[0]]
    for idx in path[1:]:
        ov = _overlap(word, blocks[idx])
        word += blocks[idx][ov:]
    return word

def _verify_superperm(word, m):
    """Check if word contains all permutations of 1..m."""
    if m <= 1:
        return True
    all_perms = set(permutations(range(1, m + 1)))
    found = set()
    for i in range(len(word) - m + 1):
        w = tuple(int(c) for c in word[i:i + m])
        if len(set(w)) == m:
            found.add(w)
    return found == all_perms

def construct(n):
    start_time = time.time()
    deadline = start_time + 110

    word = "1"
    for m in range(2, n + 1):
        if time.time() > deadline:
            break

        perms, seen = [], set()
        for i in range(len(word) - (m - 1) + 1):
            window = word[i:i + m - 1]
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)

        if m < n:
            # For intermediate steps, use simple greedy
            new = ""
            for p in perms:
                block = p + str(m) + p
                ov = _overlap(new, block)
                new += block[ov:]
            word = new
        else:
            # For the final step, use TSP-based greedy with multiple starting points
            blocks = [p + str(m) + p for p in perms]
            num_blocks = len(blocks)
            
            # Try multiple starting positions
            best_word = None
            best_len = float('inf')
            
            # Try several starting indices
            start_indices = list(range(min(num_blocks, 20)))
            # Also try reversed
            for si in start_indices:
                if time.time() > deadline - 10:
                    break
                w = _greedy_tsp(blocks, si)
                if len(w) < best_len:
                    best_len = len(w)
                    best_word = w
            
            # Try reversed blocks
            rev_blocks = [p + str(m) + p for p in reversed(perms)]
            for si in range(min(len(rev_blocks), 10)):
                if time.time() > deadline - 5:
                    break
                w = _greedy_tsp(rev_blocks, si)
                if len(w) < best_len:
                    best_len = len(w)
                    best_word = w
            
            if best_word is None:
                # Fallback: simple greedy
                best_word = ""
                for p in perms:
                    block = p + str(m) + p
                    ov = _overlap(best_word, block)
                    best_word += block[ov:]
            
            word = best_word

    return word
# EVOLVE-BLOCK-END
