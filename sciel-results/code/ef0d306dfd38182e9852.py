import itertools
import time

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    
    if n > 7:
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
    
    prev_perms = list(itertools.permutations(range(1, n)))
    num_blocks = len(prev_perms)
    
    blocks = []
    for p in prev_perms:
        b = list(p) + [n] + list(p)
        blocks.append(b)
    
    block_len = len(blocks[0])
    
    overlaps = [[0]*num_blocks for _ in range(num_blocks)]
    for i in range(num_blocks):
        bi = blocks[i]
        for j in range(num_blocks):
            bj = blocks[j]
            found = 0
            for k in range(block_len, -1, -1):
                if k == 0:
                    found = 0
                    break
                if bi[-k:] == bj[:k]:
                    found = k
                    break
            overlaps[i][j] = found
    
    # Greedy initial path
    current = 0
    unvisited = set(range(1, num_blocks))
    path = [0]
    while unvisited:
        best_next = -1
        best_overlap = -1
        for j in unvisited:
            if overlaps[current][j] > best_overlap:
                best_overlap = overlaps[current][j]
                best_next = j
        path.append(best_next)
        unvisited.remove(best_next)
        current = best_next
    
    # 2-Opt local search with prefix sums for O(1) delta
    m = len(path)
    
    def compute_prefixes(path):
        fwd_prefix = [0] * (m + 1)
        rev_prefix = [0] * (m + 1)
        for k in range(m - 1):
            fwd_prefix[k + 1] = fwd_prefix[k] + overlaps[path[k]][path[k + 1]]
            rev_prefix[k + 1] = rev_prefix[k] + overlaps[path[k + 1]][path[k]]
        return fwd_prefix, rev_prefix
    
    fwd_prefix, rev_prefix = compute_prefixes(path)
    
    improved = True
    max_passes = 30
    for _ in range(max_passes):
        improved = False
        for i in range(m - 1):
            for j in range(i + 2, m):
                old_internal = fwd_prefix[j] - fwd_prefix[i + 1]
                if j - 1 > i + 1:
                    new_internal = rev_prefix[j - 1] - rev_prefix[i + 1]
                else:
                    new_internal = 0
                
                old_boundary = overlaps[path[i]][path[i + 1]]
                if j < m - 1:
                    old_boundary += overlaps[path[j]][path[j + 1]]
                
                new_boundary = overlaps[path[i]][path[j]]
                if j < m - 1:
                    new_boundary += overlaps[path[i + 1]][path[j + 1]]
                
                delta = (new_boundary + new_internal) - (old_boundary + old_internal)
                
                if delta > 0:
                    path[i + 1:j + 1] = reversed(path[i + 1:j + 1])
                    fwd_prefix, rev_prefix = compute_prefixes(path)
                    improved = True
    
    final_word = list(blocks[path[0]])
    for k in range(1, m):
        ov = overlaps[path[k - 1]][path[k]]
        final_word += blocks[path[k]][ov:]
    
    return "".join(map(str, final_word))
