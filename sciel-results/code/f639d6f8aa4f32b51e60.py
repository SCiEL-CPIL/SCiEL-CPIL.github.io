import itertools
import random

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
                overlap = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
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
        bi_str = ''.join(map(str, blocks[i]))
        for j in range(num_blocks):
            bj_str = ''.join(map(str, blocks[j]))
            found = 0
            for k in range(block_len - 1, 0, -1):
                if bi_str[-k:] == bj_str[:k]:
                    found = k
                    break
            overlaps[i][j] = found

    def calc_total_overlap(path):
        total = 0
        for k in range(len(path) - 1):
            total += overlaps[path[k]][path[k+1]]
        return total

    best_path = None
    best_overlap_sum = -1
    
    num_restarts = 15
    for restart in range(num_restarts):
        start = random.randint(0, num_blocks - 1)
        current = start
        unvisited = set(range(num_blocks))
        unvisited.remove(start)
        path = [start]
        
        while unvisited:
            best_next = -1
            best_ov = -1
            for j in unvisited:
                ov = overlaps[current][j]
                if ov > best_ov:
                    best_ov = ov
                    best_next = j
            path.append(best_next)
            unvisited.remove(best_next)
            current = best_next
        
        ov_sum = calc_total_overlap(path)
        if ov_sum > best_overlap_sum:
            best_overlap_sum = ov_sum
            best_path = path[:]
    
    improved = True
    max_passes = 5
    for _ in range(max_passes):
        improved = False
        n_path = len(best_path)
        for i in range(n_path - 1):
            for j in range(i + 2, n_path):
                A = best_path[i]
                B = best_path[i+1]
                C = best_path[j]
                D_idx = j + 1
                D = best_path[D_idx] if D_idx < n_path else None
                
                old_ov = overlaps[A][B]
                for k in range(i + 1, j):
                    old_ov += overlaps[best_path[k]][best_path[k+1]]
                if D is not None:
                    old_ov += overlaps[C][D]
                
                new_ov = overlaps[A][C]
                for k in range(j - 1, i, -1):
                    new_ov += overlaps[best_path[k]][best_path[k-1]]
                if D is not None:
                    new_ov += overlaps[B][D]
                
                if new_ov > old_ov:
                    best_path[i+1:j+1] = best_path[i+1:j+1][::-1]
                    improved = True
        if not improved:
            break
    
    final_word = list(blocks[best_path[0]])
    for k in range(1, len(best_path)):
        curr_block = blocks[best_path[k]]
        ov = overlaps[best_path[k-1]][best_path[k]]
        final_word += curr_block[ov:]
        
    return "".join(map(str, final_word))
