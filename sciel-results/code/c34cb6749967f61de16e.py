import time
import random
import itertools
import numpy as np

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    
    start_time = time.time()
    deadline = start_time + 140.0

    perms = list(itertools.permutations(range(1, n+1)))
    num_perms = len(perms)
    
    # Precompute overlap cache
    overlap_cache = np.zeros((num_perms, num_perms), dtype=np.uint8)
    for i in range(num_perms):
        p1 = perms[i]
        for j in range(num_perms):
            p2 = perms[j]
            ov = 0
            # Check suffix of p1 vs prefix of p2
            for k in range(n-1, -1, -1):
                if p1[n-k:] == p2[:k]:
                    ov = k
                    break
            overlap_cache[i][j] = ov

    best_len = float('inf')
    best_seq = None

    def local_search(seq):
        current_len = n
        overlaps = [0] * (len(seq) - 1)
        for i in range(len(seq) - 1):
            overlaps[i] = overlap_cache[seq[i]][seq[i+1]]
            current_len += n - overlaps[i]
            
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(seq) - 1):
                A = seq[i]
                B = seq[i+1]
                
                old_ov_sum = 0
                new_ov_sum = 0
                
                if i > 0:
                    old_ov_sum += overlaps[i-1]
                    new_ov_sum += overlap_cache[seq[i-1]][B]
                
                old_ov_sum += overlaps[i]
                new_ov_sum += overlap_cache[B][A]
                
                if i < len(seq) - 2:
                    old_ov_sum += overlaps[i+1]
                    new_ov_sum += overlap_cache[A][seq[i+2]]
                
                delta_len = old_ov_sum - new_ov_sum
                
                if delta_len < 0:
                    seq[i], seq[i+1] = B, A
                    
                    if i > 0:
                        overlaps[i-1] = overlap_cache[seq[i-1]][B]
                    overlaps[i] = overlap_cache[B][A]
                    if i < len(seq) - 2:
                        overlaps[i+1] = overlap_cache[A][seq[i+2]]
                    
                    current_len += delta_len
                    improved = True
        
        return current_len

    for restart in range(200):
        if time.time() > deadline:
            break

        start_idx = random.randint(0, num_perms - 1)
        seq = [start_idx]
        used = [False] * num_perms
        used[start_idx] = True
        
        for _ in range(num_perms - 1):
            current_idx = seq[-1]
            best_score = -1
            candidates = []
            
            for i in range(num_perms):
                if used[i]:
                    continue
                ov = overlap_cache[current_idx][i]
                score = ov + random.random() * 0.5
                if score > best_score:
                    best_score = score
                    candidates = [i]
                elif score == best_score:
                    candidates.append(i)
            
            next_idx = random.choice(candidates)
            seq.append(next_idx)
            used[next_idx] = True

        seq = seq[:]
        seq_len = local_search(seq)
        
        if seq_len < best_len:
            best_len = seq_len
            best_seq = seq[:]

    if best_seq is None:
        # Fallback
        word = [1]
        for m in range(2, n + 1):
            perms_list, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms_list.append(window)
            new = []
            for p in perms_list:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    word = list(perms[best_seq[0]])
    for i in range(1, len(best_seq)):
        ov = overlap_cache[best_seq[i-1]][best_seq[i]]
        word.extend(perms[best_seq[i]][ov:])
    
    return "".join(map(str, word))
