# EVOLVE-BLOCK-START
import time
import sys
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    if n <= 3:
        # Standard construction is optimal for n<=3
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
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
    
    # Generate all permutations
    all_perms = list(permutations(range(1, n + 1)))
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}
    N = len(all_perms)
    
    # Precompute overlap matrix (sparse representation)
    # overlap[i][j] = max k such that all_perms[i][-k:] == all_perms[j][:k]
    # For memory efficiency, we'll compute on the fly or use a compact representation
    
    def get_overlap(a, b):
        # a and b are tuples of length n
        # Return max k in 1..n-1 such that a[n-k:] == b[:k]
        # Since a != b, k < n
        # Quick check: last symbol of a must equal first symbol of b for k>=1
        for k in range(n - 1, 0, -1):
            if a[n-k:] == b[:k]:
                return k
        return 0
    
    # Generate standard construction sequence
    def get_standard_sequence(n):
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
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        
        # Extract sequence
        seq = []
        seen_perms = set()
        for i in range(len(word) - n + 1):
            p = tuple(word[i:i+n])
            if p not in seen_perms:
                seen_perms.add(p)
                seq.append(p)
        return seq
    
    seq = get_standard_sequence(n)
    
    # Convert to indices for efficiency
    seq_idx = [perm_to_idx[p] for p in seq]
    
    # Precompute overlap for adjacent pairs in current sequence
    # and store in a list for fast access
    def total_overlap_idx(seq_idx):
        s = 0
        for i in range(len(seq_idx) - 1):
            s += get_overlap(all_perms[seq_idx[i]], all_perms[seq_idx[i+1]])
        return s
    
    current_seq_idx = seq_idx[:]
    current_score = total_overlap_idx(current_seq_idx)
    
    # 2-opt search with small segment reversals
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for _ in range(20000):
            if time.time() > deadline - 10:
                break
            i = random.randint(0, len(current_seq_idx) - 2)
            j = random.randint(i + 1, len(current_seq_idx) - 1)
            
            if j - i > 30:
                continue
                
            old_score = 0
            new_score = 0
            
            # Left boundary
            if i > 0:
                old_score += get_overlap(all_perms[current_seq_idx[i-1]], all_perms[current_seq_idx[i]])
                new_score += get_overlap(all_perms[current_seq_idx[i-1]], all_perms[current_seq_idx[j]])
            
            # Right boundary
            if j < len(current_seq_idx) - 1:
                old_score += get_overlap(all_perms[current_seq_idx[j]], all_perms[current_seq_idx[j+1]])
                new_score += get_overlap(all_perms[current_seq_idx[i]], all_perms[current_seq_idx[j+1]])
            
            # Internal
            for k in range(i, j):
                old_score += get_overlap(all_perms[current_seq_idx[k]], all_perms[current_seq_idx[k+1]])
                new_score += get_overlap(all_perms[current_seq_idx[k+1]], all_perms[current_seq_idx[k]])
            
            delta = new_score - old_score
            if delta > 0:
                current_seq_idx[i:j+1] = current_seq_idx[i:j+1][::-1]
                current_score += delta
                improved = True
    
    # Reconstruct word from sequence
    word = list(all_perms[current_seq_idx[0]])
    for i in range(1, len(current_seq_idx)):
        prev = all_perms[current_seq_idx[i-1]]
        curr = all_perms[current_seq_idx[i]]
        k = get_overlap(prev, curr)
        word.extend(curr[k:])
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
