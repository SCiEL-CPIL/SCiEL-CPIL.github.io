import time
import random
from itertools import permutations

def construct(n):
    deadline = time.time() + 130

    # Step 1: Classic recursive construction
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
            best_overlap = 0
            max_check = min(len(new), len(block))
            for o in range(max_check, 0, -1):
                if new[-o:] == block[:o]:
                    best_overlap = o
                    break
            new += block[best_overlap:]
        word = new

    if n <= 6:
        return "".join(map(str, word))

    # Step 2: Extract permutation sequence
    all_perms = list(permutations(range(1, n + 1)))
    perm_to_id = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)

    perm_seq = []
    seen_in_seq = set()
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i + n])
        if p in perm_to_id:
            pid = perm_to_id[p]
            if pid not in seen_in_seq:
                seen_in_seq.add(pid)
                perm_seq.append(pid)

    if len(perm_seq) != num_perms:
        return "".join(map(str, word))

    # Precompute overlap matrix using NumPy for speed
    import numpy as np
    
    # Convert permutations to arrays
    perm_array = np.array(all_perms, dtype=np.int8)  # shape (5040, 7)
    
    # Compute overlap matrix: overlap[i,j] = max k such that perm[i][-k:] == perm[j][:k]
    # For n=7, max overlap is 6
    # We can vectorize this
    
    overlap_matrix = np.zeros((num_perms, num_perms), dtype=np.int8)
    
    for k in range(1, n):  # k = 1 to 6
        # Check if perm[i][n-k:] == perm[j][:k]
        # perm[i][n-k:] has shape (5040, k)
        # perm[j][:k] has shape (5040, k)
        # We need element-wise comparison
        left = perm_array[:, n-k:]  # (5040, k)
        right = perm_array[:, :k]   # (5040, k)
        
        # For each pair (i,j), check if left[i] == right[j]
        # This is a 5040x5040xk comparison
        # Use broadcasting: left[:, None, :] == right[None, :, :]
        # Shape: (5040, 1, k) == (1, 5040, k) -> (5040, 5040, k)
        # Then all along last axis -> (5040, 5040)
        
        match = np.all(left[:, None, :] == right[None, :, :], axis=2)
        overlap_matrix[match] = k
    
    # Now overlap_matrix[i,j] = max k such that perm[i][-k:] == perm[j][:k]
    # But we set it in increasing k order, so the last set value is the max.
    # Actually, we want the MAX k, so we should set in decreasing order or take max.
    # Let me redo: for each k from 6 down to 1, set overlap where match and not already set
    
    overlap_matrix = np.zeros((num_perms, num_perms), dtype=np.int8)
    for k in range(n - 1, 0, -1):  # 6, 5, 4, 3, 2, 1
        left = perm_array[:, n-k:]
        right = perm_array[:, :k]
        match = np.all(left[:, None, :] == right[None, :, :], axis=2)
        # Only set where overlap_matrix is still 0
        mask = (overlap_matrix == 0) & match
        overlap_matrix[mask] = k
    
    def sequence_length_from_matrix(seq, ov_matrix):
        total = n
        for i in range(len(seq) - 1):
            total += (n - ov_matrix[seq[i], seq[i + 1]])
        return total

    best_seq = perm_seq[:]
    best_len = sequence_length_from_matrix(best_seq, overlap_matrix)

    # Simulated annealing
    random.seed(42)
    current_seq = best_seq[:]
    current_len = best_len
    T = 10.0
    cooling = 0.9999
    min_T = 0.01
    
    start_time = time.time()
    max_time = deadline - start_time - 5  # Leave 5s for reconstruction
    
    while T > min_T and (time.time() - start_time) < max_time:
        # Choose random move: 2-opt reversal
        i = random.randint(0, len(current_seq) - 2)
        j = random.randint(i + 1, len(current_seq) - 1)
        
        # Compute delta for reversing [i, j]
        a_idx = i - 1 if i > 0 else -1
        c_idx = j + 1 if j + 1 < len(current_seq) else -1
        
        old_ov1 = overlap_matrix[current_seq[i], current_seq[i + 1]] if i + 1 <= j else 0
        # Actually, for reversal of [i, j]:
        # Old: ... A, S[i], S[i+1], ..., S[j], C ...
        # New: ... A, S[j], S[j-1], ..., S[i], C ...
        # Boundary overlaps: (A, S[i]) -> (A, S[j]) and (S[j], C) -> (S[i], C)
        # Internal overlaps are reversed but sum is the same? No, overlap is not symmetric.
        
        # Let me compute properly:
        # Old total = sum of (n - ov[S[k], S[k+1]]) for k in [i-1, j]
        #   where S[-1] is handled separately
        # Actually, let's just compute the delta:
        
        # Old overlaps affected: (S[i-1], S[i]) if i>0, (S[i], S[i+1]), ..., (S[j-1], S[j]), (S[j], S[j+1]) if j+1 < len
        # New overlaps affected: (S[i-1], S[j]) if i>0, (S[j], S[j-1]), ..., (S[i+1], S[i]), (S[i], S[j+1]) if j+1 < len
        
        delta = 0
        if i > 0:
            delta += (n - overlap_matrix[current_seq[i-1], current_seq[j]])
            delta -= (n - overlap_matrix[current_seq[i-1], current_seq[i]])
        # Internal: old has (S[i],S[i+1]),...,(S[j-1],S[j])
        # new has (S[j],S[j-1]),...,(S[i+1],S[i])
        for k in range(i, j):
            delta += (n - overlap_matrix[current_seq[k+1], current_seq[k]])
            delta -= (n - overlap_matrix[current_seq[k], current_seq[k+1]])
        if j + 1 < len(current_seq):
            delta += (n - overlap_matrix[current_seq[i], current_seq[j+1]])
            delta -= (n - overlap_matrix[current_seq[j], current_seq[j+1]])
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_seq[i:j+1] = current_seq[i:j+1][::-1]
            current_len += delta
            
            if current_len < best_len:
                best_len = current_len
                best_seq = current_seq[:]
        
        T *= cooling
    
    # Step 5: Reconstruct word
    def reconstruct_word(seq):
        if not seq:
            return ""
        result = list(all_perms[seq[0]])
        for i in range(1, len(seq)):
            ov = overlap_matrix[seq[i-1], seq[i]]
            result.extend(all_perms[seq[i]][ov:])
        return result

    best_word = reconstruct_word(best_seq)
    
    # Verify
    perm_set = set()
    for i in range(len(best_word) - n + 1):
        p = tuple(best_word[i:i + n])
        if len(set(p)) == n:
            perm_set.add(p)
    
    if len(perm_set) == num_perms:
        return "".join(map(str, best_word))
    else:
        return "".join(map(str, word))
