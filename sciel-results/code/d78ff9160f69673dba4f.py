import time
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin for safety
    
    # 1. Standard recursive construction
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
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    if n < 5:
        return "".join(map(str, word))
    
    # 2. Extract permutation sequence from the superpermutation
    # Each consecutive n-window that is a valid permutation
    perm_sequence = []
    seen_perms = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i+n])
        if len(set(window)) == n and window not in seen_perms:
            seen_perms.add(window)
            perm_sequence.append(window)
    
    # We should have n! permutations
    target_count = 1
    for i in range(2, n + 1):
        target_count *= i
    
    if len(perm_sequence) != target_count:
        # Fallback to original word if extraction failed
        return "".join(map(str, word))
    
    # 3. Build a mapping from permutation to index using Lehmer code
    def perm_to_idx(p, n):
        """Convert a permutation to its Lehmer code index"""
        idx = 0
        used = [False] * (n + 1)
        for i in range(n):
            smaller = 0
            for j in range(1, p[i]):
                if not used[j]:
                    smaller += 1
            idx += smaller * fact[n - 1 - i]
            used[p[i]] = True
        return idx
    
    # Precompute factorials
    fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = fact[i-1] * i
    
    def perm_to_idx_fast(p, n):
        idx = 0
        used = [False] * (n + 1)
        for i in range(n):
            smaller = 0
            for j in range(1, p[i]):
                if not used[j]:
                    smaller += 1
            idx += smaller * fact[n - 1 - i]
            used[p[i]] = True
        return idx
    
    # 4. Compute overlap between two consecutive permutations
    def overlap_len(p1, p2, n):
        """Maximum k such that p1[n-k:] == p2[:k]"""
        max_k = min(n, n)
        for k in range(max_k, -1, -1):
            if k == 0:
                return 0
            if p1[n-k:] == p2[:k]:
                return k
        return 0
    
    # 5. Build the word from a permutation sequence
    def build_word_from_sequence(seq, n):
        if not seq:
            return ""
        word = list(seq[0])
        for i in range(1, len(seq)):
            p1 = seq[i-1]
            p2 = seq[i]
            ov = overlap_len(p1, p2, n)
            word.extend(p2[ov:])
        return word
    
    # 6. Fast validity check using array-based counting
    # Map each permutation to an index 0..n!-1
    perm_indices = {}
    for p in perm_sequence:
        idx = perm_to_idx_fast(p, n)
        perm_indices[p] = idx
    
    # Create inverse mapping
    idx_to_perm = {}
    for p, idx in perm_indices.items():
        idx_to_perm[idx] = p
    
    def is_valid_sequence(seq, n):
        """Check if the sequence contains all n! permutations"""
        if len(seq) != target_count:
            return False
        seen = set()
        for p in seq:
            seen.add(p)
        return len(seen) == target_count
    
    def compute_total_length(seq, n):
        """Compute total length of word built from sequence"""
        if not seq:
            return 0
        length = n
        for i in range(1, len(seq)):
            ov = overlap_len(seq[i-1], seq[i], n)
            length += (n - ov)
        return length
    
    # 7. Optimize the permutation ordering
    # Current sequence from standard construction
    current_seq = list(perm_sequence)
    current_len = compute_total_length(current_seq, n)
    
    # Strategy: Try to improve the ordering by swapping adjacent permutations
    # or using a 2-opt like approach
    
    improved = True
    iterations = 0
    max_iterations = 50000
    
    while improved and time.time() < deadline and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # Randomize the order of positions to try
        positions = list(range(len(current_seq) - 1))
        random.shuffle(positions)
        
        for pos in positions:
            if time.time() > deadline:
                break
            
            # Try swapping current_seq[pos] and current_seq[pos+1]
            if pos + 1 >= len(current_seq):
                continue
            
            # Compute current cost around this position
            old_cost = 0
            if pos > 0:
                old_cost += n - overlap_len(current_seq[pos-1], current_seq[pos], n)
            if pos + 1 < len(current_seq):
                old_cost += n - overlap_len(current_seq[pos], current_seq[pos+1], n)
            
            # Try the swap
            current_seq[pos], current_seq[pos+1] = current_seq[pos+1], current_seq[pos]
            
            new_cost = 0
            if pos > 0:
                new_cost += n - overlap_len(current_seq[pos-1], current_seq[pos], n)
            if pos + 1 < len(current_seq):
                new_cost += n - overlap_len(current_seq[pos], current_seq[pos+1], n)
            
            # Accept if better or equal
            if new_cost <= old_cost:
                current_len = compute_total_length(current_seq, n)
                if new_cost < old_cost:
                    improved = True
            else:
                # Revert
                current_seq[pos], current_seq[pos+1] = current_seq[pos+1], current_seq[pos]
        
        # Also try 2-opt: reverse a segment
        if time.time() < deadline:
            for _ in range(10):  # Try a few 2-opt moves
                if time.time() > deadline:
                    break
                i = random.randint(0, len(current_seq) - 2)
                j = random.randint(i + 1, len(current_seq) - 1)
                
                old_cost = 0
                if i > 0:
                    old_cost += n - overlap_len(current_seq[i-1], current_seq[i], n)
                if j + 1 < len(current_seq):
                    old_cost += n - overlap_len(current_seq[j], current_seq[j+1], n)
                
                # Reverse segment
                segment = current_seq[i:j+1]
                segment.reverse()
                current_seq[i:j+1] = segment
                
                new_cost = 0
                if i > 0:
                    new_cost += n - overlap_len(current_seq[i-1], current_seq[i], n)
                if j + 1 < len(current_seq):
                    new_cost += n - overlap_len(current_seq[j], current_seq[j+1], n)
                
                if new_cost <= old_cost:
                    current_len = compute_total_length(current_seq, n)
                    if new_cost < old_cost:
                        improved = True
                else:
                    # Revert
                    segment = current_seq[i:j+1]
                    segment.reverse()
                    current_seq[i:j+1] = segment
    
    # 8. Build the final word from the optimized sequence
    final_word = build_word_from_sequence(current_seq, n)
    
    # 9. Final greedy removal pass on the character string
    # Try to remove redundant symbols
    if time.time() < deadline - 10:
        final_word_list = list(final_word)
        improved2 = True
        while improved2 and time.time() < deadline - 5:
            improved2 = False
            indices = list(range(len(final_word_list)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline - 5:
                    break
                
                # Try removing symbol at position i
                test_word = final_word_list[:i] + final_word_list[i+1:]
                
                # Quick check: does it still have enough length?
                if len(test_word) < target_count + n - 1:
                    continue
                
                # Check if all permutations are still present
                seen = set()
                for j in range(len(test_word) - n + 1):
                    window = tuple(test_word[j:j+n])
                    if len(set(window)) == n:
                        seen.add(window)
                
                if len(seen) == target_count:
                    final_word_list = test_word
                    improved2 = True
                    break
        
        final_word = final_word_list
    
    return "".join(map(str, final_word)) if isinstance(final_word, list) else final_word
