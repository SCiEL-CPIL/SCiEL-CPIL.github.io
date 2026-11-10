import math
import time
import random
import sys
import numpy as np

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s limit
    
    if n <= 1:
        return "1"
    
    # Generate standard superpermutation up to n-1
    word = [1]
    for m in range(2, n):
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
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    if time.time() > deadline - 15:
        return "".join(map(str, word))
    
    # Now optimize the last step (m=n)
    # Get perms of n-1 from word
    perms, seen = [], set()
    for i in range(len(word) - n + 2):
        window = tuple(word[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
            
    num_perms = len(perms)
    if num_perms == 0:
        return "".join(map(str, word))
    
    # Precompute overlap matrix for permutations of length n-1
    # overlaps[i][j] = max k such that perms[i][-k:] == perms[j][:k]
    # This determines the cost of appending block j after block i.
    # Cost = (2*(n-1) + 1) - overlaps[i][j]
    
    # Use numpy for speed
    # Convert perms to numpy array of shape (num_perms, n-1)
    perms_np = np.array(perms, dtype=np.int8)
    p_len = n - 1
    
    overlaps_np = np.zeros((num_perms, num_perms), dtype=np.int8)
    
    # For each pair (i, j), find max overlap
    # We can do this by checking all possible k from p_len-1 down to 0
    # But that's O(N^2 * p_len). N=720, p_len=6. 720^2 * 6 ~ 3M ops. Fast enough.
    
    # More efficient:
    # For each k from p_len-1 down to 1:
    #   Check if perms[i][-k:] == perms[j][:k]
    
    # Let's use a dictionary approach for prefixes/suffixes to speed up if needed,
    # but for N=720, direct comparison is likely fine in a loop if optimized.
    # However, Python loops over 720x720 are slow.
    # Let's use a vectorized approach for each k.
    
    for k in range(p_len - 1, 0, -1):
        # suffixes: perms_np[:, -k:]
        # prefixes: perms_np[:, :k]
        # We want overlaps_np[i, j] = k if suffixes[i] == prefixes[j]
        
        # Create a hash for each prefix and suffix
        # Since p_len is small, we can just use tuples in a dict for prefixes
        prefix_map = {}
        for j in range(num_perms):
            key = tuple(perms_np[j, :k].tolist())
            prefix_map[key] = j
        
        for i in range(num_perms):
            key = tuple(perms_np[i, -k:].tolist())
            if key in prefix_map:
                j = prefix_map[key]
                if i != j: # Overlap with self is not allowed in the path (usually)
                    overlaps_np[i, j] = k
                # If i==j, we don't care for the path, but technically overlap is k.
                # But we can't use the same block twice.
    
    # The objective is to maximize sum(overlaps[order[i]][order[i+1]])
    # Standard order: 0, 1, ..., N-1
    std_order = list(range(num_perms))
    
    # Calculate baseline score (sum of overlaps)
    def calc_score(order):
        score = 0
        for i in range(len(order) - 1):
            score += overlaps_np[order[i]][order[i+1]]
        return score
    
    best_order = std_order
    best_score = calc_score(best_order)
    
    # Simulated Annealing
    current_order = std_order[:]
    current_score = best_score
    
    T = 10.0
    T_min = 0.01
    alpha = 0.995
    
    # Precompute block length for final construction
    block_len = 2 * (n - 1) + 1
    
    def build_word_from_order(order):
        """Build superpermutation from a given order of permutation indices."""
        if not order:
            return []
        result = []
        # First block
        result.extend(perms[order[0]])
        result.append(n)
        result.extend(perms[order[0]])
        
        for idx in range(1, len(order)):
            prev_idx = order[idx-1]
            curr_idx = order[idx]
            
            # Get the overlap
            o = overlaps_np[prev_idx][curr_idx]
            
            # Append the rest of the current block
            # The current block is perms[curr_idx] + [n] + perms[curr_idx]
            # The overlap is the suffix of the previous block matching the prefix of the current block.
            # Previous block ends with perms[prev_idx].
            # Current block starts with perms[curr_idx].
            # Overlap o means the last o chars of prev match first o chars of curr.
            # So we append the remaining part of the current block.
            # Current block length is 2*(n-1) + 1.
            # We already have 'o' chars from the prefix of the current block (which are the suffix of the prev block).
            # So we append block[o:].
            
            curr_block = perms[curr_idx] + [n] + perms[curr_idx]
            result.extend(curr_block[o:])
            
        return result
    
    # Run SA
    while time.time() < deadline - 10:
        # Random move: swap two adjacent elements
        i = random.randint(0, len(current_order) - 2)
        current_order[i], current_order[i+1] = current_order[i+1], current_order[i]
        
        # Calculate delta
        # Old edges: (i-1, i), (i, i+1), (i+1, i+2)
        # New edges: (i-1, i+1), (i+1, i), (i, i+2)
        # We need to be careful with boundaries.
        
        # Simpler: just recalculate the whole score? N=720, so 719 additions.
        # 719 * 10000 iterations = 7M ops. Might be slow in Python loop.
        # Let's try to optimize delta calculation.
        
        # For now, let's just do a full recalculation if it's fast enough, or use numpy sum.
        # np.sum(overlaps_np[current_order[:-1], current_order[1:]])
        
        new_score = np.sum(overlaps_np[current_order[:-1], current_order[1:]]).item()
        
        delta = new_score - current_score
        
        if delta > 0:
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_order = current_order[:]
        else:
            # Accept with probability exp(-delta/T) -- note delta is negative, so -delta is positive
            # We want to maximize score, so "cost" is -score.
            # delta_cost = -new_score - (-current_score) = current_score - new_score = -delta
            # Prob = exp(-delta_cost / T) = exp(delta / T)
            if random.random() < np.exp(delta / T):
                current_score = new_score
        
        T *= alpha
        if T < T_min:
            T = T_min
            
    # Construct final word from best_order
    best_word = build_word_from_order(best_order)
    
    # Try to trim the end
    if time.time() < deadline - 2:
        candidate = list(best_word)
        # Check if we can remove the last few characters
        # The word must contain all n! permutations.
        # We can check validity by scanning.
        
        def is_valid(word_candidate):
            if len(word_candidate) < n:
                return False
            target = math.factorial(n)
            seen = set()
            for i in range(len(word_candidate) - n + 1):
                w = tuple(word_candidate[i:i+n])
                if len(set(w)) == n:
                    seen.add(w)
                    if len(seen) == target:
                        return True
            return len(seen) == target
        
        # Try removing 1 char at a time from the end
        # Since the standard construction ends with a permutation, the last n-1 chars are a permutation.
        # Removing 1 char might break the last permutation if it's only present there.
        # But usually, it's present earlier.
        
        while len(candidate) > n:
            if time.time() > deadline - 1:
                break
            test_candidate = candidate[:-1]
            if is_valid(test_candidate):
                candidate = test_candidate
            else:
                break
        best_word = candidate
    
    return "".join(map(str, best_word))
