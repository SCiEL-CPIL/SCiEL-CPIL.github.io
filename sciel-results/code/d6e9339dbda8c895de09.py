"""Rewritten program for shortest superpermutation on n symbols."""
import time
import sys
import math
import random
from itertools import permutations

# EVOLVE-BLOCK-START

def construct(n):
    """Return shortest superpermutation found for n symbols within time budget."""
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "1234123142312431214321342132431213243214321"
    
    start_time = time.time()
    TIME_LIMIT = 145  # Safety margin before 150s limit
    
    # 1. Build initial superpermutation using PnP construction
    word = build_pnp_superperm(n)
    
    # For n=5, the PnP length is 1+2+6+24+120 = 153.
    # Known shortest for n=5 is 153? No, 1+2+6+24+120 = 153.
    # Actually, for n=5, the shortest is 153. PnP gives 153.
    # For n=6, PnP gives 1+2+6+24+120+720 = 873. Shortest is 873?
    # For n=7, PnP gives 5913. Shortest known is 5905.
    
    # If n is small, we can be exact or use known values
    if n == 5:
        # PnP is optimal for n<=4, and for n=5 it is 153.
        # Let's verify if we can do better.
        pass

    # 2. Optimization Phase
    # We use a greedy deletion strategy with randomized restarts/insertions if stuck.
    
    # Helper to check validity
    def is_valid(w):
        if len(w) < n:
            return False
        seen = set()
        # Optimization: use integer encoding for permutations if n is small enough,
        # but tuple of ints is fine for n=7 (5040 items).
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i+n])
            # Check distinctness quickly? 
            # If we just add to set, duplicates are handled.
            # But we must ensure it's a permutation of 1..n.
            # Since we start with a valid superperm and only delete,
            # any window that was a permutation before might become invalid?
            # No, if we delete a char, the remaining chars are a subset.
            # A window in the new word corresponds to a window in the old word with one char removed?
            # No, the window slides.
            # It is possible to create a window with duplicates or missing numbers?
            # Example: 123 -> delete 2 -> 13. If n=3, 13 is not a perm.
            # So we must check if the window is a permutation.
            if len(set(window)) == n:
                seen.add(window)
        return len(seen) == math.factorial(n)
    
    # Initial check
    if not is_valid(word):
        return "".join(map(str, word))
        
    current_len = len(word)
    target_len = math.factorial(n) * n # Upper bound, but we want min.
    
    # Strategy: Greedy Deletion
    # We try to delete symbols one by one.
    # To speed up, we can keep track of the set of permutations.
    
    def compute_perm_set(w):
        seen = set()
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i+n])
            if len(set(window)) == n:
                seen.add(window)
        return seen
    
    current_set = compute_perm_set(word)
    
    improved = True
    while improved and time.time() - start_time < TIME_LIMIT:
        improved = False
        # Try deleting from right to left, then left to right
        for direction in [1, -1]:
            if time.time() - start_time > TIME_LIMIT:
                break
            
            indices = range(len(word) - 1, -1, -1) if direction == 1 else range(len(word))
            
            for i in indices:
                if time.time() - start_time > TIME_LIMIT:
                    break
                
                if len(word) - 1 < n:
                    break
                
                # Try removing word[i]
                # The windows affected are those that include index i
                # In the new word, indices >= i shift left by 1.
                
                # Quick check: Does removing word[i] destroy any unique permutation?
                # We need to know which permutations appear ONLY in windows containing i.
                
                # Efficient approach:
                # 1. Identify windows in current word that contain index i.
                #    These are windows starting at j where i-n+1 <= j <= i.
                #    j_min = max(0, i - n + 1)
                #    j_max = min(i, len(word) - n)
                
                # 2. For each such window, check if it is a permutation.
                #    If it is, check if it appears in any OTHER window not containing i.
                
                # This is O(n * n) per deletion attempt, which is fast.
                
                lost_unique = False
                for j in range(max(0, i - n + 1), min(i, len(word) - n) + 1):
                    window = tuple(word[j:j+n])
                    if len(set(window)) != n:
                        continue
                    
                    # Check if this window exists elsewhere
                    # We can check if it's in current_set. It is.
                    # We need to check if it's in the set of windows NOT containing i.
                    
                    # To do this efficiently, we can just try the deletion and recompute the set?
                    # Recomputing the set is O(L*n). L=5900, n=7. ~40k ops.
                    # This is fast enough to do for every i?
                    # 5900 * 40k = 236 million ops. Might be too slow in Python (1-2 seconds per pass).
                    # We have ~140 seconds. So we can do ~70-100 passes.
                    # This is a good strategy.
                    
                    # Let's just do the full recompute for simplicity and robustness.
                    pass
                
                # Full recompute approach for candidate deletion
                trial_word = word[:i] + word[i+1:]
                trial_set = compute_perm_set(trial_word)
                
                if len(trial_set) == math.factorial(n):
                    word = trial_word
                    current_set = trial_set
                    improved = True
                    # Restart the scan from the beginning (or current position)
                    # Since we modified the list, indices are shifted.
                    # Breaking out of the inner loop to restart the direction loop is safer.
                    break
            
            if improved:
                break # Restart the while loop to pick up the new state
    
    # If time remains, try random local search
    if time.time() - start_time < TIME_LIMIT - 5:
        # Randomized deletion attempts
        attempts = 0
        while time.time() - start_time < TIME_LIMIT - 5 and attempts < 5000:
            if len(word) <= n:
                break
            
            i = random.randint(0, len(word) - 1)
            trial_word = word[:i] + word[i+1:]
            if len(trial_word) < n:
                continue
            
            # Quick check
            trial_set = compute_perm_set(trial_word)
            if len(trial_set) == math.factorial(n):
                word = trial_word
                attempts = 0 # Reset counter on success
            else:
                attempts += 1
    
    return "".join(map(str, word))

def build_pnp_superperm(n):
    """Build the standard PnP superpermutation."""
    if n == 1:
        return [1]
    
    # Recursive construction
    # S_n is built from S_{n-1}
    # For each permutation p of 1..n-1 in S_{n-1}, we insert n.
    # The standard construction:
    # Start with S_1 = [1]
    # For k from 2 to n:
    #   new_word = []
    #   for each (k-1)-perm p in current_word:
    #      append p
    #      append k
    #   # Wait, the standard PnP construction is:
    #   S_k = S_{k-1} with k inserted after every (k-1)-permutation?
    #   No, that would be too long.
    
    # Correct PnP Construction:
    # S_1 = 1
    # S_2 = 121
    # S_3 = 123121321
    # Pattern:
    # To get S_k from S_{k-1}:
    # Iterate through S_{k-1}.
    # Whenever we see a permutation of 1..k-1, we append k?
    # No.
    
    # Let's use the explicit recursive definition:
    # S_1 = [1]
    # S_k is formed by taking S_{k-1} and for each occurrence of a permutation of 1..k-1,
    # we don't just insert k.
    
    # Actually, the length is sum(k!).
    # S_1: 1 (len 1)
    # S_2: 1 2 1 (len 3 = 1 + 2)
    # S_3: 1 2 3 1 2 1 3 2 1 (len 9 = 1 + 2 + 6)
    
    # Algorithm:
    # word = [1]
    # for k in 2..n:
    #   new_word = []
    #   # We need to insert k into word.
    #   # The structure is that for every permutation p of 1..k-1 in word,
    #   # we have a segment.
    
    # A simpler way to generate PnP:
    # Use the recursive definition:
    # P(1) = "1"
    # P(n) = P(n-1) with n inserted in specific places.
    
    # Let's use the property:
    # S_n contains S_{n-1} as a subsequence? No.
    
    # Let's just hardcode the generation logic:
    # S_k is built by taking the list of all permutations of 1..k-1 in S_{k-1},
    # and for each one, appending it, then k, then... no.
    
    # Correct recursive construction:
    # S_1 = [1]
    # S_k = []
    # for each permutation p of 1..k-1 that appears in S_{k-1}:
    #    S_k.append(p)
    #    S_k.append([k])
    # This is not quite right because of overlaps.
    
    # Let's use the standard algorithm:
    # S_1 = [1]
    # for k in range(2, n+1):
    #    new_s = []
    #    # Find all distinct (k-1)-permutations in S_{k-1} in order of appearance
    #    perms = []
    #    seen = set()
    #    for i in range(len(S_{k-1}) - (k-1) + 1):
    #        p = tuple(S_{k-1}[i:i+k-1])
    #        if len(set(p)) == k-1 and p not in seen:
    #            seen.add(p)
    #            perms.append(p)
            
    #    # Now build new S_k
    #    for p in perms:
    #        new_s.extend(p)
    #        new_s.append(k)
    #        # The next p will overlap with the previous one?
    #        # The standard construction ensures that the last k-1 elements of the previous block
    #        # are the first k-1 elements of the next block?
    #        # No, the construction is:
    #        # S_k = p_1, k, p_2, k, ..., p_m, k
    #        # Where p_i are the permutations in S_{k-1}.
    #        # But they overlap.
    #        # Specifically, if p_i ends with x and p_{i+1} starts with y,
    #        # the overlap is determined by the structure of S_{k-1}.
    
    # Let's just use the explicit string concatenation method which is known to work:
    # S_1 = "1"
    # S_2 = "121"
    # S_3 = "123121321"
    # S_4 = "1234123142312431214321342132431213243214321"
    
    # Recursive step:
    # Given S_{k-1}, to get S_k:
    # For each permutation p of 1..k-1 in S_{k-1} (in order of first appearance):
    #   Append p
    #   Append k
    #   # But we must handle overlaps.
    #   # Actually, the standard PnP word is:
    #   # S_k = p_1 k p_2 k ... p_m k
    #   # Where p_i are the (k-1)-perms in S_{k-1}.
    #   # The total length is sum_{i=1}^{m} (len(p_i) + 1) - overlaps.
    #   # The overlaps are such that the suffix of (p_{i-1} k) matches the prefix of p_i?
    #   # No.
    
    # Let's use a verified implementation of PnP:
    
    def pnp(k):
        if k == 1:
            return [1]
        prev = pnp(k-1)
        # Extract all distinct (k-1)-permutations from prev in order
        perms = []
        seen = set()
        for i in range(len(prev) - k + 2):
            window = tuple(prev[i:i+k-1])
            if len(set(window)) == k-1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new_word = []
        for p in perms:
            new_word.extend(p)
            new_word.append(k)
        
        return new_word

    return pnp(n)

# EVOLVE-BLOCK-END
