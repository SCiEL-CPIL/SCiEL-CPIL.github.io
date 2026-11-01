import time
import numpy as np
import random

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin for 120s limit
    
    total = q ** n
    ball_size = 1 + n * (q - 1)
    
    # Precompute place values for index conversion
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q
        
    # Precompute all words as indices
    # words[i] is the list of digits for index i
    # We need ball_indices[i] = list of indices of words within distance 1 of i
    ball_indices = [[] for _ in range(total)]
    
    # Generate words and balls
    # To be efficient, we generate the word digits on the fly or store them.
    # Storing 2187 * 7 ints is small.
    words = np.zeros((total, n), dtype=np.int32)
    for i in range(total):
        temp = i
        for j in range(n - 1, -1, -1):
            words[i, j] = temp % q
            temp //= q
            
    for i in range(total):
        w = words[i]
        ball_indices[i].append(i)
        for coord in range(n):
            orig = w[coord]
            for v in range(q):
                if v == orig:
                    continue
                new_idx = i + (v - orig) * base[coord]
                ball_indices[i].append(new_idx)
                
    # Helper to check if a set of codewords is valid
    # Returns True if valid, and optionally updates cnt array
    def is_valid_and_count(code_set, cnt=None):
        if cnt is None:
            cnt = np.zeros(total, dtype=np.int32)
        else:
            cnt[:] = 0
        for c in code_set:
            for x in ball_indices[c]:
                cnt[x] += 1
        return np.all(cnt > 0), cnt

    def get_min_count_in_ball(c, cnt):
        min_val = 255 # max int8
        for x in ball_indices[c]:
            if cnt[x] < min_val:
                min_val = cnt[x]
        return min_val

    def greedy_build():
        covered = np.zeros(total, dtype=bool)
        code = []
        # Precompute ball arrays for faster access in greedy
        # balls_np is a list of numpy arrays? No, lists are fine for small N.
        
        while not np.all(covered):
            best_c = -1
            max_new = -1
            # Iterate through all candidates
            for c in range(total):
                # Calculate new coverage
                new_cov = 0
                for x in ball_indices[c]:
                    if not covered[x]:
                        new_cov += 1
                if new_cov > max_new:
                    max_new = new_cov
                    best_c = c
                    if max_new == ball_size:
                        break # Can't get better
            code.append(best_c)
            for x in ball_indices[best_c]:
                covered[x] = True
        return code

    # Initial greedy
    current_code = set(greedy_build())
    best_code = list(current_code)
    best_size = len(best_code)
    
    # Initialize counts
    cnt = np.zeros(total, dtype=np.int32)
    for c in current_code:
        for x in ball_indices[c]:
            cnt[x] += 1
            
    def run_delete_phase(code_set, cnt_arr):
        """Removes redundant codewords. Returns True if any removed."""
        changed = False
        # Iterate over a copy of the list to allow modification
        for c in list(code_set):
            # Check if min count in ball(c) is >= 2
            min_val = 255
            for x in ball_indices[c]:
                if cnt_arr[x] < min_val:
                    min_val = cnt_arr[x]
            if min_val >= 2:
                code_set.remove(c)
                for x in ball_indices[c]:
                    cnt_arr[x] -= 1
                changed = True
        return changed

    # Local Search Loop
    while time.time() < deadline - 5.0:
        # 1. Delete Phase
        while run_delete_phase(current_code, cnt):
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = list(current_code)
                
        # 2. Swap Phase
        # Try a few random swaps
        swaps_attempted = 0
        max_swaps = 200
        while swaps_attempted < max_swaps and time.time() < deadline - 5.0:
            swaps_attempted += 1
            if len(current_code) == 0:
                break
                
            # Pick random codeword to remove
            c = random.choice(list(current_code))
            
            # Pick random candidate to add
            c_prime = random.randint(0, total - 1)
            
            if c_prime == c:
                continue
            if c_prime in current_code:
                continue
                
            # Simulate swap
            # We need to check if (current_code - {c} + {c_prime}) is valid.
            # We can do this by checking if removing c leaves any hole that c_prime doesn't cover,
            # or simply recalculating the min count.
            # Efficient check:
            # 1. Decrement counts for c
            # 2. Increment counts for c_prime
            # 3. Check min > 0
            # 4. Revert if invalid
            
            # To avoid modifying the main cnt array if it fails, we can use a temporary check.
            # Since ball_size is small (15), we can just check the specific points.
            # Points affected: ball(c) U ball(c_prime)
            
            # Let's just do the full update on a copy? No, cnt is numpy array.
            # Copying 2187 ints is fast.
            cnt_temp = cnt.copy()
            
            for x in ball_indices[c]:
                cnt_temp[x] -= 1
            for x in ball_indices[c_prime]:
                cnt_temp[x] += 1
                
            if np.all(cnt_temp > 0):
                # Swap is valid
                current_code.remove(c)
                current_code.add(c_prime)
                cnt = cnt_temp # Update main cnt
                
                # Check for improvement
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = list(current_code)
                    
                # After a swap, run delete phase again immediately
                # This is crucial for finding smaller codes
                while run_delete_phase(current_code, cnt):
                    if len(current_code) < best_size:
                        best_size = len(current_code)
                        best_code = list(current_code)

    # Final result conversion
    result = []
    for idx in best_code:
        w = []
        temp = idx
        for i in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        result.append(w)
        
    return result
