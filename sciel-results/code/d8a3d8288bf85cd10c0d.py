import time
import random
import math
import itertools
import numpy as np

def construct(q, n, radius):
    deadline = time.monotonic() + 115.0
    total = q ** n
    rng = random.Random(12345)
    
    # Precompute neighbor indices for each codeword position
    # For radius 1: self + one position changed to each other value
    neigh = []
    for idx in range(total):
        ns = [idx]
        for i in range(n):
            pow_q_i = q ** i
            digit = (idx // pow_q_i) % q
            for new_digit in range(q):
                if new_digit != digit:
                    ns.append(idx + (new_digit - digit) * pow_q_i)
        neigh.append(ns)
    
    # Convert to numpy arrays for fast operations
    neigh_arr = np.array(neigh)  # shape (total, 1+2n)
    neigh_width = neigh_arr.shape[1]
    
    def idx_to_word(idx):
        word = [0] * n
        temp = idx
        for i in range(n):
            word[i] = temp % q
            temp //= q
        return word
    
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx
    
    # Build initial code using tetracode direct sum
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        init_code = []
        for h in tetracode:
            for tail in itertools.product(range(q), repeat=n - 4):
                w = h + list(tail)
                init_code.append(word_to_idx(w))
    else:
        # Greedy initialization
        init_code = []
        remaining = np.ones(total, dtype=bool)
        rem_count = total
        while rem_count > 0:
            best_c = -1
            best_gain = -1
            for c in range(total):
                if c in init_code:
                    continue
                gain = np.sum(remaining[neigh_arr[c]])
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            if best_c == -1:
                break
            init_code.append(best_c)
            remaining[neigh_arr[best_c]] = False
            rem_count -= 1
    
    # Coverage count using numpy
    def compute_counts(code):
        counts = np.zeros(total, dtype=np.int32)
        for c in code:
            counts[neigh_arr[c]] += 1
        return counts
    
    def is_removable(c, counts):
        return np.all(counts[neigh_arr[c]] > 1)
    
    def local_search(code, max_iters=200):
        code = list(code)
        counts = compute_counts(code)
        code_set = set(code)
        
        for _ in range(max_iters):
            if time.monotonic() > deadline:
                break
            improved = False
            
            # Phase 1: Remove redundant codewords
            new_code = []
            for c in code:
                if is_removable(c, counts):
                    counts[neigh_arr[c]] -= 1
                    code_set.discard(c)
                    improved = True
                else:
                    new_code.append(c)
            code = new_code
            
            if not improved:
                break
            
            # Phase 2: Try swaps - replace one codeword with another to maintain coverage
            # For each codeword c, try replacing with c' where c' covers the same or better
            # A swap is valid if after removing c and adding c', all points are still covered
            swap_improved = True
            while swap_improved:
                swap_improved = False
                counts = compute_counts(code)
                for i, c in enumerate(code):
                    # Temporarily remove c
                    counts[neigh_arr[c]] -= 1
                    # Find uncovered points
                    uncovered = np.where(counts == 0)[0]
                    if len(uncovered) == 0:
                        # c was redundant (shouldn't happen after phase 1, but check)
                        code.pop(i)
                        code_set.discard(c)
                        counts = compute_counts(code)
                        swap_improved = True
                        break
                    # Try to find a replacement c' that covers all uncovered points
                    # and doesn't create new issues
                    # We need c' such that all uncovered points are in neigh[c']
                    # and c' is not already in code
                    found = False
                    for c_prime in range(total):
                        if c_prime in code_set:
                            continue
                        # Check if c' covers all uncovered points
                        if np.all(np.isin(uncovered, neigh_arr[c_prime])):
                            # Check that adding c' doesn't leave any point uncovered
                            # that was only covered by c
                            # Points only covered by c: those where count was 1 before removal
                            # After removal, those are uncovered. c' must cover them.
                            # Already checked above.
                            # Also need to ensure no other point becomes uncovered
                            # Points with count 0 after removal must be covered by c'
                            # That's the uncovered set. Already checked.
                            code[i] = c_prime
                            code_set.discard(c)
                            code_set.add(c_prime)
                            counts = compute_counts(code)
                            swap_improved = True
                            found = True
                            break
                    if found:
                        break
                    # Restore c
                    counts[neigh_arr[c]] += 1
            
            if not swap_improved and not improved:
                break
        
        return code
    
    best_code = local_search(init_code)
    best_size = len(best_code)
    
    # If we already have a good solution, try to improve with more aggressive search
    # Use simulated annealing style: try random removals and repairs
    
    def repair(code):
        """Add codewords to cover all points"""
        code = list(code)
        code_set = set(code)
        counts = compute_counts(code)
        uncovered = set(np.where(counts == 0)[0].tolist())
        
        while uncovered:
            if time.monotonic() > deadline:
                break
            best_c = -1
            best_gain = -1
            for c in range(total):
                if c in code_set:
                    continue
                gain = sum(1 for u in neigh[c] if u in uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            if best_c == -1 or best_gain <= 0:
                break
            code.append(best_c)
            code_set.add(best_c)
            for u in neigh[best_c]:
                counts[u] += 1
                if counts[u] == 1 and u in uncovered:
                    uncovered.discard(u)
        
        return code
    
    # Perturbation loop
    while time.monotonic() < deadline:
        # Random perturbation: remove k random codewords
        k_drop = rng.randint(1, 4)
        if len(best_code) <= k_drop:
            k_drop = max(1, len(best_code) - 1)
        
        current_code = list(best_code)
        rng.shuffle(current_code)
        current_code = current_code[k_drop:]
        
        # Repair
        repaired_code = repair(current_code)
        
        # Local search
        improved_code = local_search(repaired_code)
        
        if len(improved_code) < best_size:
            best_size = len(improved_code)
            best_code = improved_code
            
            # Try to reach optimal
            if best_size <= 73:
                break
    
    # Convert to words
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))
    
    return result
