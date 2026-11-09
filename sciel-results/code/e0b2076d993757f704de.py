import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        return [list(w) for w in itertools.product(range(q), repeat=n)]

    N = 3**7
    all_mask = (1 << N) - 1

    # Precompute ball masks as Python integers (bitsets)
    # This is much faster for bitwise AND/OR/NOT than numpy boolean arrays for this N
    def idx_to_digits(idx):
        digits = []
        for _ in range(n):
            digits.append(idx % 3)
            idx //= 3
        return digits

    words = [idx_to_digits(i) for i in range(N)]
    
    # Precompute ball masks
    ball_masks = [0] * N
    for i in range(N):
        w = words[i]
        mask = 1 << i
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(3):
                if new_val == orig_val:
                    continue
                new_w = w.copy()
                new_w[pos] = new_val
                new_idx = 0
                power = 1
                for d in new_w:
                    new_idx += d * power
                    power *= 3
                mask |= (1 << new_idx)
        ball_masks[i] = mask

    # Fast popcount
    def popcount(x):
        return bin(x).count('1')

    def greedy_randomized():
        covered = 0
        code_indices = []
        
        while covered != all_mask:
            uncovered = all_mask ^ covered
            best_gain = -1
            best_idx = -1
            
            # Optimization: Only check candidates that have some uncovered bits
            # To speed up, we can sample or just iterate all. N=2187 is small.
            # Iterate all to ensure global greedy step.
            for i in range(N):
                gain = popcount(ball_masks[i] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
                # Tie-breaking: random noise
                elif gain == best_gain:
                    # Small chance to swap for diversity
                    if random.random() < 0.1:
                        best_idx = i

            if best_gain == 0:
                break # Should not happen

            code_indices.append(best_idx)
            covered |= ball_masks[best_idx]
            
        return code_indices

    def local_search(code_indices):
        """
        Bitset-based local search with 1-swaps and deletions.
        """
        if not code_indices:
            return code_indices
            
        covered = 0
        for idx in code_indices:
            covered |= ball_masks[idx]
            
        # Verify full coverage
        if covered != all_mask:
             # Fallback to greedy if somehow broken
             return greedy_randomized()

        improved = True
        while improved:
            improved = False
            # Try deletions
            # Shuffle to randomize
            indices_list = code_indices.copy()
            random.shuffle(indices_list)
            
            for i in range(len(indices_list)):
                idx = indices_list[i]
                # If removing idx keeps coverage, remove it
                # Check if ball(idx) is covered by other codewords
                # other_covered = covered ^ ball_masks[idx] ? No.
                # We need to know if every word in ball(idx) is covered by C \ {idx}.
                
                # Efficient check:
                # Let C_mask = OR of ball_masks for all j in code_indices except i.
                # This is expensive to compute from scratch.
                # Instead, maintain counts?
                # With bitsets, we can compute:
                # covered_by_others = 0
                # for j in code_indices:
                #    if j != idx: covered_by_others |= ball_masks[j]
                # if (ball_masks[idx] & ~covered_by_others) == 0: remove
                
                # Optimization: Precompute prefix/suffix ORs? Or just do it if N is small.
                # N_code is ~190. N is 2187.
                # 190 * 2187 ops is ~400k ops. Very fast.
                
                covered_by_others = 0
                for j in code_indices:
                    if j != idx:
                        covered_by_others |= ball_masks[j]
                        
                if (ball_masks[idx] & ~covered_by_others) == 0:
                    code_indices.remove(idx)
                    covered ^= ball_masks[idx] # Update covered? 
                    # Actually, if we remove idx, covered becomes covered_by_others.
                    covered = covered_by_others
                    improved = True
                    # Restart loop as list changed
                    break
            
            if improved:
                continue

            # Try 1-swaps: Remove idx, Add j
            # This is more complex. 
            # Heuristic: If we can't remove, try to replace a "weak" codeword with a "strong" one?
            # For now, stick to deletions and re-greedy if stuck?
            
            # If no deletion possible, try a random restart of the last few steps?
            # Or just break.
            
        return code_indices

    # Run multiple restarts
    best_code = None
    best_len = float('inf')
    
    # Baseline
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    best_code = seed_code
    best_len = len(seed_code)
    
    start_time = time.time()
    deadline = start_time + 110
    
    run_count = 0
    while time.time() < deadline:
        code_indices = greedy_randomized()
        polished_indices = local_search(code_indices)
        
        if len(polished_indices) < best_len:
            best_len = len(polished_indices)
            best_code = [words[i] for i in polished_indices]
            
        run_count += 1
        
    return best_code
