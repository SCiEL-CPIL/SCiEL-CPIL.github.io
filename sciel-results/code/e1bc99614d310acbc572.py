"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def _precompute_masks(q, n, radius):
    """
    Precompute the coverage mask for every word in the space.
    Returns a list of integers (bitsets) where bit i is 1 if word i is covered.
    """
    total = q ** n
    # Generate all words as a numpy array
    all_w = np.zeros((total, n), dtype=np.int8)
    for i in range(n):
        divisor = q ** (n - 1 - i)
        all_w[:, i] = (np.arange(total) // divisor) % q

    masks = [0] * total
    # We can compute masks more efficiently by iterating over words and their neighbors
    # For q=3, n=6, radius=1, each word covers 1 + 6*2 = 13 words.
    # Total operations: 729 * 13 ~ 9k, which is very fast.
    
    for w_idx in range(total):
        word = all_w[w_idx]
        mask = 0
        # Iterate over all words in the ball? No, that's O(q^n) per word.
        # Better: iterate over the ball explicitly.
        # For radius 1, the ball consists of the word itself and words differing in 1 position.
        if radius == 1:
            # Center
            mask |= (1 << w_idx)
            for pos in range(n):
                for val in range(q):
                    if val == word[pos]:
                        continue
                    # Calculate index of neighbor
                    # index = sum(digit * q^(n-1-pos))
                    # We can calculate the difference
                    old_val = word[pos]
                    diff = val - old_val
                    # The change in index is diff * q^(n-1-pos)
                    step = q ** (n - 1 - pos)
                    new_idx = w_idx + diff * step
                    mask |= (1 << new_idx)
        else:
            # Fallback for general radius (slower, but q=3, n=6, r=1 is the main case)
            # For now, we stick to the fast path for r=1. If r>1, we might need a different approach.
            # Given the task is K_3(6,1), we assume r=1.
            # If r != 1, we compute via distance check for this specific word (expensive but rare)
            dists = np.sum(all_w != word, axis=1)
            covered_indices = np.where(dists <= radius)[0]
            for idx in covered_indices:
                mask |= (1 << int(idx))
        masks[w_idx] = mask
        
    return masks, all_w

def _check_coverage_bitset(masks, code_indices, total):
    """Check if the union of masks for code_indices covers all bits (0 to total-1)."""
    if not code_indices:
        return total == 0
    
    union_mask = 0
    for idx in code_indices:
        union_mask |= masks[idx]
    
    # Check if all bits 0..total-1 are set
    full_mask = (1 << total) - 1
    return (union_mask & full_mask) == full_mask

def _greedy_cover_bitset(masks, total, deadline):
    """
    Greedy set cover using bitsets.
    Returns a list of indices.
    """
    uncovered = (1 << total) - 1
    code = []
    
    # To speed up greedy, we can maintain a list of candidates.
    # At each step, we need to find the candidate with max (masks[cand] & uncovered).bit_count()
    # With 729 candidates, iterating through all of them is 729 * (bigint op).
    # Bigint ops are fast in Python (C implementation).
    
    candidates = list(range(total))
    
    while uncovered != 0 and time.time() < deadline:
        best_idx = -1
        best_count = -1
        
        # Optimization: If we have many candidates, sampling might be faster, 
        # but 729 is small enough to just scan.
        # However, as we add to code, we can remove them from candidates? 
        # No, a word might be useful later if it covers different uncovered parts? 
        # Actually, in set cover, once a set is chosen, it's removed from consideration 
        # in standard greedy. But here, we can just scan all remaining candidates.
        
        for c in candidates:
            m = masks[c]
            # Count bits in intersection
            # Python 3.10+ has int.bit_count()
            cnt = (m & uncovered).bit_count()
            if cnt > best_count:
                best_count = cnt
                best_idx = c
        
        if best_idx == -1 or best_count == 0:
            break
            
        code.append(best_idx)
        uncovered &= ~masks[best_idx]
        
        # Optional: remove best_idx from candidates to save time? 
        # Yes, standard greedy removes the chosen set.
        candidates.remove(best_idx)

    # If there are still uncovered bits, add them directly (as singletons)
    # This ensures validity.
    # Find indices of remaining uncovered bits
    rem = uncovered
    while rem:
        # Get lowest set bit
        lsb = rem & -rem
        idx = lsb.bit_length() - 1
        code.append(idx)
        rem ^= lsb
        
    return code

def _remove_redundant_bitset(masks, code, total, deadline):
    """
    Remove redundant codewords.
    A codeword c is redundant if the union of masks of (code \ {c}) still covers all.
    """
    code = list(code)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(code)):
            if time.time() >= deadline:
                break
            # Check if code[i] is redundant
            # Union of all masks except code[i]
            union = 0
            for j in range(len(code)):
                if j != i:
                    union |= masks[code[j]]
            
            full_mask = (1 << total) - 1
            if (union & full_mask) == full_mask:
                code.pop(i)
                changed = True
                break
    return code

def _local_search_bitset(masks, code, total, q, n, deadline):
    """
    Try to reduce code size by swapping or removing.
    1. Try removing any element (redundancy check).
    2. Try replacing an element with a random one to see if we can then remove another?
       Or simply try to find a smaller code by random restarts?
       
       For now, let's stick to a simple heuristic:
       Randomly shuffle the code, then run a greedy-like removal?
       Or try to replace a codeword with another that covers the same "critical" words?
       
       A better approach for small N:
       Iterate through code. For each word c, try to replace it with a random word r.
       If the new code is valid, keep it? No, that doesn't reduce size.
       
       To reduce size, we need to find a set of size K-1 that covers.
       We can try:
       For each pair (i, j) in code, remove i, and try to cover the words only covered by i
       using one other word (or by adjusting others). This is complex.
       
       Simpler: 
       Run greedy with a different starting point or randomization?
       
       Let's try a "swap and shrink" loop:
       1. Randomly pick an index i in code.
       2. Randomly pick a new word r not in code.
       3. Replace code[i] with r.
       4. Run _remove_redundant_bitset.
       5. If size decreased, keep.
       6. If size same, keep (might help future swaps).
    """
    code = list(code)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(code)):
            if time.time() >= deadline:
                break
            # Try 10 random swaps for this position
            for _ in range(10):
                if time.time() >= deadline:
                    break
                # Generate random word
                new_idx = random.randint(0, total - 1)
                if new_idx in code:
                    continue
                
                trial = code[:i] + [new_idx] + code[i+1:]
                
                # Check validity first (fast)
                union = 0
                for idx in trial:
                    union |= masks[idx]
                full_mask = (1 << total) - 1
                if (union & full_mask) != full_mask:
                    continue
                
                # If valid, try to remove redundant
                trial = _remove_redundant_bitset(masks, trial, total, deadline)
                
                if len(trial) < len(code):
                    code = trial
                    improved = True
                    break
                elif len(trial) == len(code) and trial != code:
                    # Keep it if it's different, might open new paths
                    # But be careful not to cycle. 
                    # For simplicity, let's just keep the first improvement or random walk.
                    code = trial
                    improved = True
                    break

    return code

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    total = q ** n
    
    # Precompute masks
    masks, all_w = _precompute_masks(q, n, radius)
    
    # Initial Greedy
    code = _greedy_cover_bitset(masks, total, deadline)
    
    # Remove Redundant
    code = _remove_redundant_bitset(masks, code, total, deadline)
    
    # Local Search
    if time.time() < deadline:
        code = _local_search_bitset(masks, code, total, q, n, deadline)
        
    # Final Redundancy Check
    code = _remove_redundant_bitset(masks, code, total, deadline)
    
    # Verify
    if not _check_coverage_bitset(masks, code, total):
        # Fallback: Return a known valid cover? 
        # The greedy with fallback should be valid.
        # If this fails, something is wrong with the bitset logic.
        # Let's return the greedy result before local search if it was valid?
        # We didn't store it. Let's just trust the bitset check.
        # If it's invalid, we must fix it.
        # Add all remaining uncovered words.
        union = 0
        for idx in code:
            union |= masks[idx]
        full_mask = (1 << total) - 1
        rem = full_mask & ~union
        while rem:
            lsb = rem & -rem
            idx = lsb.bit_length() - 1
            code.append(idx)
            rem ^= lsb
            
    # Return list of indices
    return code
# EVOLVE-BLOCK-END
