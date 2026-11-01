"""Improved program using bit-parallel greedy + local search for K_3(7,1)."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        # Fallback: direct product construction
        import itertools
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Precompute all codewords as integers 0..2186
    # Word w is represented as integer: sum(w[i] * 3^i for i in range(7))
    # Precompute the coverage mask for each of the 2187 possible codewords
    # A codeword c covers: c itself, and for each position i, c with position i changed to (c[i]+1)%3 and (c[i]+2)%3
    
    # Precompute powers of 3
    p3 = [1] * (n + 1)
    for i in range(1, n + 1):
        p3[i] = p3[i-1] * 3
    
    total_words = 3 ** n  # 2187
    
    # For each codeword index (0..2186), compute the set of words it covers as a bitmask
    # We'll store these as Python integers (bit i is set if word i is covered)
    # Precompute the 15 neighbors for each codeword
    
    # First, convert index to word tuple and back
    def idx_to_word(idx):
        w = []
        for i in range(n):
            w.append(idx % 3)
            idx //= 3
        return w
    
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx += w[i] * p3[i]
        return idx
    
    # Precompute coverage masks for all 2187 codewords
    cover_masks = [0] * total_words
    for c_idx in range(total_words):
        w = idx_to_word(c_idx)
        mask = 1 << c_idx  # cover itself
        for pos in range(n):
            for delta in [1, 2]:
                new_w = w[:]
                new_w[pos] = (new_w[pos] + delta) % 3
                new_idx = word_to_idx(new_w)
                mask |= (1 << new_idx)
        cover_masks[c_idx] = mask
    
    # Greedy algorithm
    def greedy_cover(seed=None):
        rng = random.Random(seed)
        uncovered = (1 << total_words) - 1  # all bits set
        chosen = []
        chosen_set = set()
        
        while uncovered != 0:
            # Find codeword covering most uncovered words
            best_c = -1
            best_count = -1
            # Sample a subset for speed if too slow, but 2187 is small enough
            for c in range(total_words):
                if c in chosen_set:
                    continue
                count = (cover_masks[c] & uncovered).bit_count()
                if count > best_count:
                    best_count = count
                    best_c = c
            
            if best_c < 0:
                break
            
            chosen.append(best_c)
            chosen_set.add(best_c)
            uncovered &= ~cover_masks[best_c]
        
        return chosen
    
    # Local search: try to delete codewords
    def try_delete(chosen, rng):
        """Try to delete codewords one by one. Returns (new_chosen, improved)."""
        n_chosen = len(chosen)
        # Compute coverage count for each word
        # cov[word_idx] = number of chosen codewords that cover word_idx
        cov = [0] * total_words
        for c in chosen:
            mask = cover_masks[c]
            # For each set bit in mask, increment cov
            m = mask
            while m:
                # Get lowest set bit
                lb = m & (-m)
                idx = lb.bit_length() - 1
                cov[idx] += 1
                m ^= lb
        
        new_chosen = list(chosen)
        improved = False
        
        # Try deleting each codeword
        for i in range(len(new_chosen)):
            c = new_chosen[i]
            mask = cover_masks[c]
            # Check if any word would become uncovered
            m = mask
            can_delete = True
            while m:
                lb = m & (-m)
                idx = lb.bit_length() - 1
                if cov[idx] <= 1:
                    can_delete = False
                    break
                m ^= lb
            
            if can_delete:
                # Remove it
                m = mask
                while m:
                    lb = m & (-m)
                    idx = lb.bit_length() - 1
                    cov[idx] -= 1
                    m ^= lb
                new_chosen.pop(i)
                improved = True
                # Don't increment i since we popped
        
        return new_chosen, improved
    
    # Local search: try swapping
    def try_swap(chosen, rng, max_swaps=50):
        """Try to swap codewords to reduce count or improve structure."""
        new_chosen = list(chosen)
        n_chosen = len(new_chosen)
        
        # Compute coverage counts
        cov = [0] * total_words
        for c in new_chosen:
            mask = cover_masks[c]
            m = mask
            while m:
                lb = m & (-m)
                idx = lb.bit_length() - 1
                cov[idx] += 1
                m ^= lb
        
        # Try swapping each codeword with a random other codeword
        all_codewords = list(range(total_words))
        chosen_set = set(new_chosen)
        
        for _ in range(max_swaps):
            if time.time() > deadline:
                break
            
            # Pick a random codeword from current set
            i = rng.randint(0, len(new_chosen) - 1)
            c_old = new_chosen[i]
            
            # Pick a random candidate not in current set
            candidates = [c for c in all_codewords if c not in chosen_set]
            if not candidates:
                break
            c_new = rng.choice(candidates)
            
            # Check if swap is valid: removing c_old and adding c_new keeps coverage
            # Words only covered by c_old must be covered by c_new
            mask_old = cover_masks[c_old]
            mask_new = cover_masks[c_new]
            
            # Words covered by c_old but not c_new must have cov >= 2
            only_old = mask_old & ~mask_new
            m = only_old
            valid = True
            while m:
                lb = m & (-m)
                idx = lb.bit_length() - 1
                if cov[idx] <= 1:
                    valid = False
                    break
                m ^= lb
            
            if valid:
                # Perform swap
                new_chosen[i] = c_new
                chosen_set.remove(c_old)
                chosen_set.add(c_new)
                
                # Update coverage
                m = mask_old
                while m:
                    lb = m & (-m)
                    idx = lb.bit_length() - 1
                    cov[idx] -= 1
                    m ^= lb
                m = mask_new
                while m:
                    lb = m & (-m)
                    idx = lb.bit_length() - 1
                    cov[idx] += 1
                    m ^= lb
        
        return new_chosen
    
    # Main search loop
    best_result = None
    best_size = float('inf')
    
    # Try multiple random seeds for greedy
    for seed in range(20):
        if time.time() > deadline - 5:
            break
        
        chosen = greedy_cover(seed)
        
        # Local search: repeated deletion and swap
        for iteration in range(5):
            if time.time() > deadline - 5:
                break
            
            # Try deletion
            new_chosen, improved = try_delete(chosen, random)
            if improved:
                chosen = new_chosen
            
            # Try swap
            chosen = try_swap(chosen, random, max_swaps=20)
            
            if len(chosen) < best_size:
                best_size = len(chosen)
                best_result = list(chosen)
        
        if len(chosen) < best_size:
            best_size = len(chosen)
            best_result = list(chosen)
        
        if best_size <= 156:  # Can't do better than lower bound
            break
    
    if best_result is None:
        # Fallback
        chosen = greedy_cover(0)
        best_result = chosen
    
    # Convert indices back to word lists
    result = []
    for c_idx in best_result:
        w = idx_to_word(c_idx)
        result.append(w)
    
    return result

# EVOLVE-BLOCK-END
