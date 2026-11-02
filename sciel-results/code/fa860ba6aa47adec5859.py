"""Bit-Parallel Iterative Compression for ternary covering code K_3(7,1)."""
import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Precompute the Hamming sphere of radius 1 around each possible word
    # For q=3, n=7, radius=1: each sphere has 1 + 7*(q-1) = 1 + 7*2 = 15 words
    # Total space size: 3^7 = 2187
    space_size = q ** n
    sphere_size = 1 + n * (q - 1)

    # Precompute all spheres as bitsets (using Python integers for bit operations)
    # Each sphere is a bitmask over the space_size positions
    # We index words by their integer representation in base-q
    
    def word_to_idx(word):
        idx = 0
        for v in word:
            idx = idx * q + v
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Precompute sphere bitmasks for all words
    print(f"Precomputing spheres for {space_size} words...", flush=True)
    sphere_bitmasks = [0] * space_size
    for idx in range(space_size):
        word = idx_to_word(idx)
        mask = 1 << idx
        for pos in range(n):
            for val in range(q):
                if val != word[pos]:
                    new_word = word[:]
                    new_word[pos] = val
                    mask |= 1 << word_to_idx(new_word)
        sphere_bitmasks[idx] = mask
    print("Spheres precomputed.", flush=True)

    # Initial code: tetracode direct sum construction (186 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    code_indices = [word_to_idx(w) for w in initial_code]
    print(f"Initial code size: {len(code_indices)}", flush=True)

    # Compute current coverage bitmask
    def compute_coverage(indices):
        cov = 0
        for idx in indices:
            cov |= sphere_bitmasks[idx]
        return cov

    full_mask = (1 << space_size) - 1
    coverage = compute_coverage(code_indices)
    print(f"Initial coverage: {bin(coverage).count('1')}/{space_size}", flush=True)

    # Verify initial code covers everything
    assert coverage == full_mask, "Initial code doesn't cover full space!"

    best_code = list(code_indices)
    best_size = len(code_indices)

    # Iterative compression: try to replace pairs with single new codeword
    # A pair (i, j) can be replaced by k if:
    #   sphere[i] | sphere[j] | (coverage without i,j) == coverage
    #   i.e., sphere[i] | sphere[j] is fully covered by (coverage without i,j) | sphere[k]
    # Equivalently: (sphere[i] | sphere[j]) & ~(coverage_without_ij | sphere[k]) == 0
    
    # For efficiency, we'll use the following approach:
    # coverage_without_ij = coverage & ~sphere_bitmasks[i] & ~sphere_bitmasks[j]
    # But this is expensive. Instead, we check:
    #   (sphere[i] | sphere[j]) & ~(coverage | sphere[k]) == 0
    # Wait, that's not right either. Let me think more carefully.
    #
    # We need: after removing i and j and adding k, coverage is still full.
    # new_coverage = (coverage & ~sphere[i] & ~sphere[j]) | sphere[k]
    # We need new_coverage == full_mask
    # Equivalently: (coverage & ~sphere[i] & ~sphere[j]) | sphere[k] == full_mask
    # Which means: full_mask & ~(coverage & ~sphere[i] & ~sphere[j] | sphere[k]) == 0
    # = full_mask & ~(coverage & ~sphere[i] & ~sphere[j]) & ~sphere[k] == 0
    # = (full_mask & ~coverage | sphere[i] | sphere[j]) & ~sphere[k] == 0
    # Since coverage == full_mask initially, full_mask & ~coverage == 0, so:
    # = (sphere[i] | sphere[j]) & ~sphere[k] == 0
    # = sphere[i] | sphere[j] is a subset of sphere[k]
    # That can't be right because spheres have the same size...
    #
    # Let me reconsider. The condition is:
    # (coverage \ (sphere[i] ∪ sphere[j])) ∪ sphere[k] = full_space
    # Since coverage = full_space:
    # (full_space \ (sphere[i] ∪ sphere[j])) ∪ sphere[k] = full_space
    # This is true iff sphere[i] ∪ sphere[j] ⊆ (full_space \ (sphere[i] ∪ sphere[j])) ∪ sphere[k]
    # = sphere[i] ∪ sphere[j] ⊆ sphere[k] ∪ (complement of sphere[i] ∪ sphere[j])
    # The complement part always covers things not in sphere[i]∪sphere[j], so we just need:
    # sphere[i] ∪ sphere[j] ⊆ sphere[k] ∪ (sphere[i] ∪ sphere[j])
    # Which is always true! That means we can always replace any pair? No, that's wrong.
    #
    # I made an error. Let me redo this.
    # new_coverage = (coverage & ~sphere[i] & ~sphere[j]) | sphere[k]
    # We need new_coverage = full_mask
    # (coverage & ~sphere[i] & ~sphere[j]) | sphere[k] = full_mask
    # Let U = coverage & ~sphere[i] & ~sphere[j]
    # U | sphere[k] = full_mask
    # This means every point not in U must be in sphere[k]
    # Points not in U = full_mask & ~(coverage & ~sphere[i] & ~sphere[j])
    # = (full_mask & ~coverage) | (full_mask & ~~sphere[i]) | (full_mask & ~~sphere[j])  [De Morgan]
    # Wait, ~ (A & B & C) = ~A | ~B | ~C
    # ~(coverage & ~sphere[i] & ~sphere[j]) = ~coverage | sphere[i] | sphere[j]
    # So points not in U = (full_mask & ~coverage) | (full_mask & sphere[i]) | (full_mask & sphere[j])
    # Since coverage = full_mask, full_mask & ~coverage = 0
    # So points not in U = sphere[i] | sphere[j] (intersected with full_mask, which is all)
    # So we need: sphere[i] | sphere[j] ⊆ sphere[k]
    # But |sphere[i]| = |sphere[j]| = |sphere[k]| = 15, and |sphere[i] | sphere[j]| >= 15
    # So sphere[i] | sphere[j] ⊆ sphere[k] implies sphere[i] | sphere[j] = sphere[k]
    # This is very restrictive!
    
    # Hmm, this means pair replacement is extremely limited. Let me think of a different approach.
    # 
    # Actually, the issue is that when we remove two codewords, we might uncover some points,
    # and the new codeword needs to cover those uncovered points. But the new codeword's sphere
    # might not contain all the points that were only covered by the removed codewords.
    
    # Let me reconsider: maybe I should allow adding a codeword first, then removing one.
    # Or try a different compression strategy.
    
    # Alternative: try removing one codeword at a time if its sphere is fully covered by others.
    # A codeword i is redundant if sphere[i] ⊆ (coverage \ sphere[i])
    # i.e., every point in sphere[i] is covered by at least one other codeword.
    # coverage_without_i = coverage & ~sphere[i]
    # We need: sphere[i] & ~coverage_without_i == 0
    # = sphere[i] & ~(coverage & ~sphere[i]) == 0
    # = sphere[i] & (~coverage | sphere[i]) == 0
    # = (sphere[i] & ~coverage) | (sphere[i] & sphere[i]) == 0
    # = (sphere[i] & ~coverage) | sphere[i] == 0
    # This is only true if sphere[i] = 0, which never happens. So no single codeword is ever redundant
    # in a minimal covering code? That doesn't seem right.
    
    # I think my logic is flawed. Let me just implement a simple local search:
    # 1. Try removing each codeword and check if coverage is maintained.
    # 2. Try replacing one codeword with another.
    # 3. Try adding a codeword and then removing two.

    # Let's use a different, more practical approach: greedy local search with bit-parallel ops.

    # Function to check if a set of codeword indices covers the full space
    def is_valid_coverage(indices):
        cov = 0
        for idx in indices:
            cov |= sphere_bitmasks[idx]
        return cov == full_mask

    # Local search: try to reduce code size by removing codewords
    # or by swapping one codeword for another that allows further reduction
    
    random.seed(42)
    code_list = list(code_indices)
    
    improved = True
    iteration = 0
    while improved and time.time() < deadline:
        improved = False
        iteration += 1
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, code size: {len(code_list)}", flush=True)
        
        # Try removing each codeword
        for i in range(len(code_list)):
            if time.time() > deadline:
                break
            candidate = code_list[:i] + code_list[i+1:]
            if is_valid_coverage(candidate):
                code_list = candidate
                coverage = compute_coverage(code_list)
                improved = True
                if len(code_list) < best_size:
                    best_size = len(code_list)
                    best_code = list(code_list)
                break  # restart removal loop
        
        if time.time() > deadline:
            break
        
        # Try swapping: replace codeword i with a random other word, then try to remove
        for i in range(len(code_list)):
            if time.time() > deadline:
                break
            original = code_list[i]
            # Try a few random replacements
            for attempt in range(5):
                new_word_idx = random.randint(0, space_size - 1)
                if new_word_idx in code_list:
                    continue
                candidate = code_list[:]
                candidate[i] = new_word_idx
                if not is_valid_coverage(candidate):
                    continue
                # Now try to remove some codeword
                for j in range(len(candidate)):
                    if j == i:
                        continue
                    candidate2 = candidate[:j] + candidate[j+1:]
                    if is_valid_coverage(candidate2):
                        code_list = candidate2
                        coverage = compute_coverage(code_list)
                        improved = True
                        if len(code_list) < best_size:
                            best_size = len(code_list)
                            best_code = list(code_list)
                        break
                if improved:
                    break
    
    # Final validation and return
    if time.time() < deadline:
        final_coverage = compute_coverage(best_code)
        if final_coverage == full_mask:
            return [idx_to_word(idx) for idx in best_code]
    
    # Fallback: return initial code if something went wrong
    return initial_code
