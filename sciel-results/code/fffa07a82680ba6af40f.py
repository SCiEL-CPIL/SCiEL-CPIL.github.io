"""Improved program for ternary covering code K_3(6,1)."""
# EVOLVE-BLOCK-START
import itertools
import random
import numpy as np

def construct(q, n, radius):
    # For this specific task: q=3, n=6, radius=1
    # We implement a Bitset-based Greedy Set Cover followed by Local Search (Removal)
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters (not expected in this task, but safe)
        # Simple greedy for general case
        all_words = list(itertools.product(range(q), repeat=n))
        covered = set()
        codewords = []
        for c in all_words:
            new_covered = set()
            for i in range(n):
                for v in range(q):
                    if v != c[i]:
                        w = list(c)
                        w[i] = v
                        new_covered.add(tuple(w))
            new_covered.add(tuple(c))
            if new_covered - covered:
                covered.update(new_covered)
                codewords.append(list(c))
        return codewords

    # Specific optimized solution for 3, 6, 1
    total_words = 3 ** 6
    all_words = [list(w) for w in itertools.product(range(3), repeat=6)]
    
    # Map word tuple to index
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    # Precompute coverage masks
    # cover_mask[i] is a bitset of words covered by codeword i
    cover_masks = [0] * total_words
    
    for c_idx, c in enumerate(all_words):
        mask = 0
        # Cover itself
        mask |= (1 << c_idx)
        # Cover neighbors
        for pos in range(6):
            orig_val = c[pos]
            for new_val in range(3):
                if new_val != orig_val:
                    neighbor = c[:]
                    neighbor[pos] = new_val
                    n_idx = word_to_idx[tuple(neighbor)]
                    mask |= (1 << n_idx)
        cover_masks[c_idx] = mask
    
    # Greedy Set Cover
    # uncovered is a bitset of all words initially
    uncovered = (1 << total_words) - 1
    selected_indices = []
    
    # To speed up greedy, we can just iterate. 729 candidates * 729 bits is fast.
    # We pick the one with max popcount(uncovered & mask)
    
    # Optimization: Precompute popcount? No, Python int.bit_count() is fast.
    
    while uncovered:
        best_idx = -1
        best_count = -1
        
        for i in range(total_words):
            if i in selected_indices: # This check is O(N), slow. 
                # Better to maintain a list of available or just check mask
                pass
        
        # Simple approach: iterate all, check if already selected? 
        # Actually, we don't need to skip selected if we track 'uncovered'.
        # If a word is selected, its contribution to 'uncovered' is 0.
        # But we might pick the same one again if we don't track it.
        # So we must track selected.
        
        # Let's use a set for selected for O(1) lookup, or just a list and check membership.
        # 729 items, list membership is fine.
        
        for i in range(total_words):
            if i in selected_indices:
                continue
            count = (uncovered & cover_masks[i]).bit_count()
            if count > best_count:
                best_count = count
                best_idx = i
        
        if best_idx == -1:
            break # Should not happen if radius covers everything
            
        selected_indices.append(best_idx)
        uncovered &= ~cover_masks[best_idx]
        
    # Local Search: Removal
    # We want to remove any codeword that doesn't cover any unique word.
    # A codeword c is removable if cover_masks[c] is a subset of 
    # (union of cover_masks of all other selected codewords).
    
    changed = True
    while changed:
        changed = False
        # Calculate union of all selected masks
        # This is O(N^2) if done naively for each removal check.
        # N is ~75. 75^2 * 729 bits ops is fine.
        
        # Better: For each candidate for removal, compute the union of the rest?
        # Or maintain a "coverage count" array?
        # Let's use the bitset union approach for simplicity and correctness.
        
        for i in range(len(selected_indices)):
            c_idx = selected_indices[i]
            # Union of all others
            union_others = 0
            for j in range(len(selected_indices)):
                if i != j:
                    union_others |= cover_masks[selected_indices[j]]
            
            # Check if c_idx covers any word not in union_others
            if (cover_masks[c_idx] & ~union_others) == 0:
                # Removable
                selected_indices.pop(i)
                changed = True
                break # Restart loop as indices changed
        
    # Convert indices back to lists
    result = [all_words[i] for i in selected_indices]
    
    # Final verification (optional, but good for safety)
    # If verification fails, fallback to a known valid code (e.g. the 81-word one)
    # But greedy+removal should be valid.
    
    return result
# EVOLVE-BLOCK-END
