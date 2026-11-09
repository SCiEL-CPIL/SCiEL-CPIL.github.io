"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys


def construct(q, n, radius):
    """Construct a covering code using bitset-based greedy + local search."""
    start_time = time.time()
    deadline = 145.0  # 5s safety margin before 150s kill
    
    # Precompute all words as tuples
    all_words = list(itertools.product(range(q), repeat=n))
    num_words = len(all_words)
    
    # Build word_to_idx mapping
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbor indices for each word (radius=1 case)
    # neighbors[i] = list of indices j such that d(i,j) <= radius
    neighbors = []
    for idx, word in enumerate(all_words):
        nbrs = [idx]
        for pos in range(n):
            orig = word[pos]
            for val in range(q):
                if val == orig:
                    continue
                new_word = list(word)
                new_word[pos] = val
                new_idx = word_to_idx[tuple(new_word)]
                nbrs.append(new_idx)
        neighbors.append(nbrs)
    
    # Bitset representation of coverage
    # covered_bits is a Python int where bit i is set if word i is covered
    covered_bits = 0
    full_mask = (1 << num_words) - 1
    
    # Greedy initialization
    codewords = []
    codeword_indices = []
    uncovered_bits = full_mask
    
    # Time budget for greedy phase
    greedy_deadline = start_time + 20.0
    
    while uncovered_bits != 0:
        if time.time() > greedy_deadline:
            break
        
        # Find the word that covers the most uncovered words
        best_idx = -1
        best_count = -1
        
        # Sample candidate positions from uncovered words
        # Get some uncovered indices
        unc_list = []
        tmp = uncovered_bits
        while tmp and len(unc_list) < 200:
            # Find lowest set bit
            lowbit = tmp & (-tmp)
            idx = lowbit.bit_length() - 1
            unc_list.append(idx)
            tmp &= tmp - 1
        
        # Also check some random positions
        for _ in range(min(100, num_words)):
            unc_list.append(random.randint(0, num_words - 1))
        
        for idx in unc_list:
            if not (uncovered_bits >> idx) & 1:
                continue
            count = 0
            for nbr in neighbors[idx]:
                if (uncovered_bits >> nbr) & 1:
                    count += 1
            if count > best_count:
                best_count = count
                best_idx = idx
        
        if best_idx == -1:
            # Fallback: find any uncovered word
            tmp = uncovered_bits
            while tmp:
                lowbit = tmp & (-tmp)
                best_idx = lowbit.bit_length() - 1
                break
            if best_idx == -1:
                break
        
        # Add this codeword
        codewords.append(all_words[best_idx])
        codeword_indices.append(best_idx)
        
        # Update coverage
        for nbr in neighbors[best_idx]:
            uncovered_bits &= ~(1 << nbr)
        
        if time.time() > greedy_deadline:
            break
    
    # If greedy didn't finish, complete with simple coverage
    if uncovered_bits != 0:
        tmp = uncovered_bits
        while tmp:
            lowbit = tmp & (-tmp)
            idx = lowbit.bit_length() - 1
            if idx not in codeword_indices:
                codewords.append(all_words[idx])
                codeword_indices.append(idx)
            for nbr in neighbors[idx]:
                uncovered_bits &= ~(1 << nbr)
            tmp &= tmp - 1
            if time.time() > deadline - 10:
                break
    
    # Local search phase: try to remove codewords
    ls_deadline = deadline - 5.0
    
    improved = True
    while improved and time.time() < ls_deadline:
        improved = False
        
        # Try removing each codeword in order
        for i in range(len(codewords)):
            if time.time() > ls_deadline:
                break
            
            # Check if removing codeword i still covers all
            # Compute coverage without codeword i
            test_covered = 0
            for j, ci in enumerate(codeword_indices):
                if j == i:
                    continue
                for nbr in neighbors[ci]:
                    test_covered |= (1 << nbr)
            
            if test_covered == full_mask:
                # Remove codeword i
                codewords.pop(i)
                codeword_indices.pop(i)
                improved = True
                break
    
    # Local search: swap operations
    swap_deadline = deadline - 3.0
    
    # Build a set of current codeword indices for quick lookup
    cw_set = set(codeword_indices)
    
    for iteration in range(5000):
        if time.time() > swap_deadline:
            break
        
        if len(codewords) < 2:
            break
        
        # Pick a random codeword to potentially remove
        remove_idx = random.randint(0, len(codewords) - 1)
        remove_ci = codeword_indices[remove_idx]
        
        # Compute coverage without this codeword
        test_covered = 0
        for j, ci in enumerate(codeword_indices):
            if j == remove_idx:
                continue
            for nbr in neighbors[ci]:
                test_covered |= (1 << nbr)
        
        uncovered = full_mask & ~test_covered
        
        if uncovered == 0:
            # Can just remove
            codewords.pop(remove_idx)
            codeword_indices.pop(remove_idx)
            cw_set.discard(remove_ci)
            continue
        
        # Find a replacement that covers the most uncovered words
        # Sample from uncovered words
        unc_list = []
        tmp = uncovered
        while tmp and len(unc_list) < 50:
            lowbit = tmp & (-tmp)
            idx = lowbit.bit_length() - 1
            unc_list.append(idx)
            tmp &= tmp - 1
        
        best_add = -1
        best_gain = -1
        
        for idx in unc_list:
            if idx in cw_set:
                continue
            gain = 0
            for nbr in neighbors[idx]:
                if (uncovered >> nbr) & 1:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_add = idx
        
        if best_add != -1 and best_gain > 0:
            # Perform the swap
            codewords[remove_idx] = all_words[best_add]
            codeword_indices[remove_idx] = best_add
            cw_set.discard(remove_ci)
            cw_set.add(best_add)
    
    # Final: remove any remaining redundant codewords
    final_deadline = deadline - 1.0
    improved = True
    while improved and time.time() < final_deadline:
        improved = False
        for i in range(len(codewords)):
            if time.time() > final_deadline:
                break
            test_covered = 0
            for j, ci in enumerate(codeword_indices):
                if j == i:
                    continue
                for nbr in neighbors[ci]:
                    test_covered |= (1 << nbr)
            if test_covered == full_mask:
                codewords.pop(i)
                codeword_indices.pop(i)
                improved = True
                break
    
    # Remove duplicates
    seen = set()
    unique_codewords = []
    for cw in codewords:
        key = tuple(cw)
        if key not in seen:
            seen.add(key)
            unique_codewords.append(cw)
    
    # Final validation
    final_covered = 0
    for cw in unique_codewords:
        ci = word_to_idx[tuple(cw)]
        for nbr in neighbors[ci]:
            final_covered |= (1 << nbr)
    
    if final_covered != full_mask:
        # Fallback: add missing coverage
        uncovered = full_mask & ~final_covered
        tmp = uncovered
        while tmp:
            lowbit = tmp & (-tmp)
            idx = lowbit.bit_length() - 1
            if idx not in seen:
                unique_codewords.append(all_words[idx])
                seen.add(tuple(all_words[idx]))
            for nbr in neighbors[idx]:
                uncovered &= ~(1 << nbr)
            tmp &= tmp - 1
    
    return unique_codewords
# EVOLVE-BLOCK-END
