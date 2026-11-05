import time
import random
import math

def construct(n):
    if n <= 1:
        return "1"
    
    # Step 1: Generate standard superpermutation for n-1
    prev_word = construct(n-1)
    prev_list = [int(c) for c in prev_word]
    
    # Step 2: Extract the sequence of (n-1)-permutations from prev_word
    seen = set()
    blocks = []
    for i in range(len(prev_list) - (n-2)):
        window = tuple(prev_list[i:i+n-1])
        if len(set(window)) == n-1:
            if window not in seen:
                seen.add(window)
                blocks.append(window)
                
    # Precompute overlap matrix for efficient VNS
    num_blocks = len(blocks)
    block_len = n - 1
    overlap = [[0] * num_blocks for _ in range(num_blocks)]
    
    for i in range(num_blocks):
        for j in range(num_blocks):
            if i == j:
                overlap[i][j] = block_len
                continue
            max_k = min(block_len, block_len)
            for k in range(max_k, 0, -1):
                if blocks[i][-k:] == blocks[j][:k]:
                    overlap[i][j] = k
                    break
    
    # Precompute cost contributions for O(1) delta updates
    # For an ordering, total = base + sum of (1 + block_len - overlap[order[k]][order[k+1]]) for each adjacent pair
    # base = 2 * block_len + 1 (first block: b + [n] + b, but we count the first block as block_len + 1 + block_len)
    # Actually: first block contributes (block_len + 1 + block_len) = 2*block_len + 1
    # Each subsequent block contributes (1 + block_len + block_len - overlap) = 2*block_len + 1 - overlap
    # So total = (2*block_len + 1) + sum_{k=0}^{m-2} (2*block_len + 1 - overlap[order[k]][order[k+1]])
    # where m = num_blocks
    
    def calc_length_from_order(order):
        """Calculate total length given an ordering of block indices."""
        if not order:
            return 0
        total = 2 * block_len + 1
        for idx in range(len(order) - 1):
            curr_idx = order[idx]
            next_idx = order[idx + 1]
            total += 1 + block_len + block_len - overlap[curr_idx][next_idx]
        return total
    
    # For O(1) delta updates, we need to track the current total and be able to compute
    # the change when we modify the ordering.
    # The total depends on adjacent pairs. When we reverse a segment [i..j], only the pairs
    # at the boundaries change: (i-1, i) becomes (i-1, j), and (j, j+1) becomes (i, j+1).
    # Also, the internal pairs are reversed but their overlaps are symmetric? No, overlap is not symmetric.
    
    # Let's use a different approach: maintain the order as a list and compute deltas carefully.
    
    # Initial order: standard construction order (0, 1, 2, ..., num_blocks-1)
    best_order = list(range(num_blocks))
    best_len = calc_length_from_order(best_order)
    
    start_time = time.time()
    time_limit = 135  # Safety margin: evaluator kills at 150s
    
    def check_time():
        return time.time() - start_time < time_limit
    
    # VNS: Variable Neighborhood Search with O(1) delta updates
    # For 2-opt (reverse segment i..j):
    #   Old cost contributions from boundary pairs:
    #     - If i > 0: overlap[best_order[i-1]][best_order[i]]
    #     - If j < num_blocks-1: overlap[best_order[j]][best_order[j+1]]
    #   New cost contributions:
    #     - If i > 0: overlap[best_order[i-1]][best_order[j]]
    #     - If j < num_blocks-1: overlap[best_order[i]][best_order[j+1]]
    #   Delta = new - old
    
    def try_2opt(i, j):
        """Try reversing segment [i..j]. Return (delta, new_order) if improved."""
        nonlocal best_len, best_order
        if i >= j:
            return False
        
        old_delta = 0
        new_delta = 0
        
        # Left boundary: (i-1, i) -> (i-1, j)
        if i > 0:
            old_delta -= overlap[best_order[i-1]][best_order[i]]
            new_delta -= overlap[best_order[i-1]][best_order[j]]
        
        # Right boundary: (j, j+1) -> (i, j+1)
        if j < num_blocks - 1:
            old_delta -= overlap[best_order[j]][best_order[j+1]]
            new_delta -= overlap[best_order[i]][best_order[j+1]]
        
        # Note: the sign is tricky. Total length includes -overlap for each adjacent pair.
        # So if overlap increases, length decreases.
        # Let me re-derive:
        # total = base + sum(2*block_len + 1 - overlap[a][b]) for adjacent pairs (a,b)
        # = base + num_adjacent*(2*block_len+1) - sum(overlap[a][b])
        # So increasing sum of overlaps decreases total length.
        
        # Delta in total = -(new_sum_overlaps - old_sum_overlaps) = old_sum_overlaps - new_sum_overlaps
        # For the boundary pairs:
        # old_sum includes overlap[best_order[i-1]][best_order[i]] (if i>0) and overlap[best_order[j]][best_order[j+1]] (if j<num_blocks-1)
        # new_sum includes overlap[best_order[i-1]][best_order[j]] (if i>0) and overlap[best_order[i]][best_order[j+1]] (if j<num_blocks-1)
        
        delta = 0
        if i > 0:
            delta += overlap[best_order[i-1]][best_order[i]] - overlap[best_order[i-1]][best_order[j]]
        if j < num_blocks - 1:
            delta += overlap[best_order[j]][best_order[j+1]] - overlap[best_order[i]][best_order[j+1]]
        
        if delta < 0:
            # Accept: reverse the segment
            best_order[i:j+1] = best_order[i:j+1][::-1]
            best_len += delta
            return True
        return False
    
    def try_oropt(i, j, new_pos):
        """Try moving segment [i..j] to position new_pos. Return True if improved."""
        nonlocal best_len, best_order
        if i >= j:
            return False
        
        # Remove segment [i..j] from best_order, then insert at new_pos
        # This is more complex. Let's compute the delta carefully.
        
        # When we remove [i..j], the adjacent pairs affected are:
        # - (i-1, i) removed (if i > 0)
        # - (j, j+1) removed (if j < num_blocks-1)
        # - (i-1, j+1) added (if i > 0 and j < num_blocks-1)
        
        # Then we insert at new_pos in the remaining list.
        # The remaining list has elements: [0..i-1] + [j+1..num_blocks-1]
        # We insert the segment after position new_pos in this remaining list.
        
        # This is getting complex. Let's just compute the new order and compare.
        # For efficiency, let's limit Or-opt to small segments.
        
        segment = best_order[i:j+1]
        rest = best_order[:i] + best_order[j+1:]
        
        # Adjust new_pos for the removal
        if new_pos > j:
            new_pos -= (j - i + 1)
        
        new_order = rest[:new_pos] + segment + rest[new_pos:]
        
        new_len = calc_length_from_order(new_order)
        if new_len < best_len:
            best_order = new_order
            best_len = new_len
            return True
        return False
    
    # Local search with multiple neighborhood structures
    improved = True
    while improved and check_time():
        improved = False
        
        # Neighborhood 1: 2-opt (reverse segments) - O(1) delta
        for i in range(num_blocks - 1):
            if not check_time():
                break
            for j in range(i + 1, num_blocks):
                if not check_time():
                    break
                if try_2opt(i, j):
                    improved = True
        
        # Neighborhood 2: Or-opt (move segments) - O(n!) but only for small segments
        if not check_time():
            break
        for seg_len in [1, 2]:
            if not check_time():
                break
            for i in range(num_blocks - seg_len + 1):
                if not check_time():
                    break
                for j in range(i + seg_len, num_blocks):
                    if not check_time():
                        break
                    if try_oropt(i, i + seg_len - 1, j):
                        improved = True
    
    # Reconstruct the word from best_order
    word = []
    for idx in best_order:
        block = blocks[idx]
        # Add block, then n, then block (with overlap)
        if not word:
            word.extend(block)
            word.append(n)
            word.extend(block)
        else:
            # Find overlap between end of current word and start of (block + [n] + block)
            full_block = list(block) + [n] + list(block)
            max_overlap = min(len(word), len(full_block))
            actual_overlap = 0
            for k in range(max_overlap, 0, -1):
                if word[-k:] == full_block[:k]:
                    actual_overlap = k
                    break
            word.extend(full_block[actual_overlap:])
    
    return "".join(map(str, word))
