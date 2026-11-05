import time
import random
import math

def construct(n):
    if n <= 1:
        return "1"
    
    # Generate the base superpermutation for n-1 using the standard recursive method
    # This gives us the list of blocks (permutations of 1..n-1) in the standard order
    def get_standard_blocks(m):
        # Returns the list of blocks for a superpermutation of size m
        # A block is a tuple of m-1 distinct digits
        if m == 2:
            # Superpermutation of 2: "121"
            # Blocks of size 1: (1,), (2,)
            return [(1,), (2,)]
        
        # Get blocks for m-1
        prev_blocks = get_standard_blocks(m-1)
        
        # In the standard construction, the order of blocks for m is determined by
        # taking the superpermutation of m-1, identifying the (m-1)-permutations in order,
        # and then for each permutation P, the segment is P + [m] + P.
        # The next segment starts with the next permutation Q.
        # The overlap is between the suffix of P and the prefix of Q.
        
        # Actually, the standard construction is:
        # S_m is built from S_{m-1}.
        # S_{m-1} contains all (m-1)-perms.
        # Let the sequence of (m-1)-perms in S_{m-1} be P_1, P_2, ..., P_k.
        # Then S_m is formed by: P_1, m, P_1, P_2, m, P_2, ..., P_k, m, P_k
        # Wait, that's not quite right. The standard construction is:
        # S_m = P_1, m, P_1, P_2, m, P_2, ...
        # But P_1 and P_2 overlap in the original S_{m-1}.
        # Specifically, if S_{m-1} = w_1 w_2 ... w_L, and the (m-1)-perms appear at indices i_1, i_2, ...
        # The standard construction inserts m after each (m-1)-perm and repeats the perm.
        
        # Let's just generate the standard superpermutation string for m-1 and extract the blocks.
        # This is recursive and might be slow, but n is small (max 7).
        
        prev_word = [1]
        for k in range(2, m):
            perms, seen = [], set()
            for i in range(len(prev_word) - k + 2):
                window = tuple(prev_word[i:i + k - 1])
                if len(set(window)) == k - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [k] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            prev_word = new
            
        # Now extract the (m-1)-permutations from prev_word in order of appearance
        blocks = []
        seen = set()
        for i in range(len(prev_word) - (m - 2)):
            window = tuple(prev_word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                blocks.append(window)
                
        return blocks

    # Get the standard blocks for n
    blocks = get_standard_blocks(n)
    num_blocks = len(blocks)
    
    # Precompute overlaps between all pairs of blocks
    # overlap[i][j] = length of longest suffix of blocks[i] that is a prefix of blocks[j]
    overlap = [[0] * num_blocks for _ in range(num_blocks)]
    for i in range(num_blocks):
        for j in range(num_blocks):
            if i == j:
                overlap[i][j] = 0 # Not used directly in this way, but let's compute it
            b1 = blocks[i]
            b2 = blocks[j]
            max_k = min(len(b1), len(b2))
            for k in range(max_k, 0, -1):
                if b1[-k:] == b2[:k]:
                    overlap[i][j] = k
                    break
    
    # The length of the superpermutation given an order of blocks is:
    # L = len(blocks[0]) + 1 + len(blocks[0]) + sum_{i=1}^{k-1} (1 + len(blocks[i]) + len(blocks[i]) - overlap[order[i-1]][order[i]])
    # L = 2 * len(block) + 1 + sum_{i=1}^{k-1} (1 + 2 * len(block) - overlap[order[i-1]][order[i]])
    # Since all blocks have the same length (n-1):
    # L = 2*(n-1) + 1 + sum_{i=1}^{k-1} (1 + 2*(n-1) - overlap[order[i-1]][order[i]])
    # L = 2*(n-1) + 1 + (k-1) + 2*(n-1)*(k-1) - sum_{i=1}^{k-1} overlap[order[i-1]][order[i]]
    # L = 2*(n-1)*k + k - sum_{i=1}^{k-1} overlap[order[i-1]][order[i]]
    # To minimize L, we need to maximize the sum of overlaps.
    
    def calc_length(order):
        # order is a list of indices into blocks
        total_overlap = 0
        for i in range(len(order) - 1):
            total_overlap += overlap[order[i]][order[i+1]]
        return 2 * (n - 1) * len(order) + len(order) - total_overlap
    
    # Initialize with standard order
    best_order = list(range(num_blocks))
    best_len = calc_length(best_order)
    
    start_time = time.time()
    time_limit = 110
    
    # Variable Neighborhood Search (VNS)
    # Neighborhoods:
    # 1. Swap two blocks
    # 2. Or-opt: move a segment of length 1, 2, or 3 to a new position
    # 3. 2-opt: reverse a segment
    
    def vns(order, current_len):
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            
            # Neighborhood 1: Swap
            for i in range(num_blocks - 1):
                if time.time() - start_time > time_limit:
                    break
                for j in range(i + 1, num_blocks):
                    new_order = order[:]
                    new_order[i], new_order[j] = new_order[j], new_order[i]
                    new_len = calc_length(new_order)
                    if new_len < current_len:
                        current_len = new_len
                        order = new_order
                        improved = True
            
            if time.time() - start_time > time_limit:
                break
                
            # Neighborhood 2: Or-opt (move segment of length 1, 2, 3)
            for seg_len in [1, 2, 3]:
                if time.time() - start_time > time_limit:
                    break
                for i in range(num_blocks - seg_len):
                    if time.time() - start_time > time_limit:
                        break
                    segment = order[i:i+seg_len]
                    for j in range(num_blocks - seg_len + 1):
                        if j == i or (j < i + seg_len and j >= i - seg_len):
                            continue
                        new_order = order[:]
                        # Remove segment
                        del new_order[i:i+seg_len]
                        # Adjust j if j > i
                        if j > i:
                            j -= seg_len
                        # Insert segment at j
                        new_order[j:j] = segment
                        new_len = calc_length(new_order)
                        if new_len < current_len:
                            current_len = new_len
                            order = new_order
                            improved = True
                            
            if time.time() - start_time > time_limit:
                break
                
            # Neighborhood 3: 2-opt (reverse segment)
            for i in range(num_blocks - 1):
                if time.time() - start_time > time_limit:
                    break
                for j in range(i + 1, num_blocks):
                    new_order = order[:]
                    new_order[i:j+1] = new_order[i:j+1][::-1]
                    new_len = calc_length(new_order)
                    if new_len < current_len:
                        current_len = new_len
                        order = new_order
                        improved = True
                        
        return order, current_len
    
    best_order, best_len = vns(best_order, best_len)
    
    # Reconstruct the word from best_order
    # The word is: block[order[0]], n, block[order[0]], block[order[1]], n, block[order[1]], ...
    # Wait, the structure is:
    # Start with block[order[0]]
    # Then n
    # Then block[order[0]] (again)
    # Then for each next block order[i]:
    #   Overlap with previous block
    #   Add n
    #   Add block[order[i]]
    #   Add block[order[i]] (again) -- wait, no.
    
    # Let's trace the standard construction again.
    # S_n = P_1, n, P_1, P_2, n, P_2, ...
    # This means:
    # 1. Output P_1
    # 2. Output n
    # 3. Output P_1 (this completes the first "P n P")
    # 4. For next P_2:
    #    - The current end is P_1.
    #    - We need to output P_2, n, P_2.
    #    - But P_1 and P_2 might overlap.
    #    - Actually, the standard construction doesn't overlap P_1 and P_2 in this way.
    #    - The standard construction is:
    #      S_n = P_1, n, P_1, P_2, n, P_2, ...
    #      This is a concatenation. The overlap is handled by the fact that
    #      P_1 (end of first part) and P_2 (start of second part) might share characters?
    #      No, in the standard construction, the blocks are distinct permutations.
    #      The overlap is between the suffix of the previous P and the prefix of the next P.
    #      But in the string "P_1 n P_1 P_2 n P_2", the transition is from P_1 to P_2.
    #      So we can overlap the suffix of P_1 with the prefix of P_2.
    
    # So the reconstruction is:
    word = []
    first_block = blocks[best_order[0]]
    word.extend(first_block)
    word.append(n)
    word.extend(first_block)
    
    for i in range(1, len(best_order)):
        prev_block = blocks[best_order[i-1]]
        curr_block = blocks[best_order[i]]
        
        # Find overlap between suffix of prev_block and prefix of curr_block
        max_k = min(len(prev_block), len(curr_block))
        overlap_len = 0
        for k in range(max_k, 0, -1):
            if prev_block[-k:] == curr_block[:k]:
                overlap_len = k
                break
        
        # Add n
        word.append(n)
        # Add curr_block, skipping the overlapping part
        word.extend(curr_block[overlap_len:])
        
        # Wait, the structure is: ... prev_block, n, curr_block, n, curr_block
        # But we already added prev_block.
        # So we add n, then curr_block (with overlap), then n, then curr_block?
        # No, the pattern is P, n, P, P', n, P'
        # So after the first P, n, P, we have P', n, P'
        # The transition is from P to P'.
        # So we add n, then P' (with overlap), then n, then P'?
        # No, the "n, P" is part of the next block's segment.
        # Let's re-read the standard construction.
        # S_n = P_1, n, P_1, P_2, n, P_2, ...
        # This is:
        # [P_1] [n] [P_1] [P_2] [n] [P_2] ...
        # So after [P_1] [n] [P_1], we add [P_2] [n] [P_2].
        # The overlap is between [P_1] (end of previous) and [P_2] (start of next).
        # So we add P_2 (with overlap), then n, then P_2.
        
        # But in my loop above, I added n before curr_block.
        # That would be: ... prev_block, n, curr_block, ...
        # This matches the pattern if we consider that the 'n' belongs to the transition.
        # But the pattern is P, n, P.
        # So the 'n' is between the two copies of P.
        # In the sequence P_1, n, P_1, P_2, n, P_2:
        # The first 'n' is between the two P_1s.
        # The second 'n' is between the two P_2s.
        # So the structure is:
        # P_1
        # n
        # P_1 (end of first segment)
        # P_2 (start of second segment, overlaps with P_1)
        # n
        # P_2 (end of second segment)
        
        # So my reconstruction should be:
        # Start: P_1, n, P_1
        # For each next P_i:
        #   Overlap P_{i-1} and P_i
        #   Add P_i[overlap:]
        #   Add n
        #   Add P_i
        
        # Let's fix the reconstruction.
        
    # Let's restart the reconstruction logic.
    word = []
    first_block = blocks[best_order[0]]
    word.extend(first_block)
    word.append(n)
    word.extend(first_block)
    
    for i in range(1, len(best_order)):
        prev_block = blocks[best_order[i-1]]
        curr_block = blocks[best_order[i]]
        
        # Find overlap between suffix of prev_block and prefix of curr_block
        max_k = min(len(prev_block), len(curr_block))
        overlap_len = 0
        for k in range(max_k, 0, -1):
            if prev_block[-k:] == curr_block[:k]:
                overlap_len = k
                break
        
        # Add curr_block, skipping the overlapping part
        word.extend(curr_block[overlap_len:])
        # Add n
        word.append(n)
        # Add curr_block again
        word.extend(curr_block)
        
    return "".join(map(str, word))
