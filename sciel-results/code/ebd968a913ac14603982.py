import time
import math

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    
    # Generate the base superpermutation for n-1 to extract blocks
    # We need the list of blocks (permutations of 1..n-1) in the order they appear
    # in the standard recursive construction of S_{n-1}.
    
    def get_blocks(m):
        """Returns the list of blocks (tuples of length m) in standard recursive order for S_m."""
        if m == 1:
            return [(1,)]
        if m == 2:
            return [(1, 2), (2, 1)]
        
        prev_blocks = get_blocks(m - 1)
        # The standard construction S_m = P_1 m P_1 P_2 m P_2 ...
        # The blocks of S_m are just the blocks of S_{m-1} in the same order?
        # No, the blocks of S_m are the permutations of 1..m-1 that appear in S_m.
        # In S_m = P_1 m P_1 P_2 m P_2 ..., the windows of length m-1 are:
        # - Inside P_1: windows from P_1
        # - Crossing P_1 m: last m-2 of P_1 + m (not a perm of 1..m-1)
        # - Inside m P_1: m + first m-2 of P_1 (not a perm)
        # - Inside P_1 P_2: last part of P_1 + first part of P_2?
        # Actually, the set of blocks in S_m is exactly the set of blocks in S_{m-1}.
        # The order is determined by the first appearance.
        # In the standard construction, the order of blocks in S_m is the same as in S_{m-1}.
        # Let's verify for m=3.
        # S_2 blocks: (1,2), (2,1).
        # S_3 = 12 3 12 21 3 21.
        # Windows of length 2:
        # 12, 23(no), 31(no), 12, 21, 13(no), 32(no), 21.
        # Perms of 1,2: 12, 21.
        # Order of first appearance: 12 (at 0), 21 (at 4).
        # So blocks are [(1,2), (2,1)]. Same order.
        
        # For m=4.
        # S_3 blocks: (1,2,3), (2,1,3), (2,3,1), (3,2,1), (3,1,2), (1,3,2). 
        # Wait, S_3 = 123121321.
        # Perms of 1,2,3:
        # 123 (0)
        # 231 (1)
        # 312 (2)
        # 121 (no)
        # 213 (4)
        # 132 (5)
        # 321 (6)
        # So order: 123, 231, 312, 213, 132, 321.
        
        # If the order is always the same, we can just compute the standard order recursively.
        # However, we want to REORDER them. So we just need the SET of blocks.
        # But the "standard" starting point for optimization should be the standard order.
        
        # Let's just generate the standard S_{n-1} string and extract blocks from it.
        # This is safer.
        pass

    # Generate standard S_{n-1}
    def standard_superperm(m):
        if m == 1:
            return [1]
        if m == 2:
            return [1, 2, 1]
        prev = standard_superperm(m - 1)
        # Extract blocks of length m-1 from prev
        blocks = []
        seen = set()
        for i in range(len(prev) - (m - 1) + 1):
            w = tuple(prev[i:i + m - 1])
            if len(set(w)) == m - 1 and w not in seen:
                seen.add(w)
                blocks.append(w)
        
        res = []
        for b in blocks:
            res.extend(b)
            res.append(m)
            res.extend(b)
        return res

    # Get blocks for n-1
    prev_word = standard_superperm(n - 1)
    block_size = n - 1
    blocks = []
    seen = set()
    for i in range(len(prev_word) - block_size + 1):
        w = tuple(prev_word[i:i + block_size])
        if len(set(w)) == block_size and w not in seen:
            seen.add(w)
            blocks.append(w)
            
    k_blocks = len(blocks)
    # k_blocks should be (n-1)!
    
    # Precompute overlap matrix
    # overlap[i][j] = max k such that blocks[i][-k:] == blocks[j][:k]
    overlap = [[0] * k_blocks for _ in range(k_blocks)]
    for i in range(k_blocks):
        bi = blocks[i]
        for j in range(k_blocks):
            if i == j:
                continue
            bj = blocks[j]
            max_k = block_size
            # Check from largest possible overlap down
            # Since they are permutations, overlap can be at most block_size - 1 
            # (if they are different, they can't be identical, so max overlap is block_size-1? 
            # No, if bi=(1,2,3) and bj=(2,3,4), overlap is 2.
            # If bi=(1,2,3) and bj=(1,2,3), overlap is 3, but i!=j so this is not possible for distinct perms?
            # Wait, distinct permutations CAN have large overlaps.
            # Example: 123 and 234 (if n=4). Overlap 2.
            # Max overlap is block_size - 1?
            # If bi ends in 12 and bj starts with 12, overlap 2.
            # If bi is 123 and bj is 231, overlap 1 (3? no. 23? no. 31? no. 1? yes if bi ends 1 and bj starts 1? No bi ends 3).
            # Actually, max overlap is block_size - 1 is not strictly true.
            # But it's small.
            for k in range(block_size, 0, -1):
                if bi[-k:] == bj[:k]:
                    overlap[i][j] = k
                    break

    # Initial order: standard order (index 0 to k_blocks-1)
    current_order = list(range(k_blocks))
    
    # Function to calculate total overlap sum
    def calc_total_overlap(order):
        total = 0
        for i in range(len(order) - 1):
            total += overlap[order[i]][order[i+1]]
        return total

    best_overlap = calc_total_overlap(current_order)
    best_order = current_order[:]
    
    start_time = time.time()
    time_limit = 140 # Leave 10s for construction and safety
    
    # 2-Opt Local Search
    # We maintain prefix sums of overlaps for O(1) cost calculation of 2-opt moves.
    # P_f[i] = sum_{m=0}^{i-1} overlap[order[m]][order[m+1]]
    # P_b[i] = sum_{m=1}^{i} overlap[order[m]][order[m-1]]
    
    def get_prefix_sums(order):
        pf = [0] * len(order)
        pb = [0] * len(order)
        for i in range(1, len(order)):
            pf[i] = pf[i-1] + overlap[order[i-1]][order[i]]
            pb[i] = pb[i-1] + overlap[order[i]][order[i-1]]
        return pf, pb

    pf, pb = get_prefix_sums(current_order)
    total_overlap = pf[-1]
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i in range(k_blocks - 1):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, k_blocks):
                # Reverse segment [i, j]
                # Old edges:
                # 1. (i-1, i) if i > 0
                # 2. Internal: sum_{m=i}^{j-1} overlap[order[m]][order[m+1]] = pf[j] - pf[i]
                # 3. (j, j+1) if j < k_blocks-1
                
                # New edges:
                # 1. (i-1, j) if i > 0
                # 2. Internal reversed: sum_{m=i+1}^{j} overlap[order[m]][order[m-1]] = pb[j] - pb[i]
                # 3. (i, j+1) if j < k_blocks-1
                
                old_val = 0
                new_val = 0
                
                if i > 0:
                    old_val += overlap[current_order[i-1]][current_order[i]]
                    new_val += overlap[current_order[i-1]][current_order[j]]
                
                old_val += pf[j] - pf[i]
                new_val += pb[j] - pb[i]
                
                if j < k_blocks - 1:
                    old_val += overlap[current_order[j]][current_order[j+1]]
                    new_val += overlap[current_order[i]][current_order[j+1]]
                
                if new_val > old_val:
                    # Apply move
                    current_order[i:j+1] = current_order[i:j+1][::-1]
                    total_overlap += new_val - old_val
                    if total_overlap > best_overlap:
                        best_overlap = total_overlap
                        best_order = current_order[:]
                    # Update prefix sums?
                    # Instead of updating incrementally (which is complex), we can just recalc if we find improvement?
                    # No, recalc is O(N).
                    # Better: Just mark as improved and restart the scan?
                    # Or update pf, pb.
                    # Given N=720, O(N) update is fast.
                    pf, pb = get_prefix_sums(current_order)
                    improved = True
                    if time.time() - start_time > time_limit:
                        break
            if time.time() - start_time > time_limit:
                break

    # Reconstruct word using best_order
    word = []
    for idx, block_idx in enumerate(best_order):
        block = blocks[block_idx]
        if idx == 0:
            word.extend(block)
            word.append(n)
            word.extend(block)
        else:
            # Append n
            word.append(n)
            # Append block, overlapping with previous block
            prev_block = blocks[best_order[idx-1]]
            # Find overlap
            # We have precomputed overlap[prev_idx][curr_idx]
            ov = overlap[best_order[idx-1]][block_idx]
            word.extend(block[ov:])
            # Append block again?
            # The structure is P_i n P_i.
            # We have appended P_{i-1} n P_{i-1} in the previous step?
            # No, the previous step ended with P_{i-1}.
            # So we are at ... P_{i-1}.
            # We append n.
            # We append P_i (with overlap).
            # Now we need to append P_i again?
            # The pattern is P_1 n P_1 P_2 n P_2 ...
            # So yes, after appending P_i (the first copy), we must append P_i (the second copy).
            # But wait, the "second copy" of P_i is actually the start of the next segment's overlap context?
            # No, the next segment is P_{i+1}.
            # The end of the current segment is P_i.
            # So we just need to ensure the word ends with P_i.
            # In the step "Append P_i (with overlap)", we appended `block[ov:]`.
            # Does this result in the word ending with `block`?
            # Yes, because `block` is a permutation, and we are appending its suffix.
            # Wait, if `ov` is small, we append most of `block`.
            # The word ends with `block`.
            # So we are done for this block?
            # Let's check n=3.
            # Blocks: 12, 21.
            # Order: 12, 21.
            # Step 0: Block 12.
            # Word: 12, 3, 12.
            # Ends with 12.
            # Step 1: Block 21.
            # Append 3. Word: 12 3 12 3.
            # Append 21 with overlap.
            # Overlap(12, 21) is 1 (digit 2).
            # Append 21[1:] = 1.
            # Word: 12 3 12 3 1.
            # This is 1231231.
            # This is NOT a superpermutation. It misses 213, 132, 321?
            # 123, 231, 312, 123, 231, 312.
            # It only has 3 permutations.
            
            # The standard construction is S_n = P_1 n P_1 P_2 n P_2 ...
            # This means we append P_i, then n, then P_i.
            # So for each block, we add 2 * |P_i| + 1 characters, minus overlaps.
            # My loop structure:
            # 1. Append block (first copy)
            # 2. Append n
            # 3. Append block (second copy)
            
            # Let's re-evaluate the reconstruction.
            # For i=0:
            #   word.extend(block)
            #   word.append(n)
            #   word.extend(block)
            #   End of word: block.
            
            # For i=1:
            #   word.append(n)
            #   word.extend(block[overlap:]) # This is the FIRST copy of the new block?
            #   word.extend(block) # This is the SECOND copy?
            
            # If I do this, the length is larger.
            # Let's trace n=3 with this logic.
            # Block 12:
            # Word: 12 3 12.
            # Block 21:
            # Append 3. Word: 12 3 12 3.
            # Append 21[1:] = 1. Word: 12 3 12 3 1.
            # Append 21. Word: 12 3 12 3 1 2 1.
            # Length 9.
            # 123123121.
            # Perms: 123, 231, 312, 123, 231, 312, 121(no), 213, 132, 321?
            # 123, 231, 312, 123, 231, 312, 121, 213, 132, 321.
            # Missing 213? No, 213 is at index 6 (0-based 6? 123123121 -> 123, 231, 312, 123, 231, 312, 121, 213, 132, 321).
            # Wait, 123123121 has length 9.
            # Windows:
            # 0: 123
            # 1: 231
            # 2: 312
            # 3: 123
            # 4: 231
            # 5: 312
            # 6: 121 (no)
            # 7: 213
            # 8: 132
            # 9: 321? No, length 9 means indices 0..8.
            # Window at 7 is 121? No.
            # 1 2 3 1 2 3 1 2 1
            # 0: 123
            # 1: 231
            # 2: 312
            # 3: 123
            # 4: 231
            # 5: 312
            # 6: 123
            # 7: 231
            # 8: 312
            # This is just repeating 123123...
            # So my logic is flawed.
            
            # The correct standard construction is:
            # S_n = P_1 n P_1 P_2 n P_2 ...
            # This implies:
            # Start with P_1.
            # Append n.
            # Append P_1.
            # Append P_2 (with overlap with P_1).
            # Append n.
            # Append P_2.
            # Append P_3 (with overlap with P_2).
            # ...
            
            # So for each block P_i:
            # 1. Append n.
            # 2. Append P_i (first copy, with overlap with previous P_{i-1}).
            # 3. Append P_i (second copy, NO overlap with first copy? Or overlap with itself?)
            # If we append P_i twice, the second P_i overlaps with the first P_i completely?
            # If we append P_i, and then append P_i again, the overlap is len(P_i).
            # So we add 0 characters?
            # Then the segment is just P_i n.
            # This leads to the short word which is invalid.
            
            # The key is that the "Second Copy" of P_i is actually the "First Copy" of the NEXT block's context?
            # No.
            # Let's look at S_3 = 123121321.
            # P_1 = 12. P_2 = 21.
            # 12 3 12 21 3 21.
            # 12 (P1)
            # 3 (n)
            # 12 (P1)
            # 21 (P2, overlap with 12 is 1? 12 ends 2, 21 starts 2. So append 1? No, 21 is 2,1. 12 ends 2. So append 1? Then we have 12 1. Not 21.)
            # Wait. 12 21. Overlap is 2? No. 12 and 21. Suffix 2, Prefix 2. Overlap 1.
            # So 12 + 21[1:] = 12 + 1 = 121?
            # But S_3 has 12 21.
            # 1 2 2 1.
            # This contains 22.
            # The standard construction DOES NOT overlap P_1 and P_2 in the way I thought?
            # Or does it?
            # 123121321.
            # 12 3 12 1 3 21?
            # 1 2 3 1 2 1 3 2 1.
            # 12, 23, 31, 12, 21, 13, 32, 21.
            # Blocks: 12, 21.
            # The sequence is 12, 3, 12, 1, 3, 21.
            # This is P_1, n, P_1, P_2[1:], n, P_2.
            # So the overlap is applied to the FIRST copy of the new block?
            # And the SECOND copy is full?
            # Yes.
            
            # So the algorithm is:
            # word = []
            # for i, block in enumerate(ordered_blocks):
            #   if i == 0:
            #     word.extend(block)
            #   else:
            #     # Append block with overlap with previous block
            #     ov = overlap[prev_idx][curr_idx]
            #     word.extend(block[ov:])
            #   word.append(n)
            #   word.extend(block)
            
            # Let's check n=3.
            # Block 12:
            # word: 12
            # append 3: 12 3
            # append 12: 12 3 12
            # Block 21:
            # ov(12, 21) = 1.
            # append 21[1:] = 1. word: 12 3 12 1
            # append 3: 12 3 12 1 3
            # append 21: 12 3 12 1 3 21
            # Result: 123121321.
            # This matches!
            
            pass

    # Reconstruct word correctly
    word = []
    for idx, block_idx in enumerate(best_order):
        block = blocks[block_idx]
        if idx == 0:
            word.extend(block)
        else:
            prev_idx = best_order[idx-1]
            ov = overlap[prev_idx][block_idx]
            word.extend(block[ov:])
        
        word.append(n)
        word.extend(block)
        
    return "".join(map(str, word))
