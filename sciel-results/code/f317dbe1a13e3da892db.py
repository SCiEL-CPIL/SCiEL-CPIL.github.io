import time
import random
import math

def construct(n):
    if n <= 1:
        return "1"
    
    # Recursively build the superpermutation, but at the top level (n=7) we use VNS.
    # For n < 7, we use the standard recursive construction to get the base blocks.
    
    if n < 7:
        # Standard recursive construction for smaller n
        if n == 1:
            return "1"
        prev_word = construct(n - 1)
        prev_list = [int(c) for c in prev_word]
        
        # Extract blocks
        seen = set()
        blocks = []
        for i in range(len(prev_list) - (n - 2)):
            window = tuple(prev_list[i:i + n - 1])
            if len(set(window)) == n - 1 and window not in seen:
                seen.add(window)
                blocks.append(window)
        
        # Reconstruct
        word = []
        for p in blocks:
            block = list(p) + [n] + list(p)
            # Find max overlap
            max_k = min(len(word), len(block))
            overlap = 0
            for k in range(max_k, -1, -1):
                if k == 0:
                    overlap = 0
                    break
                if word[len(word) - k:] == block[:k]:
                    overlap = k
                    break
            word += block[overlap:]
        return "".join(map(str, word))
    
    # For n = 7, we do the heavy lifting
    # 1. Get the superpermutation for n=6
    prev_word = construct(6)
    prev_list = [int(c) for c in prev_word]
    
    # 2. Extract blocks (permutations of 1..6) in order of first appearance
    seen = set()
    blocks = []
    for i in range(len(prev_list) - 4):
        window = tuple(prev_list[i:i + 6])
        if len(set(window)) == 6 and window not in seen:
            seen.add(window)
            blocks.append(window)
    
    # 3. Precompute overlap matrix
    # overlap[i][j] = max k such that blocks[i][-k:] == blocks[j][:k]
    num_blocks = len(blocks)
    overlap = [[0] * num_blocks for _ in range(num_blocks)]
    
    for i in range(num_blocks):
        bi = blocks[i]
        for j in range(num_blocks):
            bj = blocks[j]
            # Max possible overlap is 5 (since blocks are length 6, and they are permutations, 
            # full overlap 6 would mean bi == bj, which implies i==j, but we handle i!=j mostly)
            # Actually, if i==j, overlap is 6.
            max_k = 6
            for k in range(max_k, -1, -1):
                if k == 0:
                    break
                if bi[len(bi) - k:] == bj[:k]:
                    overlap[i][j] = k
                    break
    
    # 4. Define cost function for a given ordering
    # Cost = sum of (1 + len(next_block) - overlap[prev][next]) for each transition
    # Base cost = len(first_block) + 1 + len(first_block) ?
    # Let's trace the word construction for a specific order [b0, b1, b2]
    # Word starts with b0.
    # Then we add n.
    # Then we add b0 again? No.
    # Standard construction:
    # S_n = b0 n b0 b1 n b1 ...
    # Let's check the length formula from the seed:
    # total_len = len(b0) + 1 + len(b0)
    # added for b1: 1 + len(b1) + len(b1) - overlap(b0, b1)
    # This implies the segment for b0 is b0 n b0.
    # The segment for b1 is b1 n b1.
    # The overlap is between the suffix of (b0 n b0) which is b0, and prefix of (b1 n b1) which is b1.
    # So yes, the cost of transition i->j is 1 + len(bj) - overlap[i][j] ?
    # Wait, the seed formula was: added = 1 + len(next_b) + len(next_b) - overlap.
    # This looks like it's adding the full next block twice?
    # Let's re-read the seed's calc_length.
    # "total_len = len(block_order[0]) + 1 + len(block_order[0])" -> len(b0) + 1 + len(b0)
    # "added = 1 + len(next_b) + len(next_b) - overlap"
    # This implies that for each block b, we are adding len(b) + 1 + len(b) - overlap_from_prev?
    # No, the first block adds len(b0) + 1 + len(b0).
    # Subsequent blocks add 1 + 2*len(bj) - overlap.
    # This corresponds to the string: b0 n b0 b1 n b1 ...
    # Let's verify length of "b0 n b0 b1 n b1"
    # If b0 = "12", b1 = "21".
    # b0 n b0 = "12312" (len 5)
    # b1 n b1 = "21321" (len 5)
    # Overlap between "12312" and "21321"?
    # Suffix of "12312" is "12". Prefix of "21321" is "21". No overlap?
    # Wait, the seed says overlap is between b0 and b1.
    # Suffix of b0 is "2". Prefix of b1 is "2". Overlap 1.
    # Total length = 5 + (1 + 5 - 1) = 10.
    # Actual string: "12312" + "321" (b1 n b1 minus overlap "2")?
    # "12312" + "321" = "12312321". Length 8.
    # My manual count: 1,2,3,1,2,3,2,1.
    # Does it contain 12, 21?
    # 12312321
    # Windows of size 2: 12, 23, 31, 12, 23, 32, 21.
    # Yes.
    # So the formula in the seed seems to be:
    # Cost(order) = len(b0) + 1 + len(b0) + sum_{i=1}^{k-1} (1 + 2*len(bi) - overlap[order[i-1]][order[i]])
    # Since all blocks have length n-1, let L = n-1.
    # Cost = 2L + 1 + sum_{i=1}^{k-1} (1 + 2L - overlap)
    # Cost = 2L + 1 + (k-1)(1+2L) - sum(overlaps)
    # To minimize cost, we need to maximize sum of overlaps.
    
    def get_cost(order):
        # order is a list of indices
        if not order:
            return 0
        L = n - 1
        k = len(order)
        # Base
        cost = 2 * L + 1
        for i in range(1, k):
            cost += 1 + 2 * L - overlap[order[i-1]][order[i]]
        return cost

    # 5. VNS Search
    start_time = time.time()
    time_limit = 115
    
    # Initial order: standard order (0 to num_blocks-1)
    current_order = list(range(num_blocks))
    current_cost = get_cost(current_order)
    
    best_order = current_order[:]
    best_cost = current_cost
    
    # Local search moves
    def try_2opt(order, cost):
        # Try reversing a segment
        n_blocks = len(order)
        for i in range(n_blocks - 1):
            for j in range(i + 1, n_blocks):
                # Calculate delta cost
                # Affected edges: (i-1, i), (i, i+1) ... (j, j+1)
                # Old cost parts:
                # If i > 0: edge (i-1, i)
                # If j < n_blocks - 1: edge (j, j+1)
                # Internal edges i..j are reversed.
                
                # This is complex to calculate delta quickly without re-evaluating the whole sum.
                # Given n_blocks = 720, and we have ~115s, we can afford O(N) per move?
                # 720^2 = 518,400 moves.
                # If each move takes O(N) = 720 ops, total 370M ops. Might be too slow in Python.
                # We need O(1) delta calculation.
                
                # Let's stick to simple 2-opt (swap adjacent) and Or-opt (move single block) first?
                # Or just re-evaluate if it's fast enough with NumPy?
                # No, let's use the property that cost is sum of edge weights.
                
                # Cost = C_base + sum_{i=0}^{k-2} (1 + 2L - overlap[order[i]][order[i+1]])
                # Let W[i][j] = 1 + 2L - overlap[i][j]
                # Cost = C_base + sum W[order[i]][order[i+1]]
                
                # For 2-opt reversing [i, j]:
                # Old edges: (i-1, i), (i, i+1), ..., (j-1, j), (j, j+1)
                # New edges: (i-1, j), (j, j-1), ..., (i+1, i), (i, j+1)
                
                delta = 0
                if i > 0:
                    delta -= W[order[i-1]][order[i]]
                    delta += W[order[i-1]][order[j]]
                if j < n_blocks - 1:
                    delta -= W[order[j]][order[j+1]]
                    delta += W[order[i]][order[j+1]]
                
                # Internal edges
                # Old: W[order[i]][order[i+1]] + ... + W[order[j-1]][order[j]]
                # New: W[order[j]][order[j-1]] + ... + W[order[i+1]][order[i]]
                
                # We can precompute prefix sums of W along the current order?
                # But W is not fixed, it depends on the indices.
                # However, for a fixed order, the internal cost change is:
                # sum_{m=i}^{j-1} W[order[m]][order[m+1]] - sum_{m=i}^{j-1} W[order[j - (m-i)]][order[j - (m-i) - 1]]
                # This is still O(j-i).
                
                # Given the time limit, maybe we should just do random restarts with greedy construction?
                # Or simple adjacent swaps?
                
                # Let's try adjacent swaps (2-opt with j=i+1)
                # This is O(1) delta.
                
                pass # Placeholder, we'll implement adjacent swaps and random moves in the main loop

    # Precompute W
    W = [[0] * num_blocks for _ in range(num_blocks)]
    L = n - 1
    for i in range(num_blocks):
        for j in range(num_blocks):
            W[i][j] = 1 + 2 * L - overlap[i][j]
    
    # Main VNS loop
    # We will use a mix of 2-opt (reversal) and random restarts if stuck.
    # To make 2-opt fast, we can precompute the sum of internal edges?
    # Actually, for n=720, even O(N) per move is ~720 * 720 * 720 = 370M.
    # Python can do ~10M simple ops/sec. So 370M is ~37s. This is feasible!
    # So we can just calculate the delta in O(j-i) for each 2-opt move.
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        # Iterate through all possible 2-opt moves
        # To avoid getting stuck in a local minimum, we can do a full pass.
        for i in range(num_blocks - 1):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, num_blocks):
                # Calculate delta
                delta = 0
                if i > 0:
                    delta -= W[current_order[i-1]][current_order[i]]
                    delta += W[current_order[i-1]][current_order[j]]
                if j < num_blocks - 1:
                    delta -= W[current_order[j]][current_order[j+1]]
                    delta += W[current_order[i]][current_order[j+1]]
                
                # Internal edges
                # Old sum
                old_sum = 0
                for m in range(i, j):
                    old_sum += W[current_order[m]][current_order[m+1]]
                
                # New sum (reversed)
                new_sum = 0
                for m in range(i, j):
                    # The edge at position m in the new order corresponds to
                    # original indices j-(m-i) and j-(m-i)-1
                    idx1 = current_order[j - (m - i)]
                    idx2 = current_order[j - (m - i) - 1]
                    new_sum += W[idx1][idx2]
                
                delta += new_sum - old_sum
                
                if delta < 0:
                    # Accept move
                    current_order[i:j+1] = current_order[i:j+1][::-1]
                    current_cost += delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_order = current_order[:]
    
    # Reconstruct word from best_order
    word = []
    for idx in best_order:
        block = list(blocks[idx]) + [n] + list(blocks[idx])
        max_k = min(len(word), len(block))
        overlap_val = 0
        for k in range(max_k, -1, -1):
            if k == 0:
                overlap_val = 0
                break
            if word[len(word) - k:] == block[:k]:
                overlap_val = k
                break
        word += block[overlap_val:]
    
    return "".join(map(str, word))
