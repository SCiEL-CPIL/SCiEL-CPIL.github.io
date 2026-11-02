import itertools
import time
import random
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "1234124132413241234"

    # For n=5, 6, 7, use the recursive construction as a baseline.
    # Then, if time permits, try to improve the ordering of blocks for the last step
    # using a heuristic search on the (n-1)-permutations.
    
    # The classic recursive construction builds S_n from S_{n-1} by taking each permutation p
    # of the (n-1)-symbols that appears as a window in S_{n-1}, forming the block p + [n] + p,
    # and concatenating them with maximum overlap.
    # The order of these blocks is determined by their appearance in S_{n-1}.
    # We can try to reorder these blocks to get a better overlap sum.

    # Step 1: Generate the standard S_{n-1}
    def generate_standard_superperm(m):
        word = [1]
        for k in range(2, m + 1):
            perms, seen = [], set()
            for i in range(len(word) - k + 2):
                window = tuple(word[i:i + k - 1])
                if len(set(window)) == k - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [k] + list(p)
                # Find max overlap between new and block
                max_ov = min(len(new), len(block))
                ov = 0
                for o in range(max_ov, -1, -1):
                    if o == 0:
                        break
                    if new[len(new) - o:] == block[:o]:
                        ov = o
                        break
                new += block[ov:]
            word = new
        return word

    # Generate S_{n-1}
    base_word = generate_standard_superperm(n - 1)
    
    # Extract the unique (n-1)-permutations from base_word in order of appearance
    perms, seen = [], set()
    for i in range(len(base_word) - (n - 1) + 1):
        window = tuple(base_word[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
    
    # Number of blocks is (n-1)!
    num_blocks = len(perms)
    
    # Create blocks: p + [n] + p
    blocks = []
    for p in perms:
        blocks.append(list(p) + [n] + list(p))
    
    # Precompute overlap matrix
    # overlaps[i][j] = max k such that blocks[i][-k:] == blocks[j][:k]
    # Note: The maximum possible overlap is n-1 (since the blocks are of form P n P,
    # the suffix is P and prefix is P, so they can overlap by up to len(P)=n-1 if P matches).
    # Actually, the overlap can be larger?
    # Block i: P_i, n, P_i
    # Block j: P_j, n, P_j
    # Suffix of i: ... P_i
    # Prefix of j: P_j ...
    # If P_i == P_j, overlap is n-1.
    # Can it be more?
    # Suffix of i is P_i. Prefix of j is P_j.
    # If P_i[-k:] == P_j[:k], then overlap is at least k.
    # Since P_i and P_j are permutations of 1..n-1, they have length n-1.
    # The max overlap is n-1.
    
    overlaps = [[0] * num_blocks for _ in range(num_blocks)]
    
    # Optimization: Only need to check k from n-1 down to 0.
    # Actually, we can just check k=n-1, n-2, ..., 1.
    # If blocks[i][-k:] == blocks[j][:k], then overlaps[i][j] = k.
    
    # We can vectorize this or just loop. num_blocks is 720 for n=7.
    # 720*720 = 518400 pairs. Each check is O(n). 518400 * 7 is ~3.6M ops. Fast.
    
    for i in range(num_blocks):
        bi = blocks[i]
        for j in range(num_blocks):
            bj = blocks[j]
            # Check max overlap
            max_k = n - 1
            found = 0
            for k in range(max_k, 0, -1):
                if bi[-k:] == bj[:k]:
                    found = k
                    break
            overlaps[i][j] = found
    
    # Now we have the overlap matrix.
    # We want to find a permutation of the blocks that maximizes the sum of overlaps.
    # Total length = sum(len(block)) - sum(overlaps).
    # sum(len(block)) is constant = num_blocks * (2*(n-1) + 1).
    # So we want to maximize sum(overlaps).
    
    # This is the TSP problem on a complete directed graph with edge weights = overlaps.
    # We can use a heuristic: Greedy + 2-Opt.
    
    # Initial solution: The order from the standard construction.
    # The standard construction uses the order in which the permutations appear in S_{n-1}.
    # This is exactly the order in `perms`.
    # So the initial path is [0, 1, ..., num_blocks-1].
    
    path = list(range(num_blocks))
    
    def calc_overlap_sum(path):
        total = 0
        for i in range(len(path) - 1):
            total += overlaps[path[i]][path[i+1]]
        return total
    
    initial_sum = calc_overlap_sum(path)
    
    # 2-Opt local search
    # Try to reverse a segment of the path.
    # Old: ... A, B, C, D ...
    # New: ... A, D, C, B ...
    # Wait, 2-Opt on directed graph is tricky.
    # Standard 2-Opt: Reverse path[i+1..j].
    # Old edges: (A, B), (B, ...), ..., (C, D)
    # New edges: (A, C), (C, ...), ..., (B, D)
    # Where A=path[i], B=path[i+1], C=path[j], D=path[j+1]
    
    # Delta = New_Sum - Old_Sum
    # Old_Sum_segment = ov(A,B) + sum(ov(path[k], path[k+1]) for k in i+1..j-1) + ov(C,D)
    # New_Sum_segment = ov(A,C) + sum(ov(path[k+1], path[k]) for k in i+1..j-1) + ov(B,D)
    # Note: The internal edges are reversed.
    
    # We can precompute the sum of overlaps for the path.
    # And for the reversed segments, we need the reverse overlaps.
    # rev_overlaps[i][j] = overlaps[j][i]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(num_blocks - 2):
            if time.time() > deadline:
                break
            A = path[i]
            B = path[i+1]
            for j in range(i+1, num_blocks - 1):
                if time.time() > deadline:
                    break
                C = path[j]
                D = path[j+1]
                
                # Calculate delta
                # Old: ov(A,B) + ov(C,D) + sum(ov(path[k], path[k+1]) for k in i+1..j-1)
                # New: ov(A,C) + ov(B,D) + sum(ov(path[k+1], path[k]) for k in i+1..j-1)
                
                # We need the sum of overlaps for the segment path[i+1..j].
                # Let's maintain a prefix sum of overlaps for the current path.
                # But the path changes.
                
                # For simplicity and correctness, let's just calculate the delta directly
                # if the segment is short, or use a prefix sum array that gets updated.
                
                # Given num_blocks=720, O(N^3) is 373M ops, which is too slow in Python.
                # We need O(N^2) per iteration or better.
                
                # Let's use a prefix sum of overlaps for the current path.
                # prefix[k] = sum(ov(path[t], path[t+1]) for t in 0..k-1)
                # Then sum(ov(path[i+1..j-1])) = prefix[j-1] - prefix[i+1]
                
                # We also need the sum of reversed overlaps for the segment.
                # rev_prefix[k] = sum(ov(path[t+1], path[t]) for t in 0..k-1)
                
                # Let's maintain these arrays.
                
                pass # Placeholder, logic below
                
    # Let's implement the 2-Opt with prefix sums properly.
    
    def get_prefix_sums(path):
        prefix = [0] * (len(path))
        for i in range(len(path) - 1):
            prefix[i+1] = prefix[i] + overlaps[path[i]][path[i+1]]
        return prefix
    
    def get_rev_prefix_sums(path):
        # rev_prefix[i+1] = rev_prefix[i] + overlaps[path[i+1]][path[i]]
        rev_prefix = [0] * (len(path))
        for i in range(len(path) - 1):
            rev_prefix[i+1] = rev_prefix[i] + overlaps[path[i+1]][path[i]]
        return rev_prefix
    
    # Initialize prefix sums
    prefix = get_prefix_sums(path)
    rev_prefix = get_rev_prefix_sums(path)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(num_blocks - 2):
            if time.time() > deadline:
                break
            A = path[i]
            B = path[i+1]
            
            # We need to find j > i+1 such that reversing path[i+1..j] improves the sum.
            # Let's iterate j from i+1 to num_blocks-2.
            # To make it faster, we can break early if we find an improvement?
            # Or just do one pass.
            
            for j in range(i+1, num_blocks - 1):
                if time.time() > deadline:
                    break
                C = path[j]
                D = path[j+1]
                
                # Old segment sum:
                # ov(A,B) + sum(ov(path[k], path[k+1]) for k in i+1..j-1) + ov(C,D)
                # sum(ov(path[k], path[k+1]) for k in i+1..j-1) = prefix[j] - prefix[i+1]
                
                old_sum = overlaps[A][B] + (prefix[j] - prefix[i+1]) + overlaps[C][D]
                
                # New segment sum:
                # ov(A,C) + sum(ov(path[k+1], path[k]) for k in i+1..j-1) + ov(B,D)
                # sum(ov(path[k+1], path[k]) for k in i+1..j-1) = rev_prefix[j] - rev_prefix[i+1]
                
                new_sum = overlaps[A][C] + (rev_prefix[j] - rev_prefix[i+1]) + overlaps[B][D]
                
                if new_sum > old_sum:
                    # Perform the reversal
                    path[i+1:j+1] = path[i+1:j+1][::-1]
                    
                    # Update prefix sums?
                    # It's expensive to update the whole array.
                    # But we can just recalculate them if we accept the change.
                    # Or we can update them incrementally.
                    
                    # For simplicity, let's recalculate if we make a change.
                    # This makes the loop O(N^3) in worst case if many changes.
                    # But hopefully few changes.
                    
                    prefix = get_prefix_sums(path)
                    rev_prefix = get_rev_prefix_sums(path)
                    improved = True
                    break # Restart the i loop? Or continue?
                    # Usually 2-Opt restarts the i loop after a change.
        
    # Final check on time
    if time.time() > deadline - 5.0:
        # Build the word and return
        pass

    # Build the final word
    final_word = list(blocks[path[0]])
    for k in range(1, len(path)):
        prev_block = blocks[path[k-1]]
        curr_block = blocks[path[k]]
        ov = overlaps[path[k-1]][path[k]]
        final_word += curr_block[ov:]
        
    return "".join(map(str, final_word))

# For n=5, we can hardcode or use the same logic.
# The logic above works for n=5 as well.
# Let's test if the logic handles n=5 correctly.
# n=5, num_blocks = 4! = 24. 24*24 = 576. Very fast.
# n=6, num_blocks = 5! = 120. 120*120 = 14400. Fast.
# n=7, num_blocks = 6! = 720. 720*720 = 518400. Fast.

# One issue: The standard construction for n=5 is length 153.
# The recursive construction gives 153.
# Can we do better?
# The problem is known to be hard.
# For n=5, the shortest is 153.
# For n=6, the shortest is 1759? No, 1+2+6+24+120+720 = 873?
# Wait, the recursive construction length is sum(i! for i=1..n).
# n=1: 1
# n=2: 1+2=3
# n=3: 1+2+6=9
# n=4: 1+2+6+24=33
# n=5: 1+2+6+24+120=153
# n=6: 1+2+6+24+120+720=873
# n=7: 1+2+6+24+120+720+5040=5913

# The task says shortest known for n=7 is 5905.
# So we can save 8 symbols.
# My heuristic tries to improve the overlap sum for the last step.
# The last step adds (n-1)! blocks of length 2(n-1)+1.
# Total added length = (n-1)! * (2n-1) - sum(overlaps).
# We want to maximize sum(overlaps).
# The maximum possible overlap is n-1.
# If we could achieve n-1 for all transitions, the sum would be (n-1)! * (n-1).
# But we can't, because the graph is not complete with max weights.
# Actually, the graph IS complete, but the weights vary.
# The weight overlaps[i][j] is the length of the longest suffix of block i that is a prefix of block j.
# Since block i ends with P_i and block j starts with P_j, the overlap is the longest common prefix/suffix of P_i and P_j?
# No, it's the longest k such that P_i[-k:] == P_j[:k].
# This is exactly the longest suffix of P_i that is a prefix of P_j.
# Since P_i and P_j are permutations, this k is at most n-1.
# And it is 0 if the last element of P_i is not the first element of P_j.
# So the weight is 0 or at least 1 (if last of P_i == first of P_j, then k>=1).
# Actually, if P_i[-1] == P_j[0], then k>=1.
# If P_i[-2:] == P_j[:2], then k>=2.
# So the weight is the length of the longest chain of matching suffix/prefix.

# The 2-Opt should find a good ordering.

# Let's ensure the function returns a string.

# Re-check the code for n=5.
# If n=5, it generates S4, extracts perms, builds blocks, computes overlaps, runs 2-Opt.
# This should be fast.

# One potential bug: `path[i+1:j+1] = path[i+1:j+1][::-1]`
# This modifies the list in place.
# Then we recalculate prefix sums.
# This is correct.

# Let's add a fallback if the heuristic fails or takes too long.
# The current code has a deadline check.

# Final verification of the return type.
# The task asks for a string of digits.
# `"".join(map(str, final_word))` does that.

# Let's run a mental test for n=3.
# n=3, base_word = S2 = [1, 2, 1].
# Perms of length 2 in S2: (1,2), (2,1).
# Blocks:
# (1,2) -> [1,2,3,1,2]
# (2,1) -> [2,1,3,2,1]
# Overlaps:
# ov(0,0) = 4 (12312 suffix 2312? No. 12312 vs 12312. Suffix 12312 == Prefix 12312? No, length 5. Max k=4? 2312 vs 1231? No. 312 vs 231? No. 12 vs 31? No. 2 vs 1? No. 0?
# Wait, block is [1,2,3,1,2].
# Suffix [1,2] == Prefix [1,2]. So ov(0,0) = 2?
# Let's check the code.
# bi = [1,2,3,1,2]. bj = [1,2,3,1,2].
# k=2: bi[-2:] = [1,2]. bj[:2] = [1,2]. Match. found=2.
# k=3: bi[-3:] = [3,1,2]. bj[:3] = [1,2,3]. No.
# So ov(0,0) = 2.
# ov(0,1): bi=[1,2,3,1,2], bj=[2,1,3,2,1].
# k=2: [1,2] vs [2,1]. No.
# k=1: [2] vs [2]. Yes. found=1.
# ov(1,0): bi=[2,1,3,2,1], bj=[1,2,3,1,2].
# k=2: [2,1] vs [1,2]. No.
# k=1: [1] vs [1]. Yes. found=1.
# ov(1,1): bi=[2,1,3,2,1], bj=[2,1,3,2,1].
# k=2: [2,1] vs [2,1]. Yes. found=2.

# Path [0, 1]:
# Sum = ov(0,1) = 1.
# Length = len(b0) + len(b1) - 1 = 5 + 5 - 1 = 9.
# Word: [1,2,3,1,2] + [1,3,2,1] (overlap 1) = [1,2,3,1,2,1,3,2,1].
# This is "123121321". Correct.

# Path [1, 0]:
# Sum = ov(1,0) = 1.
# Length = 9.
# Word: [2,1,3,2,1] + [2,3,1,2] = [2,1,3,2,1,2,3,1,2].
# "213212312". Also a valid superpermutation.

# So the logic holds.

# The code is ready.

# One small optimization:
# In the 2-Opt loop, we break after the first improvement found for a given i.
# This is standard "best improvement" vs "first improvement".
# "First improvement" is faster but might get stuck in local minima.
# "Best improvement" is slower but better.
# Given the time limit, "First improvement" with restarts is a good compromise.
# The current code does "First improvement" for the inner loop j, and breaks the inner loop,
# then restarts the outer loop i? No, it breaks the inner loop, then the `improved` flag is set,
# and the `while` loop continues, restarting the `for i` loop.
# This is correct.

# Let's double check the `break` logic.
# `for i`:
#   `for j`:
#     if improvement:
#       reverse
#       update prefixes
#       improved = True
#       break # breaks inner for j
#   # if improved, the for i loop continues?
#   # No, we want to restart the for i loop from 0.
#   # So we should break the outer for i loop as well?
#   # Or just let it finish and rely on the `while` loop?
#   # If we let it finish, we might find more improvements in the same pass.
#   # But the prefix sums are stale for the rest of the pass.
#   # So we MUST break the outer loop if we make a change.
    
# Let's fix the code to break the outer loop on change.

def construct(n):
    start_time = time.time()
    deadline = start_time + 145.0

    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "1234124132413241234"

    def generate_standard_superperm(m):
        word = [1]
        for k in range(2, m + 1):
            perms, seen = [], set()
            for i in range(len(word) - k + 2):
                window = tuple(word[i:i + k - 1])
                if len(set(window)) == k - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [k] + list(p)
                max_ov = min(len(new), len(block))
                ov = 0
                for o in range(max_ov, -1, -1):
                    if o == 0:
                        break
                    if new[len(new) - o:] == block[:o]:
                        ov = o
                        break
                new += block[ov:]
            word = new
        return word

    base_word = generate_standard_superperm(n - 1)
    
    perms, seen = [], set()
    for i in range(len(base_word) - (n - 1) + 1):
        window = tuple(base_word[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
    
    num_blocks = len(perms)
    
    blocks = []
    for p in perms:
        blocks.append(list(p) + [n] + list(p))
    
    overlaps = [[0] * num_blocks for _ in range(num_blocks)]
    
    for i in range(num_blocks):
        bi = blocks[i]
        for j in range(num_blocks):
            bj = blocks[j]
            max_k = n - 1
            found = 0
            for k in range(max_k, 0, -1):
                if bi[-k:] == bj[:k]:
                    found = k
                    break
            overlaps[i][j] = found
    
    path = list(range(num_blocks))
    
    def get_prefix_sums(path):
        prefix = [0] * (len(path))
        for i in range(len(path) - 1):
            prefix[i+1] = prefix[i] + overlaps[path[i]][path[i+1]]
        return prefix
    
    def get_rev_prefix_sums(path):
        rev_prefix = [0] * (len(path))
        for i in range(len(path) - 1):
            rev_prefix[i+1] = rev_prefix[i] + overlaps[path[i+1]][path[i]]
        return rev_prefix
    
    prefix = get_prefix_sums(path)
    rev_prefix = get_rev_prefix_sums(path)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(num_blocks - 2):
            if time.time() > deadline:
                break
            A = path[i]
            B = path[i+1]
            
            for j in range(i+1, num_blocks - 1):
                if time.time() > deadline:
                    break
                C = path[j]
                D = path[j+1]
                
                old_sum = overlaps[A][B] + (prefix[j] - prefix[i+1]) + overlaps[C][D]
                new_sum = overlaps[A][C] + (rev_prefix[j] - rev_prefix[i+1]) + overlaps[B][D]
                
                if new_sum > old_sum:
                    path[i+1:j+1] = path[i+1:j+1][::-1]
                    prefix = get_prefix_sums(path)
                    rev_prefix = get_rev_prefix_sums(path)
                    improved = True
                    break # Break inner loop
            if improved:
                break # Break outer loop to restart with new prefix sums
    
    final_word = list(blocks[path[0]])
    for k in range(1, len(path)):
        prev_block = blocks[path[k-1]]
        curr_block = blocks[path[k]]
        ov = overlaps[path[k-1]][path[k]]
        final_word += curr_block[ov:]
        
    return "".join(map(str, final_word))
