import time
import random
from itertools import permutations
import numpy as np

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # For small n, use the standard construction which is optimal
    if n <= 3:
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    # For n >= 4, we try to improve the standard construction
    # Generate standard superpermutation for n-1 to get the order
    # Actually, we just need the order of permutations of 1..n-1 as they appear in S_{n-1}
    
    def get_standard_order(n_prev):
        word = [1]
        for m in range(2, n_prev + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        # Extract the order of permutations of 1..n_prev in the final word
        # The permutations are the windows of size n_prev that are valid perms
        # But we want the order of the *blocks* used in the last step?
        # The last step used permutations of 1..n_prev-1 to build blocks for n_prev.
        # Wait, the order of blocks in S_n is the order of permutations of 1..n-1 in S_{n-1}.
        # So we need the order of permutations of 1..n-1 in S_{n-1}.
        # Let's just return the word S_{n-1} and extract the order from it.
        return word

    # Get S_{n-1}
    s_prev = get_standard_order(n - 1)
    
    # Extract the order of permutations of 1..n-1 in s_prev
    # A permutation of 1..n-1 is a window of size n-1 with distinct elements
    order = []
    seen = set()
    for i in range(len(s_prev) - (n - 1) + 1):
        window = tuple(s_prev[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            order.append(window)
            
    # Generate blocks for n
    blocks = []
    for p in order:
        blocks.append(list(p) + [n] + list(p))
        
    M = len(blocks)
    
    # Precompute overlap matrix efficiently using reversed strings
    # O[i][j] = max k such that blocks[i][-k:] == blocks[j][:k]
    # This is equivalent to: longest prefix of blocks[j] that matches a suffix of blocks[i]
    # Which is: longest prefix of blocks[j] that matches a suffix of reversed(blocks[i])
    
    # Convert blocks to tuples for hashing
    block_tuples = [tuple(b) for b in blocks]
    
    # Build a dict mapping (prefix_length, prefix_tuple) -> list of block indices
    # Actually, we want for each block j, for each possible overlap k,
    # check if blocks[i][-k:] == blocks[j][:k]
    
    # Better approach: for each block i, compute reversed suffixes
    # and for each block j, compute prefixes
    # Use a trie-like structure or just hash-based lookup
    
    # For efficiency with M=5040, let's use a different approach:
    # For each possible overlap length k (from max possible down to 1),
    # group blocks by their last k elements and first k elements
    
    max_block_len = max(len(b) for b in blocks)
    
    # Build dictionaries for each overlap length
    # suffix_dict[k][suffix_tuple] = list of block indices ending with that suffix
    # prefix_dict[k][prefix_tuple] = list of block indices starting with that prefix
    
    # But this is expensive. Let's use a smarter approach:
    # For each pair (i,j), we can compute overlap by checking
    # if the suffix of block i matches the prefix of block j.
    
    # Key insight: blocks have a specific structure: p + [n] + p where p is a permutation
    # So blocks[i][-k:] and blocks[j][:k] have specific patterns.
    
    # Let's just compute it directly but more efficiently using string matching
    # Convert blocks to strings for fast comparison
    
    block_strings = ["".join(map(str, b)) for b in blocks]
    
    # For each block i, compute its reversed string
    # The overlap is the longest prefix of block_strings[j] that is a suffix of block_strings[i]
    
    # Use KMP-like approach or just try lengths from max down
    # Since max overlap is bounded by block length (~2n-1), and n is small (7),
    # max overlap is about 13. So we only need to check k from 13 down to 0.
    
    overlap_matrix = [[0] * M for _ in range(M)]
    max_ov_possible = max_block_len - 1
    
    for i in range(M):
        si = block_strings[i]
        for j in range(M):
            sj = block_strings[j]
            max_ov = 0
            # Try from longest possible overlap down
            for k in range(min(len(si), len(sj)), 0, -1):
                if si[-k:] == sj[:k]:
                    max_ov = k
                    break
            overlap_matrix[i][j] = max_ov

    # Precompute block lengths
    block_lengths = [len(b) for b in blocks]
    
    def evaluate(order_idx):
        # Calculate length
        length = block_lengths[order_idx[0]]
        for i in range(1, M):
            length += block_lengths[order_idx[i]] - overlap_matrix[order_idx[i-1]][order_idx[i]]
        return length

    def build_string(order_idx):
        s = blocks[order_idx[0]]
        for i in range(1, M):
            ov = overlap_matrix[order_idx[i-1]][order_idx[i]]
            s = s + blocks[order_idx[i]][ov:]
        return s

    def is_valid(s):
        # Check if all permutations of 1..n are in s
        # n is small (7), so we can just check all windows
        # Use a set of seen permutations
        seen_perms = set()
        for i in range(len(s) - n + 1):
            window = tuple(s[i:i+n])
            if len(set(window)) == n:
                seen_perms.add(window)
        return len(seen_perms) == n

    # Initial evaluation
    current_order = list(range(M))
    current_string = build_string(current_order)
    current_length = len(current_string)
    is_current_valid = is_valid(current_string)
    
    best_order = current_order
    best_string = current_string
    best_length = current_length
    
    if not is_current_valid:
        # Should not happen for standard construction
        return "".join(map(str, best_string))

    # Local Search with 2-opt moves (reversing subsequences) and swaps
    # Since the standard construction is valid, we start from a valid state.
    # Moves might make it invalid. We only accept if valid and shorter.
    
    # Precompute the total length contribution for quick evaluation
    # total_len = sum of block_lengths - sum of overlaps between consecutive blocks
    
    def compute_total_overlap(order_idx):
        total_ov = 0
        for i in range(1, M):
            total_ov += overlap_matrix[order_idx[i-1]][order_idx[i]]
        return total_ov
    
    def compute_total_length(order_idx):
        return sum(block_lengths) - compute_total_overlap(order_idx)
    
    # Initial evaluation
    current_order = list(range(M))
    current_string = build_string(current_order)
    current_length = len(current_string)
    is_current_valid = is_valid(current_string)
    
    best_order = current_order[:]
    best_string = current_string
    best_length = current_length
    
    if not is_current_valid:
        # Should not happen for standard construction
        return "".join(map(str, best_string))
    
    # Local Search
    # We will try to improve the length by:
    # 1. Swapping two blocks
    # 2. Reversing a subsequence (2-opt)
    # Since the standard construction is valid, we start from a valid state.
    # Moves might make it invalid. We only accept if valid and shorter.
    
    # Use hill climbing with random restarts
    # Time limit check
    while time.time() < deadline:
        # Choose move type: 60% 2-opt, 40% swap
        if random.random() < 0.6:
            # 2-opt: reverse a subsequence
            if M < 2:
                continue
            i = random.randint(0, M - 2)
            j = random.randint(i + 1, M - 1)
            
            # Compute new length efficiently
            # The only overlaps that change are:
            # - overlap between order[i-1] and order[i] (if i > 0)
            # - overlap between order[j] and order[j+1] (if j < M-1)
            # - all overlaps within the reversed segment [i, j]
            
            old_len = current_length
            
            # Compute new overlaps
            new_ov = 0
            
            # Overlaps before i
            for k in range(1, i):
                new_ov += overlap_matrix[current_order[k-1]][current_order[k]]
            
            # Overlap at position i-1 to i (if i > 0)
            if i > 0:
                new_ov += overlap_matrix[current_order[i-1]][current_order[j]]  # reversed: j comes first
            
            # Overlaps within reversed segment [i, j]
            # Original: order[i], order[i+1], ..., order[j]
            # Reversed: order[j], order[j-1], ..., order[i]
            for k in range(i, j):
                new_ov += overlap_matrix[current_order[j - (k - i)]][current_order[j - (k - i) + 1]]
                # Wait, that's not right. Let me think again.
                # Reversed segment: positions i to j become order[j], order[j-1], ..., order[i]
                # So consecutive pairs are: (order[j], order[j-1]), (order[j-1], order[j-2]), ..., (order[i+1], order[i])
            
            # Let me redo this more carefully
            new_ov = 0
            # Overlaps before i (unchanged)
            for k in range(1, i):
                new_ov += overlap_matrix[current_order[k-1]][current_order[k]]
            
            # Overlap between position i-1 and new position i (which is old j)
            if i > 0:
                new_ov += overlap_matrix[current_order[i-1]][current_order[j]]
            
            # Overlaps within reversed segment
            # New order in segment: order[j], order[j-1], ..., order[i]
            for k in range(j - 1, i - 1, -1):
                new_ov += overlap_matrix[current_order[k]][current_order[k - 1]]
            
            # Overlap between new position j (which is old i) and position j+1 (if j < M-1)
            if j < M - 1:
                new_ov += overlap_matrix[current_order[i]][current_order[j + 1]]
            
            # Overlaps after j (unchanged)
            for k in range(j + 1, M):
                new_ov += overlap_matrix[current_order[k-1]][current_order[k]]
            
            new_length = sum(block_lengths) - new_ov
            
            if new_length >= best_length:
                continue
            
            # Apply the move
            new_order = current_order[:]
            new_order[i:j+1] = new_order[i:j+1][::-1]
            
            new_string = build_string(new_order)
            
            if is_valid(new_string):
                best_order = new_order
                best_string = new_string
                best_length = new_length
                current_order = new_order
                current_string = new_string
        else:
            # Swap two blocks
            if M < 2:
                continue
            i, j = random.sample(range(M), 2)
            
            # Compute new length
            # Only overlaps involving positions i and j change
            old_len = current_length
            
            new_ov = 0
            # Overlaps not involving i or j
            for k in range(1, M):
                if k - 1 == i or k - 1 == j or k == i or k == j:
                    continue
                new_ov += overlap_matrix[current_order[k-1]][current_order[k]]
            
            # Overlaps involving i and j
            # Case 1: i and j are not adjacent
            if abs(i - j) > 1:
                # Overlap before i
                if i > 0:
                    new_ov += overlap_matrix[current_order[i-1]][current_order[j]]  # j now at position i
                # Overlap after i (now j)
                if i < M - 1 and i + 1 != j:
                    new_ov += overlap_matrix[current_order[j]][current_order[i + 1]]
                # Overlap before j
                if j > 0 and j - 1 != i:
                    new_ov += overlap_matrix[current_order[j-1]][current_order[i]]  # i now at position j
                # Overlap after j
                if j < M - 1:
                    new_ov += overlap_matrix[current_order[i]][current_order[j + 1]]
            else:
                # i and j are adjacent
                a, b = min(i, j), max(i, j)
                # Overlap before a
                if a > 0:
                    new_ov += overlap_matrix[current_order[a-1]][current_order[b]]  # b now at position a
                # Overlap after b
                if b < M - 1:
                    new_ov += overlap_matrix[current_order[a]][current_order[b + 1]]  # a now at position b
            
            new_length = sum(block_lengths) - new_ov
            
            if new_length >= best_length:
                continue
            
            # Apply the swap
            new_order = current_order[:]
            new_order[i], new_order[j] = new_order[j], new_order[i]
            
            new_string = build_string(new_order)
            
            if is_valid(new_string):
                best_order = new_order
                best_string = new_string
                best_length = new_length
                current_order = new_order
                current_string = new_string

    return "".join(map(str, best_string))
