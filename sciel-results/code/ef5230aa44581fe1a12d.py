import time
import random
import math

def construct(n):
    if n <= 1:
        return "1"
    
    # Step 1: Generate standard superpermutation for n-1
    # We use the standard recursive construction for the base case
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
                
    k = len(blocks)
    L = n - 1
    
    # Precompute overlaps between all pairs of blocks
    # overlap[i][j] = max k such that blocks[i][-k:] == blocks[j][:k]
    overlap_matrix = [[0] * k for _ in range(k)]
    for i in range(k):
        bi = blocks[i]
        for j in range(k):
            bj = blocks[j]
            max_ov = L
            for ov in range(max_ov, 0, -1):
                if bi[-ov:] == bj[:ov]:
                    overlap_matrix[i][j] = ov
                    break
            # if no overlap found, it remains 0

    # Base length calculation
    # Length = 2L + 1 + sum_{i=0}^{k-2} (1 + 2L - overlap(order[i], order[i+1]))
    # = 2L + 1 + (k-1)(1+2L) - sum(overlaps)
    # Minimize length <=> Maximize sum(overlaps)
    
    best_order = list(range(k))
    
    def calc_total_overlap(order):
        total = 0
        for idx in range(len(order) - 1):
            total += overlap_matrix[order[idx]][order[idx+1]]
        return total

    best_overlap = calc_total_overlap(best_order)
    
    start_time = time.time()
    time_limit = 115 # Leave 5s for safety
    
    # Or-Opt Local Search
    # Move segment of length 1 or 2
    
    def try_or_opt(order, seg_len):
        nonlocal best_overlap
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            for i in range(k - seg_len + 1):
                if time.time() - start_time > time_limit:
                    break
                # Segment is order[i] ... order[i+seg_len-1]
                # Try moving to position j
                # j is the start index in the new list
                # We iterate j from 0 to k-seg_len
                # Skip if j is in [i, i+seg_len-1] (same position)
                for j in range(k - seg_len + 1):
                    if j >= i and j < i + seg_len:
                        continue
                    # Calculate delta
                    # Case 1: j > i + seg_len - 1 (move right)
                    # Case 2: j < i (move left)
                    
                    # Indices involved in edges:
                    # Remove:
                    # 1. (i-1, i) if i > 0
                    # 2. (i+seg_len-1, i+seg_len) if i+seg_len < k
                    # 3. (j-1, j) if j > 0  (Note: if j > i, indices shift? No, we are calculating delta on the *current* order before modification)
                    #    Actually, it's easier to define the edges in terms of the *current* list indices.
                    
                    # Let's define the edges removed and added in the *current* list context.
                    # The segment S is at indices [i, i+seg_len-1].
                    # We want to move it to start at j.
                    
                    # If j > i:
                    #   The element at j shifts to i+seg_len? No.
                    #   Original: ... A, S, B ... C, D ...
                    #   Where A is at i-1, S is i..i+seg_len-1, B is at i+seg_len.
                    #   C is at j-1, D is at j.
                    #   New: ... A, B ... C, S, D ...
                    #   Edges removed: (A, S_start), (S_end, B), (C, D)
                    #   Edges added: (A, B), (C, S_start), (S_end, D)
                    #   Note: Internal edges of S are preserved.
                    
                    # If j < i:
                    #   Original: ... C, D ... A, S, B ...
                    #   Where C is at j-1, D is at j.
                    #   A is at i-1, S is i..i+seg_len-1, B is at i+seg_len.
                    #   New: ... C, S, D ... A, B ...
                    #   Edges removed: (C, D), (A, S_start), (S_end, B)
                    #   Edges added: (C, S_start), (S_end, D), (A, B)
                    
                    # It turns out the set of removed and added edges is the same in both cases, 
                    # just the "middle" part of the list shifts.
                    # Removed:
                    #  e1 = (i-1, i) if i > 0
                    #  e2 = (i+seg_len-1, i+seg_len) if i+seg_len < k
                    #  e3 = (j-1, j) if j > 0
                    # Added:
                    #  a1 = (i-1, i+seg_len) if i > 0 and i+seg_len < k
                    #  a2 = (j-1, i) if j > 0
                    #  a3 = (i+seg_len-1, j) if i+seg_len < k
                    # Wait, if j < i, the "gap" is different.
                    # Let's just calculate the overlap sum change directly using the indices.
                    
                    old_ov = 0
                    # Edge before i
                    if i > 0:
                        old_ov += overlap_matrix[order[i-1]][order[i]]
                    # Edge after segment
                    if i + seg_len < k:
                        old_ov += overlap_matrix[order[i+seg_len-1]][order[i+seg_len]]
                    # Edge at j (if j is not part of the segment being moved)
                    # If j > i, the edge at j is (j-1, j)
                    # If j < i, the edge at j is (j-1, j)
                    # But if j is inside the segment, we skip. We already skipped.
                    if j > 0:
                        # We need to be careful: if j < i, the index j is before i.
                        # The edge (j-1, j) is removed.
                        old_ov += overlap_matrix[order[j-1]][order[j]]
                    elif j == 0:
                        # No edge before 0
                        pass
                    # Wait, if j > 0, we remove edge (j-1, j).
                    # If j == 0, there is no edge before it.
                    # But what if j == i? Skipped.
                    # What if j == i+seg_len? 
                    #   Then we are moving S to be immediately after itself? No, j is start index.
                    #   If j == i+seg_len, we move S to position i+seg_len.
                    #   Original: ... A, S, B ...
                    #   New: ... A, B, S ...
                    #   Removed: (A,S), (S,B), (B, B_next)? No.
                    #   If j == i+seg_len, the edge (j-1, j) is (i+seg_len-1, i+seg_len) which is (S_end, B).
                    #   This is the same as e2. So we would double count?
                    #   Let's refine the edge identification.
                    
                    # Let's use a robust method:
                    # Identify the 3 edges removed and 3 edges added explicitly.
                    
                    # Removed Edges:
                    # 1. Left boundary of S: (i-1, i) if i > 0
                    # 2. Right boundary of S: (i+seg_len-1, i+seg_len) if i+seg_len < k
                    # 3. Destination boundary:
                    #    If j > i: The edge that was at j, (j-1, j).
                    #    If j < i: The edge that was at j, (j-1, j).
                    #    Note: If j > i, the indices i..i+seg_len-1 are shifted? No, we are calculating delta on the static array `order` before modification.
                    #    So we just look at `order` at indices i and j.
                    
                    # Added Edges:
                    # 1. Left of S connects to what was after S: (i-1, i+seg_len) if i > 0 and i+seg_len < k
                    #    If i == 0, no left edge.
                    #    If i+seg_len == k, no right edge.
                    #    If both exist, we connect them.
                    # 2. Left of destination connects to start of S: (j-1, i) if j > 0
                    # 3. End of S connects to what was at j: (i+seg_len-1, j) if i+seg_len < k
                    #    If j == 0, no edge after j? No, j is the start index of the new S.
                    #    The element that was at j moves to i+seg_len?
                    #    Let's trace:
                    #    Original: [0..i-1], [i..i+s-1], [i+s..j-1], [j..k-1]
                    #    New: [0..i-1], [i+s..j-1], [i..i+s-1], [j..k-1]  (if j > i)
                    #    Edges in New:
                    #    ... (i-1, i+s), (i+s, i+s+1)...(j-2, j-1), (j-1, i), (i, i+1)...(i+s-2, i+s-1), (i+s-1, j), (j, j+1)...
                    #    Edges in Old:
                    #    ... (i-1, i), (i, i+1)...(i+s-2, i+s-1), (i+s-1, i+s), (i+s, i+s+1)...(j-2, j-1), (j-1, j), (j, j+1)...
                    #    
                    #    Removed:
                    #    - (i-1, i) if i>0
                    #    - (i+s-1, i+s) if i+s<k
                    #    - (j-1, j) if j>0
                    #    
                    #    Added:
                    #    - (i-1, i+s) if i>0 and i+s<k
                    #    - (j-1, i) if j>0
                    #    - (i+s-1, j) if i+s<k
                    #    
                    #    This logic holds for j > i.
                    #    
                    #    What if j < i?
                    #    Original: [0..j-1], [j..i-1], [i..i+s-1], [i+s..k-1]
                    #    New: [0..j-1], [i..i+s-1], [j..i-1], [i+s..k-1]
                    #    Edges in New:
                    #    ... (j-1, i), (i, i+1)...(i+s-2, i+s-1), (i+s-1, j), (j, j+1)...(i-2, i-1), (i-1, i+s), (i+s, i+s+1)...
                    #    Edges in Old:
                    #    ... (j-1, j), (j, j+1)...(i-2, i-1), (i-1, i), (i, i+1)...(i+s-2, i+s-1), (i+s-1, i+s), (i+s, i+s+1)...
                    #    
                    #    Removed:
                    #    - (j-1, j) if j>0
                    #    - (i-1, i) if i>0
                    #    - (i+s-1, i+s) if i+s<k
                    #    
                    #    Added:
                    #    - (j-1, i) if j>0
                    #    - (i+s-1, j) if i+s<k
                    #    - (i-1, i+s) if i>0 and i+s<k
                    #    
                    #    The sets of removed and added edges are identical in composition, just the indices j and i are swapped in the "destination" role.
                    #    So the formula is the same.
                    
                    new_ov = 0
                    if i > 0 and i + seg_len < k:
                        new_ov += overlap_matrix[order[i-1]][order[i+seg_len]]
                    if j > 0:
                        new_ov += overlap_matrix[order[j-1]][order[i]]
                    if i + seg_len < k:
                        new_ov += overlap_matrix[order[i+seg_len-1]][order[j]]
                    
                    delta = new_ov - old_ov
                    if delta > 0:
                        # Perform the move
                        segment = order[i:i+seg_len]
                        # Remove
                        del order[i:i+seg_len]
                        # Insert
                        # Adjust j if j > i
                        insert_idx = j
                        if j > i:
                            # When we remove seg_len items at i, the index j shifts left by seg_len
                            insert_idx = j - seg_len
                        order[insert_idx:insert_idx] = segment
                        best_overlap += delta
                        improved = True
                        if time.time() - start_time > time_limit:
                            break
            return improved

    # Run Or-Opt for seg_len 1 and 2
    for _ in range(20): # Max 20 passes
        if not try_or_opt(best_order, 1):
            break
        if not try_or_opt(best_order, 2):
            break
        if time.time() - start_time > time_limit:
            break

    # Reconstruct the word
    word = []
    for idx in best_order:
        p = blocks[idx]
        block = list(p) + [n] + list(p)
        if not word:
            word.extend(block)
        else:
            # Find max overlap k such that word[-k:] == block[:k]
            # We know the overlap is overlap_matrix[prev_idx][idx]
            # But to be safe and simple, we can use the matrix if we track prev_idx
            # However, the generic check is O(L) and L is small (6).
            max_k = min(len(word), len(block))
            overlap = 0
            for k_val in range(max_k, -1, -1):
                if k_val == 0:
                    overlap = 0
                    break
                if word[-k_val:] == block[:k_val]:
                    overlap = k_val
                    break
            word.extend(block[overlap:])
            
    return "".join(map(str, word))
