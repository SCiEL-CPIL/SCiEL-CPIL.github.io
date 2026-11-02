import itertools
import sys

def construct(n):
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    
    if n > 7:
        # Fallback to standard construction for larger n
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

    # Heuristic for n=7 (and potentially 4,5,6)
    # Generate all permutations of 1..n-1
    prev_perms = list(itertools.permutations(range(1, n)))
    
    # Create blocks: p + [n] + p
    blocks = []
    for p in prev_perms:
        b = list(p) + [n] + list(p)
        blocks.append(b)
        
    num_blocks = len(blocks)
    
    # Precompute overlaps
    # M[i][j] = max k such that blocks[i][-k:] == blocks[j][:k]
    # Optimization: Use string suffix/prefix matching or simple loop
    # Since block length is small (2n-1 <= 13), simple loop is fine.
    overlaps = [[0]*num_blocks for _ in range(num_blocks)]
    
    for i in range(num_blocks):
        bi = blocks[i]
        bi_str = "".join(map(str, bi))
        for j in range(num_blocks):
            bj = blocks[j]
            bj_str = "".join(map(str, bj))
            max_k = min(len(bi), len(bj))
            found = 0
            # Check from longest to shortest
            for k in range(max_k, -1, -1):
                if k == 0:
                    found = 0
                    break
                if bi_str[-k:] == bj_str[:k]:
                    found = k
                    break
            overlaps[i][j] = found

    # Greedy TSP Heuristic
    # Start with block 0
    current = 0
    unvisited = set(range(1, num_blocks))
    path = [0]
    
    while unvisited:
        best_next = -1
        best_overlap = -1
        for j in unvisited:
            if overlaps[current][j] > best_overlap:
                best_overlap = overlaps[current][j]
                best_next = j
        path.append(best_next)
        unvisited.remove(best_next)
        current = best_next
        
    def calc_length(path):
        total = len(blocks[path[0]])
        for k in range(len(path)-1):
            total += len(blocks[path[k+1]]) - overlaps[path[k]][path[k+1]]
        return total

    # 2-Opt Local Search
    # We want to maximize total overlap, which minimizes length.
    # Delta = New_Overlap_Sum - Old_Overlap_Sum
    # If Delta > 0, accept.
    
    improved = True
    max_iterations = 50 # Limit iterations to ensure time safety
    iter_count = 0
    
    while improved and iter_count < max_iterations:
        improved = False
        iter_count += 1
        for i in range(len(path)-1):
            for j in range(i+2, len(path)):
                # Reverse path[i+1:j+1]
                # Old: ... A, B, ..., C, D ...
                # New: ... A, C, ..., B, D ...
                # A = path[i], B = path[i+1], C = path[j], D = path[j+1] (if j < len-1)
                
                # Overlaps involved:
                # Old: ov(A,B) + ov(B, next_B) ... ov(prev_C, C) + ov(C,D)
                # New: ov(A,C) + ov(C, prev_C) ... ov(next_B, B) + ov(B,D)
                
                # Internal overlaps in the reversed segment are the same set, just reversed order?
                # No, ov(X,Y) != ov(Y,X) generally.
                # However, the sum of overlaps in the internal segment changes.
                # This makes 2-opt calculation complex if we don't recalculate the whole segment.
                # Given N=720, recalculating the delta for the segment is O(N) per pair (i,j).
                # Total O(N^3) per iteration? 720^3 is 373 million. Too slow for 50 iterations.
                
                # We need a faster delta calculation or fewer iterations.
                # Or just accept that 2-opt might be slow and limit iterations to 5-10.
                # Or use Or-opt (move a single node).
                
                # Let's try Or-opt (move single node) which is O(N) delta.
                # Remove node at pos k, insert at pos m.
                # This is O(N) to check delta for each m.
                # Total O(N^2) per pass.
                # 720^2 = 518k. Fast.
                
                pass # Skip 2-opt, do Or-opt instead

    # Implement Or-Opt
    improved = True
    max_iterations = 20
    iter_count = 0
    while improved and iter_count < max_iterations:
        improved = False
        iter_count += 1
        for k in range(1, len(path)-1):
            node = path[k]
            prev_node = path[k-1]
            next_node = path[k+1]
            
            # Remove node
            # Old overlaps: ov(prev, node) + ov(node, next)
            old_ov = overlaps[prev_node][node] + overlaps[node][next_node]
            
            # New overlap at removal point: ov(prev, next)
            new_ov_removal = overlaps[prev_node][next_node]
            
            best_gain = 0
            best_pos = -1
            
            # Try inserting node at every other position m
            # Inserting between path[m] and path[m+1]
            for m in range(0, len(path)-1):
                if m == k-1 or m == k:
                    continue # Skip original position and immediate neighbors (already considered or invalid)
                
                left_node = path[m]
                right_node = path[m+1]
                
                # If we insert node between left_node and right_node:
                # New overlaps: ov(left_node, node) + ov(node, right_node)
                new_ov_insertion = overlaps[left_node][node] + overlaps[node][right_node]
                
                # Gain = (new_ov_insertion + new_ov_removal) - old_ov
                # Note: new_ov_removal is constant for this k.
                gain = (new_ov_insertion + new_ov_removal) - old_ov
                
                if gain > best_gain:
                    best_gain = gain
                    best_pos = m
            
            if best_gain > 0 and best_pos != -1:
                # Perform move
                path.pop(k)
                # Adjust best_pos if k < best_pos
                if k < best_pos:
                    best_pos -= 1
                path.insert(best_pos+1, node)
                improved = True

    # Build the string
    final_word = list(blocks[path[0]])
    for k in range(1, len(path)):
        prev_block = blocks[path[k-1]]
        curr_block = blocks[path[k]]
        ov = overlaps[path[k-1]][path[k]]
        final_word += curr_block[ov:]
        
    return "".join(map(str, final_word))
