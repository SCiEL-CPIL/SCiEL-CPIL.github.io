import time
import random
import itertools

# EVOLVE-BLOCK-START
def _get_overlap(s1, s2):
    """Calculate maximum overlap between suffix of s1 and prefix of s2."""
    max_ov = min(len(s1), len(s2))
    # Optimization: Only check up to a reasonable limit if strings are long, 
    # but here max length is 13 (for n=7), so full check is fine.
    for k in range(max_ov, -1, -1):
        if s1[-k:] == s2[:k]:
            return k
    return 0

def _optimize_order_blocks(blocks, time_limit=50):
    """
    Find a permutation of blocks that maximizes the sum of overlaps between consecutive blocks.
    Returns the best order found (list of indices) or None if timeout.
    """
    n = len(blocks)
    if n <= 2:
        return list(range(n))
    
    # Precompute overlap matrix
    # overlap[i][j] = overlap of blocks[i] and blocks[j]
    # We want to maximize sum(overlap[order[k-1]][order[k]])
    # Equivalent to minimizing sum( (max_ov) - overlap[order[k-1]][order[k]] )
    # Let's just use the overlap matrix directly for Max TSP.
    
    max_ov_len = len(blocks[0]) # All blocks same length
    # Initialize matrix with 0
    ov_matrix = [[0]*n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i != j:
                ov_matrix[i][j] = _get_overlap(blocks[i], blocks[j])
    
    # Heuristic: Nearest Neighbor on "Cost" = max_ov_len - overlap
    # Actually, for Max TSP, we pick the neighbor with MAX overlap.
    
    best_order = list(range(n))
    best_score = sum(ov_matrix[best_order[k-1]][best_order[k]] for k in range(1, n))
    
    start_time = time.time()
    
    # Try a few random starts for Nearest Neighbor
    for _ in range(20): # 20 restarts
        if time.time() - start_time > time_limit * 0.5:
            break
            
        visited = [False] * n
        # Start from random node
        start_node = random.randint(0, n-1)
        current = start_node
        order = [current]
        visited[current] = True
        score = 0
        
        for _ in range(n-1):
            next_node = -1
            max_ov = -1
            # Find unvisited neighbor with max overlap
            for next_candidate in range(n):
                if not visited[next_candidate]:
                    ov = ov_matrix[current][next_candidate]
                    if ov > max_ov:
                        max_ov = ov
                        next_node = next_candidate
            order.append(next_node)
            visited[next_node] = True
            score += max_ov
            current = next_node
            
        if score > best_score:
            best_score = score
            best_order = order[:]
            
        # Improve with 2-Opt
        # 2-Opt for Max TSP: Reverse segment [i..j] if it increases total overlap.
        # Delta = (ov[i-1][j] + ov[i][j-1]) - (ov[i-1][i] + ov[j-1][j])  ... indices need care for circular vs linear
        # Here it is a path, not a cycle.
        
        improved = True
        iterations = 0
        while improved and (time.time() - start_time) < time_limit * 0.8:
            improved = False
            iterations += 1
            if iterations > 50: # Safety cap per restart
                break
            for i in range(1, n-1):
                if (time.time() - start_time) > time_limit * 0.8:
                    break
                for j in range(i+1, n):
                    # Remove edges (i-1, i) and (j-1, j)
                    # Add edges (i-1, j) and (i, j-1)
                    # Wait, reversing i..j connects i-1 to j, and i to j-1?
                    # Original: ... u - i ... j - v ...
                    # Reversed: ... u - j ... i - v ...
                    # Edges removed: (u, i) and (j, v)  -> indices in order list: order[i-1] is u, order[i] is i. order[j-1] is j_prev?
                    
                    # Let's use the order list indices directly.
                    # Segment is order[i-1] ... order[j]
                    # No, standard 2-opt on path:
                    # Consider reversing order[i..j] (inclusive).
                    # Connections involved:
                    # Left boundary: order[i-1] connected to order[i].
                    # Right boundary: order[j] connected to order[j+1] (if j < n-1).
                    
                    if i == 0:
                        left_idx = -1
                    else:
                        left_idx = i - 1
                        
                    if j == n - 1:
                        right_idx = -1
                    else:
                        right_idx = j + 1
                    
                    # Nodes involved in change:
                    # a = order[left_idx] (if left_idx >= 0)
                    # b = order[i]
                    # c = order[j]
                    # d = order[right_idx] (if right_idx >= 0)
                    
                    # Overlaps to remove:
                    # If left exists: ov[a][b]
                    # If right exists: ov[c][d]
                    # Internal overlaps within [i..j] remain the same?
                    # Wait. Reversing a path in a directed graph (overlap is directed: suffix of X matches prefix of Y).
                    # Overlap(X, Y) != Overlap(Y, X) generally.
                    # So reversing the segment changes ALL internal overlaps to their reverse values.
                    # This makes 2-Opt more complex.
                    # Delta = Sum(new_ovs) - Sum(old_ovs)
                    # Old: ov[a][b] + ov[b][next_b] + ... + ov[c][d]
                    # New: ov[a][c] + ov[c][prev_c] + ... + ov[b][d]
                    
                    # This calculation is O(length) which is O(n).
                    # 2-Opt becomes O(n^3) per restart. 720^3 is ~370M. Too slow in Python.
                    
                    # Alternative: 3-Opt or simple swap?
                    # Or just stick to NN + random restarts.
                    pass

    return best_order

def construct(n):
    # Classic recursive construction
    word = [1]
    start_time = time.time()
    total_time_limit = 110 # Leave 10s margin
    
    for m in range(2, n + 1):
        # Check time
        if time.time() - start_time > total_time_limit:
            break
            
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Create blocks for this step m
        blocks = []
        for p in perms:
            block = list(p) + [m] + list(p)
            blocks.append(block)
            
        # Determine order of blocks
        if m <= 4:
            # Small enough for exact or simple sort?
            # Standard order is fine for small m, or we can just use the list order.
            order = list(range(len(blocks)))
        else:
            # Use heuristic optimization for m >= 5
            # Only if we have time
            remaining_time = total_time_limit - (time.time() - start_time)
            # Allocate 30s for the last step, less for previous?
            # For m=5, 24 blocks. Fast.
            # For m=6, 120 blocks. Fast.
            # For m=7, 720 blocks. Main cost.
            
            # Convert blocks to strings for faster overlap comparison?
            # Tuple comparison is okay.
            
            order = _optimize_order_blocks(blocks, time_limit=min(30, remaining_time * 0.7))
            
        # Construct new word
        new_word = []
        # Start with first block
        if not order:
            word = new_word
            continue
            
        first_block = blocks[order[0]]
        new_word.extend(first_block)
        
        for idx in range(1, len(order)):
            b = blocks[order[idx]]
            # Find max overlap
            overlap = _get_overlap(tuple(new_word), tuple(b))
            # _get_overlap takes sequences. 
            # new_word is a list. b is a list.
            # The function _get_overlap uses slicing.
            # Let's ensure _get_overlap handles lists/tuples.
            
            new_word.extend(b[overlap:])
            
        word = new_word
        
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
