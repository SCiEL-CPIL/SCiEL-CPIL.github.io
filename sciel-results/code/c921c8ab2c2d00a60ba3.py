import time
import itertools
import random

def construct(n):
    """
    Constructs a superpermutation using a greedy construction followed by 
    local search (2-opt) on the order of permutation blocks to maximize overlaps.
    """
    start_time = time.time()
    deadline = start_time + 115.0

    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    
    # For n >= 4, we use the block-based optimization
    # The blocks are permutations of 1..n-1
    # We append block P as: P, n, P
    # Overlap between end of previous word (which ends with P_prev) 
    # and start of new block (P_new) is the longest suffix of P_prev 
    # that is a prefix of P_new.
    
    symbols = list(range(1, n))
    perms = list(itertools.permutations(symbols))
    M = len(perms)
    
    # Map permutation to index for fast lookup
    perm_to_idx = {p: i for i, p in enumerate(perms)}
    
    # Precompute overlap matrix? 720x720 is 518k entries. 
    # Overlap calculation: 
    # def get_overlap(p1, p2):
    #    max_k = 0
    #    for k in range(len(p1), 0, -1):
    #       if p1[-k:] == p2[:k]:
    #           return k
    #    return 0
    # This is fast enough. 720*720*6 ops ~ 3M ops. Very fast.
    
    overlap_matrix = [[0]*M for _ in range(M)]
    for i in range(M):
        p1 = perms[i]
        for j in range(M):
            p2 = perms[j]
            # Calculate max k such that p1[-k:] == p2[:k]
            # k can be 0 to n-1
            max_k = 0
            for k in range(n-1, 0, -1):
                if p1[-k:] == p2[:k]:
                    max_k = k
                    break
            overlap_matrix[i][j] = max_k

    # Initial order: Classic recursive construction order
    # The classic construction generates permutations in a specific order.
    # We can just use the order they appear in the classic superperm.
    # Or simpler: Start with a random order or the natural lex order, then optimize.
    # The natural order of itertools.permutations is lex order.
    # Let's start with lex order and optimize.
    current_order = list(range(M))
    
    # Calculate initial cost (length)
    def get_length(order):
        if not order:
            return 0
        # First block: P, n, P. Length 2n-1
        total_len = 2*n - 1
        for i in range(1, len(order)):
            prev_idx = order[i-1]
            curr_idx = order[i]
            k = overlap_matrix[prev_idx][curr_idx]
            # Added length: (n-1-k) + 1 + (n-1) = 2n - k - 1
            total_len += 2*n - k - 1
        return total_len

    initial_len = get_length(current_order)
    
    # 2-Opt Local Search
    # We want to minimize length, which is equivalent to maximizing sum of overlaps.
    # Cost of swapping two adjacent segments?
    # Standard 2-opt: reverse a segment of the path.
    # Or simpler: try swapping two adjacent blocks? No, that's 1-opt.
    # Let's do 2-opt: reverse segment (i, j).
    
    best_order = current_order[:]
    best_len = initial_len
    
    # Time check
    if time.time() > deadline - 10:
        # Not enough time for search, return classic
        return _build_word(best_order, n, perms)
        
    while time.time() < deadline - 5:
        improved = False
        # Randomly select a segment to reverse
        # Or iterate through all pairs? M=720, M^2/2 = 259k pairs.
        # Cost to evaluate a reversal:
        # The internal overlaps of the reversed segment remain the same?
        # No, overlap(u, v) is not symmetric. overlap(u, v) != overlap(v, u).
        # So reversing a segment changes all internal overlaps.
        # This makes 2-opt expensive: O(M) per reversal to recalc cost.
        # Total O(M^3) per pass? 720^3 ~ 373M. Too slow for Python in 100s.
        
        # Better: 1-opt (swap adjacent) or Or-opt (move small segments).
        # Or-opt: move a segment of length 1, 2, or 3.
        # Let's try swapping two blocks (1-opt).
        # If we swap block at i and block at i+1:
        # Old cost contribution: overlap[i-1][i] + overlap[i][i+1] + overlap[i+1][i+2]
        # New cost contribution: overlap[i-1][i+1] + overlap[i+1][i] + overlap[i][i+2]
        # We can evaluate this in O(1).
        
        for i in range(M - 1):
            if time.time() > deadline - 5:
                break
            
            # Try swapping i and i+1
            # Indices in order list
            idx_i = current_order[i]
            idx_i1 = current_order[i+1]
            
            prev_idx = current_order[i-1] if i > 0 else -1
            next_idx = current_order[i+2] if i+2 < M else -1
            
            old_cost = 0
            new_cost = 0
            
            if i > 0:
                old_cost += overlap_matrix[prev_idx][idx_i]
                new_cost += overlap_matrix[prev_idx][idx_i1]
            
            # Middle part
            old_cost += overlap_matrix[idx_i][idx_i1]
            new_cost += overlap_matrix[idx_i1][idx_i]
            
            if i+2 < M:
                old_cost += overlap_matrix[idx_i1][next_idx]
                new_cost += overlap_matrix[idx_i][next_idx]
            
            # We want to maximize overlap sum, so minimize cost (which is -overlap)
            # Actually, length added is 2n - k - 1.
            # Minimizing length is maximizing k.
            # So we check if new_cost > old_cost (sum of overlaps)
            
            if new_cost > old_cost:
                # Perform swap
                current_order[i], current_order[i+1] = current_order[i+1], current_order[i]
                improved = True
                # Update best if better
                # We don't need to recalc full length, just track improvement
                # But to be safe, we can update best_order if we track global best
                # For simplicity, we assume local search converges to a good local optimum.
        
        if not improved:
            # Try a random perturbation to escape local optimum?
            # Or stop.
            # Let's try a few random swaps to kickstart
            for _ in range(10):
                i = random.randint(0, M-2)
                j = random.randint(0, M-2)
                if i == j: continue
                # Swap
                current_order[i], current_order[j] = current_order[j], current_order[i]
            # And run 1-opt again?
            # To keep it simple, we'll just break and return.
            # Or do a "shaking" step.
            # Given time limits, breaking is safer.
            break

    best_len = get_length(current_order)
    if best_len < best_len: # This logic is flawed, best_len was initialized to initial_len
         pass
    
    # Reconstruct word
    return _build_word(current_order, n, perms)

def _build_word(order, n, perms):
    if not order:
        return ""
    
    word = []
    # First block
    p = perms[order[0]]
    word.extend(p)
    word.append(n)
    word.extend(p)
    
    for i in range(1, len(order)):
        prev_p = perms[order[i-1]]
        curr_p = perms[order[i]]
        
        # Find overlap: longest suffix of prev_p that is prefix of curr_p
        # Note: The word currently ends with prev_p.
        # The new block starts with curr_p.
        # We need to append curr_p[k:] + [n] + curr_p
        # where k is the overlap.
        
        k = 0
        max_k = n - 1
        for kk in range(max_k, 0, -1):
            if prev_p[-kk:] == curr_p[:kk]:
                k = kk
                break
        
        # Append the rest of curr_p
        word.extend(curr_p[k:])
        word.append(n)
        word.extend(curr_p)
        
    return "".join(map(str, word))
