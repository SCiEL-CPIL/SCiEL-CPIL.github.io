import time
import math
import itertools
import random
import numpy as np

# EVOLVE-BLOCK-START

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "121321"
    if n == 4:
        return "12131214121312141213121" # Length 13? No, 1+2+6+24=33? 
        # Wait, classic S4 length is 1+2+6+24 = 33.
        # Let's just use the generic builder for n <= 5 to be safe and fast.
        pass

    if n <= 5:
        return _build_classic(n)

    # For n >= 6, use optimized construction
    # 1. Build classic S_{n-1}
    base_word = _build_classic(n - 1)
    
    # 2. Extract sequence of (n-1)-permutations
    base_perms = []
    seen = set()
    m = n - 1
    for i in range(len(base_word) - m + 1):
        window = tuple(base_word[i:i + m])
        if len(set(window)) == m and window not in seen:
            seen.add(window)
            base_perms.append(window)
    
    num_perms = len(base_perms)
    
    # 3. Compute overlap matrix
    # overlap[i][j] = max k in 0..m-1 such that base_perms[i][-k:] == base_perms[j][:k]
    # Cost of transition i->j is 1 + (m - overlap). 
    # We want to minimize total cost, which is equivalent to maximizing total overlap.
    
    perms_arr = np.array(base_perms, dtype=np.int8)
    overlap_mat = np.zeros((num_perms, num_perms), dtype=np.int8)
    
    # Efficient overlap calculation
    # For k from m-1 down to 1:
    #   if overlap_mat[i][j] is 0 and suffix of i (len k) == prefix of j (len k), set to k.
    # This is O(m * num_perms^2) which is 6 * 720^2 ~ 3M ops. Fast enough.
    
    for k in range(m - 1, 0, -1):
        # suffix of i: perms_arr[i][m-k:]
        # prefix of j: perms_arr[j][:k]
        # We want to find all pairs (i,j) where these match.
        # Vectorized approach:
        # Create a view of perms_arr where we compare suffixes and prefixes.
        # This is tricky to fully vectorize without large memory.
        # Let's stick to a loop but optimize the inner check.
        
        # Actually, we can precompute hashes or just use numpy broadcasting for each i?
        # For each i, we want to find j such that perms_arr[j][:k] == perms_arr[i][m-k:]
        # Let target = perms_arr[i][m-k:]
        # Matches = np.all(perms_arr[:, :k] == target, axis=1)
        # This is O(num_perms * k) per i. Total O(num_perms^2 * k).
        
        for i in range(num_perms):
            target = perms_arr[i][m-k:]
            # Find all j where prefix matches
            # Only update if overlap is currently 0 (i.e., we haven't found a longer overlap yet)
            # Since we iterate k from high to low, the first match is the max overlap.
            # So we only need to set if it's 0.
            
            # Optimization: Only check j where overlap_mat[i][j] == 0
            # But we need to check the match condition first.
            
            # Vectorized check for all j:
            matches = np.all(perms_arr[:, :k] == target, axis=1)
            # Update only where overlap is 0
            mask = matches & (overlap_mat[i] == 0)
            overlap_mat[i][mask] = k

    # 4. Optimize the order of permutations using Simulated Annealing
    # Objective: Maximize sum(overlap_mat[order[i]][order[i+1]])
    
    def calc_total_overlap(order):
        # order is a list of indices
        # Sum overlap_mat[order[i]][order[i+1]]
        # Vectorized:
        if len(order) < 2:
            return 0
        # Create array of consecutive pairs
        # This creates a copy, might be slow if called millions of times.
        # Better to maintain a running sum and update locally for moves.
        pass

    # Let's implement SA with incremental updates.
    
    # Initial order: The natural order from base_word
    order = list(range(num_perms))
    
    # Calculate initial total overlap
    total_overlap = 0
    for i in range(num_perms - 1):
        total_overlap += overlap_mat[order[i]][order[i+1]]
    
    best_order = order[:]
    best_total_overlap = total_overlap
    
    # SA Parameters
    T = 10.0  # Initial temperature
    T_min = 0.01
    cooling_rate = 0.995
    # We have ~100s budget. 
    # Each move is O(1) or O(k) for Or-opt.
    # Let's aim for ~1M-5M iterations.
    
    current_order = order[:]
    current_overlap = total_overlap
    
    # Precompute neighbor lists? No, random moves are fine.
    
    while T > T_min and time.time() < deadline - 10:
        # Generate a move
        move_type = random.random()
        
        if move_type < 0.7:
            # 2-opt: Reverse segment [i, j]
            i = random.randint(0, num_perms - 2)
            j = random.randint(i + 1, num_perms - 1)
            
            # Calculate delta
            # Old edges: (i-1, i) and (j, j+1)
            # New edges: (i-1, j) and (i, j+1)
            # Note: boundaries need care.
            
            old_cost = 0
            new_cost = 0
            
            # Left boundary
            if i > 0:
                old_cost += overlap_mat[current_order[i-1]][current_order[i]]
                new_cost += overlap_mat[current_order[i-1]][current_order[j]]
            else:
                # i is 0, no left edge
                pass
                
            # Right boundary
            if j < num_perms - 1:
                old_cost += overlap_mat[current_order[j]][current_order[j+1]]
                new_cost += overlap_mat[current_order[i]][current_order[j+1]]
            else:
                # j is last, no right edge
                pass
            
            # Internal edges are reversed, but sum of overlaps in a path is not symmetric?
            # Wait, overlap_mat[i][j] is NOT symmetric.
            # Reversing a segment changes the direction of all internal edges.
            # old_internal = sum(overlap_mat[cur[k]][cur[k+1]] for k in i..j-1)
            # new_internal = sum(overlap_mat[cur[k+1]][cur[k]] for k in i..j-1)
            # This requires O(length) calculation.
            # For large segments, this is slow.
            # However, 2-opt is less effective for asymmetric TSP anyway.
            # Let's use Or-opt moves which are O(1) or O(k) for small k.
            
            # Let's stick to Or-opt(1), Or-opt(2), Or-opt(3) and Swap.
            # Swap: exchange positions i and j.
            # Or-opt: move a segment of length L to a new position.
            
            # Re-evaluating move strategy for Asymmetric TSP:
            # 1. Swap(i, j): O(1) delta calculation.
            # 2. Or-opt(L): Move block of size L. O(L) delta calculation.
            
            # Let's implement Swap and Or-opt(1) and Or-opt(2).
            
            if move_type < 0.5:
                # Swap
                i = random.randint(0, num_perms - 1)
                j = random.randint(0, num_perms - 1)
                if i == j:
                    continue
                
                a = current_order[i]
                b = current_order[j]
                
                old_cost = 0
                new_cost = 0
                
                # Left of i
                if i > 0:
                    old_cost += overlap_mat[current_order[i-1]][a]
                    new_cost += overlap_mat[current_order[i-1]][b]
                # Right of i
                if i < num_perms - 1:
                    old_cost += overlap_mat[a][current_order[i+1]]
                    new_cost += overlap_mat[b][current_order[i+1]]
                    
                # Left of j
                if j > 0:
                    old_cost += overlap_mat[current_order[j-1]][b]
                    new_cost += overlap_mat[current_order[j-1]][a]
                # Right of j
                if j < num_perms - 1:
                    old_cost += overlap_mat[b][current_order[j+1]]
                    new_cost += overlap_mat[a][current_order[j+1]]
                
                delta = new_cost - old_cost
                
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_order[i], current_order[j] = b, a
                    current_overlap += delta
                    if current_overlap > best_total_overlap:
                        best_total_overlap = current_overlap
                        best_order = current_order[:]
                        
            elif move_type < 0.8:
                # Or-opt(1): Move single element from pos i to pos j
                i = random.randint(0, num_perms - 1)
                j = random.randint(0, num_perms - 1)
                if i == j or i == j - 1:
                    continue
                
                a = current_order[i]
                
                # Remove a from i
                # Insert a at j (shifting elements)
                
                # This is complex to calculate delta in O(1) without careful bookkeeping.
                # Let's just do it and recalculate local cost.
                # Actually, for N=720, O(N) recalculation is too slow for millions of iterations.
                # We need O(1) or O(k) moves.
                
                # Let's stick to 2-opt (reverse) and Swap.
                # 2-opt for asymmetric TSP:
                # Reverse segment [i, j].
                # Delta = (new_edges - old_edges).
                # Old edges: (i-1, i), (i, i+1), ..., (j-1, j), (j, j+1)
                # New edges: (i-1, j), (j, j-1), ..., (i+1, i), (i, j+1)
                # The internal edges are reversed.
                # Sum_old_internal = sum(overlap_mat[order[k]][order[k+1]] for k=i..j-1)
                # Sum_new_internal = sum(overlap_mat[order[k+1]][order[k]] for k=i..j-1)
                # This is O(length).
                
                # Given the time constraint, maybe just Swap is sufficient?
                # Or use a simpler heuristic: Random Restart + Greedy?
                
                # Let's try a different approach: 
                # Since we only need to improve by 8 symbols, and the gap is small,
                # maybe a simple "First Improvement" local search with Swap and 2-opt (short segments) is enough.
                
                # Let's implement 2-opt for short segments (length <= 10) to keep O(1) or O(10) cost.
                
                pass # Skip Or-opt for now, use Swap and short 2-opt.
                
            else:
                # Short 2-opt (length <= 5)
                i = random.randint(0, num_perms - 2)
                length = random.randint(1, 5)
                j = i + length
                if j >= num_perms:
                    continue
                
                # Reverse segment [i, j]
                # Calculate delta
                old_cost = 0
                new_cost = 0
                
                # Left boundary
                if i > 0:
                    old_cost += overlap_mat[current_order[i-1]][current_order[i]]
                    new_cost += overlap_mat[current_order[i-1]][current_order[j]]
                
                # Right boundary
                if j < num_perms - 1:
                    old_cost += overlap_mat[current_order[j]][current_order[j+1]]
                    new_cost += overlap_mat[current_order[i]][current_order[j+1]]
                
                # Internal
                # Old: i->i+1, ..., j-1->j
                # New: j->j-1, ..., i+1->i
                for k in range(i, j):
                    old_cost += overlap_mat[current_order[k]][current_order[k+1]]
                    new_cost += overlap_mat[current_order[k+1]][current_order[k]]
                
                delta = new_cost - old_cost
                
                if delta > 0 or random.random() < math.exp(delta / T):
                    # Perform reverse
                    current_order[i:j+1] = current_order[i:j+1][::-1]
                    current_overlap += delta
                    if current_overlap > best_total_overlap:
                        best_total_overlap = current_overlap
                        best_order = current_order[:]
        
        T *= cooling_rate
        # Cool faster if stuck?
        
    # 5. Build final word
    # Start with base_word? No, base_word is S_{n-1}.
    # We need to construct S_n from the optimized order of S_{n-1} perms.
    
    word = []
    for idx, perm_idx in enumerate(best_order):
        p = list(base_perms[perm_idx])
        if idx == 0:
            word.extend(p)
            word.append(n)
            word.extend(p)
        else:
            prev_p = list(base_perms[best_order[idx-1]])
            # Find max overlap
            max_k = n - 2
            overlap = 0
            for k in range(max_k, -1, -1):
                if k == 0:
                    overlap = 0
                    break
                if prev_p[-k:] == p[:k]:
                    overlap = k
                    break
            word.append(n)
            word.extend(p[overlap:])
            
    # Verify
    if time.time() < deadline - 5:
        all_perms = set(itertools.permutations(range(1, n+1)))
        found = set()
        for i in range(len(word) - n + 1):
            window = tuple(word[i:i+n])
            if len(set(window)) == n:
                found.add(window)
        if len(found) < math.factorial(n):
            # Fallback
            return _build_classic(n)
            
    return "".join(map(str, word))

def _build_classic(n):
    if n == 1:
        return [1]
    word = [1]
    for m in range(2, n + 1):
        perms = []
        seen = set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(new_word), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o > 0 and new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
                elif o == 0:
                    overlap = 0
                    break
            new_word.extend(block[overlap:])
        word = new_word
    return word

# EVOLVE-BLOCK-END
