import time
import itertools
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    if n == 1:
        return "1"
        
    word = [1]
    
    for m in range(2, n + 1):
        # Extract blocks: windows of length m-1 in word that are perms of 1..m-1
        # Note: In the standard construction, the windows appearing are exactly the perms of 1..m-1
        # and they appear in the order of the previous word's construction.
        # However, we want to reorder them.
        
        target_set = set(range(1, m))
        blocks = []
        seen = set()
        
        # The windows in the current word (which is a superpermutation of 1..m-1)
        # will contain all permutations of 1..m-1.
        # We extract them in the order they appear.
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(window) == m - 1 and set(window) == target_set:
                if window not in seen:
                    seen.add(window)
                    blocks.append(window)
                    
        # We have 'blocks', a list of tuples, each of length m-1.
        # We need to order them to minimize the total length of the resulting word.
        # The resulting word is formed by:
        # block_0 + [m] + block_0
        # then block_1 + [m] + block_1, overlapping with the previous part.
        # Actually, the standard construction inserts m between the first and second half of the perm?
        # No, the standard construction is:
        # For each perm P in the previous word's perm sequence:
        #   Add P, then m, then P again.
        # So the "block" is effectively P + [m] + P.
        # Let's call the unit U_p = P + [m] + P.
        # We want to order these units U_p to maximize overlap between end of U_{prev} and start of U_{curr}.
        
        units = []
        for p in blocks:
            # Unit is p + [m] + p
            # p is a tuple of length m-1
            unit = list(p) + [m] + list(p)
            units.append(unit)
            
        N = len(units)
        
        if N == 1:
            word = units[0]
            continue
            
        # Precompute overlap matrix
        # overlap[i][j] = max k such that units[i][-k:] == units[j][:k]
        # Max possible k is len(units[i]) - 1, but practically small.
        # Since units are of form P + [m] + P, the end is P and start is P.
        # So overlap is determined by suffix of P_i and prefix of P_j.
        
        # To save time, we only compute overlaps as needed or precompute if N is small enough.
        # For m=7, N=720. Matrix is 720x720.
        # We can compute it on the fly in the greedy step if we cache it, or precompute.
        # Precomputation: 720*720 = 518k pairs.
        # Each check is fast.
        
        # Let's use a greedy nearest neighbor to get an initial order.
        # Start with a random unit or the first one.
        # To be deterministic, start with index 0.
        
        remaining = list(range(N))
        current_order = [0]
        remaining.remove(0)
        
        # Cache for overlaps
        # We can compute overlap(i, j) quickly.
        def get_overlap(i, j):
            u1 = units[i]
            u2 = units[j]
            # Max overlap is len(u1)-1
            # But we know the structure.
            # u1 ends with p_i, u2 starts with p_j.
            # p_i is u1[:m-1], p_j is u2[:m-1].
            # So we just need overlap of suffix of p_i and prefix of p_j.
            pi = u1[:m-1]
            pj = u2[:m-1]
            max_k = m - 2 # Max possible match is m-2
            for k in range(max_k, 0, -1):
                if pi[-k:] == pj[:k]:
                    return k
            return 0

        # Greedy
        while remaining:
            curr = current_order[-1]
            best_next = -1
            best_overlap = -1
            for j in remaining:
                ov = get_overlap(curr, j)
                if ov > best_overlap:
                    best_overlap = ov
                    best_next = j
            current_order.append(best_next)
            remaining.remove(best_next)
            
        # Local Search: 2-opt style
        # Try to improve the order by reversing segments or swapping.
        # We define a cost function: total length = sum(len(unit)) - sum(overlaps).
        # Minimizing length == Maximizing sum of overlaps.
        
        def total_overlap(order):
            ov_sum = 0
            for i in range(len(order) - 1):
                ov_sum += get_overlap(order[i], order[i+1])
            return ov_sum
        
        current_ov = total_overlap(current_order)
        
        # Simple local search: try swapping adjacent elements
        # And 2-opt: reverse a segment
        # Given time, we might just do a few passes of adjacent swaps if they improve.
        
        improved = True
        max_iters = 10 # Limit local search iterations to avoid timeout
        iter_count = 0
        
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            if time.time() > deadline - 5.0: break
            
            # Try adjacent swaps
            for i in range(len(current_order) - 1):
                # Check time
                if time.time() > deadline - 5.0: break
                
                # Swap i and i+1
                current_order[i], current_order[i+1] = current_order[i+1], current_order[i]
                new_ov = total_overlap(current_order)
                if new_ov > current_ov:
                    current_ov = new_ov
                    improved = True
                else:
                    # Swap back
                    current_order[i], current_order[i+1] = current_order[i+1], current_order[i]
                    
            # Try 2-opt (reverse segment) - might be too slow for 720 if done naively
            # We'll stick to adjacent swaps for safety, or maybe small reversals.
            # Actually, adjacent swaps alone might get stuck in local minima.
            # Let's add a "reverse segment of length 2, 3, 4" check.
            
            for length in [2, 3, 4]:
                if time.time() > deadline - 5.0: break
                for i in range(len(current_order) - length):
                    # Reverse segment i to i+length
                    segment = current_order[i:i+length]
                    segment.reverse()
                    current_order[i:i+length] = segment
                    
                    new_ov = total_overlap(current_order)
                    if new_ov > current_ov:
                        current_ov = new_ov
                        improved = True
                    else:
                        # Reverse back
                        segment.reverse()
                        current_order[i:i+length] = segment
                        
        # Construct the new word
        new_word = []
        for idx, u_idx in enumerate(current_order):
            unit = units[u_idx]
            if idx == 0:
                new_word.extend(unit)
            else:
                prev_unit = units[current_order[idx-1]]
                # Find overlap
                # We already know how to find it, but let's recompute for safety in construction
                # Or use the same logic
                # Overlap is between end of prev_unit and start of unit
                # prev_unit ends with p_prev, unit starts with p_curr
                p_prev = prev_unit[:m-1]
                p_curr = unit[:m-1]
                max_k = m - 2
                ov = 0
                for k in range(max_k, 0, -1):
                    if p_prev[-k:] == p_curr[:k]:
                        ov = k
                        break
                new_word.extend(unit[ov:])
                
        word = new_word
        
    return "".join(map(str, word))
