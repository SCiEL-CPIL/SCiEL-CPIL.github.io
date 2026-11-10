import itertools
import random
import time

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    if n <= 4:
        # For small n, standard is optimal or close to it
        # 1: 1
        # 2: 121
        # 3: 123121321
        # 4: ...
        # Just use the standard recursive method for n <= 6 to be safe/fast
        pass

    # Standard recursive construction
    def standard_superpermutation(k):
        word = [1]
        for m in range(2, k + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                # Find max overlap
                max_ov = 0
                for o in range(min(len(new), len(block)), -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        max_ov = o
                        break
                new += block[max_ov:]
            word = new
        return word

    # Get standard S6
    s6_std = standard_superpermutation(6)
    std_len6 = len(s6_std)
    
    # Try to improve S6 using Simulated Annealing
    # State: list of permutations of 1..6
    # We represent the state as a list of tuples.
    # Initial state: order of permutations in s6_std
    
    # Extract order of perms from s6_std
    def get_perm_order(word, m):
        order = []
        seen = set()
        for i in range(len(word) - m + 1):
            p = tuple(word[i:i+m])
            if p not in seen:
                seen.add(p)
                order.append(p)
        return order

    order6 = get_perm_order(s6_std, 6)
    
    # Precompute overlaps for n=6
    # overlap[i][j] = max k such that order6[i][-k:] == order6[j][:k]
    # Actually, we just need the overlap value.
    # For SA, we need to quickly evaluate cost change.
    
    # Cost function: Total length of superpermutation
    # Length = N + (N-1) - Sum(Overlaps)
    # So minimizing Length is maximizing Sum(Overlaps).
    
    def calc_overlap(p1, p2):
        # p1, p2 are tuples of length 6
        # Max k such that p1[-k:] == p2[:k]
        for k in range(5, -1, -1):
            if p1[-k:] == p2[:k]:
                return k
        return 0

    # Initial sum of overlaps
    current_order = order6
    current_overlap_sum = 0
    for i in range(len(current_order) - 1):
        current_overlap_sum += calc_overlap(current_order[i], current_order[i+1])
        
    best_order = current_order
    best_overlap_sum = current_overlap_sum
    
    # SA Parameters
    T = 10.0
    T_min = 0.1
    alpha = 0.999
    steps = 50000
    
    current_T = T
    step = 0
    
    while step < steps and time.time() < deadline:
        # Move: Swap two random permutations
        i, j = random.sample(range(len(current_order)), 2)
        
        # Calculate delta overlap
        # Remove edges (i-1, i), (i, i+1), (j-1, j), (j, j+1)
        # Add edges (i-1, j), (j, i), (i, j-1), (j-1, i+1) ... depends on positions
        
        # Simpler: Just recalculate total overlap for the affected region?
        # Or just recalculate total overlap for whole list? 
        # 720 items. Recalculating 719 overlaps takes ~719 * 5 ops = 3600 ops.
        # 50000 steps * 3600 = 180M ops. Might be too slow in Python.
        
        # Optimization: Store overlap matrix?
        # 720x720 matrix. 518k entries.
        # Overlap depends on the specific permutations.
        # We can precompute overlap for all pairs of permutations in the set.
        
        if step == 0:
            # Precompute overlap matrix for all pairs in current_order
            # Map permutation to index
            perm_to_idx = {p: i for i, p in enumerate(current_order)}
            N = len(current_order)
            ov_mat = [[0]*N for _ in range(N)]
            for i in range(N):
                for j in range(N):
                    if i != j:
                        ov_mat[i][j] = calc_overlap(current_order[i], current_order[j])
            
        # For a swap of i and j (i < j)
        # Affected edges:
        # (i-1, i), (i, i+1), (j-1, j), (j, j+1)
        # New edges:
        # (i-1, j), (j, i), (i, j-1), (j-1, i+1)
        # Note: if j == i+1, the logic is different.
        
        # To keep it simple and robust, let's just do a full recalculation if N is small enough?
        # N=720. Full recalc is O(N).
        # 50k steps * 720 = 36M operations. This is fine in Python if optimized.
        
        new_order = current_order.copy()
        new_order[i], new_order[j] = new_order[j], new_order[i]
        
        new_overlap_sum = 0
        for k in range(len(new_order) - 1):
            new_overlap_sum += calc_overlap(new_order[k], new_order[k+1])
            
        delta = new_overlap_sum - current_overlap_sum
        
        if delta > 0 or random.random() < pow(2.718, delta / current_T):
            current_order = new_order
            current_overlap_sum = new_overlap_sum
            if current_overlap_sum > best_overlap_sum:
                best_overlap_sum = current_overlap_sum
                best_order = current_order.copy()
                
        current_T *= alpha
        step += 1
        
    # Construct S6 from best_order
    # We need to build the string.
    # The string is formed by overlapping the permutations.
    # Start with best_order[0]
    s6_best = list(best_order[0])
    for i in range(1, len(best_order)):
        p1 = best_order[i-1]
        p2 = best_order[i]
        ov = calc_overlap(p1, p2)
        s6_best.extend(p2[ov:])
        
    len6_best = len(s6_best)
    
    # Expand to S7
    # Use the standard expansion logic on s6_best
    s7 = []
    # We need to extract the order of permutations of 1..6 from s6_best
    # And for each, create a block with 7.
    
    # Actually, the standard expansion code in the prompt works on the *word* directly.
    # It finds windows of length 6 in s6_best.
    
    # Let's reuse the standard expansion logic but starting from s6_best
    word = s6_best
    m = 7
    perms, seen = [], set()
    for i in range(len(word) - m + 2):
        window = tuple(word[i:i + m - 1])
        if len(set(window)) == m - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
            
    new = []
    for p in perms:
        block = list(p) + [m] + list(p)
        overlap = 0
        for o in range(min(len(new), len(block)), -1, -1):
            if new[len(new) - o:] == block[:o]:
                overlap = o
                break
        new += block[overlap:]
        
    s7_final = new
    
    # Compare with standard S7
    s7_std = standard_superpermutation(7)
    
    if len(s7_final) < len(s7_std):
        return "".join(map(str, s7_final))
    else:
        return "".join(map(str, s7_std))
