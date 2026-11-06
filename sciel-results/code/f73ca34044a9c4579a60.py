import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    n_triples = len(all_triples)
    
    # Precompute masks for triples
    triple_masks = [0] * n_triples
    for i, t in enumerate(all_triples):
        m = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks[i] = m
        
    # Precompute conflicts (linearity violations)
    # t1 and t2 conflict if they share >= 2 points
    conflicts = [set() for _ in range(n_triples)]
    for i in range(n_triples):
        mi = triple_masks[i]
        for j in range(i + 1, n_triples):
            mj = triple_masks[j]
            if (mi & mj).bit_count() >= 2:
                conflicts[i].add(j)
                conflicts[j].add(i)
                
    # Precompute subsets containing each triple
    # subsets_of[t] = list of indices S (0 to 2^16-1) such that triple t is a subset of S
    # S is represented as a bitmask.
    subsets_of = [[] for _ in range(n_triples)]
    
    # To speed up generation, we can iterate all masks and check? 
    # 65536 * 560 is ~36M ops. Might be slow in Python.
    # Better: For each triple t, generate all supersets.
    # t has 3 bits set. Remaining 13 bits can be anything.
    # There are 2^13 = 8192 supersets.
    # Total ops: 560 * 8192 ~ 4.6M. This is fine.
    
    for i, t in enumerate(all_triples):
        mask_t = triple_masks[i]
        # Get the bits NOT in t
        # We need to iterate all subsets of the complement of t
        # Complement mask
        comp_mask = ((1 << 16) - 1) ^ mask_t
        # Iterate submasks of comp_mask
        # Standard submask iteration:
        sub = comp_mask
        subsets = []
        while True:
            S = mask_t | sub
            subsets.append(S)
            if sub == 0:
                break
            sub = (sub - 1) & comp_mask
        subsets_of[i] = subsets
        
    # Weights for independent sets
    # weight[S] = 2^{|S|}
    # We want to minimize sum of weights of independent sets.
    # Independent set S: hit_count[S] == 0
    # hit_count[S] = number of selected triples contained in S
    
    # Initialize state
    selected = [False] * n_triples
    hit_count = [0] * (1 << 16)
    
    # Greedy initialization
    # Shuffle triples and add if no conflict
    order = list(range(n_triples))
    random.shuffle(order)
    current_conflict_count = [0] * n_triples # Number of selected conflicts for each triple
    
    for idx in order:
        # Check if adding idx causes conflict
        # Conflict if any already selected triple is in conflicts[idx]
        # We can track this by checking current_conflict_count? 
        # No, current_conflict_count[idx] is the number of SELECTED triples that conflict with idx.
        # If current_conflict_count[idx] == 0, it's safe to add.
        if current_conflict_count[idx] == 0:
            selected[idx] = True
            # Update conflict counts for neighbors
            for neighbor in conflicts[idx]:
                current_conflict_count[neighbor] += 1
            # Update hit_counts
            for S in subsets_of[idx]:
                hit_count[S] += 1
                
    # Calculate initial energy
    def calculate_energy():
        energy = 0
        # Linearity penalty
        # Count pairs of selected triples that conflict
        # Since we maintain a valid linear system in greedy, this is 0 initially.
        # In SA, we might violate.
        lin_violations = 0
        for i in range(n_triples):
            if selected[i]:
                # Count selected neighbors
                # This is O(N) per step, too slow for every energy calc.
                # We should maintain lin_violations incrementally.
                pass
                
        # Independence penalty
        for S in range(1 << 16):
            if hit_count[S] == 0:
                # weight is 2^popcount(S)
                # Precompute weights?
                # 2^k is fast.
                energy += (1 << S.bit_count())
        return energy
        
    # Let's maintain energy incrementally in the SA loop.
    # Initial energy:
    energy = 0
    for S in range(1 << 16):
        if hit_count[S] == 0:
            energy += (1 << S.bit_count())
            
    lin_violations = 0
    # In greedy, we ensured no conflicts, so lin_violations = 0.
    
    C_HARD = 10 ** 10
    
    def get_alpha():
        # Find largest S with hit_count[S] == 0
        for k in range(16, 0, -1):
            # Check if any subset of size k is independent
            # We can just iterate all S and check size? 
            # 65536 iterations is fast.
            found = False
            for S in range(1 << 16):
                if S.bit_count() == k and hit_count[S] == 0:
                    found = True
                    break
            if found:
                return k
        return 0

    best_alpha = get_alpha()
    best_selected = selected[:]
    
    # Simulated Annealing
    T = 10000.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Pick random triple
        idx = random.randint(0, n_triples - 1)
        
        if selected[idx]:
            # Try to remove
            # Delta E from independence:
            # S was hit by idx. If hit_count[S] becomes 0, it becomes independent.
            # Energy increases by sum of weights of S where hit_count[S] becomes 0.
            delta_e_ind = 0
            for S in subsets_of[idx]:
                if hit_count[S] == 1:
                    delta_e_ind += (1 << S.bit_count())
            
            # Delta E from linearity:
            # Removing idx reduces conflicts with selected neighbors.
            # Current violations involving idx: number of selected neighbors.
            # We need to know how many selected neighbors idx has.
            # We can compute this on the fly or maintain it.
            # Let's compute on the fly: 
            # sum(1 for j in conflicts[idx] if selected[j])
            # This is O(deg). Deg is small.
            sel_conflicts = sum(1 for j in conflicts[idx] if selected[j])
            # These sel_conflicts pairs were contributing to lin_violations.
            # Each pair is counted once? 
            # lin_violations counts pairs {i,j} such that both selected and conflict.
            # If we remove idx, we remove sel_conflicts pairs.
            delta_e_lin = -C_HARD * sel_conflicts
            
            delta_e = delta_e_ind + delta_e_lin
            
            if delta_e < 0 or random.random() < pow(2.71828, -delta_e / T):
                # Accept
                selected[idx] = False
                for neighbor in conflicts[idx]:
                    current_conflict_count[neighbor] -= 1
                for S in subsets_of[idx]:
                    hit_count[S] -= 1
                energy += delta_e
                lin_violations -= sel_conflicts
                
                # Check alpha
                alpha = get_alpha()
                if alpha < best_alpha:
                    best_alpha = alpha
                    best_selected = selected[:]
                    
        else:
            # Try to add
            # Check conflicts
            sel_conflicts = sum(1 for j in conflicts[idx] if selected[j])
            
            delta_e_ind = 0
            for S in subsets_of[idx]:
                if hit_count[S] == 0:
                    # S was independent, now hit. Energy decreases.
                    delta_e_ind -= (1 << S.bit_count())
            
            delta_e_lin = C_HARD * sel_conflicts
            
            delta_e = delta_e_ind + delta_e_lin
            
            if delta_e < 0 or random.random() < pow(2.71828, -delta_e / T):
                # Accept
                selected[idx] = True
                for neighbor in conflicts[idx]:
                    current_conflict_count[neighbor] += 1
                for S in subsets_of[idx]:
                    hit_count[S] += 1
                energy += delta_e
                lin_violations += sel_conflicts
                
                # Check alpha
                # Optimization: Only check if lin_violations == 0?
                # If lin_violations > 0, the solution is invalid. 
                # We should probably restart or penalize heavily.
                # The C_HARD penalty should keep it near 0.
                # But get_alpha on an invalid system might be misleading?
                # The task requires a VALID system.
                # So we should only update best if lin_violations == 0.
                if lin_violations == 0:
                    alpha = get_alpha()
                    if alpha < best_alpha:
                        best_alpha = alpha
                        best_selected = selected[:]
        
        T *= cooling
        if T < 1.0:
            T = 1.0
            # Optional: Restart with fresh greedy if stuck?
            # For now, just keep cooling to 1.

    # Ensure best_selected is valid
    # Re-verify linearity
    # And return the triples
    
    # If best_alpha is still high, try a few more greedy restarts if time permits?
    # The SA should handle it.
    
    result_triples = []
    for i in range(n_triples):
        if best_selected[i]:
            result_triples.append(list(all_triples[i]))
            
    return result_triples
