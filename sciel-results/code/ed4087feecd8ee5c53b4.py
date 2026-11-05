import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    n_points = 16
    # Generate all possible triples
    all_triples = []
    for i in range(n_points):
        for j in range(i + 1, n_points):
            for k in range(j + 1, n_points):
                all_triples.append((i, j, k))
    
    num_triples = len(all_triples)
    triple_index = {t: idx for idx, t in enumerate(all_triples)}
    
    # Precompute conflicts: conflict[i] is a list of indices j such that triple i and j share >= 2 points
    # Two triples (a,b,c) and (x,y,z) conflict if |{a,b,c} intersect {x,y,z}| >= 2
    conflicts = [[] for _ in range(num_triples)]
    for i in range(num_triples):
        t1 = set(all_triples[i])
        for j in range(i + 1, num_triples):
            t2 = set(all_triples[j])
            if len(t1.intersection(t2)) >= 2:
                conflicts[i].append(j)
                conflicts[j].append(i)
                
    # Function to check if a set of triples (by indices) is linear
    # We maintain a boolean array 'in_system'
    
    # Function to find max independent set size using backtracking
    # Returns the size of the largest independent set found, or a lower bound if search is cut short
    def max_independent_set_size(system_triples, target_size=8):
        # system_triples is a list of triple indices
        # We check if there is an independent set of size target_size
        # If yes, return target_size
        # If no, we might return target_size - 1 or less, but for optimization we care about 8 vs 7
        
        # Quick check: if number of points covered by system is small, alpha might be large?
        # No, alpha is about points NOT in any triple.
        # Wait, independence number is the largest set of points containing NO triple.
        # So we are looking for a subset of points S such that no triple from system_triples is a subset of S.
        
        # Let's use a recursive search to find an independent set of size target_size.
        # State: current set of points S, next point to consider, remaining slots needed.
        # Pruning: If remaining points < remaining slots, fail.
        # Pruning: If we can't form a triple from S U {available points}, we might be safe? No.
        
        # Actually, it's easier to iterate over subsets if n is small, but 16 choose 8 is 12870.
        # Checking if a specific subset S is independent:
        # For each t in system_triples: if t subset S, return False.
        # This is fast.
        
        # We will iterate through all subsets of size target_size.
        # If we find one that is independent, return target_size.
        # If we check all and none are independent, return target_size - 1 (assuming we don't check lower).
        
        # To be efficient, we can shuffle points to randomize which subsets we find first?
        # No, we need to be sure.
        
        # Optimization: Precompute for each point, the list of triples containing it?
        # Or just check directly.
        
        import itertools
        
        # We only need to check if alpha >= target_size.
        # If we find one, alpha >= target_size.
        # If we don't, alpha < target_size.
        
        # We can speed this up by maintaining a "blocked" status for points?
        # No, let's just do the direct check. 12870 * len(system) is acceptable if len(system) < 50.
        # 12870 * 50 = 643,500 operations. In Python, ~0.05s.
        # We can do this ~2000 times in 100s.
        
        # Let's optimize the check itself.
        # Represent points as a 16-bit mask.
        # Precompute mask for each triple.
        
        # We'll implement a fast version using bitmasks.
        
        return -1 # Placeholder

    # Let's refine the alpha calculation
    # Precompute masks for all triples
    triple_masks = [0] * num_triples
    for idx, t in enumerate(all_triples):
        mask = 0
        for p in t:
            mask |= (1 << p)
        triple_masks[idx] = mask
        
    def has_independent_set_of_size(system_indices, size):
        # system_indices: list of indices into all_triples
        # Returns True if there exists a subset of points of size 'size' that contains no triple from system_indices
        
        # If size is 0, True
        # If size is 1, True (unless system has 1-point triples, which it doesn't)
        
        # We iterate over combinations of points of size 'size'
        # There are C(16, size) combinations.
        # For each combination, check if any triple is contained.
        
        # Optimization:
        # If the system is dense, it's harder to find an independent set.
        
        # We can use a recursive backtracking to find an independent set of size 'size'.
        # This is often faster than checking all C(16, size) if we prune.
        
        def backtrack(current_points, next_point, remaining):
            if remaining == 0:
                return True
            if next_point >= n_points:
                return False
            if (n_points - next_point) < remaining:
                return False
                
            # Option 1: Don't include next_point
            if backtrack(current_points, next_point + 1, remaining):
                return True
                
            # Option 2: Include next_point
            # Check if adding next_point creates a triple with current_points
            # A triple is formed if there are two points in current_points that form a pair with next_point in some triple?
            # No, a triple is a set of 3 points.
            # We need to check if there exists a triple t in system_indices such that t is a subset of (current_points U {next_point}).
            # This happens if:
            # 1. There is a triple t subset of current_points (already checked in previous steps? No, we only check when we add)
            #    Actually, if current_points contained a triple, we wouldn't have reached this state.
            # 2. There is a triple t such that t contains next_point and 2 points from current_points.
            # 3. There is a triple t such that t contains next_point and 1 point from current_points and 1 point > next_point?
            #    No, the triple must be fully contained in the final set.
            #    If we are building the set incrementally, we only need to ensure that we don't complete a triple.
            #    A triple is completed when we add the 3rd point.
            #    So we need to check if (current_points U {next_point}) contains any triple.
            #    This is equivalent to checking if there is a triple t in system_indices such that:
            #    - t is subset of (current_points U {next_point})
            
            # To make this fast, we can precompute for each pair of points (u, v) the list of triples containing them.
            # If we add next_point, we need to check pairs (u, next_point) for u in current_points.
            # If any such pair is in a triple, and the third point is also in current_points, then we have a triple.
            # Wait, if the third point is next_point, it's (u, v, next_point).
            # So we need to check if there is a triple (u, v, next_point) where u, v are in current_points.
            
            # Let's precompute: for each point p, and each other point q, the list of triples containing p, q, and r?
            # Simpler: For each triple t, it conflicts with adding a point if...
            
            # Let's stick to the bitmask check for simplicity and correctness, and optimize later if needed.
            # But 12870 checks is fast.
            
            return False
            
        # The backtracking above is complex to implement correctly for "no triple contained".
        # Let's use the direct enumeration of subsets of size 'size' if 'size' is 8.
        # C(16, 8) = 12870.
        
        import itertools
        
        # To speed up, we can precompute the masks of the system triples.
        sys_masks = [triple_masks[i] for i in system_indices]
        
        # We want to find a mask S of weight 'size' such that for all m in sys_masks, (S & m) != m.
        
        # We can iterate over combinations.
        for comb in itertools.combinations(range(n_points), size):
            s_mask = 0
            for p in comb:
                s_mask |= (1 << p)
            
            is_independent = True
            for m in sys_masks:
                if (s_mask & m) == m:
                    is_independent = False
                    break
            if is_independent:
                return True
                
        return False

    # Local Search
    # State: list of triple indices
    # Start with a random maximal linear system
    
    def random_maximal_system():
        available = list(range(num_triples))
        random.shuffle(available)
        system = []
        in_system = [False] * num_triples
        blocked = [False] * num_triples # Blocked by conflict with existing system triple
        
        # We just pick random triples that don't conflict
        # This is a greedy maximal independent set on the conflict graph
        # To make it random, we shuffle the order of consideration.
        
        # Better: 
        # 1. Start with empty system.
        # 2. While there are available triples:
        #    Pick a random available triple.
        #    Add it to system.
        #    Mark it and all its conflicts as unavailable.
        
        available_indices = set(range(num_triples))
        while available_indices:
            # Pick random
            t_idx = random.choice(list(available_indices))
            system.append(t_idx)
            # Remove t_idx and its conflicts
            to_remove = {t_idx}
            for c in conflicts[t_idx]:
                if c in available_indices:
                    to_remove.add(c)
            available_indices.difference_update(to_remove)
            
        return system

    def get_alpha(system_indices, current_best=8):
        # Check if alpha >= current_best
        # If yes, return current_best (or higher, but we cap at 16)
        # If no, check current_best - 1
        # We assume we are interested in 8 vs 7.
        
        # If system is empty, alpha = 16
        if not system_indices:
            return 16
            
        # Check size 8
        if has_independent_set_of_size(system_indices, 8):
            return 8 # It could be higher, but 8 is "bad" enough for now. 
                      # Actually, we want to MINIMIZE alpha.
                      # If we find an independent set of size 8, alpha >= 8.
                      # If we don't, alpha < 8.
                      # So if has_independent_set_of_size returns True, alpha is at least 8.
                      # We want to know if we can get alpha <= 7.
                      # So if it returns False, alpha <= 7.
                      # Let's return a value that reflects this.
                      # Return 8 if True, 7 if False (assuming we don't check lower).
                      # This is a lower bound on alpha if True, and an upper bound if False?
                      # No.
                      # If True: alpha >= 8.
                      # If False: alpha < 8 => alpha <= 7.
                      
                      # For minimization, we prefer False.
                      # Let's return 8 if True, 7 if False.
                      # This allows us to compare.
        else:
            # It's possible alpha is 7, 6, etc.
            # We can check 7 to be sure?
            # If we just return 7, we assume alpha=7.
            # If alpha is actually 6, returning 7 is an overestimate, but still better than 8.
            # So returning 7 is safe for optimization (we think it's 7, it might be 6, both are < 8).
            return 7

    # Simulated Annealing / Hill Climbing
    best_system = random_maximal_system()
    best_alpha = get_alpha(best_system)
    
    current_system = best_system[:]
    current_alpha = best_alpha
    
    temperature = 1.0
    cooling = 0.99
    
    # Precompute list of system indices for fast access?
    # We need to efficiently remove/add triples.
    
    # To speed up has_independent_set_of_size, we can cache results?
    # The system changes slightly each step.
    
    # Let's run the search
    steps = 0
    while time.time() < deadline - 5:
        steps += 1
        
        # Pick a triple to remove
        if not current_system:
            break
        remove_idx = random.randint(0, len(current_system) - 1)
        t_remove = current_system[remove_idx]
        
        # Find a triple to add
        # It must not conflict with (current_system \ {t_remove})
        # We need to find a random triple from the "available" set.
        
        # Reconstructing available set is O(N^2).
        # Let's just pick a random triple and check if it conflicts.
        # If it conflicts, try another.
        
        t_add = -1
        for _ in range(100): # Max 100 tries
            candidate = random.randint(0, num_triples - 1)
            if candidate == t_remove:
                continue
            if candidate in current_system:
                continue
                
            # Check conflict with all other triples in system
            conflict = False
            for t_other in current_system:
                if t_other == t_remove:
                    continue
                if candidate in conflicts[t_other]:
                    conflict = True
                    break
            if not conflict:
                t_add = candidate
                break
                
        if t_add == -1:
            # No valid swap found, try a different move or restart
            # Maybe just remove and don't add? (Reduces density, might increase alpha)
            # Or restart local search from best_system
            current_system = best_system[:]
            current_alpha = best_alpha
            continue
            
        # Perform swap
        new_system = current_system[:]
        new_system[remove_idx] = t_add
        
        # Evaluate
        new_alpha = get_alpha(new_system)
        
        # Accept?
        if new_alpha < current_alpha:
            current_system = new_system
            current_alpha = new_alpha
            if current_alpha < best_alpha:
                best_alpha = current_alpha
                best_system = current_system[:]
                # If we hit 7, we are good. Can we hit 6?
                # If best_alpha == 7, we can continue to try for 6.
                # But get_alpha only distinguishes 8 vs <8.
                # We need a better alpha estimator if we want to go lower.
                # For now, let's assume 7 is the target.
                if best_alpha <= 7:
                    # We found a system with alpha <= 7.
                    # We can try to improve further, but 7 is a big jump from 8.
                    # Let's continue to see if we can find 6.
                    # To do that, we need to modify get_alpha to check 7.
                    pass
        else:
            # Accept with probability exp(-delta / T)
            delta = new_alpha - current_alpha
            if delta > 0:
                prob = pow(2.71828, -delta / temperature)
                if random.random() < prob:
                    current_system = new_system
                    current_alpha = new_alpha
        
        temperature *= cooling
        if temperature < 0.01:
            temperature = 0.01

    # Final verification and return
    # Ensure best_system is valid
    # Convert indices to triples
    result = [list(all_triples[i]) for i in best_system]
    
    # Quick sanity check for linearity
    # (Already enforced by construction)
    
    return result

if __name__ == "__main__":
    t = construct()
    print(f"Constructed {len(t)} triples")
