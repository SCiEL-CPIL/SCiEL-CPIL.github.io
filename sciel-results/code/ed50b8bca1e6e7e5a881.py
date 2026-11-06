"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    from math import gcd
    
    # Set a safety deadline. The evaluator kills at 90s, but we want to return well before.
    deadline = time.time() + 55.0
    
    nums = list(range(2, 257))
    n = len(nums)
    
    # Precompute conflicts. 
    # conflict[i] is a set of indices j such that lcm(nums[i], nums[j]) <= 256.
    conflict = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            g = gcd(nums[i], nums[j])
            if (nums[i] // g) * nums[j] <= 256:
                conflict[i].add(j)
                conflict[j].add(i)
    
    def is_valid_add(selected, i):
        """Check if adding i to selected is valid (no conflicts)."""
        if len(selected) < len(conflict[i]):
            return not selected.intersection(conflict[i])
        else:
            return not conflict[i].intersection(selected)
    
    def local_search(init_set, max_iters=100):
        """
        Local search starting from init_set.
        Returns (best_set, best_score).
        """
        best = set(init_set)
        best_score = sum(1.0 / nums[i] for i in best)
        
        # We loop until no improvement is found or deadline is reached.
        # max_iters is a safety cap against infinite loops in edge cases.
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            improved = False
            
            # 1. Try adding one element (Best Improvement Strategy)
            best_add = -1
            best_add_val = 0.0
            for i in range(n):
                if i in best:
                    continue
                if is_valid_add(best, i):
                    val = 1.0 / nums[i]
                    if val > best_add_val:
                        best_add_val = val
                        best_add = i
            
            if best_add != -1:
                best.add(best_add)
                best_score += best_add_val
                improved = True
                # After an add, the conflict set changes significantly.
                # It is often beneficial to re-evaluate adds before swaps.
                continue 

            # 2. Try remove-one-add-one (swap)
            # Only attempt swaps if we couldn't add anything (or if we allow swaps even after adds? 
            # Usually, adding is strictly better if possible. Swaps are for optimization.)
            # Let's try to find the best swap.
            best_swap = None
            best_swap_gain = 0.0
            
            # Optimization: Iterate through candidates for j (to add) and check if removing any i works?
            # Or iterate i (to remove) and find best j?
            # Current: Iterate i in best, then j not in best.
            
            for i in best:
                if time.time() > deadline:
                    break
                # If we remove i, we can add any j that doesn't conflict with (best - {i}).
                # Let's find the best j.
                # To speed up, we can pre-filter j based on nums[j] < nums[i] (gain > 0 requires 1/j > 1/i => j < i).
                
                # We need to check conflict[j] & (best - {i}) is empty.
                # This is equivalent to (conflict[j] & best) is subset of {i}.
                
                for j in range(n):
                    if j in best:
                        continue
                    if nums[j] >= nums[i]:
                        continue
                    # Check if j conflicts with anyone in best except i
                    # conflict[j] is a set of indices.
                    # We need conflict[j] & best to be either empty or {i}.
                    # Since i is in best, if conflict[j] contains i, that's allowed.
                    # If conflict[j] contains any k in best where k != i, it's invalid.
                    
                    # Efficient check:
                    # If i is not in conflict[j], then conflict[j] & best must be empty.
                    # If i is in conflict[j], then (conflict[j] & best) - {i} must be empty.
                    
                    inter = conflict[j]
                    if i in inter:
                        # Check if any other element in best is in inter
                        # We can do this by checking if len(inter & best) > 1
                        # But inter & best creates a new set.
                        # Alternative: iterate best? No, best can be large.
                        # Let's use the set intersection.
                        # To avoid creating many sets, we can do:
                        # if len(inter & best) > 1: continue
                        # This is O(|inter|) or O(|best|).
                        
                        # A faster check for small sets:
                        # if any(k in inter for k in best if k != i):
                        # This is O(|best|).
                        
                        # Let's stick to set intersection for clarity, it's C-speed.
                        if len(inter & best) > 1:
                            continue
                    else:
                        if len(inter & best) > 0:
                            continue

                    gain = 1.0 / nums[j] - 1.0 / nums[i]
                    if gain > best_swap_gain:
                        best_swap_gain = gain
                        best_swap = (i, j)
                        
            if best_swap is not None:
                i, j = best_swap
                best.remove(i)
                best.add(j)
                best_score += best_swap_gain
                improved = True
                # After a swap, continue to check for more adds/swaps.
                continue
            
            if not improved:
                break
        
        return best, best_score

    def greedy_construct(exclude_indices):
        """
        Construct a valid set by greedily adding smallest available numbers
        not in exclude_indices.
        """
        current_set = set()
        for i in range(n):
            if i in exclude_indices:
                continue
            if is_valid_add(current_set, i):
                current_set.add(i)
        return current_set

    best_set = set()
    best_score = 0.0
    
    # 1. Standard Greedy (no exclusions)
    init_set = greedy_construct(set())
    ls_set, ls_score = local_search(init_set, max_iters=10)
    if ls_score > best_score:
        best_set = ls_set
        best_score = ls_score
        
    # 2. Exclusion Prefixes
    # The insight is that excluding small numbers allows different clusters.
    # We try excluding {2}, {2,3}, {2,3,4}, etc.
    # Note: nums[0] is 2, nums[1] is 3, etc.
    max_prefix = 20
    for p in range(1, max_prefix + 1):
        if time.time() > deadline - 10.0:
            break
        exclude = set(range(p))  # Exclude indices 0..p-1 (numbers 2..p+1)
        init_set = greedy_construct(exclude)
        ls_set, ls_score = local_search(init_set, max_iters=10)
        if ls_score > best_score:
            best_set = ls_set
            best_score = ls_score

    # 3. Random Restarts with remaining time
    random.seed(42)
    while time.time() < deadline - 2.0:
        # Generate a random start
        r = random.random()
        if r < 0.4:
            # Random subset, larger size to allow more optimization room
            # The optimal set might be larger than 50.
            # Let's sample 80-120 elements.
            k = random.randint(80, 120)
            s = set(random.sample(range(n), k))
            valid_s = set()
            # Sort by reciprocal value (descending) i.e., number ascending
            for i in sorted(s, key=lambda x: nums[x]):
                if is_valid_add(valid_s, i):
                    valid_s.add(i)
            current_set = valid_s
        elif r < 0.7:
            # Perturb best: Remove 10-20 random elements
            current_set = set(best_set)
            num_removals = random.randint(10, 20)
            for _ in range(num_removals):
                if current_set:
                    victim = random.choice(list(current_set))
                    current_set.remove(victim)
        else:
            # Greedy with random exclusions
            # Exclude a random set of small numbers to force different clusters
            exclude_count = random.randint(5, 15)
            exclude = set(random.sample(range(20), exclude_count)) # Exclude from first 20 numbers
            current_set = greedy_construct(exclude)
        
        ls_set, ls_score = local_search(current_set, max_iters=50)
        if ls_score > best_score:
            best_set = ls_set
            best_score = ls_score
            
    return [nums[i] for i in best_set]
# EVOLVE-BLOCK-END
