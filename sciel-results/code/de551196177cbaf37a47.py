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
    
    def local_search(init_set, max_iters=20):
        """
        Local search starting from init_set.
        Returns (best_set, best_score).
        """
        best = set(init_set)
        best_score = sum(1.0 / nums[i] for i in best)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            improved = False
            
            # 1. Try adding one element
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
            
            # 2. Try remove-one-add-one (swap)
            if time.time() < deadline - 5.0:
                best_swap = None
                best_swap_gain = 0.0
                
                for i in best:
                    if time.time() > deadline:
                        break
                    for j in range(n):
                        if j in best:
                            continue
                        if nums[j] >= nums[i]:
                            continue
                        
                        inter = conflict[j] & best
                        if i in inter:
                            if len(inter) > 1:
                                continue
                        else:
                            if len(inter) > 0:
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
        if random.random() < 0.5:
            # Random subset
            s = set(random.sample(range(n), 50))
            valid_s = set()
            for i in sorted(s, key=lambda x: nums[x]):
                if is_valid_add(valid_s, i):
                    valid_s.add(i)
            current_set = valid_s
        else:
            # Perturb best
            current_set = set(best_set)
            for _ in range(5):
                if current_set:
                    victim = random.choice(list(current_set))
                    current_set.remove(victim)
        
        ls_set, ls_score = local_search(current_set, max_iters=5)
        if ls_score > best_score:
            best_set = ls_set
            best_score = ls_score
            
    return [nums[i] for i in best_set]
# EVOLVE-BLOCK-END
