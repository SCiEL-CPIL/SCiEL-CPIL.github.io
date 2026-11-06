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
            
            # 3. Try remove-two-add-two (2-for-2 swap)
            if time.time() < deadline - 10.0 and len(best) >= 2:
                best_2swap = None
                best_2swap_gain = 0.0
                
                # Generate candidate pairs to remove
                remove_candidates = list(best)
                m = len(remove_candidates)
                
                for ri in range(m):
                    if time.time() > deadline:
                        break
                    r1 = remove_candidates[ri]
                    for rj in range(ri + 1, m):
                        r2 = remove_candidates[rj]
                        
                        # Compute what's free after removing r1, r2
                        # An element j is valid to add if:
                        #   j not in best, and conflict[j] & (best - {r1, r2}) is empty
                        # i.e., all conflicts of j in best are subset of {r1, r2}
                        
                        # Find all j that can be added
                        addable = []
                        for j in range(n):
                            if j in best:
                                continue
                            inter = conflict[j] & best
                            if not inter:
                                addable.append(j)
                            elif len(inter) == 1:
                                if r1 in inter or r2 in inter:
                                    addable.append(j)
                            elif len(inter) == 2:
                                if r1 in inter and r2 in inter:
                                    addable.append(j)
                        
                        if len(addable) < 2:
                            continue
                        
                        # Find best pair from addable
                        # We want max (1/nums[a] + 1/nums[b]) - (1/nums[r1] + 1/nums[r2])
                        # where a != b, both in addable, and a,b don't conflict with each other
                        best_pair_gain = 0.0
                        best_pair = None
                        for ai in range(len(addable)):
                            a = addable[ai]
                            for bi in range(ai + 1, len(addable)):
                                b = addable[bi]
                                if nums[a] == nums[b]:
                                    continue
                                # Check a and b don't conflict
                                if b in conflict[a]:
                                    continue
                                pair_gain = (1.0/nums[a] + 1.0/nums[b]) - (1.0/nums[r1] + 1.0/nums[r2])
                                if pair_gain > best_pair_gain:
                                    best_pair_gain = pair_gain
                                    best_pair = (a, b)
                        
                        if best_pair is not None and best_pair_gain > best_2swap_gain:
                            best_2swap_gain = best_pair_gain
                            best_2swap = (r1, r2, best_pair[0], best_pair[1])
                
                if best_2swap is not None:
                    r1, r2, a, b = best_2swap
                    best.discard(r1)
                    best.discard(r2)
                    best.add(a)
                    best.add(b)
                    best_score += best_2swap_gain
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
        r = random.random()
        if r < 0.3:
            # Random subset
            s = set(random.sample(range(n), 50))
            valid_s = set()
            for i in sorted(s, key=lambda x: nums[x]):
                if is_valid_add(valid_s, i):
                    valid_s.add(i)
            current_set = valid_s
        elif r < 0.6:
            # Perturb best: remove 3-8 elements
            current_set = set(best_set)
            num_remove = random.randint(3, min(8, len(current_set)))
            for _ in range(num_remove):
                if current_set:
                    victim = random.choice(list(current_set))
                    current_set.remove(victim)
        else:
            # Perturb best: remove some, add random small elements
            current_set = set(best_set)
            num_remove = random.randint(2, min(5, len(current_set)))
            for _ in range(num_remove):
                if current_set:
                    victim = random.choice(list(current_set))
                    current_set.remove(victim)
            # Try adding some small elements
            for i in range(n):
                if i not in current_set and is_valid_add(current_set, i):
                    current_set.add(i)
                    if random.random() < 0.3:
                        break
        
        ls_set, ls_score = local_search(current_set, max_iters=8)
        if ls_score > best_score:
            best_set = ls_set
            best_score = ls_score
            
    return [nums[i] for i in best_set]
# EVOLVE-BLOCK-END
