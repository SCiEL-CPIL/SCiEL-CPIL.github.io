import time
import random
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s margin
    
    N = 256
    nums = list(range(2, N + 1))
    n = len(nums)
    
    # Map number to index
    # Indices 0..254 correspond to 2..256
    
    # Precompute incompatibility bitsets
    # incompat[i] is an integer where bit j is 1 if nums[i] and nums[j] are incompatible
    # Incompatible if lcm(nums[i], nums[j]) <= 256
    # Note: i == j is not a constraint, but we only care about distinct pairs.
    # We will set bit i to 0 for self, but logic will ensure we don't check self.
    
    incompat = [0] * n
    
    # Optimization: 
    # lcm(a,b) <= 256 implies a*b <= 256 * gcd(a,b).
    # If a > 256, impossible. Max is 256.
    # If a=2, b must be such that lcm(2,b) <= 256 => b <= 128 if even, b <= 255 if odd?
    # lcm(2, b) = b if b even, 2b if b odd.
    # If b even, b <= 128.
    # If b odd, 2b <= 256 => b <= 128.
    # So 2 is incompatible with all even numbers <= 128 and all odd numbers <= 128.
    # Basically 2 is incompatible with everything <= 128.
    # 2 is compatible with odd numbers > 128 (e.g., 129: lcm=258>256) and even numbers > 128 (e.g., 130: lcm=130? No, lcm(2,130)=130<=256. Wait.
    # lcm(2, 130) = 130. 130 <= 256. So 2 is incompatible with 130.
    # lcm(2, b) = b if b even. So if b even and b <= 256, lcm <= 256.
    # So 2 is incompatible with ALL even numbers in [2, 256].
    # If b odd, lcm(2,b) = 2b. 2b <= 256 => b <= 128.
    # So 2 is incompatible with all odd numbers <= 128.
    # 2 is compatible with odd numbers > 128.
    
    # General check:
    for i in range(n):
        a = nums[i]
        mask = 0
        for j in range(n):
            if i == j:
                continue
            b = nums[j]
            if lcm(a, b) <= 256:
                mask |= (1 << j)
        incompat[i] = mask

    def get_score(S):
        # S is a set of indices
        return sum(1.0 / nums[i] for i in S)

    def is_valid(S):
        # S is a set of indices
        # Check pairwise
        lst = list(S)
        for i in range(len(lst)):
            for j in range(i+1, len(lst)):
                if incompat[lst[i]] & (1 << lst[j]):
                    return False
        return True

    def greedy_init(order):
        S = set()
        mask = 0
        for idx in order:
            if incompat[idx] & mask == 0:
                S.add(idx)
                mask |= (1 << idx)
        return S

    def local_search_1swap(S):
        # Returns improved S and score
        improved = True
        while improved:
            improved = False
            # Try swapping each u in S with best v not in S
            # To be efficient, we can pre-compute candidates for each u?
            # For now, simple loop.
            # We iterate over a copy of S to allow modification
            for u in list(S):
                # Remove u temporarily
                mask_without_u = 0
                for v in S:
                    if v != u:
                        mask_without_u |= (1 << v)
                
                # Find best v not in S compatible with mask_without_u
                best_v = -1
                best_val = 0.0
                for v in range(n):
                    if v in S:
                        continue
                    if incompat[v] & mask_without_u == 0:
                        val = 1.0 / nums[v]
                        if val > best_val:
                            best_val = val
                            best_v = v
                
                if best_v != -1:
                    val_u = 1.0 / nums[u]
                    if best_val > val_u:
                        S.remove(u)
                        S.add(best_v)
                        improved = True
                        # Restart loop to ensure convergence from new state
                        break
        return S

    def random_perturbation(S, k=2):
        S_new = set(S)
        # Remove k random elements
        to_remove = random.sample(list(S_new), k)
        for u in to_remove:
            S_new.remove(u)
        # Add k random elements? Or just let local search add?
        # Local search only swaps. It doesn't add if size is fixed?
        # My local search swaps u with v. Size remains constant.
        # To change size, we need add/remove moves.
        # Let's enhance local search to allow size change.
        # Move 1: Add v if compatible with S. (Gain 1/v)
        # Move 2: Remove u if 1/u < 0? No, always positive. So we never remove unless we add something better.
        # So "Add" move: if exists v not in S compatible with S, add it.
        # "Swap" move: remove u, add v.
        # "Double Swap": remove u1, u2, add v1, v2.
        
        # Let's implement a more flexible local search.
        pass

    def local_search_full(S):
        improved = True
        while improved:
            improved = False
            # 1. Try adding a single element
            mask_S = 0
            for v in S:
                mask_S |= (1 << v)
            
            # Find best v to add
            best_v = -1
            best_val = 0.0
            for v in range(n):
                if v in S:
                    continue
                if incompat[v] & mask_S == 0:
                    val = 1.0 / nums[v]
                    if val > best_val:
                        best_val = val
                        best_v = v
            
            if best_v != -1:
                S.add(best_v)
                improved = True
                continue # Re-evaluate with new S

            # 2. Try swapping one element
            for u in list(S):
                mask_without_u = 0
                for v in S:
                    if v != u:
                        mask_without_u |= (1 << v)
                
                best_v = -1
                best_val = 0.0
                for v in range(n):
                    if v in S:
                        continue
                    if incompat[v] & mask_without_u == 0:
                        val = 1.0 / nums[v]
                        if val > best_val:
                            best_val = val
                            best_v = v
                
                if best_v != -1:
                    val_u = 1.0 / nums[u]
                    if best_val > val_u:
                        S.remove(u)
                        S.add(best_v)
                        improved = True
                        break # Restart loop

            # 3. Try removing an element? 
            # Only if it allows adding multiple elements that sum to more.
            # This is covered by iterative add/swap if we are clever, 
            # but simple add/swap might get stuck in local minima where 
            # removing u allows adding v1, v2, v3 with sum > 1/u.
            # We can add a specific move: 
            # For each u, remove u. Then greedily add best compatible elements 
            # from the rest until no more can be added. 
            # If sum(new additions) > 1/u, keep the change.
            
            # Let's implement this "Remove and Refill" move.
            for u in list(S):
                S_temp = S - {u}
                mask_temp = 0
                for v in S_temp:
                    mask_temp |= (1 << v)
                
                gain = 0.0
                added = []
                # Greedily add best compatible
                # We need to pick v's in order of 1/v
                # This is O(N^2) per u. N=255, |S|~10. 2550 * 10 = 25k ops. Fine.
                for v in range(n): # Sorted by 1/v desc is just v asc
                    if v in S_temp:
                        continue
                    if incompat[v] & mask_temp == 0:
                        S_temp.add(v)
                        mask_temp |= (1 << v)
                        gain += 1.0 / nums[v]
                        added.append(v)
                
                val_u = 1.0 / nums[u]
                if gain > val_u:
                    S = S_temp
                    improved = True
                    break # Restart

        return S

    best_S = set()
    best_score = 0.0
    
    # Initial Greedy
    order = list(range(n)) # 2, 3, ..., 256
    S = greedy_init(order)
    S = local_search_full(S)
    score = get_score(S)
    if score > best_score:
        best_score = score
        best_S = set(S)
    
    # Random Restarts
    while time.time() < deadline:
        # Random order
        random_order = list(range(n))
        random.shuffle(random_order)
        S = greedy_init(random_order)
        S = local_search_full(S)
        score = get_score(S)
        if score > best_score:
            best_score = score
            best_S = set(S)
            
        # Also try perturbing the current best
        if best_S:
            S_pert = set(best_S)
            # Remove 1-3 random elements
            if len(S_pert) > 1:
                k = random.randint(1, min(3, len(S_pert)))
                to_remove = random.sample(list(S_pert), k)
                for u in to_remove:
                    S_pert.remove(u)
                S_pert = local_search_full(S_pert)
                score = get_score(S_pert)
                if score > best_score:
                    best_score = score
                    best_S = set(S_pert)

    return [nums[i] for i in best_S]

