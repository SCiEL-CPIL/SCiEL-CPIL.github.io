import time
import random

def construct():
    N = 512
    deadline = time.time() + 85.0  # Safety margin before 90s limit
    
    # Precompute all 4-APs in [1, N]
    # A 4-AP is (a, a+d, a+2d, a+3d) with d >= 1
    all_aps = []
    for a in range(1, N + 1):
        for d in range(1, N + 1):
            if a + 3 * d <= N:
                all_aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # For each element x, precompute which APs contain x
    # ap_index[x] = list of indices into all_aps where x appears
    ap_index = [[] for _ in range(N + 1)]
    for idx, ap in enumerate(all_aps):
        for x in ap:
            ap_index[x].append(idx)
    
    # For each AP, precompute the positions (0,1,2,3) of each element
    # ap_pos[ap_idx] = dict mapping element -> position in the AP
    ap_pos = []
    for ap in all_aps:
        pos_dict = {}
        for i, x in enumerate(ap):
            pos_dict[x] = i
        ap_pos.append(pos_dict)
    
    def base4_seed():
        """Generate a 4-AP-free set using base-4 digits 0 and 1 only."""
        S = bytearray(N + 1)
        for num in range(1, N + 1):
            n = num
            valid = True
            while n > 0:
                digit = n % 4
                if digit > 1:
                    valid = False
                    break
                n //= 4
            if valid:
                S[num] = 1
        return S
    
    def can_add(x, S):
        """Check if adding x to set S (bytearray) creates any 4-AP."""
        for ap_idx in ap_index[x]:
            ap = all_aps[ap_idx]
            # Check if all other 3 elements of this AP are in S
            count = 0
            for y in ap:
                if y == x:
                    continue
                if S[y]:
                    count += 1
            if count == 3:
                return False
        return True
    
    def greedy_random(start_set=None, seed=None):
        S_arr = bytearray(N + 1)
        if start_set is not None:
            for i in range(1, N + 1):
                if start_set[i]:
                    S_arr[i] = 1
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        for x in candidates:
            if not S_arr[x] and can_add(x, S_arr):
                S_arr[x] = 1
        return S_arr
    
    def local_search(S, max_iterations=200):
        """Swap-based local search: remove one element, try to add multiple others."""
        S = bytearray(S)
        current_size = sum(S)
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            elements = [i for i in range(1, N + 1) if S[i]]
            if not elements:
                break
            random.shuffle(elements)
            
            best_swap = None
            best_new_size = current_size
            
            # Try removing each element and greedily adding as many as possible
            for u in elements[:40]:
                if time.time() > deadline:
                    break
                S_temp = bytearray(S)
                S_temp[u] = 0
                # Try to add all possible elements
                for v in range(1, N + 1):
                    if not S_temp[v] and can_add(v, S_temp):
                        S_temp[v] = 1
                
                new_size = sum(S_temp)
                if new_size > best_new_size:
                    best_new_size = new_size
                    best_swap = S_temp.copy()
                    
            if best_swap is not None and best_new_size > current_size:
                S = best_swap
                current_size = best_new_size
            else:
                break
                
        return S, current_size
    
    def walksat(S, max_iters=50000):
        """WalkSAT-style local search minimizing violated 4-AP clauses."""
        S = bytearray(S)
        current_size = sum(S)
        
        # Count violated APs (APs where all 4 elements are in S)
        def count_violations(S):
            count = 0
            for ap in all_aps:
                if S[ap[0]] and S[ap[1]] and S[ap[2]] and S[ap[3]]:
                    count += 1
            return count
        
        # We want 0 violations, but we track size as primary objective
        # Actually, let's use a different approach: 
        # We want to maximize |S| subject to no 4-APs.
        # WalkSAT flips bits to reduce violations.
        
        violations = count_violations(S)
        if violations == 0:
            return S, current_size
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random violated AP
            violated_aps = []
            for idx, ap in enumerate(all_aps):
                if S[ap[0]] and S[ap[1]] and S[ap[2]] and S[ap[3]]:
                    violated_aps.append(idx)
            
            if not violated_aps:
                break
            
            ap_idx = random.choice(violated_aps)
            ap = all_aps[ap_idx]
            
            # Try flipping each element in the AP
            best_flip = None
            best_score = None
            
            for x in ap:
                S_temp = bytearray(S)
                S_temp[x] ^= 1  # flip bit
                new_violations = count_violations(S_temp)
                new_size = sum(S_temp)
                # Score: prefer fewer violations, then larger size
                score = (-new_violations, new_size)
                if best_score is None or score > best_score:
                    best_score = score
                    best_flip = x
            
            if best_flip is not None:
                S[best_flip] ^= 1
                current_size = sum(S)
                violations = count_violations(S)
                
                if violations == 0:
                    # Try to add more elements
                    for v in range(1, N + 1):
                        if not S[v] and can_add(v, S):
                            S[v] = 1
                    current_size = sum(S)
                    return S, current_size
        
        # If we still have violations, remove elements to fix them
        violations = count_violations(S)
        while violations > 0:
            if time.time() > deadline:
                break
            violated_aps = []
            for idx, ap in enumerate(all_aps):
                if S[ap[0]] and S[ap[1]] and S[ap[2]] and S[ap[3]]:
                    violated_aps.append(idx)
            if not violated_aps:
                break
            ap_idx = random.choice(violated_aps)
            ap = all_aps[ap_idx]
            x = random.choice(ap)
            S[x] = 0
            violations = count_violations(S)
        
        # Final polish: add any remaining elements
        for v in range(1, N + 1):
            if not S[v] and can_add(v, S):
                S[v] = 1
        
        return S, sum(S)
    
    best_arr = bytearray(N + 1)
    best_size = 0
    
    # Phase 1a: Base-4 seed
    b4 = base4_seed()
    b4_size = sum(b4)
    if b4_size > best_size:
        best_size = b4_size
        best_arr = bytearray(b4)
    
    # Phase 1b: Complete base-4 seed with greedy
    S_completed = greedy_random(start_set=b4)
    S_completed_size = sum(S_completed)
    if S_completed_size > best_size:
        best_size = S_completed_size
        best_arr = bytearray(S_completed)
    
    # Phase 2: Randomized Greedy from empty (multiple restarts)
    for i in range(2000):
        if time.time() > deadline:
            break
        S = greedy_random()
        size = sum(S)
        if size > best_size:
            best_size = size
            best_arr = bytearray(S)
    
    # Phase 3: Randomized Greedy from base-4 seed (multiple restarts)
    for i in range(1000):
        if time.time() > deadline:
            break
        S = greedy_random(start_set=b4)
        size = sum(S)
        if size > best_size:
            best_size = size
            best_arr = bytearray(S)
    
    # Phase 4: Local Search (Remove-Add with swap)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S_result, size_result = local_search(best_arr, max_iterations=100)
        if size_result > best_size:
            best_size = size_result
            best_arr = bytearray(S_result)
            improved = True
    
    # Phase 5: WalkSAT-style search from best solution
    if time.time() < deadline:
        S_w, size_w = walksat(best_arr, max_iters=30000)
        if size_w > best_size:
            best_size = size_w
            best_arr = bytearray(S_w)
    
    # Phase 6: More local search rounds
    for _ in range(5):
        if time.time() > deadline:
            break
        S_result, size_result = local_search(best_arr, max_iterations=50)
        if size_result > best_size:
            best_size = size_result
            best_arr = bytearray(S_result)
    
    # Final polish: try to add any remaining element
    S = best_arr
    for v in range(1, N + 1):
        if not S[v] and can_add(v, S):
            S[v] = 1
    
    return [i for i in range(1, N + 1) if S[i]]
