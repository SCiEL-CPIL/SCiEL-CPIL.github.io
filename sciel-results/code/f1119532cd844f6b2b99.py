import math
import time
import random

def construct():
    N = 512
    
    # 1. Precompute divisors bitmask for each number
    # div_mask[i] has bit j set if j divides i (1-indexed bits)
    div_mask = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i, N + 1, i):
            div_mask[j] |= (1 << i)
            
    # 2. Precompute compatibility
    # compat[i] is a set of j such that lcm(i, j) <= N
    compat = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            if i * j // math.gcd(i, j) <= N:
                compat[i].add(j)
                compat[j].add(i)
                
    # Helper to get union bitmask for a set M
    def get_union_mask(M):
        mask = 0
        for m in M:
            mask |= div_mask[m]
        return mask
        
    def union_size(M):
        return get_union_mask(M).bit_count()
        
    # 3. Greedy Construction
    def greedy_construct(initial=None):
        if initial is None:
            M = {1}
        else:
            M = set(initial)
            
        # Ensure initial M is a clique, otherwise prune
        # (Greedy loop below handles addition, but initial might be invalid)
        # Simple fix: keep only elements that are mutually compatible
        valid_init = set()
        for x in M:
            ok = True
            for y in valid_init:
                if x not in compat[y]:
                    ok = False
                    break
            if ok:
                valid_init.add(x)
        M = valid_init
        if not M:
            M = {1}
            
        changed = True
        while changed:
            changed = False
            best_gain = 0
            best_candidate = None
            
            current_mask = get_union_mask(M)
            current_size = current_mask.bit_count()
            
            for x in range(1, N + 1):
                if x in M:
                    continue
                # Check compatibility
                is_compat = True
                for m in M:
                    if x not in compat[m]:
                        is_compat = False
                        break
                if not is_compat:
                    continue
                
                # Calculate marginal gain
                new_mask = current_mask | div_mask[x]
                gain = new_mask.bit_count() - current_size
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = x
            
            if best_candidate is not None and best_gain > 0:
                M.add(best_candidate)
                changed = True
                
        # Prune non-maximal elements
        # If a | b and a != b, remove a
        M_list = list(M)
        new_M = set()
        for a in M_list:
            is_max = True
            for b in M_list:
                if a != b and a % b == 0:
                    is_max = False
                    break
            if is_max:
                new_M.add(a)
        return new_M

    # 4. Local Search (Swaps)
    def local_search(M):
        M = set(M)
        improved = True
        while improved:
            improved = False
            current_mask = get_union_mask(M)
            current_size = current_mask.bit_count()
            
            # Try removing each m
            for m in list(M):
                M_minus = M - {m}
                base_mask = get_union_mask(M_minus)
                
                # Try adding each x
                for x in range(1, N + 1):
                    if x in M_minus:
                        continue
                    # Check compatibility with M_minus
                    is_compat = True
                    for y in M_minus:
                        if x not in compat[y]:
                            is_compat = False
                            break
                    if not is_compat:
                        continue
                    
                    # Check if adding x improves union size
                    new_mask = base_mask | div_mask[x]
                    new_size = new_mask.bit_count()
                    
                    if new_size > current_size:
                        M = M_minus | {x}
                        current_size = new_size
                        improved = True
                        break # Restart loop with new M
                if improved:
                    break
        return M

    # 5. Main Loop
    start_time = time.time()
    DEADLINE = start_time + 85.0
    
    # Initial run
    best_M = greedy_construct()
    best_M = local_search(best_M)
    best_size = union_size(best_M)
    
    rng = random.Random(42)
    
    while time.time() < DEADLINE:
        # Random restart
        # Pick a random starting point that is a clique
        init_M = set()
        for _ in range(rng.randint(1, 3)):
            x = rng.randint(1, N)
            # Check if x is compatible with current init_M
            ok = True
            for y in init_M:
                if x not in compat[y]:
                    ok = False
                    break
            if ok:
                init_M.add(x)
        
        if not init_M:
            init_M = {1}
            
        M = greedy_construct(init_M)
        M = local_search(M)
        
        curr_size = union_size(M)
        if curr_size > best_size:
            best_size = curr_size
            best_M = M

    # Generate final set S from best_M
    final_mask = get_union_mask(best_M)
    S = []
    for i in range(1, N + 1):
        if (final_mask >> i) & 1:
            S.append(i)
            
    return S
