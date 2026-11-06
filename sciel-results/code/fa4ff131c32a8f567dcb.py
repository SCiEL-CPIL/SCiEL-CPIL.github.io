import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    N = 1000
    max_sum = 2 * N
    
    # State variables
    # S: boolean array of size N+1, S[i] is 1 if i is in the set
    # C: integer array of size max_sum+1, C[s] is the count of representations for sum s
    
    def init_greedy(random_order=True):
        S = np.zeros(N + 1, dtype=bool)
        C = np.zeros(max_sum + 1, dtype=np.int32)
        
        if random_order:
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
        else:
            candidates = list(range(1, N + 1))
            
        for x in candidates:
            if S[x]:
                continue
            # Check validity
            # Sums involving x: x + y for y in S, and 2x
            indices = np.where(S)[0]
            if len(indices) > 0:
                sums = x + indices
                # Check if any sum has count >= 2
                if np.any(C[sums] >= 2):
                    continue
            if C[2 * x] >= 2:
                continue
            
            # Add x
            S[x] = True
            if len(indices) > 0:
                C[x + indices] += 1
            C[2 * x] += 1
            
        return S, C, np.sum(S)

    def local_search(S, C, size):
        # Try to improve the set
        # We will try random insertions and swaps
        no_improvement_count = 0
        max_no_improvement = 1000 # Limit iterations per local search phase
        
        while no_improvement_count < max_no_improvement:
            # Try random insertion
            x = random.randint(1, N)
            if not S[x]:
                indices = np.where(S)[0]
                valid = True
                if len(indices) > 0:
                    sums = x + indices
                    if np.any(C[sums] >= 2):
                        valid = False
                if C[2 * x] >= 2:
                    valid = False
                
                if valid:
                    # Accept insertion
                    S[x] = True
                    if len(indices) > 0:
                        C[x + indices] += 1
                    C[2 * x] += 1
                    size += 1
                    no_improvement_count = 0
                    continue
            
            # Try random swap: remove x, add y
            # Pick x in S
            indices_in = np.where(S)[0]
            if len(indices_in) == 0:
                break
            x = indices_in[random.randint(0, len(indices_in) - 1)]
            
            # Pick y not in S
            y = random.randint(1, N)
            if S[y]:
                continue
                
            # Check if swapping x for y is valid
            # Remove x temporarily
            # We need to check if y can be added to S \ {x}
            
            # Sums for y in S \ {x}:
            # If x is in indices, we exclude it.
            # indices_ex_x = indices_in[indices_in != x]
            # But we can just check the condition for y against S, and if the only conflict is with x, it's fine.
            # Actually, simpler:
            # 1. Remove x from S and C
            # 2. Check if y can be added
            # 3. If yes, add y
            # 4. If no, restore x
            
            # Optimization: Check feasibility of y in S \ {x} without modifying S/C yet?
            # Feasibility of y: C'[y+z] < 2 for all z in S\{x}, and C'[2y] < 2.
            # C' is C with x removed.
            # C'[s] = C[s] - (1 if x+z=s for some z in S\{x} or 2x=s)
            # This is complex to check directly.
            
            # Let's just do the remove-check-add-restore. It's O(|S|).
            
            # Remove x
            S[x] = False
            indices_ex_x = np.where(S)[0]
            if len(indices_ex_x) > 0:
                C[x + indices_ex_x] -= 1
            C[2 * x] -= 1
            
            # Check y
            valid_y = True
            if len(indices_ex_x) > 0:
                sums_y = y + indices_ex_x
                if np.any(C[sums_y] >= 2):
                    valid_y = False
            if C[2 * y] >= 2:
                valid_y = False
            
            if valid_y:
                # Add y
                S[y] = True
                if len(indices_ex_x) > 0:
                    C[y + indices_ex_x] += 1
                C[2 * y] += 1
                # Size remains same, but configuration improved/changed
                no_improvement_count = 0
            else:
                # Restore x
                S[x] = True
                if len(indices_ex_x) > 0:
                    C[x + indices_ex_x] += 1
                C[2 * x] += 1
                no_improvement_count += 1
                
        return S, C, size

    best_size = 0
    best_S = np.zeros(N + 1, dtype=bool)
    
    # Multiple restarts
    while time.time() < deadline:
        S, C, size = init_greedy(random_order=True)
        if size > best_size:
            best_size = size
            best_S = S.copy()
            
        # Local search
        S, C, size = local_search(S, C, size)
        
        if size > best_size:
            best_size = size
            best_S = S.copy()
            
        # If we are close to deadline, break
        if time.time() > deadline - 5.0:
            break

    # Convert best_S to list
    result = [i for i in range(1, N + 1) if best_S[i]]
    return result
