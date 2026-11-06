import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 10s safety margin
    
    N = 256
    # Weights are 1/i. We want to maximize sum.
    # Order vertices by weight descending: 1, 2, 3, ...
    order = list(range(1, N + 1))
    weights = [0.0] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = 1.0 / i
        
    # Precompute suffix sums of weights for the upper bound
    # suffix_sum[i] = sum(weights[order[i]] + ... + weights[order[-1]])
    # Note: order is 1..N.
    suffix_sum = [0.0] * (N + 1)
    for i in range(N - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + weights[order[i]]
        
    best_score = 0.0
    best_set = set()
    
    # in_S array for O(1) lookup
    in_S = [False] * (N + 1)
    current_sum = 0.0
    
    # To speed up is_valid, we can maintain a list of elements in S
    S_list = []
    
    def is_valid_addition(x):
        # Check if x can be added to current S_list
        # 1. x is middle: a + c = 2x => c = 2x - a
        # 2. x is endpoint: a + x = 2c => c = (a+x)/2
        # 3. a is middle: x + c = 2a => c = 2a - x
        
        for a in S_list:
            # x is middle
            c = 2 * x - a
            if 1 <= c <= N and in_S[c]:
                return False
            
            # x is endpoint (a is middle? No, a is in S, x is new. If a is middle, then x and c are endpoints)
            # a is middle: x + c = 2a => c = 2a - x
            c2 = 2 * a - x
            if 1 <= c2 <= N and in_S[c2]:
                return False
            
            # x is endpoint, a is endpoint? Then midpoint is (x+a)/2.
            # If midpoint is in S, then {a, mid, x} is a 3-AP.
            s = a + x
            if s % 2 == 0:
                mid = s // 2
                if 1 <= mid <= N and in_S[mid]:
                    return False
        return True

    def solve(idx):
        nonlocal best_score, best_set, current_sum
        
        if time.time() > deadline:
            return
            
        # Pruning
        if current_sum + suffix_sum[idx] <= best_score:
            return
            
        if idx == N:
            if current_sum > best_score:
                best_score = current_sum
                best_set = set(S_list)
            return
            
        x = order[idx]
        
        # Option 1: Include x
        if is_valid_addition(x):
            # Add x
            in_S[x] = True
            S_list.append(x)
            current_sum += weights[x]
            
            solve(idx + 1)
            
            # Remove x
            current_sum -= weights[x]
            S_list.pop()
            in_S[x] = False
            
        # Option 2: Exclude x
        solve(idx + 1)

    solve(0)
    
    # If B&B didn't finish or didn't find anything (unlikely), fall back to greedy?
    # But B&B should find something.
    # Let's do a quick local search on the result to polish it.
    
    def hill_climb(S):
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            x_outs = sorted([x for x in S], reverse=True)
            x_ins = sorted([x for x in range(1, N + 1) if x not in S])
            
            for x_out in x_outs:
                if time.time() > deadline - 5.0:
                    break
                S_temp = S.copy()
                S_temp.remove(x_out)
                for x_in in x_ins:
                    if x_in >= x_out:
                        break
                    # Check validity of adding x_in to S_temp
                    # Re-use logic, but S_temp is a set, so we need to check against S_temp
                    # This is slow. Let's just do a simple check.
                    valid = True
                    for a in S_temp:
                        c = 2 * x_in - a
                        if c in S_temp:
                            valid = False
                            break
                        c2 = 2 * a - x_in
                        if c2 in S_temp:
                            valid = False
                            break
                        s = a + x_in
                        if s % 2 == 0:
                            mid = s // 2
                            if mid in S_temp:
                                valid = False
                                break
                    if valid:
                        S = S_temp
                        S.add(x_in)
                        improved = True
                        break
                if improved:
                    break
        return S

    if best_set:
        final_set = hill_climb(best_set)
    else:
        # Fallback greedy
        final_set = set()
        for x in range(1, N + 1):
             if is_valid_addition(x): # Note: is_valid_addition uses global state, need to be careful
                 # The global state is reset after B&B? No, in_S is global.
                 # Let's just return best_set if it exists, or a simple greedy set.
                 pass
        # Simple greedy fallback
        S_greedy = set()
        in_S_greedy = [False] * (N + 1)
        S_list_greedy = []
        for x in range(1, N + 1):
            valid = True
            for a in S_list_greedy:
                c = 2 * x - a
                if 1 <= c <= N and in_S_greedy[c]:
                    valid = False
                    break
                c2 = 2 * a - x
                if 1 <= c2 <= N and in_S_greedy[c2]:
                    valid = False
                    break
                s = a + x
                if s % 2 == 0:
                    mid = s // 2
                    if 1 <= mid <= N and in_S_greedy[mid]:
                        valid = False
                        break
            if valid:
                in_S_greedy[x] = True
                S_list_greedy.append(x)
                S_greedy.add(x)
        final_set = S_greedy

    return list(final_set)
