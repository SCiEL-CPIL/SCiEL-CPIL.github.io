import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    def is_valid(subset):
        # DP to check if any subset sums to 200
        reachable = [False] * 201
        reachable[0] = True
        for x in subset:
            if x > 200:
                continue
            for s in range(200 - x, -1, -1):
                if reachable[s]:
                    reachable[s + x] = True
                    if s + x == 200:
                        return False
        return True

    def find_minimal_conflict(subset, x):
        # Find a minimal subset of subset+{x} that sums to 200
        target = 200 - x
        if target < 0:
            return None
        
        # Find subset of subset summing to target
        # dp[s] = index of element that was last added to reach s
        dp = [-1] * (target + 1)
        dp[0] = -1  # base case
        parent = [None] * (target + 1)  # parent[s] = (prev_sum, element_index)
        
        for idx, y in enumerate(subset):
            if y > target:
                continue
            # iterate backwards
            for s in range(target - y, -1, -1):
                if dp[s] != -1 and dp[s + y] == -1:
                    dp[s + y] = idx
                    parent[s + y] = (s, idx)
        
        if dp[target] == -1:
            return None
        
        # Backtrack
        result = [x]
        s = target
        while s > 0:
            prev_s, idx = parent[s]
            result.append(subset[idx])
            s = prev_s
        return result

    def greedy_build(order):
        selected = []
        for x in order:
            if is_valid(selected + [x]):
                selected.append(x)
        return selected

    def local_search(initial_set):
        S = set(initial_set)
        U = set(range(1, 101)) - S
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Phase 1: Try to add elements
            for x in sorted(U):
                if time.time() > deadline:
                    return S
                if is_valid(list(S) + [x]):
                    S.add(x)
                    U.remove(x)
                    improved = True
            
            # Phase 2: Try swaps if no additions were made
            if not improved:
                swap_made = False
                u_list = list(U)
                random.shuffle(u_list)
                for x in u_list:
                    if time.time() > deadline:
                        return S
                    
                    conflict_set = find_minimal_conflict(list(S), x)
                    if not conflict_set:
                        continue
                        
                    s_list = [y for y in conflict_set if y in S]
                    random.shuffle(s_list)
                    for y in s_list:
                        new_S = (S - {y}) | {x}
                        if is_valid(list(new_S)):
                            S = new_S
                            U = set(range(1, 101)) - S
                            swap_made = True
                            improved = True
                            break
                    if improved:
                        break
        return S

    best_set = set()
    
    # Run multiple random restarts
    num_restarts = 200
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        if i == 0:
            order = list(range(100, 0, -1))
        elif i == 1:
            order = list(range(1, 101))
        elif i == 2:
            # Try to include many large numbers
            order = list(range(100, 50, -1)) + list(range(1, 51))
            random.shuffle(order[50:])
        else:
            # Randomized
            order = list(range(1, 101))
            random.shuffle(order)
            # Bias towards larger numbers
            if random.random() < 0.3:
                order = sorted(order, reverse=True, key=lambda x: random.random())
        
        current_set = greedy_build(order)
        current_set = local_search(current_set)
        
        if len(current_set) > len(best_set):
            best_set = current_set
            if len(best_set) >= 90:
                break

    if not best_set:
        # Fallback: return a valid small set
        return [1]
    
    return list(best_set)
