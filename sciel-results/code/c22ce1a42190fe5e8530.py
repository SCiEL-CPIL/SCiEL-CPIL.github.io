import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    MAX_N = 512
    MAX_SUM = 48 * 512  # 24576
    TARGET_SIZE = 48
    
    # dp[k][s] = count of subsets of size k with sum s
    # Size: (TARGET_SIZE + 1, MAX_SUM + 1)
    # We only need up to size 48.
    
    def build_dp(S):
        dp = np.zeros((TARGET_SIZE + 1, MAX_SUM + 1), dtype=np.int64)
        dp[0][0] = 1
        for x in S:
            # Update in reverse to avoid using x multiple times
            for k in range(TARGET_SIZE - 1, -1, -1):
                if np.any(dp[k]):
                    # dp[k+1][x:] += dp[k][:-x]
                    # Note: numpy slice addition
                    dp[k+1][x:] += dp[k][:-x]
        return dp

    def is_valid_from_dp(dp, S):
        # Check if any a in S is mean of subset of S\{a}
        # We need dp for S\{a}.
        # We can compute dp_excl_a from dp using the division trick on COUNTS.
        # Q_k[s] = P_k[s] - Q_{k-1}[s-a]
        
        n = len(S)
        for a in S:
            # Compute Q (dp for S\{a})
            # Q is (n, MAX_SUM+1)
            # We only need to check k >= 2
            # Q_0 = P_0 (which is 1 at sum 0)
            # Q_k = P_k - shift(Q_{k-1}, a)
            
            # Optimization: We don't need the full Q array, just check the targets.
            # But the recurrence is sequential.
            
            # Let's just compute the full Q for this a.
            # Cost O(N * MAX_SUM).
            # Total O(N^2 * MAX_SUM).
            # 48 * 24000 * 48 = 55M ops. Too slow per check.
            
            # Alternative: Rebuild dp for S\{a} from scratch?
            # O(N^2 * MAX_SUM) again.
            
            # Is there a faster way?
            # Maybe just check the specific targets?
            # Target for k is k*a.
            # We need count of subsets of size k from S\{a} with sum k*a.
            # This is exactly Q_k[k*a].
            
            # Let's try to compute only the necessary values?
            # No, the recurrence needs the whole row.
            
            # Let's stick to the rebuild strategy but optimize it.
            # Actually, 55M numpy ops is fast (~0.05s).
            # If we do 1000 checks, 50s. Too slow.
            
            # We need to reduce the number of checks or make check faster.
            
            # Idea: Only check validity when we actually change the set.
            # In SA, we propose a move, check validity, then accept/reject.
            # So we check every move.
            
            # Let's try to limit the search space.
            # Start with a good greedy set.
            
            pass

    # Fallback: Use the bitset rebuild method which is O(N^2) bit ops.
    # 48^2 = 2304 ops.
    # 10,000 checks = 23M ops. This is feasible in Python if optimized.
    
    def is_valid_bitset(S):
        n = len(S)
        if n == 0: return True
        
        dp = [0] * (n + 1)
        dp[0] = 1
        for x in S:
            for k in range(n - 1, -1, -1):
                if dp[k]:
                    dp[k+1] |= (dp[k] << x)
        
        for a in S:
            # Compute Q for S\{a}
            # Q_k = P_k - (Q_{k-1} << a)
            # This is valid for COUNTS, not SETS.
            # So we CANNOT use this for bitsets.
            
            # We MUST rebuild dp for S\{a}.
            # To save time, we can just iterate k and check.
            # But we need the dp for S\{a}.
            
            # Let's just rebuild.
            S_excl = [x for x in S if x != a]
            n_excl = len(S_excl)
            dp_excl = [0] * (n_excl + 1)
            dp_excl[0] = 1
            for x in S_excl:
                for k in range(n_excl - 1, -1, -1):
                    if dp_excl[k]:
                        dp_excl[k+1] |= (dp_excl[k] << x)
                        
            for k in range(2, n_excl + 1):
                target = k * a
                if target > 24576: break
                if (dp_excl[k] >> target) & 1:
                    return False
        return True

    # This is O(N^3) bit ops.
    # 48^3 = 110k ops per check.
    # 1000 checks = 110M ops. ~10s in Python.
    # Feasible for ~500-1000 iterations.

    # Let's try to improve the initial set.
    
    def greedy_start():
        S = []
        dp = [0] * (TARGET_SIZE + 1)
        dp[0] = 1
        nums = list(range(1, MAX_N + 1))
        random.shuffle(nums)
        
        for x in nums:
            if len(S) >= TARGET_SIZE: break
            # Check if x can be added
            # 1. x not mean of S
            # 2. No a in S is mean of S U {x}
            
            # Just use is_valid_bitset(S + [x])
            if is_valid_bitset(S + [x]):
                S.append(x)
                # Update dp? No, is_valid_bitset rebuilds.
                
        return S

    best_set = greedy_start()
    best_size = len(best_set)
    
    current_set = best_set[:]
    current_size = len(current_set)
    
    T = 5.0
    T_min = 0.1
    cooling_rate = 0.999
    
    rng = random.Random(42)
    
    while time.time() < deadline:
        move_type = rng.random()
        
        if move_type < 0.6:
            # Add
            if current_size < MAX_N:
                current_set_set = set(current_set)
                x = rng.randint(1, MAX_N)
                if x not in current_set_set:
                    new_set = current_set + [x]
                    if is_valid_bitset(new_set):
                        new_size = len(new_set)
                        current_set = new_set
                        current_size = new_size
                        if current_size > best_size:
                            best_set = current_set[:]
                            best_size = current_size
                            
        elif move_type < 0.9:
            # Remove
            if current_size > 0:
                idx = rng.randint(0, current_size - 1)
                new_set = current_set[:idx] + current_set[idx+1:]
                new_size = len(new_set)
                delta = new_size - current_size
                if rng.random() < math.exp(delta / T):
                    current_set = new_set
                    current_size = new_size
                    
        else:
            # Swap
            if current_size > 0:
                current_set_set = set(current_set)
                u_idx = rng.randint(0, current_size - 1)
                u = current_set[u_idx]
                v = rng.randint(1, MAX_N)
                if v not in current_set_set:
                    new_set = current_set[:]
                    new_set[u_idx] = v
                    if is_valid_bitset(new_set):
                        new_size = len(new_set)
                        current_set = new_set
                        current_size = new_size
                        if current_size > best_size:
                            best_set = current_set[:]
                            best_size = current_size
        
        T = max(T_min, T * cooling_rate)

    return best_set

import math
