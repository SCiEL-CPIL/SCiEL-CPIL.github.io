import random

def construct():
    max_n = 512
    max_size = 48
    best_set = []
    
    def can_add(S, x, dp):
        # Check 1: x is not a mean of any subset of S
        # dp[k] is bitset of sums of subsets of S of size k
        for k in range(2, len(S) + 1):
            if (dp[k] >> (k * x)) & 1:
                return False
        
        # Check 2: Adding x does not make any a in S a mean of a subset of S U {x}
        # This implies subset V = {x} U U where U subset of S\{a}, |U|=j, sum(U) = (j+1)a - x
        for a in S:
            for j in range(1, len(S)): # j is size of U, so j+1 is size of V
                target = (j + 1) * a - x
                if target < 0:
                    continue
                # We need subset of S\{a} of size j summing to target
                # First check if S has such a sum
                if (dp[j] >> target) & 1:
                    # It's possible that all such subsets contain a.
                    # If target < a, then no subset containing a can sum to target (since elements > 0).
                    # So if target < a, it's valid.
                    if target < a:
                        return False
                    else:
                        # Check S\{a} explicitly
                        S_minus_a = [y for y in S if y != a]
                        dp2 = [0] * (j + 1)
                        dp2[0] = 1
                        for y in S_minus_a:
                            for k in range(j, 0, -1):
                                dp2[k] |= (dp2[k-1] << y)
                        if (dp2[j] >> target) & 1:
                            return False
        return True

    # Greedy with randomization
    for _ in range(100): # 100 runs
        candidates = list(range(1, max_n + 1))
        random.shuffle(candidates)
        S = []
        dp = [0] * (max_size + 1)
        dp[0] = 1
        
        for x in candidates:
            if len(S) >= max_size:
                break
            if can_add(S, x, dp):
                S.append(x)
                # Update dp
                for k in range(len(S), 0, -1):
                    dp[k] |= (dp[k-1] << x)
        
        if len(S) > len(best_set):
            best_set = S
            if len(best_set) == max_size:
                break
                
    return best_set
