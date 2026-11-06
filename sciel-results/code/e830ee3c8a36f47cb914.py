import random
import time

def construct():
        start_time = time.time()
        time_limit = 55  # Leave 5s buffer
        
        def is_valid(S):
            n = len(S)
            if n < 3: # Need at least 3 elements for a mean of 2 others? No, 1 element is mean of 2 others requires 3 elements total.
                return True
            
            # To speed up, sort S?
            # The DP is independent of order.
            
            # We check each a
            for a in S:
                others = [y for y in S if y != a]
                m = len(others)
                if m < 2:
                    continue
                
                # DP for others
                # Max sum of others
                max_sum = sum(others)
                dp = [0] * (m + 1)
                dp[0] = 1
                
                for y in others:
                    for k in range(m, 0, -1):
                        dp[k] |= (dp[k-1] << y)
                
                for k in range(2, m + 1):
                    target = k * a
                    if target > max_sum:
                        break
                    if (dp[k] >> target) & 1:
                        return False
            return True

        def greedy_random():
            S = set()
            available = list(range(1, 513))
            random.shuffle(available)
            for x in available:
                if len(S) == 48:
                    break
                # Check if adding x is valid
                # To speed up, we can check only if x is not the mean of existing subsets?
                # No, x might cause existing elements to become invalid?
                # If S is valid, adding x:
                # 1. x must not be mean of subset of S.
                # 2. Existing a in S must not be mean of subset of S \ {a} U {x}.
                #    This means a was not mean of subset of S \ {a}, but might be mean of subset involving x.
                #    So we must re-check all a in S.
                # Thus, is_valid(S U {x}) is required.
                
                if is_valid(list(S) + [x]):
                    S.add(x)
            return S

        best_set = greedy_random()
        
        # Local Search
        current_set = set(best_set)
        
        while time.time() - start_time < time_limit:
            if len(current_set) == 48:
                break
            
            available = [x for x in range(1, 513) if x not in current_set]
            if not available:
                break
            
            # Try to add
            x = random.choice(available)
            if is_valid(list(current_set) + [x]):
                current_set.add(x)
                if len(current_set) > len(best_set):
                    best_set = set(current_set)
                    if len(best_set) == 48:
                        break
            else:
                # Try to swap
                if current_set:
                    y = random.choice(list(current_set))
                    z = random.choice(available)
                    if y == z: continue
                    new_set = set(current_set)
                    new_set.remove(y)
                    new_set.add(z)
                    if is_valid(list(new_set)):
                        current_set = new_set
                        # We don't necessarily update best_set here unless size increased,
                        # but a swap keeps size same. It might unblock future additions.
        
        return list(best_set)
