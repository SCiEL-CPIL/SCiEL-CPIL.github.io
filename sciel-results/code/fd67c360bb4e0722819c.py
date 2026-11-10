import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 512
    target_size = 48
    
    def find_violation(S):
        S_list = list(S)
        for a in S_list:
            # dp[k] bitset
            # We need to iterate over others
            # To avoid list creation, we can just iterate S and skip a
            # But we need to know the count of others for dp size?
            # dp size is fixed max 48.
            # We can just use a fixed size array of 49.
            
            dp = [0] * 49
            dp[0] = 1
            count = 0
            for x in S_list:
                if x == a:
                    continue
                count += 1
                # Update dp
                # Iterate k from count down to 1
                for k in range(count, 0, -1):
                    if dp[k-1]:
                        dp[k] |= (dp[k-1] << x)
            
            # Check violations
            for k in range(2, count + 1):
                target_sum = k * a
                if (dp[k] >> target_sum) & 1:
                    return a
        return None

    best_set = []
    
    for restart in range(10000):
        if time.time() > deadline:
            break
            
        S = random.sample(range(1, N + 1), target_size)
        current_S = list(S)
        
        for step in range(200):
            if time.time() > deadline:
                break
                
            v = find_violation(current_S)
            
            if v is None:
                # Valid
                while len(current_S) < target_size:
                    added = False
                    for _ in range(50):
                        new_x = random.randint(1, N)
                        if new_x in current_S:
                            continue
                        test_S = current_S + [new_x]
                        if find_violation(test_S) is None:
                            current_S = test_S
                            added = True
                            break
                    if not added:
                        break
                
                if len(current_S) > len(best_set):
                    best_set = current_S
                
                if len(current_S) == target_size:
                    return current_S
                break
            
            current_S.remove(v)
            
            for _ in range(20):
                new_x = random.randint(1, N)
                if new_x in current_S:
                    continue
                test_S = current_S + [new_x]
                if find_violation(test_S) is None:
                    current_S = test_S
                    break
            
        if len(current_S) > len(best_set):
            best_set = current_S

    return best_set
