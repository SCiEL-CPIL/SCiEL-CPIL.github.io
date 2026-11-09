import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N_MAX = 512
    TARGET_SIZE = 48
    
    def is_valid(A):
        n = len(A)
        if n < 3:
            return True
        for a in A:
            others = [x for x in A if x != a]
            max_sum = sum(others)
            dp = [0] * n
            dp[0] = 1
            for x in others:
                for k in range(n-1, 0, -1):
                    dp[k] |= (dp[k-1] << x)
            for k in range(2, n):
                target = k * a
                if target > max_sum:
                    break
                if (dp[k] >> target) & 1:
                    return False
        return True

    def find_violation(A):
        n = len(A)
        for a in A:
            others = [x for x in A if x != a]
            max_sum = sum(others)
            dp = [0] * n
            dp[0] = 1
            for x in others:
                for k in range(n-1, 0, -1):
                    dp[k] |= (dp[k-1] << x)
            for k in range(2, n):
                target = k * a
                if target > max_sum:
                    break
                if (dp[k] >> target) & 1:
                    return a, others
        return None, None

    best_valid = []
    
    # Initialize
    A = random.sample(range(1, N_MAX + 1), TARGET_SIZE)
    
    # Check initial
    if is_valid(A):
        return A
        
    while time.time() < deadline:
        # Check if current A is valid
        if is_valid(A):
            if len(A) > len(best_valid):
                best_valid = A
            if len(A) == TARGET_SIZE:
                return A
                
        # Find violation and repair
        a, others = find_violation(A)
        if a is None:
            # Should be valid
            if len(A) > len(best_valid):
                best_valid = A
            return A
            
        # Repair
        if random.random() < 0.5:
            A.remove(a)
        else:
            if others:
                rem = random.choice(others)
                A.remove(rem)
            else:
                A.remove(a)
                
        # Add new
        if len(A) < TARGET_SIZE:
            A_set = set(A)
            while True:
                new_val = random.randint(1, N_MAX)
                if new_val not in A_set:
                    A.append(new_val)
                    break
                    
    return best_valid
