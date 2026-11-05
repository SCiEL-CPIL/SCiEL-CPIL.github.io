import random
import time

def construct():
    deadline = time.time() + 85
    best_size = 0
    best_set = []
    
    # Try a few deterministic orders first
    orders = [
        list(range(1, 513)),
        list(range(512, 0, -1)),
        list(range(1, 513, 2)), # odds
        list(range(2, 513, 2))  # evens
    ]
    
    # Add random orders
    for i in range(50):
        c = list(range(1, 513))
        random.shuffle(c)
        orders.append(c)
        
    for order in orders:
        if time.time() > deadline:
            break
        if len(order) == 0:
            continue
            
        S = []
        max_sum = 512 * 48
        sums_by_size = [0] * 49
        sums_by_size[0] = 1
        
        for x in order:
            if time.time() > deadline:
                break
            if len(S) >= 48:
                break
                
            valid = True
            n = len(S)
            
            # 1. x is mean of subset of S
            for k in range(2, n + 1):
                if (sums_by_size[k] >> (k * x)) & 1:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # 2. y in S is mean of subset containing x
            for y in S:
                for k in range(1, n + 1):
                    target = (k + 1) * y - x
                    if 0 <= target <= max_sum:
                        if (sums_by_size[k] >> target) & 1:
                            valid = False
                            break
                if not valid:
                    break
            
            if valid:
                S.append(x)
                # Update sums
                new_sums = [0] * 49
                for i in range(n + 1):
                    new_sums[i] = sums_by_size[i]
                for k in range(n, -1, -1):
                    new_sums[k+1] |= (sums_by_size[k] << x)
                sums_by_size = new_sums
                
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            
    return best_set
