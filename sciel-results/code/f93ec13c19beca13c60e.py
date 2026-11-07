import time

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    # sums_by_cardinality[k] is an integer bitmask where bit s is 1 if sum s is possible with subset size k
    sums_by_cardinality = [1]  # Index 0: bit 0 is 1 (sum 0)
    S = []
    
    for x in range(1, 257):
        if time.time() > deadline:
            break
        
        max_k = len(sums_by_cardinality)
        
        # Calculate X masks
        # X[k] = sums_by_cardinality[k-1] << x
        X = [0] * (max_k + 1)
        for k in range(1, max_k + 1):
            X[k] = sums_by_cardinality[k-1] << x
            
        # Check collisions
        valid = True
        for i in range(1, max_k + 1):
            if X[i] == 0:
                continue
            for j in range(0, max_k + 1):
                if i == j:
                    continue
                # Check against old sums
                if sums_by_cardinality[j] & X[i]:
                    valid = False
                    break
                # Check against new sums X[j]
                # Note: X[j] is defined for j in 0..max_k
                if X[j] & X[i]:
                    valid = False
                    break
            if not valid:
                break
                
        if valid:
            # Update sums_by_cardinality
            sums_by_cardinality.append(X[max_k])
            for k in range(max_k - 1, 0, -1):
                sums_by_cardinality[k] |= X[k]
            S.append(x)
            
    return [int(x) for x in S]
