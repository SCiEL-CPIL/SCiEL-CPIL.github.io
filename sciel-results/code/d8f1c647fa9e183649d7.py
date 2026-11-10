import time

def construct():
    start_time = time.time()
    deadline = 80.0 # Safety margin
    
    p = 1009
    # Precompute inverses
    inv = [0] * (p + 1)
    for i in range(1, p):
        # Fermat's little theorem: i^(p-2) mod p
        # Or just brute force for small p
        # Since p is small, we can just loop
        pass
        
    # Faster inverse calculation for all i
    # inv[i] * i % p == 1
    # We can just loop j for each i, but O(N^2) is fine.
    # Or use pow(i, p-2, p)
    for i in range(1, p):
        inv[i] = pow(i, p - 2, p)

    max_len = 0
    best_start = 2
    
    # L can range from 2 to 1007
    # If L=1, inv[1]=1, conflict.
    # If L=1008, inv[1008]=1008, conflict.
    
    for L in range(2, 1008):
        if time.time() > deadline:
            break
            
        R = L
        # Expand R as far as possible
        while R < 1007:
            next_R = R + 1
            # Check if next_R conflicts with any x in [L, R]
            # Conflict if inv[next_R] is in [L, R]
            if L <= inv[next_R] <= R:
                break
            R = next_R
            
        current_len = R - L + 1
        if current_len > max_len:
            max_len = current_len
            best_start = L
            
    return [best_start, max_len]
