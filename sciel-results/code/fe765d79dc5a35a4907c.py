# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    deadline = 85.0  # 85s deadline, leaving 5s safety margin before 90s kill
    
    N = 1000000
    # Initialize phi[i] = i
    phi = list(range(N + 1))
    
    # Sieve for totient - O(N log log N)
    for i in range(2, N + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, N + 1, i):
                phi[j] = phi[j] // i * (i - 1)
        
        # Check deadline periodically
        if i % 100000 == 0 and time.time() > deadline:
            break
    
    # Find smallest n with phi[n] divisible by 120
    for n in range(1, N + 1):
        if phi[n] % 120 == 0:
            return n
        if time.time() > deadline:
            break
    
    return -1
# EVOLVE-BLOCK-END
