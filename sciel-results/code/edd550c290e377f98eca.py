import time

def construct():
    N = 1000000
    # Deadline guard, though this should be very fast
    start_time = time.time()
    DEADLINE = 80  # Leave 10s margin
    
    # Step 1: Compute Smallest Prime Factor (SPF)
    spf = list(range(N + 1))
    for i in range(2, int(N**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Step 2: Compute Largest Prime Factor (LPF)
    lpf = [0] * (N + 1)
    lpf[1] = 1
    for i in range(2, N + 1):
        if spf[i] == i:
            lpf[i] = i
        else:
            lpf[i] = max(spf[i], lpf[i // spf[i]])
            
    if time.time() - start_time > DEADLINE:
        return 10000

    # Step 3: Find n minimizing max(lpf[n], lpf[n+1])
    min_val = float('inf')
    best_n = 10000
    
    # We can stop early if min_val becomes very small, but let's just scan.
    # Range is 10000 to 999999
    for n in range(10000, N):
        if time.time() - start_time > DEADLINE:
            break
        current_max = max(lpf[n], lpf[n + 1])
        if current_max < min_val:
            min_val = current_max
            best_n = n
            
    return best_n
