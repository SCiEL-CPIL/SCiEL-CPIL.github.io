import time

def construct():
    N = 123457
    start_time = time.time()
    deadline = start_time + 50.0  # 10s safety margin
    
    lpf = [0] * (N + 1)
    # Sieve of Eratosthenes variant to find largest prime factor
    for i in range(2, N + 1):
        if lpf[i] == 0: # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i
                
    best_a = 1
    best_score = float('inf')
    
    # Optimization: Check a=1 and a=N-1 first? No, loop is fast.
    # But we can break if best_score == 2 (theoretical min for N>2)
    
    for a in range(1, N):
        if time.time() > deadline:
            break
        b = N - a
        pa = lpf[a] if a > 1 else 1
        pb = lpf[b] if b > 1 else 1
        current_max = pa if pa > pb else pb
        if current_max < best_score:
            best_score = current_max
            best_a = a
            if best_score == 2:
                break
                
    return [best_a, N - best_a]
