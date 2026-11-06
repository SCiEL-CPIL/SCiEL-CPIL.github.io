import time

def construct():
    N = 123457
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50  # 50s limit to be safe
    
    # Precompute largest prime factors for 1..N
    # lpf[i] will store the largest prime factor of i
    lpf = [1] * (N + 1)
    
    # Sieve for Largest Prime Factor
    # Iterate p from N down to 2
    # If lpf[p] is still 1, p is prime
    # Mark all multiples of p with p as their largest prime factor
    # Since we go downwards, the first prime that touches a number is its largest prime factor
    for p in range(N, 1, -1):
        if lpf[p] == 1:
            # p is prime
            for j in range(p, N + 1, p):
                lpf[j] = p
                
    # Check time
    if time.time() > deadline:
        # Fallback if something went wrong (unlikely)
        return [1, N - 1]

    best_max_val = N + 1
    best_a = 1
    
    # Find the pair (a, b) with a+b=N minimizing max(P(a), P(b))
    for a in range(1, N):
        b = N - a
        # P(a) is lpf[a], P(b) is lpf[b]
        current_max = lpf[a] if lpf[a] > lpf[b] else lpf[b]
        
        if current_max < best_max_val:
            best_max_val = current_max
            best_a = a
            
            # Optional: Early termination if we found 1? 
            # P(1)=1. If a=1, P(1)=1. P(N-1).
            # If we find a pair where max is 1, it must be (1, N-1) and P(N-1)=1? No, N-1 > 1 so P(N-1) > 1.
            # So min possible max is at least 2.
            # If best_max_val == 2, we can't get lower (since P(n)>=2 for n>1, and one of a,b must be >1? 
            # If a=1, b=N-1. max(1, P(N-1)).
            # If a>1 and b>1, then P(a)>=2, P(b)>=2, so max>=2.
            # So 2 is the global minimum possible value.
            if best_max_val == 2:
                break

    return [best_a, N - best_a]
