def construct():
    N = 123457
    # Sieve for largest prime factor
    lpf = [0] * (N + 1)
    lpf[1] = 1
    for i in range(2, N + 1):
        if lpf[i] == 0: # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i
    
    best_val = float('inf')
    best_a = 1
    
    # We can optimize by only checking a up to N//2
    # due to symmetry max(P(a), P(b)) is same for (a, b) and (b, a)
    for a in range(1, N // 2 + 1):
        b = N - a
        val = max(lpf[a], lpf[b])
        if val < best_val:
            best_val = val
            best_a = a
            # Optional: if best_val == 2, we can't do better (min prime is 2, P(1)=1)
            # Actually P(1)=1, so min possible max is 1? 
            # If a=1, b=N-1. P(1)=1. P(N-1) is likely large.
            # If a and b are both 1? No, sum is fixed.
            # Min possible value for P(n) is 1 if n=1.
            # Can we have a=1, b=123456? P(1)=1, P(123456)=23. Max=23.
            # Can we have both small?
            # If we find val=2, it means both are powers of 2.
            # 2^k + 2^m = 123457 (odd). Impossible. One must be odd.
            # So min prime factor for odd number is >=3.
            # So max(P(a), P(b)) >= 3.
            if best_val == 3:
                break
    return [best_a, N - best_a]
