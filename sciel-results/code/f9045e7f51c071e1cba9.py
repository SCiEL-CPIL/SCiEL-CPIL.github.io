import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    N = 123457
    
    # Fast path: Check for 5-smooth decomposition
    # 5-smooth numbers are of the form 2^a * 3^b * 5^c
    # We generate all 5-smooth numbers <= N
    smooth_5 = set()
    pow2 = [1]
    while pow2[-1] <= N:
        pow2.append(pow2[-1] * 2)
    pow3 = [1]
    while pow3[-1] <= N:
        pow3.append(pow3[-1] * 3)
    pow5 = [1]
    while pow5[-1] <= N:
        pow5.append(pow5[-1] * 5)
        
    for p2 in pow2:
        if p2 > N: break
        for p3 in pow3:
            if p2 * p3 > N: break
            for p5 in pow5:
                val = p2 * p3 * p5
                if val > N: break
                smooth_5.add(val)
                
    # Check if any a in smooth_5 has N-a also in smooth_5
    for a in smooth_5:
        b = N - a
        if b in smooth_5:
            # Both are 5-smooth, so max(P(a), P(b)) <= 5
            # Since P(1)=1, and N>2, min possible is >1.
            # If a=1, b=N-1. If N-1 is 5-smooth, max is 5.
            # This is likely the global minimum.
            return [a, b]

    # Fallback: Full scan
    lpf = [1] * (N + 1)
    for i in range(2, N + 1):
        if lpf[i] == 1:
            for j in range(i, N + 1, i):
                lpf[j] = i
                
    best_max_pf = float('inf')
    best_a = 1
    best_b = N - 1
    
    for a in range(1, N):
        if time.time() - start_time > DEADLINE:
            break
        b = N - a
        pa = lpf[a]
        pb = lpf[b]
        m = pa if pa > pb else pb
        if m < best_max_pf:
            best_max_pf = m
            best_a = a
            best_b = b
            
    return [best_a, best_b]
