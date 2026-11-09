import time

def _pf(n):
    if n <= 1:
        return set()
    s = set()
    while n % 2 == 0:
        s.add(2); n //= 2
    while n % 3 == 0:
        s.add(3); n //= 3
    f = 5
    while f * f <= n:
        while n % f == 0:
            s.add(f); n //= f
        f += 2
    if n > 1:
        s.add(n)
    return s

def _has_only_primes(n, allowed):
    """Check if n's prime factors are all in allowed set."""
    if n <= 1:
        return True
    for p in allowed:
        while n % p == 0:
            n //= p
    return n == 1

def construct():
    deadline = time.time() + 85
    N = 12
    M = 210  # 2*3*5*7
    
    # Try to find 12 distinct positive integers i such that all i+j have primes only from {2,3,5,7}
    # Then use x = M*i
    allowed = {2, 3, 5, 7}
    
    # Search for good i values
    best_i = None
    best_score = float('inf')
    
    # Start with greedy: pick i values that work with each other
    candidates = list(range(1, 50))
    
    for start in range(1, 30):
        if time.time() > deadline - 10:
            break
        current = [start]
        for c in candidates:
            if c in current:
                continue
            # Check if c works with all current
            ok = True
            for d in current:
                s = c + d
                if not _has_only_primes(s, allowed):
                    ok = False
                    break
            if ok:
                current.append(c)
                if len(current) >= N:
                    break
        if len(current) >= N:
            # Evaluate full score
            pr = set()
            for i in range(N):
                for j in range(i+1, N):
                    pr |= _pf(current[i] + current[j])
            sc = len(pr)
            if sc < best_score:
                best_score = sc
                best_i = current[:]
            if best_score <= 4:
                break
    
    if best_i is None:
        # Fallback: use multiples of 210 directly, accept whatever prime factors arise
        best_i = list(range(1, N+1))
    
    result = [M * x for x in best_i[:N]]
    
    # Verify constraints
    assert len(set(result)) == N
    assert all(1 <= x <= 10000 for x in result)
    
    return result
# EVOLVE-BLOCK-END
