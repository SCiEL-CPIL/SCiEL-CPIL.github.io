import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin before 90s kill
    
    N = 123457
    
    def generate_smooth(primes, limit):
        """Generate all smooth numbers (with prime factors in primes) up to limit."""
        smooth = set([1])
        for p in primes:
            new_smooth = set()
            for s in smooth:
                v = s
                while v <= limit:
                    new_smooth.add(v)
                    v *= p
            smooth |= new_smooth
        return smooth
    
    # Strategy 1: Try 7-smooth decomposition (primes: 2, 3, 5, 7)
    # If both a and N-a are 7-smooth, max(P(a),P(b)) <= 7
    smooth7 = generate_smooth([2, 3, 5, 7], N)
    for a in smooth7:
        if a < N:
            b = N - a
            if b in smooth7:
                return [a, b]
    
    # Strategy 2: Try 11-smooth decomposition (primes: 2, 3, 5, 7, 11)
    smooth11 = generate_smooth([2, 3, 5, 7, 11], N)
    for a in smooth11:
        if a < N:
            b = N - a
            if b in smooth11:
                return [a, b]
    
    # Strategy 3: Fall back to LPF sieve
    lpf = [1] * (N + 1)
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    is_prime[1] = False
    
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    
    for i in range(2, N + 1):
        if is_prime[i]:
            for j in range(i, N + 1, i):
                lpf[j] = i
    
    best_a = 1
    best_val = max(lpf[1], lpf[N - 1])
    
    for a in range(1, N):
        if time.time() > deadline:
            break
        b = N - a
        val = max(lpf[a], lpf[b])
        if val < best_val:
            best_val = val
            best_a = a
    
    return [best_a, N - best_a]
