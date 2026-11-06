import math

def construct():
    N = 123457
    
    # Generate primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    
    # Function to generate all K-smooth numbers <= N
    def get_smooths(K):
        # Primes <= K
        p_list = [p for p in primes if p <= K]
        smooths = set()
        
        def dfs(idx, current):
            if current > N:
                return
            smooths.add(current)
            for i in range(idx, len(p_list)):
                next_val = current * p_list[i]
                if next_val > N:
                    break
                dfs(i, next_val)
                
        dfs(0, 1)
        return smooths

    # Iterate through primes as potential K
    for K in primes:
        smooths = get_smooths(K)
        # Check if there is a pair a, b in smooths such that a + b = N
        for a in smooths:
            b = N - a
            if b > 0 and b in smooths:
                return [a, b]
        
        # If K is large enough, we might have missed something? 
        # Actually, if K >= N, smooths contains all numbers 1..N.
        # So a=1, b=N-1 will always be found.
        # So the loop will terminate.
        
    # Fallback (should not be reached)
    return [1, N-1]
