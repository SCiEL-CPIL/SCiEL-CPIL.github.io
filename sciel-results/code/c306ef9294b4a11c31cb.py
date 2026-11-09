"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from heapq import heappush, heappop

def construct():
    """
    Return the smallest n in [1, 1000000] such that phi(n) is divisible by 120.
    Uses a direct constructive search over prime power components.
    """
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    TARGET = 120
    MAX_N = 1000000
    
    # Generate prime powers p^k such that p^k <= MAX_N
    # For each, compute phi(p^k) = p^(k-1) * (p-1)
    # We'll use Dijkstra-like search: state = (n, g) where g = gcd(phi_accumulated, 120)
    # We want to reach g = 120, minimizing n.
    
    # First, get primes up to MAX_N
    def sieve_primes(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    primes = sieve_primes(MAX_N)
    
    # For each prime, generate all powers p^k <= MAX_N with their phi values
    # State: (n, phi_gcd) - we use a priority queue ordered by n
    # phi_gcd represents gcd of the accumulated phi with 120
    
    # Initialize: n=1, phi=1, gcd(1,120)=1
    # Use Dijkstra: dist[g] = minimum n achieving gcd state g
    # Actually, we need to track the full phi modulo something... 
    # Better: track the accumulated phi's divisibility by 2^a * 3^b * 5^c
    
    # Since 120 = 2^3 * 3^1 * 5^1, we can track (v2, v3, v5) where v_p is the exponent of p in phi
    # We need v2>=3, v3>=1, v5>=1
    
    # For each prime power p^k, phi(p^k) = p^(k-1)*(p-1)
    # We can compute the (v2, v3, v5) of phi(p^k)
    
    def prime_factorization_exponents(x, primes_list):
        """Return dict of prime->exponent for x, using primes_list up to sqrt(x)"""
        factors = {}
        temp = x
        for p in primes_list:
            if p * p > temp:
                break
            while temp % p == 0:
                factors[p] = factors.get(p, 0) + 1
                temp //= p
        if temp > 1:
            factors[temp] = factors.get(temp, 0) + 1
        return factors
    
    # Generate candidate prime powers
    # For each prime p, for k=1,2,... while p^k <= MAX_N:
    #   pp = p^k
    #   phi_pp = p^(k-1) * (p-1)
    #   Compute (v2, v3, v5) of phi_pp
    
    candidates = []  # list of (pp, phi_pp, v2, v3, v5)
    for p in primes:
        pp = p
        k = 1
        while pp <= MAX_N:
            if k == 1:
                phi_pp = p - 1
            else:
                phi_pp = phi_pp * p  # phi(p^k) = p^(k-1)*(p-1) = p * phi(p^(k-1))
            
            # Compute v2, v3, v5 of phi_pp
            v2 = 0
            v3 = 0
            v5 = 0
            temp = phi_pp
            while temp % 2 == 0:
                v2 += 1
                temp //= 2
            while temp % 3 == 0:
                v3 += 1
                temp //= 3
            while temp % 5 == 0:
                v5 += 1
                temp //= 5
            
            candidates.append((pp, phi_pp, v2, v3, v5))
            
            if pp > MAX_N // p:
                break
            pp *= p
            k += 1
    
    # Sort candidates by pp (prime power value) to explore smaller n first
    candidates.sort(key=lambda x: x[0])
    
    # Dijkstra: state is (v2, v3, v5), capped at (3,1,1) since we only care about reaching those
    # dist[v2][v3][v5] = minimum n to achieve at least these exponents
    # We use a priority queue: (n, v2, v3, v5)
    
    INF = float('inf')
    # dist[v2][v3][v5] for v2 in 0..3, v3 in 0..1, v5 in 0..1
    dist = [[[INF for _ in range(2)] for _ in range(2)] for _ in range(4)]
    dist[0][0][0] = 1  # Start with n=1, phi=1, so v2=v3=v5=0
    
    # Priority queue: (n, v2, v3, v5)
    pq = [(1, 0, 0, 0)]
    
    best = INF
    
    while pq:
        if time.time() > deadline:
            break
        
        n, v2, v3, v5 = heappop(pq)
        
        if n > dist[v2][v3][v5]:
            continue
        
        if v2 >= 3 and v3 >= 1 and v5 >= 1:
            best = n
            break
        
        # Try adding each candidate prime power
        for pp, phi_pp, cv2, cv3, cv5 in candidates:
            new_n = n * pp
            if new_n > MAX_N:
                continue
            
            new_v2 = min(v2 + cv2, 3)
            new_v3 = min(v3 + cv3, 1)
            new_v5 = min(v5 + cv5, 1)
            
            if new_n < dist[new_v2][new_v3][new_v5]:
                dist[new_v2][new_v3][new_v5] = new_n
                heappush(pq, (new_n, new_v2, new_v3, new_v5))
    
    if best < INF:
        return best
    
    # Fallback: use sieve
    N = 1000000
    phi = list(range(N + 1))
    for i in range(2, N + 1):
        if phi[i] == i:
            for j in range(i, N + 1, i):
                phi[j] -= phi[j] // i
    for n in range(1, N + 1):
        if phi[n] % 120 == 0:
            return n
    
    return 1800
# EVOLVE-BLOCK-END
