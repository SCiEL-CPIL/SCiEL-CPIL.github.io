import time
import random
from math import gcd

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit
    
    N = 512
    MAX_A = N - 1  # 511
    
    # Precompute prime factors for each number 1..511
    prime_factors = [set() for _ in range(N)]
    for a in range(2, N):
        x = a
        p = 2
        while p * p <= x:
            while x % p == 0:
                prime_factors[a].add(p)
                x //= p
            p += 1
        if x > 1:
            prime_factors[a].add(x)
    
    # Precompute coprime checks using sets of prime factors
    # is_coprime[a][b] is True if gcd(a,b)==1
    # We can just check intersection of prime_factors[a] and prime_factors[b]
    
    def is_coprime(a, b):
        return not (prime_factors[a] & prime_factors[b])
    
    # Precompute compatibility matrix for speed
    # compat[a] is a set of b such that gcd(a,b)==1
    compat = [set() for _ in range(N)]
    for a in range(1, N):
        for b in range(1, N):
            if is_coprime(a, b):
                compat[a].add(b)
                
    def score(S):
        return sum(1.0 / (N - a) for a in S)
    
    best_S = set()
    best_score = 0.0
    
    # Greedy initialization
    def greedy_init():
        S = set()
        # Always include 1? It's coprime to all.
        # Let's just iterate from high weight to low weight
        candidates = list(range(MAX_A, 0, -1))
        random.shuffle(candidates) # Add some randomness for restarts
        # Actually, pure greedy from highest weight is a good baseline
        # Let's do a greedy that picks highest weight available
        S = set()
        for a in range(MAX_A, 0, -1):
            if S.issubset(compat[a]):
                S.add(a)
        return S
    
    # Local Search
    def local_search(S):
        current_score = score(S)
        improved = True
        while improved and (time.time() - start_time) < time_limit - 5.0:
            improved = False
            # Try swaps
            for a_out in list(S):
                best_a_in = None
                best_gain = 0
                S_minus = S - {a_out}
                # Candidates must be coprime to all in S_minus
                # And a_in != a_out
                for a_in in range(MAX_A, 0, -1):
                    if a_in in S:
                        continue
                    if S_minus.issubset(compat[a_in]):
                        gain = 1.0 / (N - a_in) - 1.0 / (N - a_out)
                        if gain > best_gain:
                            best_gain = gain
                            best_a_in = a_in
                if best_a_in is not None:
                    S.remove(a_out)
                    S.add(best_a_in)
                    current_score += best_gain
                    improved = True
                    break
            # Try adds
            if not improved:
                best_a_in = None
                best_gain = 0
                for a_in in range(MAX_A, 0, -1):
                    if a_in in S:
                        continue
                    if S.issubset(compat[a_in]):
                        gain = 1.0 / (N - a_in)
                        if gain > best_gain:
                            best_gain = gain
                            best_a_in = a_in
                if best_a_in is not None:
                    S.add(best_a_in)
                    current_score += best_gain
                    improved = True
            # Try removals (if it allows better swaps later, but usually swaps cover this)
            # We can also try random restarts within local search if stuck
        return S, current_score

    # Run multiple restarts
    num_restarts = 0
    while (time.time() - start_time) < time_limit - 10.0:
        num_restarts += 1
        # Initialize with greedy
        S = greedy_init()
        # Run local search
        S, sc = local_search(S)
        if sc > best_score:
            best_score = sc
            best_S = set(S)
            
    # Final polish on best_S
    S = set(best_S)
    S, sc = local_search(S)
    if sc > best_score:
        best_score = sc
        best_S = set(S)
        
    return list(best_S)
