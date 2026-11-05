"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    # Sieve for primes up to 20000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                step = i
                start = i * i
                is_prime[start::step] = [False] * len(is_prime[start::step])
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes_20k = sieve(20000)
    
    def is_smooth(n, P):
        if n <= 1:
            return True
        for p in P:
            while n % p == 0:
                n //= p
            if n == 1:
                return True
        return n == 1
    
    best_result = None
    best_score = float('inf')
    
    for num_primes in range(3, 12):
        if time.time() > deadline - 5:
            break
        
        P = primes_20k[:num_primes]
        
        # Generate smooth numbers up to 20000
        smooth = set()
        for n in range(2, 20001):
            if is_smooth(n, P):
                smooth.add(n)
        
        found_for_this_P = False
        for max_range in [100, 200, 500, 1000, 2000, 5000, 10000]:
            if time.time() > deadline - 3:
                break
            
            candidates = list(range(1, max_range + 1))
            n_cand = len(candidates)
            
            # Build adjacency
            adj = [set() for _ in range(n_cand)]
            for i in range(n_cand):
                ci = candidates[i]
                for j in range(i + 1, n_cand):
                    cj = candidates[j]
                    if ci + cj in smooth:
                        adj[i].add(j)
                        adj[j].add(i)
            
            # Find 12-clique
            degrees = [len(adj[i]) for i in range(n_cand)]
            order = sorted(range(n_cand), key=lambda x: -degrees[x])
            
            found = None
            def backtrack(clique, start_idx):
                nonlocal found
                if len(clique) == 12:
                    found = clique[:]
                    return True
                if found is not None:
                    return True
                remaining = 12 - len(clique)
                for idx in range(start_idx, len(order)):
                    v = order[idx]
                    ok = all(v in adj[u] for u in clique)
                    if not ok:
                        continue
                    if len(order) - idx - 1 < remaining - 1:
                        break
                    clique.append(v)
                    if backtrack(clique, idx + 1):
                        return True
                    clique.pop()
                return False
            
            backtrack([], 0)
            
            if found is not None:
                result = sorted([candidates[v] for v in found])
                distinct_primes = set()
                for i in range(12):
                    for j in range(i + 1, 12):
                        s = result[i] + result[j]
                        for p in P:
                            while s % p == 0:
                                distinct_primes.add(p)
                                s //= p
                        if s > 1:
                            distinct_primes.add(s)
                score = len(distinct_primes)
                if score < best_score:
                    best_score = score
                    best_result = result
                found_for_this_P = True
                break
        
        if found_for_this_P and best_score <= num_primes:
            break
    
    if best_result is not None:
        return best_result
    return list(range(1, 13))
# EVOLVE-BLOCK-END
