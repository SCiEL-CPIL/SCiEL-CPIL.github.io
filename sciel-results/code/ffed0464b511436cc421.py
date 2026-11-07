import math

def construct():
    n = 256
    # Generate primes up to n
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, n + 1) if is_prime[i]]
    
    # Calculate max power for each prime
    max_power = {}
    for p in primes:
        val = p
        while val * p <= n:
            val *= p
        max_power[p] = val
        
    # Base sum
    base_sum = 1 + sum(max_power.values())
    
    # Find all pairs (p, q) where combining is beneficial
    # M_pq = max x <= n such that P(x) is subset of {p, q}
    # We only care if P(x) == {p, q} or {p} or {q}
    # If P(x) == {p}, x is a power of p, max is max_power[p]
    # If P(x) == {p, q}, x is p^a q^b
    
    edges = []
    for i in range(len(primes)):
        for j in range(i + 1, len(primes)):
            p = primes[i]
            q = primes[j]
            # Find max x with factors in {p, q}
            best_x = 0
            # x = p^a * q^b
            # a, b >= 0
            # Since p, q >= 2, a, b are small
            # Max a: p^a <= n
            max_a = 0
            temp = 1
            while temp * p <= n:
                temp *= p
                max_a += 1
            max_b = 0
            temp = 1
            while temp * q <= n:
                temp *= q
                max_b += 1
                
            for a in range(max_a + 1):
                for b in range(max_b + 1):
                    val = (p ** a) * (q ** b)
                    if val <= n and val > best_x:
                        best_x = val
            
            # Check if best_x uses both p and q?
            # If best_x is a power of p, it's max_power[p]
            # If best_x is a power of q, it's max_power[q]
            # We want to replace max_power[p] + max_power[q] with best_x
            # Only if best_x uses BOTH p and q?
            # If best_x = p^a, then it's just max_power[p].
            # Then sum is max_power[p] + max_power[q].
            # If we use best_x, we lose max_power[q].
            # So we only swap if best_x (using both) > max_power[p] + max_power[q]
            
            # Check if best_x is divisible by p and q
            if best_x % p == 0 and best_x % q == 0:
                gain = best_x - max_power[p] - max_power[q]
                if gain > 0:
                    edges.append((gain, i, j, best_x))
            
    # Solve Max Weight Matching on the graph defined by edges
    # Nodes are indices in primes list
    # Since edges are sparse, we can use a simple greedy or branch and bound
    # Given small number of edges, greedy is likely sufficient, but let's do a simple DP/Backtracking if edges are few
    
    # Sort edges by gain descending
    edges.sort(key=lambda x: -x[0])
    
    used = [False] * len(primes)
    selected_edges = []
    
    # Greedy Matching
    for gain, i, j, val in edges:
        if not used[i] and not used[j]:
            used[i] = True
            used[j] = True
            selected_edges.append((i, j, val))
            
    # Construct result
    result = [1]
    for idx, p in enumerate(primes):
        if not used[idx]:
            result.append(max_power[p])
            
    for i, j, val in selected_edges:
        result.append(val)
        
    return result
