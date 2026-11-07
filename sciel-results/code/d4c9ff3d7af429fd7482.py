import time
from math import isqrt, gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85

    N = 256

    # Sieve to get primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]

    # For each prime, find the largest prime power p^k <= N
    prime_powers = {}
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        prime_powers[p] = power

    # Baseline sum
    baseline_sum = 1 + sum(prime_powers[p] for p in primes)

    # Identify beneficial merges
    # We build a graph where nodes are primes, edges are pairs (p,q) with p*q <= N and gain > 0
    # gain = p*q - P_p - P_q
    
    # Map prime to index for easy lookup
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # Adjacency list for positive gain edges
    # graph[i] = list of (j, gain)
    graph = [[] for _ in range(num_primes)]
    
    for i in range(num_primes):
        p = primes[i]
        Pp = prime_powers[p]
        for j in range(i + 1, num_primes):
            q = primes[j]
            Pq = prime_powers[q]
            prod = p * q
            if prod > N:
                break
            gain = prod - Pp - Pq
            if gain > 0:
                graph[i].append((j, gain))
                graph[j].append((i, gain))

    # Find connected components
    visited = [False] * num_primes
    components = []
    
    for i in range(num_primes):
        if not visited[i]:
            # BFS/DFS to find component
            stack = [i]
            visited[i] = True
            comp = []
            while stack:
                node = stack.pop()
                comp.append(node)
                for neighbor, _ in graph[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)
            if comp:
                components.append(comp)

    total_max_gain = 0
    matching_pairs = [] # List of (p, q) to include in result instead of P_p, P_q

    # For each component, find max weight matching
    # Since components are small, we can use DP on bitmask
    for comp in components:
        if len(comp) == 1:
            continue # No edges, no gain
        
        # Map component nodes to 0..k-1
        k = len(comp)
        node_to_local = {node: i for i, node in enumerate(comp)}
        
        # Build local weight matrix
        # weights[i][j] = gain if edge exists, else 0
        weights = [[0] * k for _ in range(k)]
        for node in comp:
            for neighbor, gain in graph[node]:
                if neighbor in node_to_local:
                    i = node_to_local[node]
                    j = node_to_local[neighbor]
                    weights[i][j] = gain
                    weights[j][i] = gain
        
        # DP for max weight matching on k nodes
        # dp[mask] = max gain for matching in subgraph induced by mask
        size = 1 << k
        dp = [0] * size
        
        for mask in range(1, size):
            # Find lowest set bit
            lowbit = mask & (-mask)
            i = lowbit.bit_length() - 1
            remaining = mask ^ lowbit
            
            # Option 1: i is unmatched
            dp[mask] = dp[remaining]
            
            # Option 2: i is matched with j in remaining
            for j in range(k):
                if (remaining & (1 << j)) and weights[i][j] > 0:
                    new_mask = remaining ^ (1 << j)
                    candidate = weights[i][j] + dp[new_mask]
                    if candidate > dp[mask]:
                        dp[mask] = candidate
        
        full_mask = (1 << k) - 1
        max_gain = dp[full_mask]
        total_max_gain += max_gain
        
        # Reconstruct matching for this component
        mask = full_mask
        while mask:
            lowbit = mask & (-mask)
            i = lowbit.bit_length() - 1
            remaining = mask ^ lowbit
            
            matched_j = -1
            for j in range(k):
                if (remaining & (1 << j)) and weights[i][j] > 0:
                    new_mask = remaining ^ (1 << j)
                    if weights[i][j] + dp[new_mask] == dp[mask]:
                        matched_j = j
                        break
            
            if matched_j != -1:
                p = comp[i]
                q = comp[matched_j]
                matching_pairs.append((p, q))
                mask = remaining ^ (1 << matched_j)
            else:
                mask = remaining

    # Construct result
    matched_primes = set()
    for p, q in matching_pairs:
        matched_primes.add(p)
        matched_primes.add(q)
        
    result = [1]
    for p in primes:
        if p not in matched_primes:
            result.append(prime_powers[p])
            
    for p, q in matching_pairs:
        result.append(p * q)
        
    # Verify
    if len(set(result)) != len(result):
        result = [1] + [prime_powers[p] for p in primes]
    else:
        valid = True
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                if gcd(result[i], result[j]) != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            result = [1] + [prime_powers[p] for p in primes]
            
    result = [int(x) for x in result]
    return result
