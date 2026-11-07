import math

def construct():
    N = 256
    
    # 1. Sieve of Eratosthenes to get primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i*i, N+1, i):
                is_prime[j] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]
    
    # 2. Calculate max power of each prime <= N
    max_powers = {}
    for p in primes:
        val = p
        power = p
        while power * p <= N:
            power *= p
        max_powers[p] = power
        
    # 3. Calculate gains for all pairs of primes
    # Gain = M_pq - (M_p + M_q)
    # We only consider pairs where M_pq > M_p + M_q
    pairs = []
    n_primes = len(primes)
    for i in range(n_primes):
        p = primes[i]
        Mp = max_powers[p]
        for j in range(i + 1, n_primes):
            q = primes[j]
            Mq = max_powers[q]
            
            # Find max p^a * q^b <= N with a>=1, b>=1
            max_pq = 0
            a = 1
            pa = p
            while pa <= N:
                b = 1
                pq = pa * q
                while pq <= N:
                    if pq > max_pq:
                        max_pq = pq
                    b += 1
                    pq = pa * (q ** b)
                    # Optimization: if q**b grows fast, break early
                    if q ** b > N // pa:
                        break
                a += 1
                pa = p ** a
                if pa > N:
                    break
                    
            gain = max_pq - (Mp + Mq)
            if gain > 0:
                pairs.append((gain, p, q, max_pq))
                
    # 4. Sort pairs by gain descending
    pairs.sort(key=lambda x: x[0], reverse=True)
    
    # 5. Greedy selection of disjoint pairs
    selected_pairs = {} # p -> (q, val), q -> (p, val)
    used_primes = set()
    
    for gain, p, q, val in pairs:
        if p not in used_primes and q not in used_primes:
            used_primes.add(p)
            used_primes.add(q)
            selected_pairs[p] = (q, val)
            selected_pairs[q] = (p, val)
            
    # 6. Construct the result
    result = [1]
    for p in primes:
        if p in selected_pairs:
            # If p is part of a selected pair, the value is already added
            # We need to add it only once.
            # Let's just collect values
            pass
        else:
            result.append(max_powers[p])
            
    # Add values from selected pairs
    # To avoid duplicates, we iterate through the pairs we selected
    # selected_pairs maps p->(q,val) and q->(p,val)
    # We can just iterate through the keys and add if key < partner?
    # Or simply:
    added_vals = set()
    for p in selected_pairs:
        q, val = selected_pairs[p]
        if p < q: # Only add once
             result.append(val)
             
    # Ensure distinctness and type
    # The logic ensures distinctness because primes are distinct and pairs are disjoint.
    # 1 is distinct from all others.
    
    return result
