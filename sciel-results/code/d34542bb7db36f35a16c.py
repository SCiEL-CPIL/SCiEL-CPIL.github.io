import time
from math import log2

def construct():
    """
    Return strictly increasing integers in [2,128] such that products of every
    nonempty consecutive block are distinct. Maximize length.
    """
    start_time = time.time()
    deadline = start_time + 80.0
    
    # Precompute prime factorizations
    primes_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    prime_to_idx = {p: i for i, p in enumerate(primes_list)}
    num_primes = len(primes_list)
    
    def factorize(n):
        vec = [0] * num_primes
        temp = n
        for i, p in enumerate(primes_list):
            if p * p > temp:
                break
            while temp % p == 0:
                vec[i] += 1
                temp //= p
        if temp > 1:
            if temp in prime_to_idx:
                vec[prime_to_idx[temp]] += 1
        return tuple(vec)
    
    factorizations = {n: factorize(n) for n in range(2, 129)}
    
    def is_valid_fast(seq):
        n = len(seq)
        if n == 0:
            return True
        
        # Prefix sums of factorization vectors
        prefix = [tuple([0] * num_primes)]
        for i in range(n):
            prev = prefix[-1]
            vec = factorizations[seq[i]]
            curr = tuple(prev[k] + vec[k] for k in range(num_primes))
            prefix.append(curr)
        
        products_set = set()
        for i in range(n):
            for j in range(i + 1, n + 1):
                diff = tuple(prefix[j][k] - prefix[i][k] for k in range(num_primes))
                if diff in products_set:
                    return False
                products_set.add(diff)
        
        return True
    
    all_primes = sorted(primes_list)
    composites = [n for n in range(2, 129) if n not in prime_to_idx]
    
    def num_distinct_pf(n):
        return sum(1 for v in factorizations[n] if v > 0)
    
    # Multiple strategies, keep best
    best_seq = list(all_primes)
    best_len = len(best_seq)
    
    # Strategy 1: Greedy add composites sorted by distinct PF count
    for order in [
        sorted(composites, key=num_distinct_pf),
        sorted(composites, key=lambda x: -num_distinct_pf(x)),
        sorted(composites, key=lambda x: x),
        sorted(composites, key=lambda x: -x),
    ]:
        if time.time() > deadline:
            break
        
        current = list(all_primes)
        for c in order:
            if time.time() > deadline:
                break
            
            lo, hi = 0, len(current)
            while lo < hi:
                mid = (lo + hi) // 2
                if current[mid] < c:
                    lo = mid + 1
                else:
                    hi = mid
            pos = lo
            if pos < len(current) and current[pos] == c:
                continue
            
            new_seq = current[:pos] + [c] + current[pos:]
            if is_valid_fast(new_seq):
                current = new_seq
        
        if len(current) > best_len:
            best_len = len(current)
            best_seq = current
    
    # Strategy 2: Try removing one prime and adding two composites
    for p_idx in range(len(all_primes)):
        if time.time() > deadline:
            break
        
        base = [x for i, x in enumerate(all_primes) if i != p_idx]
        for c1 in composites:
            if time.time() > deadline:
                break
            for c2 in composites:
                if time.time() > deadline:
                    break
                if c1 >= c2:
                    continue
                if c1 in all_primes or c2 in all_primes:
                    continue
                
                candidate = sorted(base + [c1, c2])
                if len(candidate) != len(base) + 2:
                    continue
                if is_valid_fast(candidate):
                    if len(candidate) > best_len:
                        best_len = len(candidate)
                        best_seq = candidate
    
    return sorted(best_seq)
