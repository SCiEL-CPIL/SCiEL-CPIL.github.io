def construct():
    import time
    from math import isqrt

    start_time = time.time()
    DEADLINE = 85.0

    def sieve(n):
        is_prime = bytearray(b'\x01') * (n + 1)
        if n >= 0: is_prime[0] = 0
        if n >= 1: is_prime[1] = 0
        for i in range(2, isqrt(n) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = 0
        return [i for i in range(n+1) if is_prime[i]]
    
    primes = sieve(10000)
    n = len(primes)
    max_val = 10000

    def greedy_init():
        seq = [primes[0]]
        last_gap = 0
        idx = 0
        while True:
            if idx + 1 < n:
                g = primes[idx+1] - primes[idx]
                if g >= last_gap:
                    seq.append(primes[idx+1])
                    last_gap = g
                    idx += 1
                    continue
            found = False
            for j in range(idx + 1, n):
                g = primes[j] - primes[idx]
                if g >= last_gap:
                    seq.append(primes[j])
                    last_gap = g
                    idx = j
                    found = True
                    break
            if not found:
                break
        return seq

    best_seq = greedy_init()
    best_len = len(best_seq)
    
    def dfs(idx, last_gap, seq):
        nonlocal best_seq, best_len
        
        if time.time() - start_time > DEADLINE:
            return False
        
        current_len = len(seq)
        
        if last_gap > 0:
            max_add = (10000 - primes[idx]) // last_gap
        else:
            max_add = n - idx - 1
            
        if current_len + max_add <= best_len:
            return True
        
        if current_len > best_len:
            best_len = current_len
            best_seq = list(seq)
            
        for j in range(idx + 1, n):
            gap = primes[j] - primes[idx]
            if gap < last_gap:
                continue
            
            if gap > 0:
                new_max_add = (10000 - primes[j]) // gap
            else:
                new_max_add = 0
            
            if current_len + 1 + new_max_add <= best_len:
                break
            
            seq.append(primes[j])
            if not dfs(j, gap, seq):
                return False
            seq.pop()
            
        return True

    dfs(0, 0, [primes[0]])
    
    return best_seq
