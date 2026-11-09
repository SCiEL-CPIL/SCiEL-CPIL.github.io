import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s margin
    
    # Primes up to 250
    primes = []
    for i in range(2, 250):
        is_p = True
        for p in primes:
            if p * p > i:
                break
            if i % p == 0:
                is_p = False
                break
        if is_p:
            primes.append(i)
    
    prime_to_bit = {p: 1 << i for i, p in enumerate(primes)}
    
    def get_vector(n):
        v = 0
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                if count % 2 == 1:
                    v ^= prime_to_bit[p]
        if temp > 1:
            v ^= prime_to_bit[temp]
        return v

    target_vec = get_vector(210)
    
    best_result = [224, 240]
    
    for M in range(213, 240):
        if time.time() > deadline:
            break
        candidates = list(range(211, M + 1))
        vectors = [get_vector(x) for x in candidates]
        
        vec_to_indices = {}
        for i, v in enumerate(vectors):
            if v not in vec_to_indices:
                vec_to_indices[v] = []
            vec_to_indices[v].append(i)
            
        n_cand = len(candidates)
        for i in range(n_cand):
            for j in range(i + 1, n_cand):
                needed_vec = vectors[i] ^ vectors[j] ^ target_vec
                if needed_vec in vec_to_indices:
                    for z_idx in vec_to_indices[needed_vec]:
                        if z_idx != i and z_idx != j:
                            x = candidates[i]
                            y = candidates[j]
                            z = candidates[z_idx]
                            return [x, y, z]
                            
    return best_result
