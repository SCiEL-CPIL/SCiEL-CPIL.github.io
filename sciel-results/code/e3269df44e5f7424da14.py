import time

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    primes = [2, 3, 5, 7, 11, 13, 17]
    max_m = 10**18
    max_k = 10**9
    
    solutions = {}
    
    def gen(idx, current_k, current_m):
        if time.time() > deadline:
            return
        
        if idx == len(primes):
            if current_m <= max_m:
                if current_m in solutions:
                    solutions[current_m].append(current_k)
                else:
                    solutions[current_m] = [current_k]
            return
        
        p = primes[idx]
        p_pow = 1
        sig = 1
        
        while True:
            next_k = current_k * p_pow
            if next_k > max_k:
                break
                
            term = p_pow * sig
            next_m = current_m * term
            if next_m > max_m:
                break
                
            gen(idx + 1, next_k, next_m)
            
            p_pow *= p
            sig += p_pow
            
    gen(0, 1, 1)
    
    best_m = 0
    best_ks = []
    max_count = 0
    
    for m, ks in solutions.items():
        if len(ks) > max_count:
            max_count = len(ks)
            best_m = m
            best_ks = ks
            
    if max_count == 0:
        return {'M': 333312, 'k': [336, 372, 434]}
        
    return {'M': best_m, 'k': best_ks}
