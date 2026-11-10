import time

def construct():
    best_seq = []
    best_len = 0
    max_val = 128
    start_time = time.time()
    time_limit = 50  # Leave 10s buffer
    
    # Initial seed: all primes in [2, 128]
    # This guarantees a valid sequence of length 31
    primes = [n for n in range(2, max_val + 1) if all(n % i != 0 for i in range(2, int(n**0.5) + 1))]
    best_seq = primes
    best_len = len(primes)
    
    def get_new_products(seq, new_val):
        new_prods = []
        prod = 1
        for i in range(len(seq), -1, -1):
            if i == 0:
                new_prods.append(new_val)
            else:
                prod *= seq[-i]
                new_prods.append(prod * new_val)
        return new_prods

    def dfs(start_val, current_seq, current_products):
        nonlocal best_seq, best_len
        
        if time.time() - start_time > time_limit:
            return

        if len(current_seq) > best_len:
            best_len = len(current_seq)
            best_seq = list(current_seq)
            
        if len(current_seq) + (max_val - start_val + 1) <= best_len:
            return
            
        for next_val in range(start_val, max_val + 1):
            if time.time() - start_time > time_limit:
                return

            new_prods = get_new_products(current_seq, next_val)
            
            conflict = False
            for p in new_prods:
                if p in current_products:
                    conflict = True
                    break
            
            if conflict:
                continue
                
            new_products = current_products.union(new_prods)
            new_seq = current_seq + [next_val]
            
            dfs(next_val + 1, new_seq, new_products)

    dfs(2, [], set())
    return best_seq
