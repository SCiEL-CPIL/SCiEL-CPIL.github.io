def construct():
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    LIMIT = 10**18
    
    best_n = 2
    best_divs = 1
    
    def search(idx, max_exp, current_n, current_divs):
        nonlocal best_n, best_divs
        
        if current_divs > best_divs:
            best_divs = current_divs
            best_n = current_n
            
        if idx >= len(primes):
            return
            
        for e in range(max_exp, 0, -1):
            next_n = current_n * (primes[idx] ** e)
            if next_n > LIMIT:
                continue
            search(idx + 1, e, next_n, current_divs * (e + 1))
            
    search(0, 60, 1, 1)
    
    return best_n
