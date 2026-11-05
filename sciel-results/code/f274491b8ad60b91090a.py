import time
from math import factorial

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    target = factorial(12)  # 479001600
    
    # Prime factorization of 12!: 2^10 * 3^5 * 5^2 * 7 * 11
    def factorize(n):
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors
    
    target_factors = factorize(target)
    
    # Generate all divisors > 12 with their factorizations
    divisors_info = []
    i = 1
    while i * i <= target:
        if target % i == 0:
            for d in (i, target // i):
                if d > 12:
                    divisors_info.append((d, factorize(d)))
        i += 1
    
    divisors_info.sort(key=lambda x: x[0])
    divisors = [d[0] for d in divisors_info]
    div_factors = [d[1] for d in divisors_info]
    
    best_max = target
    best_set = [target]
    
    def can_divide(rem, div):
        for p, e in div.items():
            if rem.get(p, 0) < e:
                return False
        return True
    
    def sub_factors(rem, div):
        new = dict(rem)
        for p, e in div.items():
            new[p] = new.get(p, 0) - e
            if new[p] == 0:
                del new[p]
        return new
    
    def prod_from_factors(f):
        r = 1
        for p, e in f.items():
            r *= p ** e
        return r
    
    def dfs(idx, rem, chosen, chosen_set, curr_max):
        nonlocal best_max, best_set
        
        if not rem:
            if curr_max < best_max:
                best_max = curr_max
                best_set = list(chosen)
            return
        
        if len(chosen) >= 12:
            return
            
        if curr_max >= best_max:
            return
        
        if time.time() > deadline:
            return
        
        rem_prod = prod_from_factors(rem)
        
        # If remaining product is a single valid factor, finish
        if rem_prod > 12 and rem_prod not in chosen_set:
            fm = max(curr_max, rem_prod)
            if fm < best_max:
                best_max = fm
                best_set = list(chosen) + [rem_prod]
            return
        
        for i in range(idx, len(divisors)):
            d = divisors[i]
            df = div_factors[i]
            
            if d in chosen_set:
                continue
            if not can_divide(rem, df):
                continue
            
            nm = max(curr_max, d)
            if nm >= best_max:
                continue
            
            nr = sub_factors(rem, df)
            nrp = prod_from_factors(nr)
            
            # If after taking d, remaining is a single valid factor
            if nrp > 12 and nrp not in chosen_set:
                fm = max(nm, nrp)
                if fm < best_max:
                    best_max = fm
                    best_set = list(chosen) + [d, nrp]
                continue
            
            chosen.append(d)
            chosen_set.add(d)
            dfs(i + 1, nr, chosen, chosen_set, nm)
            chosen.pop()
            chosen_set.remove(d)
            
            if time.time() > deadline:
                return
    
    chosen = []
    chosen_set = set()
    dfs(0, target_factors, chosen, chosen_set, 0)
    
    return best_set
