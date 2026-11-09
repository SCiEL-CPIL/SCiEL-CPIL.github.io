"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    N = 256
    
    def prime_factors(n):
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors
    
    pfs = {}
    for i in range(2, N + 1):
        pfs[i] = prime_factors(i)
    
    def is_compatible(num, sel):
        for s in sel:
            if s == 1:
                continue
            if pfs[num] & pfs[s]:
                return False
        return True
    
    # Greedy: descending order
    selected = {1}
    for num in range(N, 1, -1):
        if is_compatible(num, selected):
            selected.add(num)
    
    current_sum = sum(selected)
    
    # Local search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding
        for num in range(N, 1, -1):
            if num not in selected and is_compatible(num, selected):
                selected.add(num)
                current_sum += num
                improved = True
                break
        
        if time.time() >= deadline:
            break
        
        # Try 1-swap
        for remove in sorted(selected - {1}):
            if time.time() >= deadline:
                break
            sel2 = selected - {remove}
            for add in range(N, 1, -1):
                if add not in sel2 and is_compatible(add, sel2):
                    if current_sum - remove + add > current_sum:
                        selected = sel2 | {add}
                        current_sum += add - remove
                        improved = True
                        break
                if time.time() >= deadline:
                    break
    
    return sorted(selected)
# EVOLVE-BLOCK-END

