import time
from math import factorial

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin
    
    target = factorial(12)
    
    # Precompute divisors for all divisors of target
    # First, get all divisors of target
    all_divs = []
    for i in range(1, int(target**0.5) + 1):
        if target % i == 0:
            all_divs.append(i)
            if i != target // i:
                all_divs.append(target // i)
    all_divs = set(all_divs)
    
    # For each divisor, precompute its divisors > 12
    # This might be slow if done naively, but 792 divisors is small.
    # Actually, we can just generate divisors on the fly in DFS, 
    # but precomputing is safer for speed.
    
    divisors_cache = {}
    for d in all_divs:
        divs_of_d = []
        for j in range(1, int(d**0.5) + 1):
            if d % j == 0:
                if j > 12:
                    divs_of_d.append(j)
                other = d // j
                if other != j and other > 12:
                    divs_of_d.append(other)
        divisors_cache[d] = sorted(divs_of_d)
        
    best_max = target
    best_path = [target]
    
    def dfs(rem, last, count, path):
        nonlocal best_max, best_path, start_time
        
        if time.time() - start_time > deadline:
            return
            
        if rem == 1:
            if 1 <= count <= 12:
                current_max = path[-1]
                if current_max < best_max:
                    best_max = current_max
                    best_path = list(path)
            return
            
        if count == 12:
            # Must use rem as the last factor
            if rem > last and rem < best_max and rem > 12:
                # Check if rem is a valid factor (it is, by construction of rem being a divisor)
                # And distinct (rem > last)
                new_path = path + [rem]
                if rem < best_max:
                    best_max = rem
                    best_path = new_path
            return
            
        # Get candidate divisors
        candidates = divisors_cache.get(rem, [])
        
        # Filter and sort
        # We want d > last and d < best_max
        # Also, we can include rem itself if it's a valid candidate
        # Note: rem is in divisors_cache[rem] if rem > 12
        
        valid_candidates = []
        for d in candidates:
            if d > last and d < best_max:
                valid_candidates.append(d)
                
        # Sort ascending to try smaller factors first (allows more factors, potentially smaller max)
        valid_candidates.sort()
        
        for d in valid_candidates:
            # Pruning: 
            # If we pick d, the next factors must be > d.
            # The minimum product of the remaining (count + 1) to 12 factors 
            # is not a simple bound because count is variable.
            # But if rem // d < d + 1, we can't pick any more factors > d.
            # So if rem // d != 1, we need rem // d to have a divisor > d.
            # This is handled by the recursive call returning early if no candidates.
            
            dfs(rem // d, d, count + 1, path + [d])
            
            if time.time() - start_time > deadline:
                return

    dfs(target, 12, 0, [])
    
    return best_path
