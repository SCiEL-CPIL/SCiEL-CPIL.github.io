from math import gcd

def construct():
    # Generate candidate starting sets: divisors of m for m in 1..512
    # We know the best m has max divisors. Let's find them all and try expanding.
    
    def get_divisors(n):
        divs = []
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
        return divs

    def is_compatible(num, current_set):
        for x in current_set:
            if (num // gcd(num, x)) * x > 512:
                return False
        return True

    best_size = 0
    best_set = []
    
    # Try starting with divisors of numbers with many divisors
    # To be efficient, we can just try all m, but limit expansion time.
    # Since 512 is small, we can try a few promising m's.
    # The m's with max divisors are around 360, 420, 480, 504.
    
    # Let's just iterate m from 1 to 512, get divisors, sort by size desc.
    candidates = []
    for m in range(1, 513):
        divs = get_divisors(m)
        candidates.append((len(divs), m, divs))
        
    # Sort by length descending
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Try expanding the top few candidates
    # We can also try starting with empty set and greedy adding?
    # Greedy adding usually yields smaller sets than starting with a dense core.
    
    for _, m, divs in candidates[:20]: # Try top 20 seeds
        current_set = set(divs)
        # Try to add other numbers from 1 to 512
        # We should try adding numbers that are "small" or "smooth"?
        # Just iterate all numbers.
        for num in range(1, 513):
            if num in current_set:
                continue
            if is_compatible(num, current_set):
                current_set.add(num)
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
            
    return best_set
