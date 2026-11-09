import math

def count_distinct_prime_factors(n):
    count = 0
    d = 2
    while d * d <= n:
        if n % d == 0:
            count += 1
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        count += 1
    return count

def construct():
    limit = 40
    candidates = list(range(2, limit + 1))
    
    # Sort by number of distinct prime factors, then by value
    candidates.sort(key=lambda x: (count_distinct_prime_factors(x), x))
    
    selected = []
    products = {1}
    
    for x in candidates:
        new_products = {p * x for p in products}
        if not new_products.intersection(products):
            selected.append(x)
            products.update(new_products)
            
    return selected
