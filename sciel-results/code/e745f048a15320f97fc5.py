"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _subset_products_collide(existing_products, new_num):
    """Check if adding new_num to the set creates any subset product collision.
    existing_products is the set of all subset products of the current set.
    If no collision, return the updated set of subset products; else return None."""
    new_products = set()
    for p in existing_products:
        new_products.add(p * new_num)
    # Check for collisions: any product in new_products that's also in existing_products
    # (excluding the case where it's just the new_num itself times 1, which is fine)
    # Actually, we need: the full set of subset products after adding new_num must be
    # existing_products ∪ {x * new_num : x in existing_products}
    # and all elements in this union must be distinct.
    collision = False
    for p in new_products:
        if p in existing_products:
            # This product was already achievable without new_num, so collision
            collision = True
            break
    if collision:
        return None
    # Also check within new_products for duplicates (shouldn't happen since existing_products
    # has no duplicates and multiplication by fixed new_num is injective)
    full_set = existing_products | new_products
    return full_set

def construct():
    deadline = time.time() + 55  # safety margin
    
    # All candidate integers 2..40
    candidates = list(range(2, 41))
    
    # Try greedy with different orderings
    best_result = []
    
    # Order 1: primes first, then prime powers, then composites
    def _is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    primes = [x for x in candidates if _is_prime(x)]
    prime_powers = []
    composites = []
    for x in candidates:
        if _is_prime(x):
            continue
        # Check if x is a prime power
        is_pp = False
        for p in primes:
            t = x
            while t % p == 0:
                t //= p
            if t == 1:
                is_pp = True
                break
        if is_pp:
            prime_powers.append(x)
        else:
            composites.append(x)
    
    order1 = primes + prime_powers + composites
    
    # Order 2: by number of prime factors (fewer first)
    def _num_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    
    order2 = sorted(candidates, key=lambda x: (_num_prime_factors(x), x))
    
    # Order 3: descending (larger numbers first)
    order3 = sorted(candidates, reverse=True)
    
    # Order 4: ascending
    order4 = sorted(candidates)
    
    for order in [order1, order2, order3, order4]:
        if time.time() > deadline:
            break
        current_set = {1}  # empty product
        selected = []
        for num in order:
            if time.time() > deadline:
                break
            result = _subset_products_collide(current_set, num)
            if result is not None:
                current_set = result
                selected.append(num)
        if len(selected) > len(best_result):
            best_result = selected[:]
    
    # Try local search: try to swap out elements to get more in
    if time.time() < deadline - 5:
        # Try removing one element and trying to add two others
        for i in range(len(best_result)):
            if time.time() > deadline:
                break
            # Remove best_result[i]
            reduced = best_result[:i] + best_result[i+1:]
            # Rebuild subset products
            current_set = {1}
            for num in reduced:
                current_set = _subset_products_collide(current_set, num)
            # Try adding each candidate not in reduced
            added = False
            for num in candidates:
                if num in reduced:
                    continue
                if time.time() > deadline:
                    break
                result = _subset_products_collide(current_set, num)
                if result is not None:
                    new_selected = reduced + [num]
                    if len(new_selected) > len(best_result):
                        best_result = new_selected
                        added = True
                        break
    
    return sorted(best_result)
# EVOLVE-BLOCK-END
