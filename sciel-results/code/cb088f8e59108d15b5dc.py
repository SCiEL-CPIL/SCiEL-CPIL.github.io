"""Improved program for large integer sets with no two-other-product divisibility."""
# EVOLVE-BLOCK-START
import time
import random

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _is_valid_set(S_list):
    n = len(S_list)
    if n < 3:
        return True
    for i in range(n):
        a = S_list[i]
        for j in range(n):
            if j == i:
                continue
            b = S_list[j]
            for k in range(n):
                if k == i or k == j:
                    continue
                c = S_list[k]
                if (b * c) % a == 0:
                    return False
    return True

def _can_add(S_list, x):
    """Check if adding x to S_list keeps the set valid.
    Only need to check triples involving x."""
    n = len(S_list)
    # Triple (x, b, c): x | b*c ?
    for i in range(n):
        b = S_list[i]
        for j in range(i + 1, n):
            c = S_list[j]
            if (b * c) % x == 0:
                return False
    # Triple (b, x, c): b | x*c ?
    # Triple (b, c, x): c | b*x ?
    for i in range(n):
        b = S_list[i]
        for j in range(i + 1, n):
            c = S_list[j]
            if (x * c) % b == 0:
                return False
            if (b * x) % c == 0:
                return False
    return True

def construct():
    deadline = time.time() + 80  # Safety margin for 90s limit
    n_max = 256
    
    # Generate all primes
    primes = [x for x in range(2, n_max + 1) if _is_prime(x)]
    
    # Start with all primes
    S = primes[:]
    S.sort()
    best = S[:]
    best_size = len(S)
    
    # Generate composites
    composites = [x for x in range(2, n_max + 1) if not _is_prime(x)]
    
    # Try multiple greedy orders
    orders = [
        sorted(composites, key=lambda x: x),
        sorted(composites, key=lambda x: -x),
        sorted(composites, key=lambda x: (len([p for p in primes if x % p == 0]), x)),  # fewer prime factors first
        sorted(composites, key=lambda x: (-len([p for p in primes if x % p == 0]), x)),
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        current_S = best[:]
        for x in order:
            if time.time() > deadline:
                break
            if x in current_S:
                continue
            if _can_add(current_S, x):
                current_S.append(x)
        
        if len(current_S) > best_size:
            best = current_S[:]
            best_size = len(S)
            best.sort()
    
    # Local search: try to swap elements to enable more additions
    all_nums = set(range(2, n_max + 1))
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_set = set(best)
        for x in list(current_set):
            # Try removing x
            new_S = [v for v in best if v != x]
            # Try adding each composite not in new_S
            for y in sorted(all_nums - set(new_S)):
                if time.time() > deadline:
                    break
                if _can_add(new_S, y):
                    candidate = new_S + [y]
                    # Try to add more
                    for z in sorted(all_nums - set(candidate)):
                        if time.time() > deadline:
                            break
                        if _can_add(candidate, z):
                            candidate.append(z)
                    if len(candidate) > best_size:
                        best = candidate
                        best_size = len(candidate)
                        current_set = set(best)
                        improved = True
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Final verification
    if not _is_valid_set(best):
        return [int(x) for x in primes]
    
    return [int(x) for x in sorted(best)]

# EVOLVE-BLOCK-END
