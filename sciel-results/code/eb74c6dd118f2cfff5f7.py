import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s limit
    
    N = 256
    candidates = list(range(2, N + 1))
    
    def is_prime(n):
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
    
    primes = [x for x in candidates if is_prime(x)]
    composites = [x for x in candidates if not is_prime(x)]
    
    def check_valid(x, S_list):
        """Check if x can be added to S_list without violating constraints."""
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                if (a * b) % x == 0 or (x * b) % a == 0 or (x * a) % b == 0:
                    return False
        return True
    
    # Start with all primes (they form a valid set since for distinct primes p,q,r: p ∤ q*r)
    S = set(primes)
    S_list = sorted(primes)
    
    # Try to add composites in descending order (larger composites are less likely to divide products)
    for x in sorted(composites, reverse=True):
        if time.time() > deadline - 10:
            break
        if check_valid(x, S_list):
            S.add(x)
            S_list.append(x)
            S_list.sort()
    
    best_S = set(S)
    best_size = len(S)
    
    # Local search: try to add elements, then do 2-for-1 swaps
    for _ in range(10):
        if time.time() > deadline - 5:
            break
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            S_list = sorted(S)
            not_in_S = sorted(set(candidates) - S, reverse=True)
            
            # Try adding any compatible element
            for x in not_in_S:
                if time.time() > deadline - 3:
                    break
                if check_valid(x, S_list):
                    S.add(x)
                    S_list = sorted(S)
                    best_size = len(S)
                    improved = True
                    break
            
            # Try 2-for-1 swaps: remove x, add y and z
            if not improved:
                for x in S_list:
                    if time.time() > deadline - 3:
                        break
                    S_without_x = [y for y in S_list if y != x]
                    for y in not_in_S:
                        if time.time() > deadline - 3:
                            break
                        if y in S:
                            continue
                        if not check_valid(y, S_without_x):
                            continue
                        S_with_y = S_without_x + [y]
                        for z in not_in_S:
                            if time.time() > deadline - 3:
                                break
                            if z in S or z == y:
                                continue
                            if check_valid(z, S_with_y):
                                S = set(S_with_y) | {z}
                                best_size = len(S)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
    
    result = sorted(S)
    return [int(x) for x in result]
