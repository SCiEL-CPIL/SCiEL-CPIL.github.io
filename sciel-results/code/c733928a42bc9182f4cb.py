import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline with margin
    
    best_set = [0, 1, 3]
    best_score = -1
    
    def compute_score(U):
        U = sorted(set(U))
        if 0 not in U:
            U = [0] + U
        n = len(U)
        max_u = max(U)
        
        # Compute U+U
        sum_set = set()
        for i in range(n):
            for j in range(i, n):
                sum_set.add(U[i] + U[j])
        # Note: U+U includes all pairs, but since we're dealing with a set,
        # U+U = {a+b : a in U, b in U}
        # We only need to compute a+b for all a,b in U
        sum_set = set()
        for a in U:
            for b in U:
                sum_set.add(a + b)
        
        # Compute U-U
        diff_set = set()
        for a in U:
            for b in U:
                diff_set.add(a - b)
        
        if max_u == 0:
            return 1.0
        
        ratio = len(diff_set) / len(sum_set)
        if ratio <= 0:
            return -float('inf')
        
        denom = math.log(2 * max_u + 1)
        if denom == 0:
            return 1.0
        
        return 1.0 + math.log(ratio) / denom
    
    best_score = compute_score(best_set)
    
    # Try various constructions
    
    # 1. Powers of 2
    for n in range(2, 25):
        if time.time() > deadline - 5:
            break
        U = [0] + [2**i for i in range(n)]
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_set = U
    
    # 2. Powers of 3
    for n in range(2, 20):
        if time.time() > deadline - 5:
            break
        U = [0] + [3**i for i in range(n)]
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_set = U
    
    # 3. Mixed: small base then large gaps
    for base in [2, 3, 4, 5]:
        for n in range(2, 15):
            if time.time() > deadline - 5:
                break
            U = [0]
            val = 1
            for i in range(n):
                U.append(val)
                val *= base
            s = compute_score(U)
            if s > best_score:
                best_score = s
                best_set = U
    
    # 4. Arithmetic progression with a twist
    for n in range(2, 30):
        if time.time() > deadline - 5:
            break
        for d in [2, 3, 5]:
            U = [0] + [i * d for i in range(1, n)]
            s = compute_score(U)
            if s > best_score:
                best_score = s
                best_set = U
    
    # 5. Quadratic-like: i^2
    for n in range(2, 50):
        if time.time() > deadline - 5:
            break
        U = [0] + [i**2 for i in range(1, n)]
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_set = U
    
    # 6. i^3
    for n in range(2, 30):
        if time.time() > deadline - 5:
            break
        U = [0] + [i**3 for i in range(1, n)]
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_set = U
    
    # 7. Exponential with varying base
    for base in [2, 3, 4, 5, 7, 10]:
        for n in range(2, 12):
            if time.time() > deadline - 5:
                break
            U = [0]
            val = 1
            for i in range(n):
                U.append(val)
                val = val * base + 1  # a_i = base^i * (base-1)/(base-1) ... let's just do geometric
            # Actually let's do proper geometric
            U = [0] + [base**i for i in range(n)]
            s = compute_score(U)
            if s > best_score:
                best_score = s
                best_set = U
    
    # 8. Try sets with a dense beginning and sparse end
    for n_dense in range(2, 10):
        for n_sparse in range(1, 10):
            if time.time() > deadline - 5:
                break
            U = list(range(n_dense))  # 0, 1, ..., n_dense-1
            # Add sparse elements
            base_val = n_dense * 10
            for j in range(n_sparse):
                U.append(base_val * (j + 1))
            s = compute_score(U)
            if s > best_score:
                best_score = s
                best_set = U
    
    # 9. Try based on prime numbers
    def is_prime(x):
        if x < 2:
            return False
        if x < 4:
            return True
        if x % 2 == 0 or x % 3 == 0:
            return False
        i = 5
        while i * i <= x:
            if x % i == 0 or x % (i + 2) == 0:
                return False
            i += 6
        return True
    
    for n in range(2, 50):
        if time.time() > deadline - 5:
            break
        primes = []
        x = 2
        while len(primes) < n:
            if is_prime(x):
                primes.append(x)
            x += 1
        U = [0] + primes
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_set = U
    
    # 10. Try Fibonacci-like
    for n in range(2, 20):
        if time.time() > deadline - 5:
            break
        fibs = [0, 1]
        while len(fibs) < n:
            fibs.append(fibs[-1] + fibs[-2])
        U = fibs[:n]
        if 0 not in U:
            U = [0] + U
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_set = U
    
    # 11. Try sets where elements are multiples of a large number plus small offsets
    for large_mult in [10, 20, 50, 100]:
        for n in range(2, 10):
            if time.time() > deadline - 5:
                break
                for offset in range(5):
                    U = [0]
                    for i in range(1, n):
                        U.append(i * large_mult + offset)
                    s = compute_score(U)
                    if s > best_score:
                        best_score = s
                        best_set = U
    
    # 12. Try sets with quadratic growth: i*(i+1)//2
    for n in range(2, 50):
        if time.time() > deadline - 5:
            break
        U = [0] + [i*(i+1)//2 for i in range(1, n)]
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_set = U
    
    # 13. Try sets based on powers with a shift
    for base in [2, 3]:
        for n in range(2, 15):
            if time.time() > deadline - 5:
                break
            U = [0] + [base**i - 1 for i in range(1, n)]
            s = compute_score(U)
            if s > best_score:
                best_score = s
                best_set = U
    
    # 14. Try combining small AP with large geometric
    for n_ap in range(2, 8):
        for n_geo in range(1, 8):
            if time.time() > deadline - 5:
                break
            U = list(range(n_ap))
            start_val = n_ap * 5
            for j in range(n_geo):
                U.append(start_val * (2 ** j))
            s = compute_score(U)
            if s > best_score:
                best_score = s
                best_set = U
    
    # 15. Try sets with rapidly growing gaps
    for n in range(2, 12):
        if time.time() > deadline - 5:
            break
        U = [0]
        val = 1
        for i in range(n):
            U.append(val)
            val = val * 2 + i  # super-exponential
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_set = U
    
    # 16. Try sets where U is a subgroup-like structure
    # For a prime p, take elements that are powers of a primitive root
    for p in [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
        if time.time() > deadline - 5:
            break
        # Find primitive root
        def primitive_root(p):
            for g in range(2, p):
                order = p - 1
                factors = set()
                temp = order
                d = 2
                while d * d <= temp:
                    if temp % d == 0:
                        factors.add(d)
                        while temp % d == 0:
                            temp //= d
                    d += 1
                if temp > 1:
                    factors.add(temp)
                ok = True
                for f in factors:
                    if pow(g, order // f, p) == 1:
                        ok = False
                        break
                if ok:
                    return g
            return None
        
        g = primitive_root(p)
        if g is None:
            continue
        # Take all powers
        U = set()
        val = 1
        for i in range(p - 1):
            U.add(val)
            val = (val * g) % p
        U = sorted(list(U))
        if 0 not in U:
            U = [0] + U
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_set = U
    
    # 17. Try larger primes for multiplicative subgroups
    for p in [53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        if time.time() > deadline - 5:
            break
        def primitive_root(p):
            for g in range(2, p):
                order = p - 1
                factors = set()
                temp = order
                d = 2
                while d * d <= temp:
                    if temp % d == 0:
                        factors.add(d)
                        while temp % d == 0:
                            temp //= d
                    d += 1
                if temp > 1:
                    factors.add(temp)
                ok = True
                for f in factors:
                    if pow(g, order // f, p) == 1:
                        ok = False
                        break
                if ok:
                    return g
            return None
        
        g = primitive_root(p)
        if g is None:
            continue
        # Take a subgroup of size d where d divides p-1
        for d in range(2, p):
            if (p - 1) % d != 0:
                continue
            if time.time() > deadline - 5:
                break
            step = (p - 1) // d
            U = set()
            val = 1
            for i in range(d):
                U.add(val)
                val = (val * pow(g, step, p)) % p
            U = sorted(list(U))
            if 0 not in U:
                U = [0] + U
            s = compute_score(U)
            if s > best_score:
                best_score = s
                best_set = U
    
    # 18. Try sets with a specific structure: {0} union {k * m : k in K} where K is a sum-free set
    # Sum-free sets in [1, n]
    for n in range(10, 100):
        if time.time() > deadline - 5:
            break
        # Largest sum-free set in [1, n] is {floor(n/2)+1, ..., n}
        U = [0] + list(range(n // 2 + 1, n + 1))
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_set = U
    
    # 19. Try odd numbers
    for n in range(2, 100):
        if time.time() > deadline - 5:
            break
        U = [0] + [2 * i + 1 for i in range(n)]
        s = compute_score(U)
        if s > best_score:
            best_score = s
            best_set = U
    
    # 20. Try sets with a gap structure
    for gap in [2, 3, 5, 10]:
        for n in range(2, 30):
            if time.time() > deadline - 5:
                break
            U = [0]
            val = 1
            for i in range(n):
                U.append(val)
                val += gap
            s = compute_score(U)
            if s > best_score:
                best_score = s
                best_set = U
    
    return best_set
