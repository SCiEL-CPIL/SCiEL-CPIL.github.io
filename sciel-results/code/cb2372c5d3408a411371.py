import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55 # Safety margin

    # Primes in [2, 256]
    is_prime = [True] * 257
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(256**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, 257, i):
                is_prime[j] = False

    range_max = 128

    def search(limit):
        # Iterate A
        for A in itertools.combinations(range(1, limit + 1), 3):
            if time.time() > deadline:
                return None
            
            # Find all valid b in [1, limit] such that a+b is prime for all a in A
            valid_bs = []
            for b in range(1, limit + 1):
                # Check if all sums are prime
                ok = True
                for a in A:
                    if not is_prime[a + b]:
                        ok = False
                        break
                if ok:
                    valid_bs.append(b)
            
            if len(valid_bs) < 3:
                continue
                
            # Try all combinations of 3 from valid_bs
            for B in itertools.combinations(valid_bs, 3):
                # Check distinctness of sums
                sums = []
                for a in A:
                    for b in B:
                        sums.append(a + b)
                if len(set(sums)) == 9:
                    return (list(A), list(B))
        return None

    # Try increasing limits
    for limit in [20, 30, 40, 50, 60, 80, 100, 128]:
        if time.time() > deadline:
            break
        res = search(limit)
        if res:
            return res

    # Fallback (should not happen if solution exists in 128)
    return [1, 7, 13], [4, 10, 16]
