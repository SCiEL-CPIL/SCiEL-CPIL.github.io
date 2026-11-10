import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin

    # Helper to count distinct prime factors of all pairwise sums
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

    def count_primes(arr):
        all_primes = set()
        for i in range(len(arr)):
            for j in range(i+1, len(arr)):
                all_primes.update(prime_factors(arr[i] + arr[j]))
        return len(all_primes)

    # Strategy: Use a small set of primes and find a clique in the smooth-sum graph
    # We try increasing sizes of prime sets, starting with 2, 3, 5
    
    # Precompute small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    best_result = None
    best_count = 100

    # Try prime sets of size 3, 4, 5...
    # For each prime set P, build graph where edge (u,v) exists if u+v is P-smooth
    # Find a clique of size 12
    
    for p_size in range(3, 6):  # Try prime sets of size 3, 4, 5
        if time.time() > deadline:
            break
            
        # Generate combinations of small primes of size p_size
        # Always include 2 (since sums of even numbers are even)
        # Actually, let's try with and without 2
        
        # For efficiency, focus on sets that include 2
        other_primes = small_primes[1:p_size+1]  # primes after 2
        
        # Try different combinations
        for combo in itertools.combinations(other_primes, p_size-1):
            P = [2] + list(combo)
            P_set = set(P)
            
            # Build smoothness check for this P
            def is_smooth(n):
                for p in P_set:
                    while n % p == 0:
                        n //= p
                return n == 1
            
            # Search for clique of size 12 in [1, 200]
            L = 200
            valid = [set() for _ in range(L+1)]
            
            for i in range(1, L+1):
                for j in range(i+1, L+1):
                    if is_smooth(i+j):
                        valid[i].add(j)
                        valid[j].add(i)
            
            # Greedy clique finding
            nodes = list(range(1, L+1))
            nodes.sort(key=lambda x: len(valid[x]), reverse=True)
            
            clique = []
            def find_clique(start, current):
                if len(current) == 12:
                    return current[:]
                if len(current) + (len(nodes) - start) < 12:
                    return None
                for k in range(start, len(nodes)):
                    node = nodes[k]
                    if all(node in valid[c] for c in current):
                        res = find_clique(k+1, current + [node])
                        if res:
                            return res
                return None
            
            clique = find_clique(0, [])
            
            if clique:
                # Scale to fit in [1, 10000]
                max_val = max(clique)
                scale = 1
                while max_val * scale * 2 <= 10000:
                    scale *= 2
                result = [x * scale for x in clique]
                
                if len(set(result)) == 12 and all(1 <= x <= 10000 for x in result):
                    count = count_primes(result)
                    if count < best_count:
                        best_count = count
                        best_result = result
                        if count <= 3:  # Can't do much better
                            return result

    # Fallback: Random local search
    if best_result is None or time.time() < deadline - 10:
        random.seed(42)
        
        # Start with multiples of 30 (2*3*5)
        base = list(range(30, 30*12, 30))
        current = base[:]
        current_count = count_primes(current)
        
        for iteration in range(1000):
            if time.time() > deadline:
                break
            
            i = random.randint(0, 11)
            old = current[i]
            new = random.randint(1, 10000)
            
            if new in current:
                continue
            
            current[i] = new
            new_count = count_primes(current)
            
            if new_count < current_count:
                current_count = new_count
                if current_count < best_count:
                    best_count = current_count
                    best_result = current[:]
                    if best_count <= 3:
                        break
            else:
                current[i] = old
        
        if best_result is None:
            best_result = current
    
    return best_result
