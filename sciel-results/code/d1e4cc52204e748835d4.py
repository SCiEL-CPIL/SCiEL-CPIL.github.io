import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 80  # seconds, safety margin before 90s kill
    
    N = 210
    MAX_VAL = 10210
    MAX_SIZE = 100
    
    # Sieve to get primes up to MAX_VAL
    def sieve(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    primes = sieve(MAX_VAL)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # Target: parity vector of 210
    target = 0
    temp = N
    for i, p in enumerate(primes):
        if temp % p == 0:
            cnt = 0
            while temp % p == 0:
                temp //= p
                cnt += 1
            if cnt % 2 == 1:
                target |= (1 << i)
    # temp should be 1 now since 210 = 2*3*5*7
    
    # Incremental Gaussian elimination over GF(2)
    # basis: list where basis[i] has leading bit at position i
    # We'll maintain a reduced row echelon form
    basis = [0] * num_primes  # basis[i] = vector with leading bit i, or 0 if no such vector
    basis_count = 0
    
    # For solution extraction, we need to track which original vectors contributed
    # Use a parallel structure: for each basis vector, store the set of original indices
    # But sets are expensive. Instead, store as bitset of original indices.
    # Max candidates we'll process: maybe up to a few thousand. Use Python int as bitset.
    
    basis_orig = [0] * num_primes  # bitset of original indices in this basis vector
    
    # Store all vectors and their original indices
    all_vectors = []
    
    def compute_parity(n):
        """Compute parity vector of n as a Python int bitset."""
        vec = 0
        temp = n
        for i, p in enumerate(primes):
            if p * p > temp:
                break
            if temp % p == 0:
                cnt = 0
                while temp % p == 0:
                    temp //= p
                    cnt += 1
                if cnt % 2 == 1:
                    vec |= (1 << i)
        if temp > 1:
            # temp is a prime factor
            idx = prime_to_idx.get(temp)
            if idx is not None:
                vec |= (1 << idx)
        return vec
    
    def try_solve(target_vec, basis, basis_orig, all_vectors, num_primes):
        """Try to express target_vec as XOR of basis vectors. Return list of original indices or None."""
        remaining = target_vec
        solution_bits = 0
        for i in range(num_primes - 1, -1, -1):
            if (remaining >> i) & 1:
                if basis[i] == 0:
                    return None  # not in span
                remaining ^= basis[i]
                solution_bits ^= basis_orig[i]
        if remaining != 0:
            return None
        # Extract indices from solution_bits
        indices = []
        b = solution_bits
        while b:
            low = b & (-b)
            idx = low.bit_length() - 1
            indices.append(idx)
            b ^= low
        return indices
    
    best_solution = None
    best_max = MAX_VAL + 1
    
    for n in range(N + 1, MAX_VAL + 1):
        if time.time() - start_time > DEADLINE:
            break
        
        vec = compute_parity(n)
        orig_idx = len(all_vectors)
        all_vectors.append((n, vec))
        
        # Insert into basis
        v = vec
        orig_bit = (1 << orig_idx)
        for i in range(num_primes - 1, -1, -1):
            if not ((v >> i) & 1):
                continue
            if basis[i] != 0:
                v ^= basis[i]
                orig_bit ^= basis_orig[i]
            else:
                basis[i] = v
                basis_orig[i] = orig_bit
                basis_count += 1
                break
        
        # Check if target is in span
        if basis_count >= 1:  # at least one basis vector
            sol = try_solve(target, basis, basis_orig, all_vectors, num_primes)
            if sol is not None:
                # Check size constraint
                if len(sol) <= MAX_SIZE:
                    # Get the actual numbers
                    selected = [all_vectors[i][0] for i in sol]
                    cur_max = max(selected)
                    if cur_max < best_max:
                        best_max = cur_max
                        best_solution = selected
                        # We found a solution; we can potentially stop early
                        # but let's continue a bit more to see if we can do better
                        # Actually, since we're going in order of n, the first solution
                        # found will have the smallest possible max value
                        # Wait, no - we're adding vectors one at a time, and checking
                        # if target is in span. The first n where it becomes in span
                        # gives us the minimal window [211, n]. But the solution might
                        # not include n. The max of the solution could be less than n.
                        # Actually, the solution is a subset of {211, ..., n}, so max <= n.
                        # And for any n' < n, the target was not in span of {211,...,n'},
                        # so no solution exists with all elements <= n'.
                        # Therefore, the first n where a solution exists gives the minimal
                        # possible max value.
                        break
                else:
                    # Solution exists but too large. Need to find a smaller solution.
                    # This is harder. For now, continue searching.
                    # Actually, if the solution is too large, we might need to go further
                    # to find a different combination. But this is complex.
                    # Let's just note it and continue.
                    pass
    
    if best_solution is not None:
        return list(best_solution)
    
    # Fallback: return a trivial solution
    # 4*210 = 840, and 840*210 = 210^2 * 4 = (2*210)^2 which is a perfect square
    return [4 * N]
