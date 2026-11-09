import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    max_val = 10210
    
    # Generate primes up to max_val
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(max_val)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    def get_vector(n):
        v = 0
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                if count % 2 == 1:
                    v |= (1 << prime_to_idx[p])
        if temp > 1:
            if temp in prime_to_idx:
                v |= (1 << prime_to_idx[temp])
            # else temp is prime > max_val? No, temp <= n <= max_val
        return v

    target_vec = get_vector(210)
    
    basis = []
    combos = []
    
    # We iterate M from 211 upwards
    # Index k corresponds to value 211 + k
    for k in range(1, max_val - 210):
        if time.time() > deadline:
            break
            
        val = 211 + k
        v = get_vector(val)
        
        # Reduce v by current basis
        v_curr = v
        c_curr = 1 << k
        
        for j in range(len(basis)):
            # Find highest set bit
            # Python int doesn't have bit_length? It does.
            # But we need to match the pivot.
            # Let's store the pivot index for each basis vector for speed.
            # Actually, just checking the highest bit is fine for small dimensions.
            # To be safe and fast, let's store the highest bit index.
            
            # We need the highest set bit of v_curr and basis[j].
            # If they are the same, we XOR.
            
            # Optimization: Precompute highest bit?
            # For now, simple check.
            
            # Get highest bit index
            # If v_curr is 0, break.
            if v_curr == 0:
                break
            
            # Find highest set bit
            # v_curr.bit_length() - 1
            hb_curr = v_curr.bit_length() - 1
            hb_basis = basis[j].bit_length() - 1
            
            if hb_curr == hb_basis:
                v_curr ^= basis[j]
                c_curr ^= combos[j]
        
        if v_curr != 0:
            basis.append(v_curr)
            combos.append(c_curr)
            
        # Check if target is in span
        t_curr = target_vec
        c_target = 0
        
        for j in range(len(basis)):
            if t_curr == 0:
                break
            hb_t = t_curr.bit_length() - 1
            hb_basis = basis[j].bit_length() - 1
            if hb_t == hb_basis:
                t_curr ^= basis[j]
                c_target ^= combos[j]
                
        if t_curr == 0:
            # Solution found
            # Extract indices from c_target
            indices = []
            mask = c_target
            idx = 0
            while mask:
                if mask & 1:
                    indices.append(idx)
                mask >>= 1
                idx += 1
            
            # Convert indices to values
            # Index i corresponds to value 211 + i
            # Wait, k is the index in the loop.
            # The bit position in c_target corresponds to the loop index k.
            # So if bit k is set, the value is 211 + k.
            
            result = []
            for i in indices:
                result.append(211 + i)
                
            # Verify
            prod = 210
            for x in result:
                prod *= x
            r = math.isqrt(prod)
            if r * r == prod:
                # Check distinctness and range
                if len(set(result)) == len(result) and all(210 < x <= max_val for x in result):
                    if len(result) <= 100:
                        return result
            # If invalid, continue? 
            # Theoretically, the linear algebra solution should be valid.
            # If it fails verification, there might be a bug or the 100 limit is hit.
            # If 100 limit is hit, we might need a different solution.
            # But for now, let's assume it's fine.
            
    return [840] # Fallback
