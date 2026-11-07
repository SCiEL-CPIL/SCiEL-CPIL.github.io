import math
import time

# EVOLVE-BLOCK-START

def construct():
    # Primes up to 10210
    max_val = 10210
    # Sieve of Eratosthenes
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if is_prime[i]:
            for j in range(i*i, max_val + 1, i):
                is_prime[j] = False
    
    primes = [i for i, prime in enumerate(is_prime) if prime]
    prime_index = {p: i for i, p in enumerate(primes)}
    
    # Calculate masks for all numbers
    # mask[n] is the bitmask of prime indices in the square-free part of n
    masks = [0] * (max_val + 1)
    for n in range(2, max_val + 1):
        temp = n
        mask = 0
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                if count % 2 == 1:
                    mask ^= (1 << prime_index[p])
        if temp > 1:
            # temp is a prime
            mask ^= (1 << prime_index[temp])
        masks[n] = mask
        
    target_mask = masks[210]
    
    # Basis storage
    # Each entry: (pivot_bit, vector_mask, coeff_mask)
    # pivot_bit is the index of the highest set bit in vector_mask
    # coeff_mask is a bitmask of the original numbers (211, 212, ...) used
    # We map the original number x to an index idx = x - 211
    # So coeff_mask bit idx is 1 if x is in the combination
    
    basis = {} # pivot_bit -> (vector_mask, coeff_mask)
    
    # We iterate M from 211 upwards
    for M in range(211, max_val + 1):
        v = masks[M]
        idx = M - 211
        c = 1 << idx
        
        # Reduce v by existing basis
        # We process bits from high to low
        # However, Python integers don't have a fixed size, but we can find the highest bit
        # Let's iterate through set bits of v
        temp_v = v
        temp_c = c
        while temp_v:
            # Find highest set bit
            # bit_length - 1
            pivot = temp_v.bit_length() - 1
            if pivot in basis:
                b_vec, b_coeff = basis[pivot]
                temp_v ^= b_vec
                temp_c ^= b_coeff
            else:
                # Add to basis
                basis[pivot] = (temp_v, temp_c)
                break
        
        # Check if target is in span
        # Reduce target_mask by basis
        t_v = target_mask
        t_c = 0
        # We need to reduce t_v
        # We must ensure we use the basis in a consistent way (e.g., highest bit first)
        # The basis is stored by pivot. We should process pivots in descending order of pivot bit
        # Or simply, while t_v != 0, find highest bit, if in basis, eliminate.
        # This works because the basis vectors are in row-echelon form (unique pivot for each)
        
        # Optimization: We can check if target is in span more efficiently?
        # No, reduction is O(D) where D is dimension. D is small.
        
        check_v = target_mask
        check_c = 0
        while check_v:
            pivot = check_v.bit_length() - 1
            if pivot in basis:
                b_vec, b_coeff = basis[pivot]
                check_v ^= b_vec
                check_c ^= b_coeff
            else:
                break
        
        if check_v == 0:
            # Solution found
            # check_c contains the indices of the selected numbers
            # We need to check if the number of selected numbers is <= 100
            num_selected = check_c.bit_count()
            if num_selected <= 100:
                result = []
                # Extract indices
                c_mask = check_c
                while c_mask:
                    # Get lowest set bit
                    lsb = c_mask & -c_mask
                    idx_found = lsb.bit_length() - 1
                    x_found = idx_found + 211
                    result.append(x_found)
                    c_mask ^= lsb
                return result
            else:
                # If num_selected > 100, we might need a different solution.
                # For small M, this is unlikely. If it happens, we continue searching for larger M?
                # Or we can try to find a shorter combination.
                # Given the constraints, if M is small, dim is small, weight is small.
                # If M is large, dim is large, but we can likely find a short solution.
                # For now, we assume the first solution found is valid.
                # If not, we could continue the loop, but the basis grows.
                # Let's assume it's valid.
                pass
                
    # Fallback: return a trivial solution if loop finishes (should not happen)
    # 210 * (4*210) = 210^2 * 4 = (2*210)^2. 
    # 4*210 = 840.
    # 840 is <= 10210.
    # Distinct? Yes, just one number.
    # Strictly larger than 210? Yes.
    return [840]

# EVOLVE-BLOCK-END
