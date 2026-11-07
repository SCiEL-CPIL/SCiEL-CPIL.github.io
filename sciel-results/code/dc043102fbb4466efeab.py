import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50 # Safety margin

    N = 210
    MAX_VAL = 10210
    
    # 1. Sieve of Eratosthenes
    sieve = [True] * (MAX_VAL + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.isqrt(MAX_VAL)) + 1):
        if sieve[i]:
            for j in range(i*i, MAX_VAL + 1, i):
                sieve[j] = False
    primes = [i for i in range(2, MAX_VAL + 1) if sieve[i]]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # 2. Precompute squarefree kernels as bitmasks
    # kernel[x] is an int where bit i is 1 if prime[i] has odd exponent in x
    kernels = [0] * (MAX_VAL + 1)
    for x in range(2, MAX_VAL + 1):
        n = x
        mask = 0
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                exp = 0
                while n % p == 0:
                    n //= p
                    exp += 1
                if exp % 2 == 1:
                    mask |= (1 << prime_to_idx[p])
        if n > 1:
            mask |= (1 << prime_to_idx[n])
        kernels[x] = mask
        
    target = kernels[N]
    
    # 3. Incremental Gaussian Elimination
    # basis: list of tuples (mask, origin_mask)
    # origin_mask is a bitmask of indices relative to start (211 is index 0)
    basis = []
    
    for M in range(N + 1, MAX_VAL + 1):
        if time.time() > deadline:
            break
            
        x = M
        v_x = kernels[x]
        # Index of x in the set {211, ..., M}
        # 211 -> 0, 212 -> 1, ..., M -> M - 211
        idx = M - (N + 1)
        orig_mask = 1 << idx
        
        # Reduce v_x with current basis
        temp_mask = v_x
        temp_orig = orig_mask
        
        # We need to process basis in order of leading bit
        # To make reduction efficient, we should keep basis sorted or use a map
        # For simplicity, since basis size is small, we can just iterate.
        # But to be efficient, let's maintain a dictionary: leading_bit -> (mask, origin)
        # Actually, standard Gaussian elimination adds to the end and reduces against all.
        # Let's just iterate through the list.
        
        for i in range(len(basis)):
            b_mask, b_orig = basis[i]
            # Check if temp_mask has the leading bit of b_mask
            # Leading bit is the highest set bit
            lb_b = b_mask.bit_length() - 1
            if temp_mask & (1 << lb_b):
                temp_mask ^= b_mask
                temp_orig ^= b_orig
        
        if temp_mask == 0:
            # v_x is in the span of previous vectors
            # This doesn't expand the span, so we don't add to basis
            # But we might have found a new representation?
            # No, we just need to know if TARGET is in the span.
            # The span hasn't changed.
            pass
        else:
            # Add to basis
            basis.append((temp_mask, temp_orig))
            # Note: For efficient future reductions, we might want to keep basis in RREF.
            # But since we check target at every step, and M is small, simple list is okay.
            # However, if we don't keep it reduced, the "leading bit" check might not work perfectly if basis vectors aren't reduced against each other.
            # Let's maintain proper RREF.
            # When adding new vector, we should reduce it against existing basis,
            # AND reduce existing basis against the new vector? No, that's expensive.
            # Standard approach:
            # 1. Reduce new vector against existing basis.
            # 2. If non-zero, add to basis.
            # 3. To make future reductions fast, we can sort by leading bit.
            # Let's just sort the basis by leading bit after adding?
            # Or better, maintain a dict keyed by leading bit.
            
            # Let's restart the basis management to be robust:
            # Use a dictionary `basis_dict` mapping leading_bit -> (mask, origin)
            # This ensures we always check the vector with the highest possible leading bit first.
            
    # Wait, the loop above modifies basis in place. 
    # Let's refine the loop to use a dict for O(1) leading bit lookup.
    
    # Re-implementing the core loop properly:
    
    basis_dict = {} # leading_bit -> (mask, origin_mask)
    
    for M in range(N + 1, MAX_VAL + 1):
        if time.time() > deadline:
            break
            
        x = M
        v_x = kernels[x]
        idx = M - (N + 1)
        orig_mask = 1 << idx
        
        temp_mask = v_x
        temp_orig = orig_mask
        
        # Reduce temp_mask using basis_dict
        # We need to eliminate bits from highest to lowest
        # Get current highest bit
        while temp_mask:
            lb = temp_mask.bit_length() - 1
            if lb in basis_dict:
                b_mask, b_orig = basis_dict[lb]
                temp_mask ^= b_mask
                temp_orig ^= b_orig
            else:
                # This bit cannot be eliminated
                # So this is the new leading bit for this vector
                break
        
        if temp_mask == 0:
            # Dependent, no change to span
            pass
        else:
            # New independent vector
            # Its leading bit is lb (from the loop above)
            # Wait, the loop breaks when it finds a bit NOT in basis_dict.
            # That bit is the leading bit of the reduced vector.
            basis_dict[lb] = (temp_mask, temp_orig)
            
        # Check if target is in the span of current basis
        # Reduce target using basis_dict
        t_mask = target
        t_orig = 0
        while t_mask:
            lb = t_mask.bit_length() - 1
            if lb in basis_dict:
                b_mask, b_orig = basis_dict[lb]
                t_mask ^= b_mask
                t_orig ^= b_orig
            else:
                # Target has a bit that cannot be eliminated
                # So it's not in the span
                t_mask = -1 # Mark as failure
                break
        
        if t_mask == 0:
            # Found solution!
            # t_orig contains the indices
            selected_indices = []
            # t_orig is a bitmask
            # We need to convert it to a list of x values
            # Index i corresponds to x = 211 + i
            # But t_orig might have bits set for indices > M-211?
            # No, we only added bits up to M-211.
            # However, the basis vectors might have originated from earlier indices.
            # The origin masks are cumulative.
            
            # Extract indices
            # t_orig is an int.
            # Iterate through bits
            # Max index is M - 211
            for i in range(M - N):
                if (t_orig >> i) & 1:
                    selected_indices.append(N + 1 + i)
            
            # Verify
            # Check distinctness (guaranteed by bitmask)
            # Check range (guaranteed by construction)
            # Check size
            if 1 <= len(selected_indices) <= 100:
                # Verify perfect square
                prod = N
                for s in selected_indices:
                    prod *= s
                if _is_perfect_square(prod):
                    return [int(x) for x in selected_indices]
            else:
                # If size > 100, we might need a different solution?
                # But for small M, size is small.
                # If this happens, we might need to search for a smaller weight solution.
                # But let's assume the first found is valid.
                pass

    # Fallback
    return [4 * N]

def _is_perfect_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n
