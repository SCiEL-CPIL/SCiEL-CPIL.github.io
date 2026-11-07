import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin for 90s limit
    
    # 1. Sieve of Eratosthenes for primes up to 10210
    limit = 10210
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, val in enumerate(is_prime) if val]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    # 2. Compute parity vector for 210
    def get_parity_vector(n):
        vec = 0
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
                    vec ^= (1 << prime_to_idx[p])
        if temp > 1:
            # temp is a prime
            if temp in prime_to_idx:
                vec ^= (1 << prime_to_idx[temp])
            else:
                # This shouldn't happen if limit is sufficient
                pass
        return vec

    target_vec = get_parity_vector(210)
    
    # 3. Iterate K from 211 upwards
    # We store vectors v(211) ... v(K)
    # And we maintain a basis to check span membership efficiently?
    # Or just re-run elimination for each K?
    # Re-running is O(K^2 * D) roughly. K is small.
    
    # Let's try to find the minimal K.
    # We'll store the vectors in a list.
    vecs = []
    
    for K in range(211, 10211):
        if time.time() > deadline:
            break # Should not happen if K is small, but safety
            
        v_K = get_parity_vector(K)
        vecs.append(v_K)
        
        # Check if target_vec is in span of vecs
        # We do Gaussian elimination on [vecs | target_vec]
        # To be efficient, we can maintain a basis incrementally, 
        # but for K ~ 100, full elimination is fast enough.
        
        # Let's check span membership.
        # We reduce target_vec using the basis of vecs.
        # We need the basis of vecs.
        # We can compute the basis of vecs each time? 
        # Or maintain it.
        
        # Let's maintain a basis for the current set of vectors.
        # Basis is a list of (pivot, vector)
        # Actually, we need to know if target is in span.
        
        # Simple approach:
        # Create a list of all vectors: vecs + [target_vec]
        # Perform elimination. If target_vec reduces to 0, it's in span.
        
        # To find the solution, we need the coefficients.
        # So we do full elimination with coefficient tracking.
        
        # Matrix: columns are vecs[0], ..., vecs[K-210], target_vec
        # We want to see if target_vec is a linear combination of the others.
        
        # Let's create a list of (vector, coeff_mask)
        # coeff_mask is a bitset where bit i is 1 if original vec i is used.
        # For the target vector, we initialize coeff_mask to 0?
        # No, we want to represent target_vec as sum of vecs.
        # So we augment the matrix with an extra column for the target.
        # We perform elimination on the target column using the other columns.
        
        # Let's do this:
        # basis = [] # list of (pivot, vec, mask)
        # masks for vecs[i] is 1 << i
        
        # We iterate through vecs and add them to basis.
        # Then we try to reduce target_vec using the basis.
        
        # Actually, it's easier to just run elimination on the augmented matrix 
        # where the last column is the target, and we see if it becomes 0.
        
        # Let's implement a helper to check if target is in span of vecs.
        
        def is_in_span_and_get_coeffs(target, vecs_list):
            n_cols = len(vecs_list)
            # We work on a copy of target
            curr_target = target
            # We also track the combination of vecs_list that equals curr_target?
            # No, we want target = sum c_i v_i.
            # So we start with "current vector" = target, "current coeffs" = 0 (meaning 0*v_i)
            # Wait.
            # We want to express target as linear combo of v_i.
            # We can perform elimination on the target vector using the basis of v_i.
            
            # Let's build the basis of v_i first.
            basis = [] # list of (pivot, basis_vec, basis_mask)
            # basis_mask is a bitset of which v_i are used to form this basis vector.
            
            for i, v in enumerate(vecs_list):
                curr_v = v
                curr_mask = 1 << i
                # Reduce curr_v using existing basis
                for pivot, b_vec, b_mask in basis:
                    if curr_v & (1 << pivot):
                        curr_v ^= b_vec
                        curr_mask ^= b_mask
                if curr_v != 0:
                    # Find new pivot
                    pivot = curr_v.bit_length() - 1
                    # Insert into basis (keep sorted by pivot for efficiency? or just append)
                    # For small n, append is fine.
                    basis.append((pivot, curr_v, curr_mask))
            
            # Now reduce target using basis
            curr_t = target
            curr_t_mask = 0 # This will hold the combination of basis vectors?
            # No, we want the combination of original v_i.
            # If we reduce target using basis vectors, and each basis vector has a mask of original v_i,
            # then the mask of the reduction steps will give the final combination.
            
            # Let's re-do:
            # We want to check if target is in span.
            # We reduce target using the basis vectors.
            # If it becomes 0, it is in span.
            # The "mask" of the operations tells us which basis vectors were used.
            # But we need the mask of original v_i.
            
            # Let's track:
            # curr_t = target
            # used_basis_mask = 0 (bitset of indices in 'basis' list)
            # But 'basis' list indices are not stable if we don't sort/insert carefully?
            # Let's just store the mask of original v_i directly in the basis entries.
            
            # Re-implementing reduction of target:
            curr_t = target
            final_mask = 0
            for pivot, b_vec, b_mask in basis:
                if curr_t & (1 << pivot):
                    curr_t ^= b_vec
                    final_mask ^= b_mask
            
            if curr_t == 0:
                return True, final_mask
            else:
                return False, 0

        in_span, mask = is_in_span_and_get_coeffs(target_vec, vecs)
        
        if in_span:
            # Extract selected indices
            selected_indices = []
            for i in range(len(vecs)):
                if mask & (1 << i):
                    selected_indices.append(211 + i)
            
            # Verify
            # Product of selected * 210 must be square.
            # This is guaranteed by the linear algebra.
            # But let's double check for safety.
            # And ensure distinctness (guaranteed by indices).
            # And range (guaranteed by loop).
            
            # Return as list of ints
            return selected_indices

    # If no solution found (should not happen as 210*210 is square, 
    # but 210 is not > 210. 
    # However, we can always pick 210 * (210 * k^2) / 210 = 210 * k^2?
    # Wait, the numbers must be > 210.
    # If we pick { 210 * 2 }, is 420 > 210? Yes.
    # v(420) = v(2*210) = v(2) + v(210).
    # Target is v(210).
    # So v(210) + v(420) = v(2). Not 0.
    
    # Actually, we can always find a solution.
    # For example, pick all numbers in a range?
    # Or just return a fallback if the loop finishes (which it shouldn't).
    
    # Fallback: return a valid set, e.g. {420} is not enough.
    # { 210*2, 210*3, 210*5, 210*7 }?
    # v(420)+v(630)+v(1050)+v(1470) = v(2)+v(210) + v(3)+v(210) + v(5)+v(210) + v(7)+v(210)
    # = v(2)+v(3)+v(5)+v(7) + 4*v(210) = v(210) + 0 = v(210).
    # So {420, 630, 1050, 1470} works.
    # Max is 1470. Raw value 1260.
    # The algorithm should find something much better.
    
    return [420, 630, 1050, 1470]
