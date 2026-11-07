import math

N = 210
MAX_VAL = 10210
MAX_SIZE = 100

def get_primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def construct():
    # 1. Identify relevant primes
    # Primes up to MAX_VAL are sufficient for factoring numbers in range.
    primes = get_primes(MAX_VAL)
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # Target vector: parity of 210
    # 210 = 2 * 3 * 5 * 7
    target_mask = 0
    for p in [2, 3, 5, 7]:
        target_mask |= (1 << prime_to_idx[p])
        
    # 2. Compute vector for each number in range
    # We will iterate M from 211 to MAX_VAL
    # We need to find the smallest M such that target_mask is in the span of {v(n) for 211 <= n <= M}
    
    # To be efficient, we can just check incrementally.
    # However, finding the subset requires backtracking.
    # Let's find the minimal M first.
    
    # We can use a simple check: if we have more vectors than dimensions, span is likely full.
    # But we need exactness.
    
    # Let's build the list of (value, mask) for all n in [211, MAX_VAL]
    candidates = []
    for n in range(211, MAX_VAL + 1):
        mask = 0
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
                    mask |= (1 << prime_to_idx[p])
        if temp > 1:
            # temp is prime
            if temp in prime_to_idx:
                mask |= (1 << prime_to_idx[temp])
            else:
                # This shouldn't happen if primes list is complete up to MAX_VAL
                pass
        candidates.append((n, mask))
        
    # Now find minimal prefix length L such that target is in span of first L masks.
    # We can use Gaussian elimination to maintain the basis and check membership.
    # But we also need to recover the subset.
    
    # A simpler approach for small M:
    # Since the expected answer is small (like 42, so M=252), the search space is small.
    # We can just iterate M from 211 to, say, 500.
    # For each M, check if target is in span.
    # If yes, find the subset.
    
    def find_subset(masks, target):
        # masks is a list of (value, mask)
        # Returns a list of values forming the subset, or None
        # Uses Gaussian elimination with backtracking
        n_vars = len(masks)
        if n_vars == 0:
            return [] if target == 0 else None
            
        # We want sum c_i v_i = target
        # This is equivalent to finding a solution to A x = b
        # We can use sympy or manual Gaussian elimination.
        # Given the small size (likely < 100 vars for small M), we can use a simple method.
        # Actually, we can just use itertools if M is very small, but M might be up to 1000.
        # Let's use a standard linear algebra approach over GF(2).
        
        # Build matrix A (rows are vectors) and b (target)
        # We want to solve A^T x = b? No, sum c_i v_i = b.
        # This is a system where columns are vectors? No, vectors are rows.
        # We are looking for coefficients c such that c^T A = b^T?
        # Let's just use a library if available, or implement simple elimination.
        # Since we are in Python, we can represent vectors as ints and use bitwise ops.
        
        # Algorithm:
        # 1. Perform Gaussian elimination on the vectors to form a basis.
        # 2. Keep track of which original vectors were used to form each basis vector.
        # 3. Try to express target in terms of the basis.
        # 4. If successful, expand the basis vectors back to original vectors.
        
        basis = [] # List of (pivot_bit, basis_vector, combination_mask)
        # combination_mask is a bitmask of indices in 'masks' that sum to this basis_vector
        
        for i, (val, vec) in enumerate(masks):
            v = vec
            comb = 1 << i
            # Try to eliminate v using existing basis
            # We want to find a pivot for v
            # Standard elimination: for each bit in v, if there is a basis vector with that pivot, XOR it.
            # We need to identify pivots.
            
            # Let's store basis as dict: pivot_bit -> (vec, comb)
            # But we need to iterate bits.
            
            # Simpler: just keep a list of (vec, comb) and do elimination.
            # To be efficient, we can sort by highest set bit?
            
            # Let's use a standard approach:
            # Maintain a list of basis vectors.
            # For each new vector v:
            #   for each basis vector b:
            #     if b has a 1 in the highest set bit of v:
            #        v ^= b
            #        comb ^= b_comb
            #   if v != 0:
            #      add v to basis with comb
            #   else:
            #      v is dependent, skip (or use to find dependencies, but we just need one solution)
            
            # This finds *a* basis. To check if target is in span:
            #   t = target
            #   for each basis vector b (in reverse order of addition? or sorted by pivot?):
            #      if b has 1 in highest set bit of t:
            #         t ^= b
            #         result_comb ^= b_comb
            #   if t == 0: success
            #   else: fail
            
            # We need to be careful with the order of elimination to ensure pivots are unique.
            # Let's refine the elimination to ensure each basis vector has a unique pivot.
            
            pass # We will implement this logic in the main loop below
            
    # Let's implement the search for minimal M
    # We will maintain the basis dynamically as we add more numbers.
    
    basis = [] # List of tuples (pivot, vec, comb)
    # pivot is the index of the highest set bit in vec
    
    for i, (val, vec) in enumerate(candidates):
        v = vec
        comb = 1 << i
        
        # Eliminate v using current basis
        # We need to find the pivot for v.
        # We iterate through basis vectors? No, that's slow.
        # We can store basis in a dict keyed by pivot.
        # But we are adding one by one.
        
        # Let's use a dict for basis lookup by pivot.
        # However, the standard algorithm for maintaining a basis in GF(2):
        # 1. Find the highest set bit in v. Let it be h.
        # 2. If there is a basis vector with pivot h, XOR v with it (and comb with its comb).
        # 3. Repeat until v is 0 or v has a pivot not in the basis.
        # 4. If v is not 0, add it to basis with its new pivot.
        
        # We need to be able to find the basis vector with a specific pivot quickly.
        # Since we are iterating i from 0 to N, we can just keep a list and scan?
        # N is ~10000. Scanning 10000 items 10000 times is 10^8, might be slow in Python.
        # Let's use a dict: pivot -> (vec, comb)
        
        pass # Re-implementing with dict
        
    # Let's restart the logic with a dict-based basis for efficiency.
    
    basis_dict = {} # pivot -> (vec, comb)
    
    for i, (val, vec) in enumerate(candidates):
        v = vec
        comb = 1 << i
        
        while v:
            # Find highest set bit
            h = v.bit_length() - 1
            if h in basis_dict:
                b_vec, b_comb = basis_dict[h]
                v ^= b_vec
                comb ^= b_comb
            else:
                # v has a new pivot h
                basis_dict[h] = (v, comb)
                break
                
        # Now check if target is in the span of the current basis
        # We can do this after each addition, or just once we think we have enough.
        # To find the *minimal* M, we should check after each addition.
        # But checking is O(num_basis).
        
        # Let's check if target is in span
        t = target_mask
        res_comb = 0
        # We need to eliminate t using the basis
        # We should iterate basis vectors in descending order of pivot?
        # The elimination process above doesn't guarantee the basis is in row echelon form with unique pivots in a way that allows simple back-substitution without re-ordering.
        # Actually, the standard "insert into basis" algorithm maintains a basis where each vector has a unique pivot.
        # To test membership of t:
        #   For each pivot h in descending order (or any order, as long as we eliminate correctly):
        #      if t has bit h set:
        #         if h in basis_dict:
        #             b_vec, b_comb = basis_dict[h]
        #             t ^= b_vec
        #             res_comb ^= b_comb
        #         else:
        #             # t has a bit h that is not a pivot in basis, so t is not in span
        #             break
        #   if t == 0: in span
        
        # Let's implement this check.
        # To be efficient, we can just check if the current basis can form the target.
        
        t_check = target_mask
        c_check = 0
        found = True
        # We need to eliminate t_check using basis_dict.
        # We should process bits from highest to lowest.
        # The basis_dict keys are the pivots.
        # We can just iterate through the bits of t_check.
        
        # A better way:
        # Copy t_check.
        # While t_check != 0:
        #   h = t_check.bit_length() - 1
        #   if h in basis_dict:
        #      b_vec, b_comb = basis_dict[h]
        #      t_check ^= b_vec
        #      c_check ^= b_comb
        #   else:
        #      found = False
        #      break
        # if found and t_check == 0:
        #    We have a solution c_check.
        
        t_check = target_mask
        c_check = 0
        found = True
        while t_check != 0:
            h = t_check.bit_length() - 1
            if h in basis_dict:
                b_vec, b_comb = basis_dict[h]
                t_check ^= b_vec
                c_check ^= b_comb
            else:
                found = False
                break
                
        if found and t_check == 0:
            # We found a solution using indices in c_check
            # c_check is a bitmask of indices i in candidates
            # Extract the values
            result = []
            for j in range(len(candidates)):
                if c_check & (1 << j):
                    result.append(candidates[j][0])
                    
            # Verify the result
            # The product of kernels should be 210
            # Let's just return this.
            # We need to ensure distinctness (guaranteed by indices) and constraints.
            # The task asks for 1..100 integers.
            # If len(result) > 100, we might have a problem.
            # But for small M, the solution is likely small.
            # If we need to reduce size, we can remove pairs of identical vectors? No, they are distinct numbers.
            # We can remove any two numbers with the same kernel?
            # If we remove n1 and n2 with same kernel, the product kernel is unchanged (k*k=1).
            # So we can simplify the set by removing pairs with same kernel.
            
            # Group by kernel
            from collections import defaultdict
            kernel_map = defaultdict(list)
            for idx in range(len(candidates)):
                if c_check & (1 << idx):
                    k = candidates[idx][1]
                    kernel_map[k].append(candidates[idx][0])
            
            final_set = []
            for k, nums in kernel_map.items():
                # We can keep at most 1 number from each kernel group?
                # If we keep 2, their product kernel is 1.
                # If we keep 1, kernel is k.
                # We need the product of kept kernels to be 210.
                # If we keep 1 from each group, product is prod(k for k in groups).
                # If we keep 2 from a group, it contributes 1.
                # We want to minimize the count? No, we want to satisfy the condition.
                # The condition is already satisfied by the full set.
                # If we remove pairs, the condition remains satisfied.
                # So we can reduce the set size.
                
                # To be safe, let's just keep 1 from each group if the count is odd, 0 if even?
                # Wait.
                # Sum of vectors is Target.
                # If we have 2 vectors v, sum is 0.
                # If we have 3 vectors v, sum is v.
                # So we can replace any even number of vectors with the same kernel by 0 vectors.
                # And any odd number by 1 vector.
                # This minimizes the set size.
                
                if len(nums) % 2 == 1:
                    # Keep one
                    # Which one? The smallest? Largest?
                    # To minimize max, we should keep the smallest?
                    # No, we already fixed M. The max is determined by the largest element in the *original* solution?
                    # Actually, if we remove large elements, max might decrease.
                    # So we should keep the *smallest* element from each group to minimize the max of the final set?
                    # Wait, if we keep the smallest, the max of the set is the max of the kept elements.
                    # If we kept a large element and removed it, max decreases.
                    # So yes, keep the smallest from each group.
                    final_set.append(min(nums))
            
            # Now verify final_set
            # Product of kernels of final_set must be 210.
            # Since we reduced mod 2, the sum of kernels is the same as the original sum (Target).
            # So it should work.
            
            # Check constraints
            if not final_set:
                # Should not happen if target != 0
                continue
                
            if any(x <= 210 or x > 10210 for x in final_set):
                continue
                
            if len(final_set) > 100:
                # This is unlikely for small M
                continue
                
            # Return the set
            # We want to minimize max(final_set).
            # The current M is the max of the *searched* range.
            # But the actual max of final_set might be smaller than M.
            # We can just return final_set.
            # The score is max(final_set) - 210.
            
            return final_set
            
    # If no solution found in range, return a known good one
    return [840]

# Test at module level to ensure correctness
if __name__ == "__main__":
    res = construct()
    prod = 210
    for x in res:
        prod *= x
    sqrt_p = math.isqrt(prod)
    print(f"Result: {res}")
    print(f"Max - 210: {max(res) - 210}")
    print(f"Is square: {sqrt_p * sqrt_p == prod}")
