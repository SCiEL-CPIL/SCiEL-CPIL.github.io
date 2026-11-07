import math
import time

def construct():
    start_time = time.time()
    deadline = 90.0
    
    # 1. Sieve primes up to 10210
    max_val = 10210
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(max_val)) + 1):
        if is_prime[i]:
            for j in range(i*i, max_val + 1, i):
                is_prime[j] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    # 2. Helper to get bitset
    def get_bitset(n):
        bits = 0
        temp = n
        for p, idx in prime_to_idx.items():
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                if count % 2 == 1:
                    bits |= (1 << idx)
            if temp == 1:
                break
        if temp > 1:
            # temp is a prime > sqrt(max_val) or just a prime
            # Actually, since we iterated all primes up to sqrt, if temp > 1 it is prime
            # And it must be in prime_to_idx if it's <= max_val
            if temp in prime_to_idx:
                bits |= (1 << prime_to_idx[temp])
            else:
                # Should not happen for n <= max_val
                pass
        return bits

    target = get_bitset(210)
    
    # 3. Iterate M
    basis = {} # pivot bit index -> bitset
    vectors = []
    
    for M in range(211, 10211):
        if time.time() > deadline - 5:
            break # Safety margin
            
        v = get_bitset(M)
        vectors.append(v)
        
        # Reduce v by current basis
        # We need to maintain basis such that we can reduce quickly
        # Let's just store the basis vectors in a list and reduce
        # To make it fast, we can keep a dict pivot -> vector
        # But we also need to handle if v reduces to 0
        
        temp = v
        # Reduce temp by basis
        # We iterate through basis? No, we should reduce by highest bits.
        # Let's keep basis as a list of (pivot, vector) sorted by pivot?
        # Or just a dict.
        # To reduce:
        # While temp != 0:
        #   h = temp.bit_length() - 1
        #   if h in basis:
        #       temp ^= basis[h]
        #   else:
        #       break
        # If temp == 0, v is in span.
        # Else, v is independent. Add to basis.
        #   But wait, if we add v, we need to make sure it's reduced w.r.t existing basis?
        #   The loop above reduces it w.r.t existing pivots.
        #   So the resulting `temp` has no bits that are pivots of existing basis vectors.
        #   So its highest bit is a new pivot.
        
        while temp != 0:
            h = temp.bit_length() - 1
            if h in basis:
                temp ^= basis[h]
            else:
                break
        
        if temp != 0:
            # New basis vector
            h = temp.bit_length() - 1
            basis[h] = temp
            # Note: We don't strictly need to update other basis vectors to be reduced by this one
            # for the "is in span" check, as long as we check membership correctly.
            # Wait, if we don't update, the basis is not in RREF.
            # Does "reduce by basis" work if basis is not RREF?
            # Yes, if we always reduce by the pivot of the current highest bit.
            # As long as each basis vector has a unique pivot and no other basis vector has that bit set.
            # When we added `temp`, it had no bits corresponding to existing pivots (because we reduced it).
            # So its highest bit `h` is not a pivot of any existing vector.
            # And no existing vector has bit `h` set?
            # Wait. If an existing vector had bit `h` set, then `h` would have been its pivot or a lower bit.
            # If `h` was a pivot of an existing vector, we would have reduced by it.
            # If `h` was a non-pivot bit of an existing vector...
            # Example: Basis: {2: 0b100, 1: 0b010}.
            # New v: 0b110.
            # Reduce: h=2. 0b110 ^ 0b100 = 0b010.
            # h=1. 0b010 ^ 0b010 = 0.
            # So it works.
            # What if New v: 0b101.
            # Reduce: h=2. 0b101 ^ 0b100 = 0b001.
            # h=0. 0 not in basis. Break.
            # Add 0b001 to basis with pivot 0.
            # Basis: {2: 0b100, 1: 0b010, 0: 0b001}.
            # This is fine.
            
            # Check if target is in span
            # We need to check if target reduces to 0.
            t_temp = target
            while t_temp != 0:
                th = t_temp.bit_length() - 1
                if th in basis:
                    t_temp ^= basis[th]
                else:
                    break
            
            if t_temp == 0:
                # Solution exists
                # Solve A x = target for vectors[0..k-1]
                # k = len(vectors)
                # Use Gaussian elimination
                # We have vectors list.
                # We want c such that sum c[i] * vectors[i] == target
                # This is linear system over F2.
                # Since k is small, we can just use the basis info?
                # Or run a simple solver.
                # Let's run a solver on the specific vectors.
                # Create matrix of vectors (rows) and target (col).
                # Actually, columns are primes, rows are numbers.
                # We want column vector c.
                # V^T c = target? No.
                # sum c_i v_i = target.
                # This is a linear combination of row vectors.
                # So we are solving for coefficients of rows.
                # Standard Gaussian elimination on the matrix formed by rows v_0...v_{k-1}.
                
                # We can reuse the reduction logic.
                # We know target is in span.
                # We can find the coefficients by expressing target as combination of basis vectors,
                # and then expressing each basis vector as combination of original vectors.
                # This requires tracking coefficients.
                
                # Simpler: Just run Gaussian elimination on the k vectors to find a solution.
                # k is likely < 20.
                # 2^k is small? No, we don't iterate subsets.
                # We do Gaussian elimination.
                
                # Let's implement a simple solver for this specific case.
                # We have list `vecs` of length k.
                # We want `coeffs` of length k.
                
                # Augment matrix: rows are vecs, last col is target?
                # No, we want sum c_i v_i = t.
                # This is equivalent to finding a vector c in the null space of something?
                # No.
                
                # Let's just use the basis we built.
                # We need to track how each basis vector was formed from original vectors.
                # Let's restart the loop with coefficient tracking if we want to be efficient.
                # But k is small, so we can just solve it fresh.
                
                # Fresh solver:
                # Matrix A is k x P.
                # We want x such that A x = t (where x is column vector of coeffs).
                # Wait, A is k rows. x is k columns.
                # A x is a vector of size P.
                # We want A x = t.
                # This is a system of P equations with k unknowns.
                # We can pick any k independent equations?
                # Or just use the fact that we already reduced it.
                
                # Actually, since we already determined it's solvable, we can just find ANY solution.
                # We can use the basis vectors.
                # Let's track coefficients in the basis.
                
                # I will refactor the loop to track coefficients.
                
                pass # Placeholder for solution logic

    return [] # Fallback
