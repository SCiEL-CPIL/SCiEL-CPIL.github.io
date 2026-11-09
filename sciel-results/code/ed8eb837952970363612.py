"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    # Primes up to 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    p_count = len(primes)
    
    # Map integer to its exponent vector
    # We use Fraction for exact arithmetic in Gaussian elimination
    # Or we can use integer arithmetic with careful pivoting to avoid fractions initially,
    # but Fractions are safe for small dimensions (12x12).
    
    # To maximize speed, we can try to pick a basis greedily or use a simple
    # linear independence check.
    # Since we know the max size is 12, and we have 12 primes, the primes themselves
    # form a basis. Any other set of 12 linearly independent vectors is also valid.
    # The set of primes is the simplest valid solution of size 12.
    
    # However, to be a "stronger algorithm" and potentially find a larger set if my
    # dimension analysis was wrong (it isn't, but let's be rigorous), I will implement
    # a general maximum independent set finder.
    
    # Generate vectors for all numbers 2..40
    vectors = []
    numbers = list(range(2, 41))
    
    for n in numbers:
        v = [0] * p_count
        temp = n
        for i, p in enumerate(primes):
            while temp % p == 0:
                v[i] += 1
                temp //= p
        # temp should be 1 for all n <= 40 since we included all primes <= 40
        vectors.append(v)
    
    # Find maximum linearly independent subset
    # Use Gaussian elimination
    # We maintain a list of basis vectors and the original numbers they came from
    basis = []
    basis_nums = []
    
    # To maximize the number of elements, we just need to find a basis.
    # The size of the basis is the rank.
    # We iterate through numbers. If a vector is independent of the current basis, add it.
    # This greedy approach finds A basis, but does it find a MAXIMAL one?
    # Yes, any maximal linearly independent set is a basis, and all bases have the same size (the rank).
    # So any greedy algorithm that adds independent vectors will end up with size = rank.
    
    # However, the order might matter for speed, but not for the final size.
    # Let's just do it.
    
    for idx, n in enumerate(numbers):
        if time.time() > deadline:
            break
            
        v = vectors[idx]
        
        # Check if v is linearly independent from current basis
        # We perform row reduction on [basis + v]
        # If the rank doesn't increase, it's dependent.
        
        # Efficient check: Try to express v as linear combo of basis?
        # Or just augment basis with v and check rank.
        
        # Simple method: Add v to a copy of basis, reduce, see if we have a zero row.
        # Since dimension is small (12), this is fast.
        
        # Let's use a helper to check independence
        def is_independent(vec, basis_vecs):
            # Create a matrix of basis_vecs + [vec]
            # Perform Gaussian elimination
            # If rank(basis_vecs + [vec]) == rank(basis_vecs), then dependent.
            # Actually, just check if vec is in span(basis_vecs).
            
            # We can use the existing basis in reduced row echelon form (RREF)
            # But we haven't maintained RREF.
            
            # Let's maintain the basis in RREF to make checking O(D^2) instead of O(D^3) per step.
            pass

    # Let's implement a proper incremental RREF builder
    # Basis is a list of tuples (vector, original_number)
    # We keep the matrix of vectors in RREF.
    
    rref_rows = []
    selected_numbers = []
    
    for idx, n in enumerate(numbers):
        if time.time() > deadline:
            break
            
        v = vectors[idx]
        
        # Reduce v against current rref_rows
        # rref_rows are in RREF, meaning each row has a leading 1 in a unique column,
        # and all other entries in that column are 0.
        
        current_v = v[:]
        
        for row in rref_rows:
            # Find the pivot column of this row
            # In RREF, the first non-zero element is the pivot.
            pivot_col = -1
            for j in range(p_count):
                if row[j] != 0:
                    pivot_col = j
                    break
            
            if pivot_col == -1:
                continue
                
            # If current_v has a non-zero component at pivot_col, eliminate it
            if current_v[pivot_col] != 0:
                # Subtract (current_v[pivot_col] * row) from current_v
                # Since we are using integers, and RREF pivots are 1,
                # this keeps integers if we started with integers?
                # Wait, RREF over integers isn't always integer-preserving without scaling.
                # But we can work over Q or use Fraction.
                # Given the small size, let's use Fraction for safety and simplicity.
                
                factor = Fraction(current_v[pivot_col])
                for j in range(p_count):
                    current_v[j] = Fraction(current_v[j]) - factor * row[j]
                    
        # After reduction, if current_v is all zeros, it's dependent.
        # Otherwise, it's independent.
        
        # Check if all zero
        is_zero = True
        for val in current_v:
            if val != 0:
                is_zero = False
                break
                
        if not is_zero:
            # It is independent.
            # Add it to basis.
            # To maintain RREF, we need to:
            # 1. Add current_v to the list of rows.
            # 2. Normalize the pivot to 1.
            # 3. Eliminate this pivot from all OTHER rows (existing and new).
            
            # Find pivot col for current_v
            pivot_col = -1
            for j in range(p_count):
                if current_v[j] != 0:
                    pivot_col = j
                    break
                    
            # Normalize
            pivot_val = current_v[pivot_col]
            for j in range(p_count):
                current_v[j] = current_v[j] / pivot_val
                
            # Add to rref_rows
            rref_rows.append(current_v)
            selected_numbers.append(n)
            
            # Eliminate this pivot from all other rows
            for i, row in enumerate(rref_rows):
                if i == len(rref_rows) - 1:
                    continue
                if row[pivot_col] != 0:
                    factor = row[pivot_col]
                    for j in range(p_count):
                        row[j] = row[j] - factor * current_v[j]
                        
            # Note: The order of rows in rref_rows doesn't strictly matter for the independence check
            # as long as each row has a unique pivot column and is normalized.
            # The elimination step ensures the unique pivot property for the new row's column.
            # We don't strictly need the "zeros below" property of standard Gaussian elimination
            # if we always eliminate against ALL existing rows, we effectively maintain a set of
            # orthogonal (in the linear algebra sense) vectors where each has a unique leading term.
            # This is sufficient for the "is_independent" check: if a new vector reduces to 0,
            # it is in the span.

    # The selected_numbers contains a maximal linearly independent set.
    # Convert to list of ints
    result = [int(x) for x in selected_numbers]
    
    # Ensure types are exact Python ints
    return result

# EVOLVE-BLOCK-END
