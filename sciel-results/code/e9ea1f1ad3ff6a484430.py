import time
import itertools
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # Helper: prime factorization exponent vector for numbers 2..40
    # Primes up to 40: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    p_idx = {p: i for i, p in enumerate(primes)}
    
    def factor_vec(n):
        v = [0] * len(primes)
        for i, p in enumerate(primes):
            while n % p == 0:
                v[i] += 1
                n //= p
        return tuple(v)
    
    # Precompute exponent vectors for 2..40
    vecs = {}
    for n in range(2, 41):
        vecs[n] = factor_vec(n)
    
    # Two numbers a, b are "conflicting" if there exist integers x, y (not both 0)
    # such that x * vec(a) + y * vec(b) = 0.
    # This means vec(a) and vec(b) are linearly dependent over Q.
    # For non-zero vectors, this means one is a rational multiple of the other.
    
    def are_conflicting(a, b):
        va = vecs[a]
        vb = vecs[b]
        # Check if va = k * vb for some rational k
        # Find first non-zero in vb
        for i in range(len(vb)):
            if vb[i] != 0:
                # k = va[i] / vb[i]
                num = va[i]
                den = vb[i]
                # Check if all other components match
                for j in range(len(vb)):
                    if va[j] * den != vb[j] * num:
                        return False
                return True
        # If vb is all zeros (which it isn't for n>=2), handle separately
        return all(x == 0 for x in va)
    
    # Build conflict graph
    # Vertices: 2..40
    # Edge between a, b if they conflict
    vertices = list(range(2, 41))
    conflict = {}
    for i, a in enumerate(vertices):
        conflict[a] = set()
        for j, b in enumerate(vertices):
            if i == j:
                continue
            if are_conflicting(a, b):
                conflict[a].add(b)
    
    # Wait, the condition for distinct subset products is stronger than just pairwise independence.
    # A set S has distinct subset products if and only if the exponent vectors of elements in S
    # are linearly independent over the rationals.
    # Pairwise linear independence is NOT sufficient for linear independence of the whole set.
    # Example: v1=(1,0), v2=(0,1), v3=(1,1). v1, v2, v3 are pairwise independent but v3 = v1+v2.
    # So we need the full set to be linearly independent.
    
    # However, the "conflict graph" idea from the prompt suggests edges for linear dependence.
    # If we pick a set with no edges (no pair is dependent), is the set independent? No, as shown above.
    # So the maximum independent set in the pairwise conflict graph is NOT the answer.
    # We need a subset of vertices whose vectors are linearly independent.
    
    # This is the problem of finding the largest linearly independent subset of a set of vectors.
    # This can be solved by Gaussian elimination over the rationals!
    # The size of the maximum linearly independent subset is the rank of the matrix of vectors.
    # And any basis of that matrix gives such a set.
    
    # So the problem reduces to:
    # 1. Construct the matrix of exponent vectors for numbers 2..40.
    # 2. Find a subset of rows that forms a basis (linearly independent).
    # 3. The size of this basis is the maximum possible k.
    # 4. We just need to return any such basis.
    
    # Let's compute the rank and a basis.
    
    # The vectors are in Q^12 (12 primes).
    # We can use fractions for exact arithmetic.
    from fractions import Fraction
    
    # Matrix rows: vecs[n] for n in 2..40
    # We want to find a subset of these rows that is linearly independent and maximal.
    
    # Gaussian elimination to find a basis.
    # We'll maintain a list of basis vectors (as tuples of Fractions).
    # For each vector v, check if it's independent of the current basis.
    # If yes, add it to the basis.
    
    # To make it efficient, we can reduce each vector by the current basis.
    # If the remainder is non-zero, it's independent.
    
    basis_vecs = []  # List of (number, vector)
    
    # We process numbers in some order. To maximize the count, any order that finds a full rank basis works.
    # The rank is at most 12 (dimension of the space).
    # Actually, the rank is determined by the set of all 39 vectors.
    
    # Let's just do standard row reduction.
    # Represent vectors as lists of Fractions.
    
    def to_frac_vec(v):
        return [Fraction(x) for x in v]
    
    current_basis = []  # List of frac vectors
    selected_numbers = []
    
    # To get a "nice" set, maybe prefer smaller numbers? Or doesn't matter.
    # The task just says "maximize number of selected integers".
    
    # Let's iterate through 2..40.
    for n in range(2, 41):
        v = to_frac_vec(vecs[n])
        
        # Reduce v by current_basis
        # If v becomes zero, it's dependent.
        # Otherwise, add it to basis (and keep the reduced form for future reductions? 
        # Standard Gaussian elimination keeps the basis in row-echelon form).
        
        # Let's implement a simple check: is v in the span of current_basis?
        # We can try to eliminate v using current_basis.
        # For each basis vector b in current_basis:
        #   If v has a non-zero component in the pivot position of b,
        #   subtract (v[pivot] / b[pivot]) * b from v.
        # If v is all zeros at the end, it's dependent.
        
        # We need to maintain pivot positions for the basis.
        # Let's re-implement this properly.
        
        # Actually, simpler approach:
        # Keep the basis as a list of vectors.
        # For a new vector v, check independence by trying to express it as a linear combination.
        # Since dimension is small (12) and max basis size is 12, this is fast.
        
        # Let's use a helper to check if v is independent of basis.
        # We can do this by adding v to the basis and checking if rank increases.
        # Or simpler: reduce v by the basis in row-echelon form.
        
        # Let's maintain the basis in row-echelon form with pivot indices.
        # basis[i] has pivot at index pivot[i].
        
        # I'll restart the basis construction with proper row-echelon maintenance.
        
        pass # The loop above was just a sketch. Let's do it properly below.
        
    # Proper Gaussian Elimination
    basis = []  # List of (pivot_index, frac_vector)
    selected = []
    
    for n in range(2, 41):
        v = to_frac_vec(vecs[n])
        
        # Reduce v by existing basis
        for pivot_idx, b_vec in basis:
            if v[pivot_idx] != 0:
                factor = v[pivot_idx] / b_vec[pivot_idx]
                for j in range(len(v)):
                    v[j] -= factor * b_vec[j]
        
        # Check if v is zero
        if all(x == 0 for x in v):
            continue # Dependent
        
        # Find new pivot
        new_pivot = -1
        for j in range(len(v)):
            if v[j] != 0:
                new_pivot = j
                break
        
        # Add to basis
        basis.append((new_pivot, v))
        selected.append(n)
        
        # Check time
        if time.time() > deadline - 5:
            break
    
    # The selected numbers form a linearly independent set.
    # Its size is the rank of the set of all vectors.
    # Is it maximal? Yes, because we added every independent vector we could.
    # The size is the rank.
    
    # Let's verify the result.
    # We need to ensure all subset products are distinct.
    # This is guaranteed by linear independence of the exponent vectors.
    
    # Let's double check with a brute force verification for small sets, or just trust the math.
    # The math is solid: distinct subset products <=> linearly independent exponent vectors.
    
    # One edge case: the problem asks for distinct integers in 2..40.
    # Our selected list contains distinct integers from 2..40.
    
    # Let's verify the size.
    # The vectors are in Q^12.
    # The maximum rank is 12.
    # Can we achieve rank 12?
    # We have primes 2,3,5,7,11,13,17,19,23,29,31,37.
    # Their vectors are the standard basis vectors e_1, ..., e_12.
    # So the set of primes {2,3,5,7,11,13,17,19,23,29,31,37} is linearly independent.
    # So the rank is at least 12.
    # Since the dimension is 12, the rank is exactly 12.
    # So we can always find a set of 12.
    
    # Can we find more? No, because the dimension is 12.
    # So the maximum k is 12.
    
    # Wait, the current program scores 1.3333 with 16 numbers.
    # 16/12 = 1.3333.
    # If the max is 12, then 16 is impossible?
    # Let me re-read the problem.
    # "All 2^k subset products ... must be distinct."
    # "construct() -> 1..18 distinct integers in 2..40."
    
    # If my math says max is 12, why does the seed program have 16?
    # Let's check the seed program's set: [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    # Vectors:
    # 2: (1,0,...)
    # 4: (2,0,...)
    # 16: (4,0,...)
    # These three are linearly dependent! 2*vec(2) = vec(4), etc.
    # vec(4) = 2 * vec(2).
    # So {2, 4, 16} is linearly dependent.
    # Therefore, subset products are NOT distinct.
    # Example: 4 = 2*2. But 2 is in the set.
    # Subset {4} has product 4.
    # Subset {2, 2} is not allowed because elements are distinct in the set, but in subset product, we pick elements from the set.
    # Wait, "subset products". A subset is a collection of distinct elements from the set.
    # So we can't pick 2 twice.
    # But we can pick 2 and 4. Product 8.
    # Can we get 4 another way? No, only {4} gives 4.
    # Can we get 16? {16} gives 16. {2, 8} no 8. {4, 4} no.
    # Let's check for collisions in {2, 4, 16}.
    # Subsets:
    # {} -> 1
    # {2} -> 2
    # {4} -> 4
    # {16} -> 16
    # {2,4} -> 8
    # {2,16} -> 32
    # {4,16} -> 64
    # {2,4,16} -> 128
    # All distinct!
    
    # So my linear independence condition is too strong?
    # Linear independence of vectors v1, v2, ... vk means that the only solution to
    # c1*v1 + ... + ck*vk = 0 is c1=...=ck=0.
    # This ensures that no non-trivial integer combination is zero?
    # Actually, distinct subset products means that for any two distinct subsets A, B,
    # product(A) != product(B).
    # This is equivalent to saying that for any distinct subsets A, B,
    # sum_{i in A} v_i != sum_{i in B} v_i.
    # This is equivalent to saying that all subset sums of the vectors are distinct.
    # This is NOT the same as linear independence.
    # Linear independence implies distinct subset sums?
    # If v1, v2 are linearly independent, subset sums are 0, v1, v2, v1+v2. All distinct.
    # If v1, v2 are linearly dependent, say v2 = 2*v1.
    # Subset sums: 0, v1, 2*v1, 3*v1. All distinct.
    # Wait. If v2 = 2*v1, then {v2} sum is 2*v1, {v1, v1} is not a subset.
    # So for k=2, dependence doesn't cause collision unless v1=v2.
    
    # Let's check k=3. v1=(1,0), v2=(2,0), v3=(3,0).
    # Sums: 0, 1, 2, 3, 1+2=3, 1+3=4, 2+3=5, 1+2+3=6.
    # Collision: {v2} sum=2, {v1, ?} no.
    # {v2} sum = 2.
    # {v1, v1} not allowed.
    # Is there a collision?
    # v1=1, v2=2, v3=3.
    # Sums: 0, 1, 2, 3, 1+2=3.
    # Collision! {v3} sum=3 and {v1, v2} sum=3.
    # So {1, 2, 3} is NOT a valid set of exponents.
    # Corresponding numbers: 2^1=2, 2^2=4, 2^3=8.
    # Set {2, 4, 8}.
    # Subsets:
    # {8} -> 8
    # {2, 4} -> 8
    # Collision!
    
    # So why did {2, 4, 16} work?
    # Exponents: 1, 2, 4.
    # Sums: 0, 1, 2, 4, 1+2=3, 1+4=5, 2+4=6, 1+2+4=7.
    # All distinct!
    # So the condition is that the subset sums of the exponent vectors are distinct.
    # This is called a "superincreasing" sequence in 1D? No, just distinct subset sums.
    # In 1D, a set of integers has distinct subset sums if and only if it is "sum-free" in a specific sense?
    # Actually, a set of positive integers has distinct subset sums if and only if no element is the sum of a subset of the previous elements (if sorted).
    # This is the condition for a "complete sequence" or similar.
    # Specifically, if we sort the exponents a1 < a2 < ... < ak, we need a_i > sum(a_1..a_{i-1}) for distinct subset sums?
    # No, that's for representing all numbers.
    # For distinct subset sums, the condition is weaker.
    # Example: {1, 2} sums: 0,1,2,3. Distinct.
    # {1, 3} sums: 0,1,3,4. Distinct.
    # {2, 3} sums: 0,2,3,5. Distinct.
    # {1, 2, 4} sums: 0..7. Distinct.
    # {1, 2, 3} sums: 0,1,2,3,3,4,5,6. Collision.
    
    # So in 1D, we need the set of exponents to have distinct subset sums.
    # This is equivalent to the set being "independent" in the sense of the subset sum problem.
    
    # In higher dimensions, it's more complex.
    # But the key insight is: Linear Independence is NOT sufficient (as {1,2,3} shows? No, {1,2,3} are linearly dependent in 1D? No, in 1D any two are dependent. {1,2,3} are dependent. And they failed.)
    # Wait, {1,2} are dependent in 1D (2=2*1). But {2,4} (exponents 1,2) worked.
    # So linear dependence does NOT imply collision.
    # Collision implies linear dependence?
    # If sum(A) = sum(B) with A != B, then sum(A) - sum(B) = 0.
    # This is a linear combination with coefficients in {-1, 0, 1}.
    # So a collision implies a linear dependence.
    # But the converse is not true.
    
    # So the condition is: The set of vectors must have distinct subset sums.
    # This is a stronger condition than just "no subset sum is zero" (which is linear independence over integers with coeff 1).
    # Actually, distinct subset sums is equivalent to: for any subset S, sum(S) is unique.
    
    # How to maximize the size of such a set from 2..40?
    # This is a combinatorial search problem.
    # The seed program found 16.
    # My linear independence approach found 12.
    # So I need to search for sets with distinct subset sums.
    
    # Let's go back to the seed program's approach: it uses a known good set of 16.
    # Can we find 17 or 18?
    # The search space is huge.
    # However, we can use a greedy approach with backtracking, checking the distinct subset sum condition.
    
    # Condition check:
    # Given a set of vectors, are all subset sums distinct?
    # We can maintain a set of all subset sums.
    # When adding a new vector v, the new sums are old_sums and old_sums + v.
    # We need (old_sums + v) to be disjoint from old_sums.
    # i.e., no x in old_sums such that x + v is in old_sums.
    # Or simply, check if the new set of sums has size 2 * old_size.
    
    # Let's implement this search.
    
    def has_distinct_subset_sums(vecs_list):
        sums = {tuple([0]*len(vecs_list[0]))} # Zero vector
        for v in vecs_list:
            new_sums = set()
            for s in sums:
                new_s = tuple(s[i] + v[i] for i in range(len(v)))
                new_sums.add(new_s)
            # Check if new_sums overlaps with sums
            if len(new_sums | sums) != len(new_sums) + len(sums):
                return False
            sums = new_sums | sums
        return True
    
    # This is O(2^k * k * dim). For k=18, 2^18 = 262144.
    # 262144 * 18 * 12 ~ 56 million ops. Might be slow in Python if done naively in a search loop.
    # But we can optimize.
    
    # Let's try a greedy construction.
    # Start with empty set.
    # Add numbers from 2..40 in some order.
    # If adding n keeps the set valid, add it.
    
    # What order?
    # The seed set uses primes and powers of 2, 3, 5.
    # Let's try adding numbers in increasing order? Or random?
    
    # Let's try a simple greedy:
    # current_set = []
    # current_sums = {zero_vec}
    # for n in 2..40:
    #   v = vecs[n]
    #   new_sums = {s + v for s in current_sums}
    #   if new_sums is disjoint from current_sums:
    #       current_set.append(n)
    #       current_sums = current_sums | new_sums
    
    # Let's test this greedy.
    
    def greedy_construct():
        current_set = []
        current_sums = {tuple([0]*12)}
        
        # Try adding in order 2..40
        for n in range(2, 41):
            v = vecs[n]
            new_sums = set()
            for s in current_sums:
                new_s = tuple(s[i] + v[i] for i in range(12))
                new_sums.add(new_s)
            
            # Check disjoint
            if len(new_sums & current_sums) == 0:
                current_set.append(n)
                current_sums = current_sums | new_sums
            # Check time
            if time.time() > deadline - 10:
                break
                
        return current_set
    
    # Let's run this greedy and see how many we get.
    # If it gets 16, great. If it gets more, even better.
    # If it gets less, we can try different orders or backtracking.
    
    best_set = greedy_construct()
    best_size = len(best_set)
    
    # If best_size < 16, try the known good set.
    known_good = [2, 4, 16, 3, 9, 5, 25, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    if best_size < len(known_good):
        best_set = known_good
        best_size = len(known_good)
    
    # Can we try to add more to best_set?
    # Let's try to add any number from 2..40 not in best_set.
    
    # First, compute sums for best_set.
    def compute_sums(s):
        sums = {tuple([0]*12)}
        for n in s:
            v = vecs[n]
            new_sums = set()
            for x in sums:
                new_s = tuple(x[i] + v[i] for i in range(12))
                new_sums.add(new_s)
            sums = sums | new_sums
        return sums
    
    current_sums = compute_sums(best_set)
    current_set = set(best_set)
    
    # Try adding one more
    improved = False
    for n in range(2, 41):
        if n in current_set:
            continue
        v = vecs[n]
        new_sums = set()
        for s in current_sums:
            new_s = tuple(s[i] + v[i] for i in range(12))
            new_sums.add(new_s)
        
        if len(new_sums & current_sums) == 0:
            best_set.append(n)
            current_set.add(n)
            current_sums = current_sums | new_sums
            improved = True
            # Try to add more?
            break # Restart loop to try adding another?
            
    # If improved, try again
    while improved and time.time() < deadline - 5:
        improved = False
        for n in range(2, 41):
            if n in current_set:
                continue
            v = vecs[n]
            new_sums = set()
            for s in current_sums:
                new_s = tuple(s[i] + v[i] for i in range(12))
                new_sums.add(new_s)
            
            if len(new_sums & current_sums) == 0:
                best_set.append(n)
                current_set.add(n)
                current_sums = current_sums | new_sums
                improved = True
                break
    
    # Final verification
    # Just to be safe, verify the final set.
    final_sums = compute_sums(best_set)
    if len(final_sums) != (1 << len(best_set)):
        # Something went wrong, fallback
        return known_good
    
    return best_set
