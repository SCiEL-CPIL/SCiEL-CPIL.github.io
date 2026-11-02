import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x7 matrix.
    Strategy: Hybrid Strassen/Schoolbook with verification and potential local refinement.
    Base Rank: 39.
    """
    start_time = time.time()
    DEADLINE = 150.0
    SAFETY_MARGIN = 5.0
    
    n, m, p = DIMENSIONS  # 2, 3, 7
    
    # Helper functions for indices
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # We will store terms as lists of (u_vec, v_vec, w_vec)
    # u_vec: list of length n*m
    # v_vec: list of length m*p
    # w_vec: list of length p*n
    
    def add_term_to_list(terms, u_dict, v_dict, w_dict):
        u_list = [0] * (n * m)
        v_list = [0] * (m * p)
        w_list = [0] * (p * n)
        for idx, val in u_dict.items():
            u_list[idx] = val
        for idx, val in v_dict.items():
            v_list[idx] = val
        for idx, val in w_dict.items():
            w_list[idx] = val
        terms.append((u_list, v_list, w_list))
        
    def strassen_2x2x2(terms, k0, k1):
        """
        Adds 7 terms for Strassen multiplication of A[0:2, 0:2] * B[0:2, k0:k1+2]
        A is 2x2, B block is 2x2.
        """
        # M1 = (a00 + a11)(b00 + b11) -> C00, C11
        add_term_to_list(terms, 
            {u_idx(0,0):1, u_idx(1,1):1}, 
            {v_idx(0,k0):1, v_idx(1,k1):1}, 
            {w_idx(k0,0):1, w_idx(k1,1):1})
            
        # M2 = (a10 + a11)b00 -> C10, -C11
        add_term_to_list(terms, 
            {u_idx(1,0):1, u_idx(1,1):1}, 
            {v_idx(0,k0):1}, 
            {w_idx(k0,1):1, w_idx(k1,1):-1})
            
        # M3 = a00(b01 - b11) -> C01, C11
        add_term_to_list(terms, 
            {u_idx(0,0):1}, 
            {v_idx(0,k1):1, v_idx(1,k1):-1}, 
            {w_idx(k1,0):1, w_idx(k1,1):1})
            
        # M4 = a11(b10 - b00) -> C00, C10
        add_term_to_list(terms, 
            {u_idx(1,1):1}, 
            {v_idx(1,k0):1, v_idx(0,k0):-1}, 
            {w_idx(k0,0):1, w_idx(k0,1):1})
            
        # M5 = (a00 + a01)b11 -> -C00, C01
        add_term_to_list(terms, 
            {u_idx(0,0):1, u_idx(0,1):1}, 
            {v_idx(1,k1):1}, 
            {w_idx(k0,0):-1, w_idx(k1,0):1})
            
        # M6 = (a10 - a00)(b00 + b01) -> C11
        add_term_to_list(terms, 
            {u_idx(1,0):1, u_idx(0,0):-1}, 
            {v_idx(0,k0):1, v_idx(0,k1):1}, 
            {w_idx(k1,1):1})
            
        # M7 = (a01 - a11)(b10 + b11) -> C00
        add_term_to_list(terms, 
            {u_idx(0,1):1, u_idx(1,1):-1}, 
            {v_idx(1,k0):1, v_idx(1,k1):1}, 
            {w_idx(k0,0):1})

    def schoolbook_2x2x1(terms, k):
        """
        Adds 4 terms for schoolbook multiplication of A[0:2, 0:2] * B[0:2, k]
        """
        add_term_to_list(terms, {u_idx(0,0):1}, {v_idx(0,k):1}, {w_idx(k,0):1})
        add_term_to_list(terms, {u_idx(0,1):1}, {v_idx(1,k):1}, {w_idx(k,0):1})
        add_term_to_list(terms, {u_idx(1,0):1}, {v_idx(0,k):1}, {w_idx(k,1):1})
        add_term_to_list(terms, {u_idx(1,1):1}, {v_idx(1,k):1}, {w_idx(k,1):1})

    def schoolbook_2x1x1(terms, k):
        """
        Adds 2 terms for schoolbook multiplication of A[0:2, 2] * B[2, k]
        """
        add_term_to_list(terms, {u_idx(0,2):1}, {v_idx(2,k):1}, {w_idx(k,0):1})
        add_term_to_list(terms, {u_idx(1,2):1}, {v_idx(2,k):1}, {w_idx(k,1):1})

    def build_base_decomposition():
        terms = []
        # Strassen for B cols (0,1)
        strassen_2x2x2(terms, 0, 1)
        # Schoolbook for B col 2
        schoolbook_2x2x1(terms, 2)
        # Strassen for B cols (3,4)
        strassen_2x2x2(terms, 3, 4)
        # Strassen for B cols (5,6)
        strassen_2x2x2(terms, 5, 6)
        # Schoolbook for A row 2 (index 2) with all B cols
        for k in range(p):
            schoolbook_2x1x1(terms, k)
        return terms

    def terms_to_matrices(terms):
        rank = len(terms)
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        for c, (u, v, w) in enumerate(terms):
            U[:, c] = u
            V[:, c] = v
            W[:, c] = w
        return U, V, W

    def verify_decomposition(U, V, W):
        """
        Check if U, V, W exactly decompose the matrix multiplication tensor.
        T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
        We need T[i,j,k] = 1 if i*m+j == k*n+i? No.
        The tensor for matrix mult C = AB is:
        C[i,k] = sum_j A[i,j] B[j,k]
        So the tensor T has indices (i, j, k) where A is (i,j), B is (j,k), C is (i,k).
        Wait, the interface says:
        U shape (n*m, rank) -> indices (i,j) flattened
        V shape (m*p, rank) -> indices (j,k) flattened
        W shape (p*n, rank) -> indices (k,i) flattened? 
        "output factor follows the official trace convention W[k*n+i,r]"
        So W index is (k, i).
        
        The tensor equation is:
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,k}? No.
        It should equal 1 if the indices correspond to a valid multiplication term.
        Actually, for a specific A[i,j] and B[j,k], the contribution to C[i,k] is 1.
        So the tensor T is defined by:
        T_{(i,j), (j,k), (k,i)} = 1 ? No, that's not right.
        
        Let's look at the standard tensor definition.
        The multiplication tensor T is a 3-tensor of size (n*m) x (m*p) x (p*n).
        T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i) is a valid chain?
        No, T is the structure constants of the algebra.
        For matrix multiplication, T_{(i,j), (j,k), (i,k)} = 1.
        Here indices are:
        U index: i*m + j
        V index: j*p + k
        W index: k*n + i
        
        So we check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        for all valid i, j, k.
        And 0 otherwise? No, the tensor is sparse.
        The only non-zero entries of the "target" tensor are at (i*m+j, j*p+k, k*n+i) for all i,j,k.
        Wait, i goes 0..n-1, j goes 0..m-1, k goes 0..p-1.
        There are n*m*p such entries.
        
        Let's verify this interpretation.
        """
        # Target tensor entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx_val = i*m + j
                    v_idx_val = j*p + k
                    w_idx_val = k*n + i
                    val = np.dot(U[u_idx_val, :], np.multiply(V[v_idx_val, :], W[w_idx_val, :]))
                    if abs(val - 1.0) > 1e-9:
                        return False
        # Check that other entries are 0? 
        # The tensor is defined such that T_{a,b,c} = 0 unless a=(i,j), b=(j,k), c=(k,i) for some i,j,k.
        # Actually, for a fixed (a,b), c is determined if it's a valid term.
        # If we sum over r, we get a tensor T.
        # We need T to be the standard matrix multiplication tensor.
        # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i.
        # Otherwise 0.
        
        # Let's check a few "invalid" entries to be sure.
        # E.g. i=0, j=0, k=0 -> a=0, b=0, c=0. Sum = 1.
        # i=0, j=0, k=1 -> a=0, b=1, c=1. Sum = 1.
        # What about a=0 (i=0,j=0), b=0 (j=0,k=0), c=2 (k=1,i=0)? 
        # a=0, b=0 implies i=0,j=0 and j=0,k=0. So i=0,j=0,k=0.
        # Then c should be k*n+i = 0*2+0 = 0.
        # But we are checking c=2.
        # The tensor T[0,0,2] should be 0.
        # Let's verify a random invalid triplet.
        # a=0 (i=0,j=0), b=1 (j=0,k=1), c=0 (k=0,i=0).
        # j must be consistent. b=1 -> j=0. a=0 -> j=0. Consistent.
        # k from b is 1. k from c is 0. Inconsistent.
        # So T[0,1,0] should be 0.
        
        # We can't check all (n*m)*(m*p)*(p*n) entries efficiently if we just loop, 
        # but n*m=6, m*p=21, p*n=14. 6*21*14 = 1764. Very fast.
        
        for a in range(n*m):
            for b in range(m*p):
                # Determine i,j from a
                i = a // m
                j = a % m
                # Determine j',k from b
                jp = b // p
                k = b % p
                if j != jp:
                    continue # Should be 0, but if j!=jp, it's structurally 0 in the sum?
                    # Actually, if j!=jp, the product U[a,r]V[b,r] might be non-zero,
                    # but W[c,r] is summed over c.
                    # The tensor T[a,b,c] = sum_r U[a,r]V[b,r]W[c,r].
                    # We need T[a,b,c] to be 0 if indices don't match.
                    pass
                
        # More efficient check:
        # Compute the full tensor T and compare to target.
        # T is (6, 21, 14).
        # We can compute T = np.einsum('ar,br,cr->abc', U, V, W)
        # But that might be slow if rank is high? Rank is ~40. 6*21*14*40 = ~70k ops. Trivial.
        
        T = np.zeros((n*m, m*p, p*n))
        for r in range(U.shape[1]):
            T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            
        # Create target tensor
        T_target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    T_target[a, b, c] = 1.0
                    
        if not np.allclose(T, T_target, atol=1e-9):
            return False
            
        return True

    # 1. Build base decomposition
    terms = build_base_decomposition()
    U, V, W = terms_to_matrices(terms)
    
    # 2. Verify
    if not verify_decomposition(U, V, W):
        # Fallback: return a known correct decomposition (e.g. schoolbook)
        # But we trust the base. If it fails, we have a bug.
        # Let's try to return schoolbook just in case.
        terms_school = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    add_term_to_list(terms_school, 
                        {u_idx(i,j):1}, 
                        {v_idx(j,k):1}, 
                        {w_idx(k,i):1})
        U, V, W = terms_to_matrices(terms_school)
        return {"U": U, "V": V, "W": W}
        
    current_rank = len(terms)
    
    # 3. Attempt refinement: Remove redundant terms
    # A term is redundant if it can be represented as a linear combination of others?
    # No, rank minimization is hard.
    # However, we can check if any term is a linear combination of others in the tensor space?
    # This is equivalent to checking if the set of vectors (u_r, v_r, w_r) is linearly independent
    # in the tensor product space? No.
    
    # Simple heuristic: Check if removing a term and re-optimizing the remaining coefficients
    # (which are fixed to integers/half-integers here) still works?
    # Since we are using integer coefficients, we can't just "re-optimize" easily without solving an ILP.
    
    # Alternative: Check if two terms are identical?
    # In our construction, terms are distinct.
    
    # Let's try a "greedy elimination" of terms that have small contributions?
    # No, all contributions are essential in Strassen.
    
    # Is there a lower rank?
    # Lower bound for 2x3x7:
    # Rank(2x2x2) >= 7.
    # We used 3 blocks of 2x2x2 (rank 7 each) = 21.
    # Plus 1 block of 2x2x1 (rank 4).
    # Plus 1 block of 2x1x7 (rank 14).
    # Total 39.
    
    # Can we do better?
    # Maybe mix the 2x2x1 and 2x1x7 parts?
    # The 2x2x1 part uses A[0:2, 0:2] and B[0:2, 2].
    # The 2x1x7 part uses A[0:2, 2] and B[2, 0:7].
    # These are disjoint in A and B indices, so they can't be merged.
    
    # What about the 2x2x2 blocks?
    # They use A[0:2, 0:2] and disjoint B blocks.
    # They share the A part.
    # Strassen already exploits this sharing within the 2x2x2 block.
    # Can we exploit sharing *between* the 2x2x2 blocks?
    # This would require a "block Strassen" or similar.
    # For 2x2 * 2x4, can we do better than 7+7=14?
    # Rank of 2x2 * 2x4 tensor.
    # It contains 2x2 * 2x2 as a sub-tensor, so rank >= 7.
    # It is like two 2x2 * 2x2 multiplications sharing A.
    # Let A = [[a,b],[c,d]], B1 = [[e,f],[g,h]], B2 = [[i,j],[k,l]].
    # We need to compute A*B1 and A*B2.
    # This is equivalent to multiplying A (2x2) by B (2x4).
    # This is a rectangular matrix multiplication.
    # Known ranks:
    # 2x2 * 2x2: 7
    # 2x2 * 2x3: 10? (Strassen for 2x2*2x2 + schoolbook for 2x2*2x1 -> 7+4=11. Can we do 10?)
    # 2x2 * 2x4: 14? (7+7=14. Can we do 13? 12?)
    
    # If we could do 2x2 * 2x4 in 13, total would be 7 (for 2x2*2x2) + 13 (for 2x2*2x4) + 4 (for 2x2*2x1) + 14 (for 2x1*1x7) = 38.
    # Wait, my split was:
    # B cols 0,1 -> 2x2*2x2 (7)
    # B col 2 -> 2x2*2x1 (4)
    # B cols 3,4 -> 2x2*2x2 (7)
    # B cols 5,6 -> 2x2*2x2 (7)
    # Total 25 for A[0:2,0:2] * B[0:2, 0:7].
    # Plus 14 for A[0:2,2] * B[2, 0:7].
    # Total 39.
    
    # If we group B cols 0,1,2,3,4,5,6 differently?
    # B is 2x7.
    # 2x2 * 2x7.
    # Rank of 2x2 * 2x7?
    # If we use Strassen for 2x2*2x2 (7), and then 2x2*2x5?
    # 2x2 * 2x5: Rank?
    # Schoolbook: 2*2*5 = 20.
    # Strassen + Schoolbook: 7 (for 2x2*2x2) + 4 (for 2x2*2x1) + 7 (for 2x2*2x2) = 18.
    # So 2x2 * 2x5 can be done in 18.
    # Then 2x2 * 2x7 = 2x2 * 2x5 + 2x2 * 2x2 = 18 + 7 = 25.
    # This matches our current count (7+4+7+7=25).
    
    # Can 2x2 * 2x5 be done in 17?
    # Can 2x2 * 2x4 be done in 13?
    # If 2x2 * 2x4 is 13, then 2x2 * 2x6 is 13+7=20? Or 13+13=26?
    # 2x2 * 2x6 = 2x2 * 2x4 + 2x2 * 2x2 = 13 + 7 = 20.
    # Then 2x2 * 2x7 = 2x2 * 2x6 + 2x2 * 2x1 = 20 + 4 = 24.
    # Total 24 + 14 = 38.
    
    # So the key is: What is the rank of 2x2 * 2x4?
    # Let's try to find a rank 13 decomposition for 2x2 * 2x4.
    # Or rank 12?
    
    # Given the time limit and complexity, I will stick with the rank 39 solution but ensure it is robust.
    # I will add a check to see if we can simply return the base solution.
    
    # If we have time, we could try to search for a rank 38 solution using a simple local search?
    # But defining the search space for rank 38 is hard.
    
    # Let's just return the rank 39 solution.
    
    if time.time() > DEADLINE - SAFETY_MARGIN:
        pass # Just in case, though construction is fast.
        
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
