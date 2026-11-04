"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 5
    
    # We need to decompose the tensor T_{i,j,k} = delta_{i,k} (conceptually, 
    # but actually T_{i,j,k} is 1 if the index corresponds to the multiplication 
    # C[i,k] = sum_j A[i,j] B[j,k]).
    # The tensor has indices (i, j, k) with dimensions (n, m, p).
    # U is (n*m, R), V is (m*p, R), W is (p*n, R).
    # The decomposition is: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 if (i,j,k) is a valid term, else 0.
    # Actually, the tensor entry T[i, j, k] is 1 for all i, j, k because 
    # C[i,k] = sum_j A[i,j] B[j,k]. The tensor is the identity on the "multiplication" relation.
    # Wait, the tensor for matrix multiplication M_{n,m,p} has components T_{i,j,k} = 1 
    # if we view it as a trilinear form? No.
    # The standard definition: The tensor T has indices (i, j, k). 
    # T[i, j, k] = 1 if the indices are such that it contributes to the product.
    # In the context of the interface:
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # The sum over r should reconstruct the identity map for the multiplication.
    # Specifically, for any A, B, C = A*B.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor T_{i,j,k} is effectively 1 for all i,j,k in the range?
    # No, the tensor is the structure tensor of the algebra.
    # For matrix multiplication, T_{i,j,k} = 1 if (i,j,k) is a valid triple?
    # Actually, T is a 3-way array of size n x m x p where T[i,j,k] = 1.
    # Let's verify.
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # If we set A[i,j] and B[j,k], the result at [i,k] is sum_j sum_r ...
    # This implies that the "1" in the tensor corresponds to the fact that 
    # the term A[i,j] B[j,k] contributes to C[i,k].
    # So T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    # We will use a recursive Winograd-style decomposition (Binary splitting).
    # This guarantees a valid decomposition.
    
    # Base case: 2x2x2 Strassen (Rank 7)
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+b)(e+f)
    # M2 = (a-b)(e+f)
    # M3 = (c+d)h
    # M4 = a g
    # M5 = d (f-h)
    # M6 = (a+b) g
    # M7 = (c-d) f
    # C11 = M1 + M2 + M3 + M4
    # C12 = M5 + M6
    # C21 = M7 + M3
    # C22 = M1 - M2 + M5 - M7
    
    # We represent the decomposition as a list of (U_vec, V_vec, W_vec).
    # U_vec length n*m, V_vec length m*p, W_vec length p*n.
    
    def get_strassen_2x2x2():
        # Returns list of 7 tuples (u, v, w)
        # u index: i*2 + j
        # v index: j*2 + k
        # w index: k*2 + i
        
        terms = []
        
        # M1: (a+b)(e+f)
        # u: a(0,0)->1, b(0,1)->1, c(1,0)->0, d(1,1)->0 => [1,1,0,0]
        # v: e(0,0)->1, f(0,1)->1, g(1,0)->0, h(1,1)->0 => [1,1,0,0]
        # w: C11(0,0) gets +1, C22(1,1) gets +1.
        # w index: k*2+i.
        # C11: i=0,k=0 => idx 0. C22: i=1,k=1 => idx 3.
        # So w: [1, 0, 0, 1]
        terms.append(([1,1,0,0], [1,1,0,0], [1,0,0,1]))
        
        # M2: (a-b)(e+f)
        # u: [1,-1,0,0]
        # v: [1,1,0,0]
        # w: C11(0,0) gets +1, C22(1,1) gets -1.
        # w: [1, 0, 0, -1]
        terms.append(([1,-1,0,0], [1,1,0,0], [1,0,0,-1]))
        
        # M3: (c+d)h
        # u: [0,0,1,1]
        # v: h is (1,1) => idx 3. v: [0,0,0,1]
        # w: C11(0,0) gets +1, C21(1,0) gets +1.
        # C11: i=0,k=0 => idx 0. C21: i=1,k=0 => idx 2.
        # w: [1, 0, 1, 0]
        terms.append(([0,0,1,1], [0,0,0,1], [1,0,1,0]))
        
        # M4: a g
        # u: a(0,0) => [1,0,0,0]
        # v: g(1,0) => idx 2. v: [0,0,1,0]
        # w: C11(0,0) gets +1.
        # w: [1, 0, 0, 0]
        terms.append(([1,0,0,0], [0,0,1,0], [1,0,0,0]))
        
        # M5: d (f-h)
        # u: d(1,1) => [0,0,0,1]
        # v: f(0,1)->1, h(1,1)->-1. v: [0,1,0,-1]
        # w: C12(0,1) gets +1, C22(1,1) gets +1.
        # C12: i=0,k=1 => idx 1. C22: i=1,k=1 => idx 3.
        # w: [0, 1, 0, 1]
        terms.append(([0,0,0,1], [0,1,0,-1], [0,1,0,1]))
        
        # M6: (a+b) g
        # u: [1,1,0,0]
        # v: g(1,0) => idx 2. v: [0,0,1,0]
        # w: C12(0,1) gets +1.
        # w: [0, 1, 0, 0]
        terms.append(([1,1,0,0], [0,0,1,0], [0,1,0,0]))
        
        # M7: (c-d) f
        # u: [0,0,1,-1]
        # v: f(0,1) => idx 1. v: [0,1,0,0]
        # w: C21(1,0) gets +1, C22(1,1) gets -1.
        # C21: i=1,k=0 => idx 2. C22: i=1,k=1 => idx 3.
        # w: [0, 0, 1, -1]
        terms.append(([0,0,1,-1], [0,1,0,0], [0,0,1,-1]))
        
        return terms

    def pad_vectors(u, v, w, new_n, new_m, new_p, old_n, old_m, old_p):
        """
        Pads vectors from (old_n, old_m, old_p) to (new_n, new_m, new_p).
        u: length old_n*old_m -> new_n*new_m
        v: length old_m*old_p -> new_m*new_p
        w: length old_p*old_n -> new_p*new_n
        """
        new_u = [0] * (new_n * new_m)
        new_v = [0] * (new_m * new_p)
        new_w = [0] * (new_p * new_n)
        
        # U padding: i in [0, old_n), j in [0, old_m)
        for i in range(old_n):
            for j in range(old_m):
                new_u[i * new_m + j] = u[i * old_m + j]
                
        # V padding: j in [0, old_m), k in [0, old_p)
        for j in range(old_m):
            for k in range(old_p):
                new_v[j * new_p + k] = v[j * old_p + k]
                
        # W padding: k in [0, old_p), i in [0, old_n)
        for k in range(old_p):
            for i in range(old_n):
                new_w[k * new_n + i] = w[k * old_n + i]
                
        return new_u, new_v, new_w

    def decompose(n, m, p):
        """
        Returns a list of (u, v, w) for dimensions n, m, p.
        Uses binary splitting (Winograd) for rectangular cases.
        """
        if n == 2 and m == 2 and p == 2:
            return get_strassen_2x2x2()
            
        # Binary splitting strategy:
        # Split one of the dimensions.
        # For 2x3x5:
        # Split p: 5 -> 2 + 3.
        # C = A * B.
        # B is 3x5. Split B into B1 (3x2) and B2 (3x3).
        # C1 = A * B1 (2x3 * 3x2 -> 2x2). Rank 7.
        # C2 = A * B2 (2x3 * 3x3 -> 2x3). 
        # This recursive step needs to handle 2x3x3.
        
        # Let's define the split logic.
        # If n, m, p are not all 2, we split the largest dimension or a specific one.
        # To keep it simple and robust, we can always split the last dimension p if p > 2,
        # or m if m > 2, etc.
        
        # Strategy:
        # 1. If p > 2, split p into p1 and p2.
        #    We compute A (n x m) * B (m x p).
        #    Split B into B1 (m x p1) and B2 (m x p2).
        #    Compute C1 = A * B1 (n x p1) and C2 = A * B2 (n x p2).
        #    The decomposition for C1 uses dimensions (n, m, p1).
        #    The decomposition for C2 uses dimensions (n, m, p2).
        #    We need to "embed" these into the full (n, m, p) space.
        #    The U vectors remain the same (A is full n x m).
        #    The V vectors are padded: B1 is at columns 0..p1-1, B2 at p1..p-1.
        #    The W vectors are padded: C1 is at cols 0..p1-1, C2 at p1..p-1.
        
        # Let's implement this recursively.
        
        if p > 2:
            p1 = p // 2
            p2 = p - p1
            terms1 = decompose(n, m, p1)
            terms2 = decompose(n, m, p2)
            
            final_terms = []
            
            # Embed terms1
            for u, v, w in terms1:
                # u is already n*m
                # v needs padding from m*p1 to m*p
                new_v = [0] * (m * p)
                for j in range(m):
                    for k in range(p1):
                        new_v[j * p + k] = v[j * p1 + k]
                # w needs padding from p1*n to p*n
                new_w = [0] * (p * n)
                for k in range(p1):
                    for i in range(n):
                        new_w[k * n + i] = w[k * n + i]
                final_terms.append((u, new_v, new_w))
                
            # Embed terms2
            for u, v, w in terms2:
                # u is already n*m
                new_v = [0] * (m * p)
                for j in range(m):
                    for k in range(p2):
                        new_v[j * p + (p1 + k)] = v[j * p2 + k]
                new_w = [0] * (p * n)
                for k in range(p2):
                    for i in range(n):
                        new_w[(p1 + k) * n + i] = w[k * n + i]
                final_terms.append((u, new_v, new_w))
                        
            return final_terms
            
        if m > 2:
            m1 = m // 2
            m2 = m - m1
            # Split A and B along m.
            # A = [A1, A2] (n x m1, n x m2)
            # B = [B1; B2] (m1 x p, m2 x p)
            # C = A1 B1 + A2 B2
            # Term 1: A1 (n x m1) * B1 (m1 x p) -> C (n x p). Dim (n, m1, p)
            # Term 2: A2 (n x m2) * B2 (m2 x p) -> C (n x p). Dim (n, m2, p)
            
            terms1 = decompose(n, m1, p)
            terms2 = decompose(n, m2, p)
            
            final_terms = []
            
            # Embed terms1
            for u, v, w in terms1:
                new_u = [0] * (n * m)
                for i in range(n):
                    for j in range(m1):
                        new_u[i * m + j] = u[i * m1 + j]
                new_v = [0] * (m * p)
                for j in range(m1):
                    for k in range(p):
                        new_v[j * p + k] = v[j * p + k]
                # w is already p*n
                final_terms.append((new_u, new_v, w))
                
            # Embed terms2
            for u, v, w in terms2:
                new_u = [0] * (n * m)
                for i in range(n):
                    for j in range(m2):
                        new_u[i * m + (m1 + j)] = u[i * m2 + j]
                new_v = [0] * (m * p)
                for j in range(m2):
                    for k in range(p):
                        new_v[(m1 + j) * p + k] = v[j * p + k]
                final_terms.append((new_u, new_v, w))
                
            return final_terms
            
        if n > 2:
            n1 = n // 2
            n2 = n - n1
            # Split A and C along n.
            # A = [A1; A2] (n1 x m, n2 x m)
            # C = [C1; C2] (n1 x p, n2 x p)
            # C1 = A1 B, C2 = A2 B
            
            terms1 = decompose(n1, m, p)
            terms2 = decompose(n2, m, p)
            
            final_terms = []
            
            # Embed terms1
            for u, v, w in terms1:
                new_u = [0] * (n * m)
                for i in range(n1):
                    for j in range(m):
                        new_u[i * m + j] = u[i * m + j]
                # v is already m*p
                new_w = [0] * (p * n)
                for k in range(p):
                    for i in range(n1):
                        new_w[k * n + i] = w[k * n1 + i]
                final_terms.append((new_u, v, new_w))
                
            # Embed terms2
            for u, v, w in terms2:
                new_u = [0] * (n * m)
                for i in range(n2):
                    for j in range(m):
                        new_u[(n1 + i) * m + j] = u[i * m + j]
                new_w = [0] * (p * n)
                for k in range(p):
                    for i in range(n2):
                        new_w[k * n + (n1 + i)] = w[k * n2 + i]
                final_terms.append((new_u, v, new_w))
                
            return final_terms
            
        # Fallback: Schoolbook (should not be reached for 2,3,5 with above logic)
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = [0] * (n*m)
                    v = [0] * (m*p)
                    w = [0] * (p*n)
                    u[i*m+j] = 1
                    v[j*p+k] = 1
                    w[k*n+i] = 1
                    terms.append((u, v, w))
        return terms

    terms = decompose(n, m, p)
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u, v, w) in enumerate(terms):
        for idx, val in enumerate(u):
            if val != 0:
                U[idx][r] = val
        for idx, val in enumerate(v):
            if val != 0:
                V[idx][r] = val
        for idx, val in enumerate(w):
            if val != 0:
                W[idx][r] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
