import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will build U, V, W as lists of columns (rank terms)
    # Each term is a tuple (u_vec, v_vec, w_vec)
    # u_vec is length n*m
    # v_vec is length m*p
    # w_vec is length p*n
    
    terms = []
    
    # Helper to add a rank-1 term
    def add_term(u_indices, v_indices, w_indices, val=1):
        # u_indices: list of (i, j) or flat index
        # For efficiency, we use flat indices
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        
        for idx in u_indices:
            u[idx] = val
        for idx in v_indices:
            v[idx] = val
        for idx in w_indices:
            w[idx] = val
            
        terms.append((u, v, w))
        
    # 1. Strassen 4x4 for A[0:4, 0:4] * B[0:4, 0:4]
    # We generate the 49 terms for 4x4 Strassen
    # Base 2x2 Strassen terms
    # Indices in 2x2: i,j in 0,1
    # U indices: i*2+j
    # V indices: j*2+k
    # W indices: k*2+i
    
    strassen_2x2 = [
        # (U indices, V indices, W indices)
        # M1 = (A00+A11)(B00+B11)
        # A part: A00, A11. B part: B00, B11.
        # This is not a single rank-1 term. It's a product of sums.
        # We need to expand each Strassen step into rank-1 terms of the sub-blocks.
        # Since sub-blocks are 2x2, we use the 2x2 Strassen decomposition for each of the 7 products.
    ]
    
    # Let's define the 2x2 Strassen decomposition explicitly as 7 rank-1 terms?
    # No, Strassen 2x2 is rank 7.
    # The 7 terms are the M1..M7.
    # But M1 is a product of 2x2 matrices.
    # So M1 itself is a 2x2 matrix multiplication, which has rank 7.
    # So 4x4 Strassen has rank 7 * 7 = 49.
    
    # We need the 7 "macro" terms of Strassen 4x4.
    # Macro term t: U_t(A) V_t(B) W_t
    # U_t is a 4x4 matrix (linear combo of A blocks).
    # V_t is a 4x4 matrix (linear combo of B blocks).
    # W_t is a scalar (or matrix if we don't combine W).
    # In our format, W is part of the output basis.
    # Actually, the standard Strassen decomposition is:
    # C = sum_t W_t * (U_t A) * (V_t B) ? No.
    # C_{ik} = sum_t (sum_{j} U_{ij,t} A_{ij}) (sum_{j} V_{jk,t} B_{jk}) W_{ki,t}?
    
    # Let's just generate the 49 terms by recursion.
    # Define a function to get the 7 Strassen steps for 2x2.
    # Step s: 
    # A_part: list of (i, j, coeff)
    # B_part: list of (j, k, coeff)
    # C_part: list of (i, k, coeff)
    
    # Strassen 2x2 Steps:
    # 1. M1 = (A00+A11)(B00+B11)
    # A: (0,0,1), (1,1,1)
    # B: (0,0,1), (1,1,1)
    # C: (0,0,1), (0,1,1), (1,0,1), (1,1,1) -> Wait, C is the result of the multiplication.
    # The contribution to C is M1 * C_mask.
    # C_mask for M1:
    # C00 += M1, C01 += M1? No.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So for each macro step s, we have a linear combination of A, linear combination of B,
    # and a linear combination of the resulting C blocks.
    
    # Since the sub-blocks are 2x2, we can use the rank-1 decomposition of 2x2 multiplication
    # for the sub-product.
    
    # Rank-1 decomposition of 2x2 multiplication (Schoolbook, rank 4):
    # T1: A00 B00 -> C00
    # T2: A01 B10 -> C01
    # T3: A10 B01 -> C10
    # T4: A11 B11 -> C11
    
    # Wait, Strassen 2x2 is rank 7, not 4.
    # We should use Strassen 2x2 for the sub-products too?
    # If we use Strassen 2x2 for the sub-products, the rank is 7*7=49.
    # If we use Schoolbook 2x2 (rank 4), the rank is 7*4=28?
    # No, the "Strassen 4x4" algorithm splits into 7 products of 2x2 matrices.
    # If we compute each 2x2 product with rank 4, total rank is 28.
    # But 2x2 multiplication has rank 7 (Strassen). Can we do better?
    # No, 7 is the minimal rank for 2x2.
    # So using Schoolbook (rank 4) is impossible for exact decomposition?
    # Schoolbook is exact. Rank 4 is the rank of the 2x2 multiplication tensor?
    # No, the rank of 2x2 multiplication is 7.
    # Schoolbook uses 4 terms?
    # C00 = A00 B00 + A01 B10
    # C01 = A00 B01 + A01 B11
    # C10 = A10 B00 + A11 B10
    # C11 = A10 B01 + A11 B11
    # This is 8 terms.
    # Strassen reduces it to 7.
    # So the rank of 2x2 is 7.
    
    # So for 4x4, we have 7 products of 2x2 matrices.
    # Each 2x2 product has rank 7.
    # Total rank 49.
    
    # We need to generate the 49 terms.
    
    # Define Strassen 2x2 steps (Macro)
    # Each step has A_coeffs, B_coeffs, C_coeffs
    # A_coeffs: dict (i,j) -> coeff
    # B_coeffs: dict (j,k) -> coeff
    # C_coeffs: dict (i,k) -> coeff
    
    strassen_steps = [
        # 1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (0,1):1, (1,0):1, (1,1):1}), # M1 adds to C00, C01? No.
        # M1 contributes to C00, C11?
        # C00 = M1 + ...
        # C11 = M1 - ...
        # So C_coeffs for M1: {(0,0):1, (1,1):1}
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # 2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}), # M2 -> C10
        # 3
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1}), # M3 -> C01
        # 4
        ({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):1, (1,0):1}), # M4 -> C00, C10
        # 5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):-1, (1,1):-1}), # M5 -> C01, C11?
        # C01 = M3 + M5. So M5 -> C01 (+1).
        # C11 = M1 - M2 + M3 + M6. M5 is not in C11?
        # Formula: C11 = M1 - M2 + M3 + M6.
        # C01 = M3 + M5.
        # So M5 -> C01.
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1}),
        # 6
        ({(1,0):-1, (0,0):1}, {(0,0):1, (0,1):1}, {(1,1):1}), # M6 -> C11
        # 7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}), # M7 -> C00
    ]
    
    # Correct Strassen Coeffs:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    strassen_steps_correct = [
        # (A_coeffs, B_coeffs, C_coeffs)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}), # M1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}), # M2
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1}), # M3
        ({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):1, (1,0):1}), # M4
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1, (1,1):-1}), # M5 (Wait, C11 has -M5? No, C11 = M1-M2+M3+M6. M5 is not in C11.)
        # Let's check C01 = M3 + M5. So M5 -> C01 (+1).
        # C11 does not contain M5.
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,1):1}), # M5
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}), # M6
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}), # M7
    ]
    
    # Now, for each macro step, we need to expand the 2x2 product into 7 rank-1 terms.
    # The 2x2 product is $A_{sub} B_{sub}$.
    # We use the 2x2 Strassen decomposition for this.
    # But the A_sub and B_sub are linear combinations.
    # This is getting complex.
    
    # Alternative: Just use Schoolbook for the 2x2 sub-products?
    # Rank 4 for 2x2.
    # Total rank 7 * 4 = 28.
    # Is Schoolbook exact? Yes.
    # Is rank 4 valid for 2x2?
    # The tensor rank of 2x2 is 7.
    # But we can use any decomposition.
    # Schoolbook uses 8 terms?
    # C00 = A00B00 + A01B10 (2 terms)
    # C01 = A00B01 + A01B11 (2 terms)
    # C10 = A10B00 + A11B10 (2 terms)
    # C11 = A10B01 + A11B11 (2 terms)
    # Total 8 terms.
    # So rank 8 for 2x2 schoolbook.
    # 7 * 8 = 56.
    # Strassen 2x2 is rank 7.
    # 7 * 7 = 49.
    
    # Let's stick to 49.
    
    # We will generate the 49 terms by iterating over the 7 macro steps,
    # and for each, iterating over the 7 base Strassen terms.
    
    # Base Strassen 2x2 terms (Rank 1 in the tensor sense? No, Rank 7 total).
    # We need the 7 rank-1 decompositions of the 2x2 multiplication tensor?
    # No, the 2x2 multiplication tensor has rank 7.
    # The 7 terms of Strassen are the rank-1 components.
    
    # So for each macro step, we have a 2x2 matrix multiplication.
    # We decompose this 2x2 multiplication into 7 rank-1 terms.
    
    # The 7 rank-1 terms for 2x2 Strassen:
    # Term 1: U1 V1 W1
    # ...
    
    # We can hardcode the 49 terms or generate them.
    # Given the complexity, I will generate them programmatically.
    
    # Define the 7 Strassen terms for 2x2 as (U_idx, V_idx, W_idx)
    # U_idx: (i, j)
    # V_idx: (j, k)
    # W_idx: (i, k)
    
    # Term 1: M1 = (A00+A11)(B00+B11)
    # This is not rank 1.
    # The rank-1 terms are the building blocks.
    
    # Actually, the 7 terms of Strassen ARE the rank-1 decompositions.
    # $C = \sum_{t=1}^7 M_t$.
    # $M_t$ is a scalar.
    # $C_{ik} = \sum_t W_{ik,t} M_t$.
    # $M_t = ( \sum_{ij} U_{ij,t} A_{ij} ) ( \sum_{jk} V_{jk,t} B_{jk} )$.
    
    # So we just need the coefficients U, V, W for the 7 terms.
    
    # We already have strassen_steps_correct.
    # For each step t:
    # U_{ij,t} is the coeff of A_{ij}.
    # V_{jk,t} is the coeff of B_{jk}.
    # W_{ik,t} is the coeff of C_{ik}.
    
    # So for the 4x4 case:
    # We have 7 macro steps.
    # Each macro step involves a 2x2 matrix multiplication.
    # Let the macro step be $A_{sub} B_{sub}$.
    # $A_{sub}$ is a 2x2 matrix formed by linear combos of A blocks.
    # $B_{sub}$ is a 2x2 matrix formed by linear combos of B blocks.
    # The result is a 2x2 matrix $C_{sub}$.
    # $C_{sub}$ is added to the final C with weights.
    
    # To decompose $A_{sub} B_{sub}$ into rank-1 terms, we use the 7 Strassen terms for 2x2.
    # Let $s$ be the macro step index (0..6).
    # Let $t$ be the base Strassen term index (0..6).
    
    # The base Strassen term $t$ has:
    # $u_t = \sum_{ij} U^{base}_{ij,t} A_{sub,ij}$
    # $v_t = \sum_{jk} V^{base}_{jk,t} B_{sub,jk}$
    # $w_t = \sum_{ik} W^{base}_{ik,t} C_{sub,ik}$
    
    # Substituting $A_{sub}$ and $B_{sub}$:
    # $u_t = \sum_{ij} U^{base}_{ij,t} (\sum_{I,J} U^{macro}_{I,J,s} A_{I,J})$
    # This is a linear combination of A.
    
    # So we can combine the coefficients.
    
    # We will build the 49 terms.
    
    # First, get the base 2x2 Strassen coefficients.
    # We use the same strassen_steps_correct for the base.
    
    base_steps = strassen_steps_correct
    
    for s in range(7):
        A_macro = base_steps[s][0]
        B_macro = base_steps[s][1]
        C_macro = base_steps[s][2]
        
        for t in range(7):
            U_base = base_steps[t][0]
            V_base = base_steps[t][1]
            W_base = base_steps[t][2]
            
            u_vec = [0] * (n * m)
            v_vec = [0] * (m * p)
            w_vec = [0] * (p * n)
            
            # Expand u_vec
            # u_t = sum_{ij} U_base[ij,t] * A_sub[ij]
            # A_sub[ij] = sum_{I,J} A_macro[ij] * A[I,J]
            # So coeff of A[I,J] is sum_{ij} U_base[ij,t] * A_macro[ij]
            
            # Note: A_macro is dict (i,j) -> coeff.
            # U_base is dict (i,j) -> coeff.
            
            # We need to sum over the 2x2 indices of the sub-block.
            # i,j in 0,1.
            
            # For each I,J in 0..3:
            # I = 2*i + i_local
            # J = 2*j + j_local
            
            for i in range(2):
                for j in range(2):
                    if (i, j) in U_base:
                        coeff_u = U_base[(i, j)]
                        if (i, j) in A_macro:
                            coeff_macro = A_macro[(i, j)]
                            # Total coeff for A[I,J] where I=2i+il, J=2j+jl?
                            # No, A_sub[i,j] is the block A[2i:2i+2, 2j:2j+2].
                            # So A_sub[i,j] is a 2x2 matrix.
                            # Wait, my A_macro is for the 2x2 blocks.
                            # A_sub is a 2x2 matrix of blocks?
                            # No, in 4x4 Strassen, A is split into 2x2 blocks of size 2x2.
                            # So A_sub is a 2x2 matrix of 2x2 blocks.
                            # The multiplication is block multiplication.
                            # $A_{sub} B_{sub}$ is a 2x2 matrix of 2x2 blocks.
                            # Each block product is a 2x2 matrix multiplication.
                            
                            # So for each block position (i,j) in the 2x2 block matrix:
                            # The term is $A_{block, i, j} \times B_{block, j, k}$.
                            # This is a 2x2 matrix multiplication.
                            
                            # So we need to iterate over the block positions (i,j) and (j,k).
                            
            # This is getting too nested.
            
            # Simpler approach:
            # Just generate the 49 terms by direct index mapping.
            # I will hardcode the 49 terms or use a simpler generator.
            
    # Given the time, I will use the 68 rank decomposition which is easier to implement correctly.
    # 49 + 4 + 4 + 1 + 4 + 1 + 4 + 1 = 68.
    
    # Clear terms
    terms = []
    
    # 1. A00 B00 (4x4 Strassen, Rank 49)
    # We will use the 2x2 Strassen for the 4x4 blocks.
    # This requires implementing the 49 terms.
    # I will use a library or simple loop.
    
    # For now, I will use the 68 rank solution with Schoolbook for 4x4?
    # Schoolbook 4x4 is 16 terms? No, 4*4=16.
    # 16 + 4 + 4 + 1 + 4 + 1 + 4 + 1 = 35.
    # Wait, Schoolbook 4x4 rank is 16.
    # 16 + 19 = 35.
    # 35 is better than 68.
    
    # Let's use Schoolbook for the 4x4 block.
    # Rank 16.
    
    for i in range(4):
        for j in range(4):
            u = [0] * 25
            v = [0] * 25
            w = [0] * 25
            u[i * 5 + j] = 1
            v[j * 5 + k] = 1 # k is 0..3
            w[k * 5 + i] = 1
            terms.append((u, v, w))
            
    # 2. A01 B10 (4x1 * 1x4, Rank 4)
    for c in range(4):
        u = [0] * 25
        v = [0] * 25
        w = [0] * 25
        u[c * 5 + 4] = 1
        v[4 * 5 + c] = 1 # j=4, k=c
        w[c * 5 + c] = 1 # k=c, i=c
        terms.append((u, v, w))
        
    # 3. A00 B01 (4x4 * 4x1, Rank 4)
    for c in range(4):
        u = [0] * 25
        v = [0] * 25
        w = [0] * 25
        u[i * 5 + c] = 1 # j=c
        v[c * 5 + 4] = 1 # j=c, k=4
        w[4 * 5 + i] = 1 # k=4, i
        terms.append((u, v, w))
        
    # 4. A01 B11 (4x1 * 1x1, Rank 1)
    u = [0] * 25
    v = [0] * 25
    w = [0] * 25
    u[i * 5 + 4] = 1
    v[4 * 5 + 4] = 1
    w[4 * 5 + i] = 1
    terms.append((u, v, w))
    
    # 5. A10 B00 (1x4 * 4x4, Rank 4)
    for c in range(4):
        u = [0] * 25
        v = [0] * 25
        w = [0] * 25
        u[4 * 5 + c] = 1
        v[c * 5 + k] = 1
        w[k * 5 + 4] = 1
        terms.append((u, v, w))
        
    # 6. A11 B10 (1x1 * 1x4, Rank 1)
    u = [0] * 25
    v = [0] * 25
    w = [0] * 25
    u[4 * 5 + 4] = 1
    v[4 * 5 + k] = 1
    w[k * 5 + 4] = 1
    terms.append((u, v, w))
    
    # 7. A10 B01 (1x4 * 4x1, Rank 4)
    for c in range(4):
        u = [0] * 25
        v = [0] * 25
        w = [0] * 25
        u[4 * 5 + c] = 1
        v[c * 5 + 4] = 1
        w[4 * 5 + 4] = 1
        terms.append((u, v, w))
        
    # 8. A11 B11 (1x1 * 1x1, Rank 1)
    u = [0] * 25
    v = [0] * 25
    w = [0] * 25
    u[4 * 5 + 4] = 1
    v[4 * 5 + 4] = 1
    w[4 * 5 + 4] = 1
    terms.append((u, v, w))
    
    # Total terms: 16 + 4 + 4 + 1 + 4 + 1 + 4 + 1 = 35.
    
    # Build U, V, W
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(terms):
        for i in range(n * m):
            if u[i] != 0:
                U[i][c] = u[i]
        for j in range(m * p):
            if v[j] != 0:
                V[j][c] = v[j]
        for k in range(p * n):
            if w[k] != 0:
                W[k][c] = w[k]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
