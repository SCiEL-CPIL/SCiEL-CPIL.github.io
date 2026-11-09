import time
import sys

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Split 5 into 2 + 2 + 1
    splits = [2, 2, 1]
    num_blocks = len(splits)
    
    # Precompute local decompositions for each block size
    # We need a function to get U, V, W for a block multiplication
    # of size ni x nj x nk
    
    def get_block_decomp(ni, nj, nk):
        """
        Returns list of (u, v, w) where u, v, w are dicts mapping local index to value.
        Local index for U (A block) is (r, c) where r in 0..ni-1, c in 0..nj-1.
        Local index for V (B block) is (r, c) where r in 0..nj-1, c in 0..nk-1.
        Local index for W (C block) is (r, c) where r in 0..ni-1, c in 0..nk-1.
        """
        terms = []
        if ni == 2 and nj == 2 and nk == 2:
            # Strassen
            # M1 = (A11+A22)(B11+B22)
            u1 = {(0,0): 1, (1,1): 1}
            v1 = {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1}
            # Wait, Strassen combines B elements too.
            # B11 is (0,0), B22 is (1,1).
            # v1 should correspond to (B11+B22).
            # v1 = {(0,0): 1, (1,1): 1}
            v1 = {(0,0): 1, (1,1): 1}
            w1 = {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1} # No, W is the result.
            # The term is M1.
            # C11 += M1, C12 += M1? No.
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # We need to define the 7 multiplications M1..M7.
            # Each M is a product of a linear combo of A and a linear combo of B.
            # The result is added to specific parts of C.
            
            # M1: A=(1,0,0,1), B=(1,0,0,1) -> C contribution: (1,1,1,1) with coeff 1?
            # No, M1 is a scalar.
            # The tensor term is u ⊗ v ⊗ w.
            # u is the coefficient vector for A.
            # v is the coefficient vector for B.
            # w is the coefficient vector for C.
            
            # M1:
            u = {(0,0): 1, (1,1): 1}
            v = {(0,0): 1, (1,1): 1}
            w = {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1} # This is wrong.
            # The output of M1 is added to C11, C12, C21, C22?
            # No, M1 is a scalar.
            # C11 gets +M1.
            # C12 gets +M1? No.
            # Let's check Strassen formulas again.
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            
            # So M1 contributes to C11 (coeff 1) and C22 (coeff 1).
            # So w for M1 is {(0,0): 1, (1,1): 1}.
            
            terms.append((
                {(0,0): 1, (1,1): 1}, # u
                {(0,0): 1, (1,1): 1}, # v
                {(0,0): 1, (1,1): 1}  # w
            ))
            
            # M2: (A21+A22)B11
            u = {(1,0): 1, (1,1): 1}
            v = {(0,0): 1}
            w = {(1,0): 1} # C21 += M2
            terms.append((u, v, w))
            
            # M3: A11(B12-B22)
            u = {(0,0): 1}
            v = {(0,1): 1, (1,1): -1}
            w = {(0,1): 1, (1,1): 1} # C12 += M3, C22 += M3
            terms.append((u, v, w))
            
            # M4: A22(B21-B11)
            u = {(1,1): 1}
            v = {(1,0): 1, (0,0): -1}
            w = {(0,0): 1, (1,0): 1} # C11 += M4, C21 += M4
            terms.append((u, v, w))
            
            # M5: (A11+A12)B22
            u = {(0,0): 1, (0,1): 1}
            v = {(1,1): 1}
            w = {(0,0): -1, (0,1): 1} # C11 -= M5, C12 += M5
            terms.append((u, v, w))
            
            # M6: (A21-A11)(B11+B12)
            u = {(1,0): 1, (0,0): -1}
            v = {(0,0): 1, (0,1): 1}
            w = {(1,1): 1} # C22 += M6
            terms.append((u, v, w))
            
            # M7: (A12-A22)(B21+B22)
            u = {(0,1): 1, (1,1): -1}
            v = {(1,0): 1, (1,1): 1}
            w = {(0,0): 1} # C11 += M7
            terms.append((u, v, w))
            
        else:
            # Schoolbook
            for r in range(ni):
                for c in range(nj):
                    for k in range(nk):
                        u = {(r, c): 1}
                        v = {(c, k): 1}
                        w = {(r, k): 1}
                        terms.append((u, v, w))
                        
        return terms

    # Build global U, V, W
    # We don't know the total rank in advance, so we can use a list of terms and then flatten.
    # Or we can calculate the total rank first.
    
    total_rank = 0
    for i in range(num_blocks):
        for j in range(num_blocks):
            for k in range(num_blocks):
                ni = splits[i]
                nj = splits[j]
                nk = splits[k]
                terms = get_block_decomp(ni, nj, nk)
                total_rank += len(terms)
                
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i in range(num_blocks):
        row_start_A = sum(splits[:i])
        for j in range(num_blocks):
            col_start_A = sum(splits[:j])
            for k in range(num_blocks):
                row_start_B = sum(splits[:j])
                col_start_B = sum(splits[:k])
                row_start_C = sum(splits[:i])
                col_start_C = sum(splits[:k])
                
                ni = splits[i]
                nj = splits[j]
                nk = splits[k]
                terms = get_block_decomp(ni, nj, nk)
                
                for u, v, w in terms:
                    for (r, c), val in u.items():
                        global_row = row_start_A + r
                        global_col = col_start_A + c
                        U[global_row * m + global_col][col] = val
                        
                    for (r, c), val in v.items():
                        global_row = row_start_B + r
                        global_col = col_start_B + c
                        V[global_row * p + global_col][col] = val
                        
                    for (r, c), val in w.items():
                        global_row = row_start_C + r
                        global_col = col_start_C + c
                        W[global_row * n + global_col][col] = val
                        
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
