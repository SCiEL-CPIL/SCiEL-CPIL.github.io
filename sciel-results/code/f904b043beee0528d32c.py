"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

import time
import numpy as np

# EVOLVE-BLOCK-START

def _merge_dicts(*dicts):
    """Merge list of dicts of index->coeff, summing coefficients."""
    result = {}
    for d in dicts:
        for idx, coeff in d.items():
            result[idx] = result.get(idx, 0) + coeff
    return {k: v for k, v in result.items() if v != 0}

def _negate_dict(d):
    return {k: -v for k, v in d.items()}

def _strassen44_terms():
    """
    Return list of 49 (a_coefs, b_coefs, c_coefs) dicts for 4x4x4 matrix multiplication
    using recursive Strassen (2 levels of 7 mults each).
    
    Each dict maps (row, col) index to integer coefficient.
    For A: (i,j) where i=row, j=col
    For B: (j,k) where j=row, k=col  
    For C: (k,i) where k=row, i=col (trace convention W[k*n+i])
    """
    terms = []
    
    # 7 block-level products for 4x4x4 using 2x2 block Strassen
    # Block positions: (br, bc) where br in {0,1}, bc in {0,1}
    # A block (br,bc) covers A[br*2+di, bc*2+dj] for di,dj in {0,1}
    # B block (br,bc) covers B[br*2+di, bc*2+dj] for di,dj in {0,1}
    # C block (br,bc) covers C[br*2+di, bc*2+dj] for di,dj in {0,1}
    
    # Strassen formulas at the block level:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B00)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M3 - M4 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M5 - M6
    
    # For each Mi, it's a 2x2 matrix product. Within each, 7 scalar mults.
    # The 2x2 result P has entries P[0,0],P[0,1],P[1,0],P[1,1]
    # These map to C block entries as specified.
    
    # Block product definitions:
    # (a_blocks, b_blocks, c_map)
    # a_blocks: dict from (br,bc) to sign for A
    # b_blocks: dict from (br,bc) to sign for B
    # c_map: dict from (pr,pc) [P position] to list of (cbr,cbc,sign)
    #   where P[pr,pc] contributes sign to C block (cbr,cbc)
    
    block_products = [
        # M1: A=(A00+A11), B=(B00+B11)
        # P[0,0]->C00:+1, P[0,1]->C00:+1, P[1,0]->C00:+1, P[1,1]->C00:+1
        # P[0,0]->C11:+1, P[0,1]->C11:+1, P[1,0]->C11:+1, P[1,1]->C11:+1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1},
         {0: [(0,0,1),(1,1,1)], 1: [(0,0,1),(1,1,1)], 2: [(0,0,1),(1,1,1)], 3: [(0,0,1),(1,1,1)]}),
        # M2: A=(A10+A11), B=B00
        # P[0,0]->C10:+1, P[0,1]->C10:+1, P[1,0]->C10:+1, P[1,1]->C10:+1
        # P[0,0]->C11:-1, P[0,1]->C11:-1, P[1,0]->C11:-1, P[1,1]->C11:-1
        ({(1,0):1, (1,1):1}, {(0,0):1},
         {0: [(1,0,1),(1,1,-1)], 1: [(1,0,1),(1,1,-1)], 2: [(1,0,1),(1,1,-1)], 3: [(1,0,1),(1,1,-1)]}),
        # M3: A=A00, B=(B10-B00)
        # P[0,0]->C00:-1, P[0,1]->C00:-1, P[1,0]->C00:-1, P[1,1]->C00:-1
        # P[0,0]->C01:+1, P[0,1]->C01:+1, P[1,0]->C01:+1, P[1,1]->C01:+1
        ({(0,0):1}, {(1,0):1, (0,0):-1},
         {0: [(0,0,-1),(0,1,1)], 1: [(0,0,-1),(0,1,1)], 2: [(0,0,-1),(0,1,1)], 3: [(0,0,-1),(0,1,1)]}),
        # M4: A=A11, B=(B01-B10)
        # P[0,0]->C00:-1, P[0,1]->C00:-1, P[1,0]->C00:-1, P[1,1]->C00:-1
        # P[0,0]->C10:+1, P[0,1]->C10:+1, P[1,0]->C10:+1, P[1,1]->C10:+1
        ({(1,1):1}, {(0,1):1, (1,0):-1},
         {0: [(0,0,-1),(1,0,1)], 1: [(0,0,-1),(1,0,1)], 2: [(0,0,-1),(1,0,1)], 3: [(0,0,-1),(1,0,1)]}),
        # M5: A=(A00+A01), B=B11
        # P[0,0]->C01:+1, P[0,1]->C01:+1, P[1,0]->C01:+1, P[1,1]->C01:+1
        # P[0,0]->C11:+1, P[0,1]->C11:+1, P[1,0]->C11:+1, P[1,1]->C11:+1
        ({(0,0):1, (0,1):1}, {(1,1):1},
         {0: [(0,1,1),(1,1,1)], 1: [(0,1,1),(1,1,1)], 2: [(0,1,1),(1,1,1)], 3: [(0,1,1),(1,1,1)]}),
        # M6: A=(A10-A00), B=(B00+B01)
        # P[0,0]->C11:-1, P[0,1]->C11:-1, P[1,0]->C11:-1, P[1,1]->C11:-1
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1},
         {0: [(1,1,-1)], 1: [(1,1,-1)], 2: [(1,1,-1)], 3: [(1,1,-1)]}),
        # M7: A=(A01-A11), B=(B10+B11)
        # P[0,0]->C00:+1, P[0,1]->C00:+1, P[1,0]->C00:+1, P[1,1]->C00:+1
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1},
         {0: [(0,0,1)], 1: [(0,0,1)], 2: [(0,0,1)], 3: [(0,0,1)]}),
    ]
    
    # Strassen 2x2 scalar products within a block product:
    # X[0,0]=x0, X[0,1]=x1, X[1,0]=x2, X[1,1]=x3
    # Y[0,0]=y0, Y[0,1]=y1, Y[1,0]=y2, Y[1,1]=y3
    # m1=(x0+x3)(y0+y3), m2=(x2+x3)(y0), m3=(x0)(y2-y0), m4=(x3)(y1-y2)
    # m5=(x0+x1)(y3), m6=(x2-x0)(y0+y1), m7=(x1-x3)(y2+y3)
    # P[0,0]=m1-m3-m4+m7, P[0,1]=m3+m5, P[1,0]=m2+m4, P[1,1]=m1-m2+m5-m6
    
    # For each scalar product k (0-6), which P positions it contributes to:
    s_c_contribs = {
        0: {0: 1, 3: 1},    # m1: P[0,0]+1, P[1,1]+1
        1: {2: 1, 3: -1},   # m2: P[1,0]+1, P[1,1]-1
        2: {0: -1, 1: 1},   # m3: P[0,0]-1, P[0,1]+1
        3: {0: -1, 2: 1},   # m4: P[0,0]-1, P[1,0]+1
        4: {1: 1, 3: 1},    # m5: P[0,1]+1, P[1,1]+1
        5: {3: -1},         # m6: P[1,1]-1
        6: {0: 1},          # m7: P[0,0]+1
    }
    
    for a_blocks, b_blocks, c_map in block_products:
        # Build X and Y coefficient matrices (2x2, each entry is dict of (idx)->coeff)
        X = [[{} for _ in range(2)] for _ in range(2)]
        for (br, bc), sign in a_blocks.items():
            for di in range(2):
                for dj in range(2):
                    idx = (br*2+di, bc*2+dj)
                    X[di][dj][idx] = sign
        
        Y = [[{} for _ in range(2)] for _ in range(2)]
        for (br, bc), sign in b_blocks.items():
            for di in range(2):
                for dj in range(2):
                    idx = (br*2+di, bc*2+dj)
                    Y[di][dj][idx] = sign
        
        # 7 scalar Strassen products
        scalar_mults = [
            (_merge_dicts(X[0][0], X[1][1]), _merge_dicts(Y[0][0], Y[1][1])),           # m1
            (_merge_dicts(X[1][0], X[1][1]), _merge_dicts(Y[0][0])),                     # m2
            (_merge_dicts(X[0][0]), _merge_dicts(Y[1][0], _negate_dict(Y[0][0]))),       # m3
            (_merge_dicts(X[1][1]), _merge_dicts(Y[0][1], _negate_dict(Y[1][0]))),       # m4
            (_merge_dicts(X[0][0], X[0][1]), _merge_dicts(Y[1][1])),                     # m5
            (_merge_dicts(X[1][0], _negate_dict(X[0][0])), _merge_dicts(Y[0][0], Y[0][1])),  # m6
            (_merge_dicts(X[0][1], _negate_dict(X[1][1])), _merge_dicts(Y[1][0], Y[1][1])),  # m7
        ]
        
        for k, (a_coefs, b_coefs) in enumerate(scalar_mults):
            # Determine C coefficients for this scalar product
            c_coefs = {}
            for p_pos, s_sign in s_c_contribs[k].items():
                pr, pc = (p_pos // 2, p_pos % 2)  # 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
                for (cbr, cbc, c_sign) in c_map[p_pos]:
                    # C entry at (cbr*2+pr, cbc*2+pc)
                    c_idx = (cbr*2+pr, cbc*2+pc)
                    total_sign = s_sign * c_sign
                    c_coefs[c_idx] = c_coefs.get(c_idx, 0) + total_sign
            
            c_coefs = {k: v for k, v in c_coefs.items() if v != 0}
            if a_coefs and b_coefs and c_coefs:
                terms.append((a_coefs, b_coefs, c_coefs))
    
    return terms

def construct():
    """
    Matrix multiplication 4x4 by 4x9.
    Returns U (16 x rank), V (36 x rank), W (36 x rank) with integer entries.
    """
    start_time = time.time()
    n, m, p = DIMENSIONS  # 4, 4, 9
    DEADLINE = start_time + 110  # safety margin before 120s limit
    
    # Strategy: Use Strassen-49 for 4x4x4 matrix multiplication on two column blocks of B
    # B is 4x9, split into B1 (4x4), B2 (4x4), B3 (4x1)
    # C = A*B1 + A*B2 + A*B3
    # rank(A*B1) using Strassen-49: 49
    # rank(A*B2) using Strassen-49: 49
    # rank(A*B3) schoolbook: 4*4*1 = 16
    # Total: 114
    
    # Generate Strassen-49 terms for 4x4x4
    strassen_terms = _strassen44_terms()
    
    if len(strassen_terms) != 49:
        # Fallback to schoolbook if Strassen generation fails
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Build the full decomposition
    # For B1 (columns 0-3): use Strassen-49 terms, B indices map to B[0:4, 0:4]
    # For B2 (columns 4-7): use Strassen-49 terms, B indices map to B[0:4, 4:8]
    # For B3 (column 8): schoolbook, B indices map to B[0:4, 8:9]
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    # Strassen terms for B1 (columns 0-3)
    for a_coefs, b_coefs, c_coefs in strassen_terms:
        U_row = [0] * 16
        V_row = [0] * 36
        W_row = [0] * 36
        
        for (i, j), val in a_coefs.items():
            U_row[i * m + j] = val
        
        for (j, k), val in b_coefs.items():
            # B1: k in 0-3
            V_row[j * p + k] = val
        
        for (k, i), val in c_coefs.items():
            # C: W[k*n+i]
            W_row[k * n + i] = val
        
        U_terms.append(U_row)
        V_terms.append(V_row)
        W_terms.append(W_row)
    
    # Strassen terms for B2 (columns 4-7)
    for a_coefs, b_coefs, c_coefs in strassen_terms:
        U_row = [0] * 16
        V_row = [0] * 36
        W_row = [0] * 36
        
        for (i, j), val in a_coefs.items():
            U_row[i * m + j] = val
        
        for (j, k), val in b_coefs.items():
            # B2: k in 4-7
            V_row[j * p + (k + 4)] = val
        
        for (k, i), val in c_coefs.items():
            W_row[k * n + i] = val
        
        U_terms.append(U_row)
        V_terms.append(V_row)
        W_terms.append(W_row)
    
    # Schoolbook terms for B3 (column 8)
    for i in range(n):
        for j in range(m):
            for k in range(1):  # k=0 for the single column
                U_row = [0] * 16
                V_row = [0] * 36
                W_row = [0] * 36
                
                U_row[i * m + j] = 1
                V_row[j * p + (8 + k)] = 1
                W_row[(8 + k) * n + i] = 1
                
                U_terms.append(U_row)
                V_terms.append(V_row)
                W_terms.append(W_row)
    
    rank = len(U_terms)
    
    # Verify correctness with random test
    if time.time() < DEADLINE:
        np.random.seed(42)
        A_test = np.random.randn(n, m).astype(float)
        B_test = np.random.randn(m, p).astype(float)
        
        U_mat = np.array(U_terms, dtype=float)
        V_mat = np.array(V_terms, dtype=float)
        W_mat = np.array(W_terms, dtype=float)
        
        # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        # Actually, the decomposition should satisfy:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # And C[i,k] = sum_r (sum_j U[i*m+j,r] * A[i,j]) * (sum_k V[j*p+k,r] * B[j,k]) * W[k*n+i,r]
        # Wait, that's not quite right. Let me think about the tensor decomposition.
        
        # The tensor T_{i,j,k} = 1 for all i,j,k (matrix multiplication tensor)
        # We need: T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Let's verify this
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        
        if max_err > 1e-6:
            # Fallback to schoolbook if verification fails
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][column] = 1
                        V[j * p + k][column] = 1
                        W[k * n + i][column] = 1
                        column += 1
            return {"U": U, "V": V, "W": W}
    
    return {"U": U_terms, "V": V_terms, "W": W_terms}

# EVOLVE-BLOCK-END
