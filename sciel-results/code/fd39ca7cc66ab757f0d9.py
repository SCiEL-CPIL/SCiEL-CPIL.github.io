"""Matrix multiplication 3x3x8 with Strassen-optimized decomposition."""
DIMENSIONS = [3, 3, 8]

import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 8
    
    U_rows = n * m  # 9
    V_rows = m * p  # 24
    W_rows = p * n  # 24
    
    def strassen_terms(i0, i1, j0, j1, k0, k1):
        """Return list of 7 dicts, each with 'U', 'V', 'W' keys mapping index->coeff."""
        terms = []
        
        # M1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
        # C[i0,k0] += M1, C[i1,k1] += M1
        terms.append({
            'U': {(i0*m+j0): 1, (i1*m+j1): 1},
            'V': {(j0*p+k0): 1, (j1*p+k1): 1},
            'W': {(k0*n+i0): 1, (k1*n+i1): 1}
        })
        
        # M2 = (A[i1,j0]+A[i1,j1]) * B[j0,k0]
        # C[i1,k0] += M2, C[i1,k1] -= M2
        terms.append({
            'U': {(i1*m+j0): 1, (i1*m+j1): 1},
            'V': {(j0*p+k0): 1},
            'W': {(k0*n+i1): 1, (k1*n+i1): -1}
        })
        
        # M3 = A[i0,j0] * (B[j0,k1]-B[j1,k1])
        # C[i0,k1] += M3, C[i1,k1] += M3
        terms.append({
            'U': {(i0*m+j0): 1},
            'V': {(j0*p+k1): 1, (j1*p+k1): -1},
            'W': {(k1*n+i0): 1, (k1*n+i1): 1}
        })
        
        # M4 = A[i1,j1] * (B[j1,k0]-B[j0,k0])
        # C[i0,k0] += M4, C[i1,k0] += M4
        terms.append({
            'U': {(i1*m+j1): 1},
            'V': {(j1*p+k0): 1, (j0*p+k0): -1},
            'W': {(k0*n+i0): 1, (k0*n+i1): 1}
        })
        
        # M5 = (A[i0,j0]+A[i0,j1]) * B[j1,k1]
        # C[i0,k0] -= M5, C[i0,k1] += M5
        terms.append({
            'U': {(i0*m+j0): 1, (i0*m+j1): 1},
            'V': {(j1*p+k1): 1},
            'W': {(k0*n+i0): -1, (k1*n+i0): 1}
        })
        
        # M6 = (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1])
        # C[i1,k1] += M6
        terms.append({
            'U': {(i1*m+j0): 1, (i0*m+j0): -1},
            'V': {(j0*p+k0): 1, (j0*p+k1): 1},
            'W': {(k1*n+i1): 1}
        })
        
        # M7 = (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1])
        # C[i0,k0] += M7
        terms.append({
            'U': {(i0*m+j1): 1, (i1*m+j1): -1},
            'V': {(j1*p+k0): 1, (j1*p+k1): 1},
            'W': {(k0*n+i0): 1}
        })
        
        return terms
    
    # Track which (i,j,k) triples are covered by Strassen
    strassen_covered = set()
    
    # Apply Strassen for 4 blocks
    k_pairs = [(0,1), (2,3), (4,5), (6,7)]
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    for k0, k1 in k_pairs:
        i0, i1 = 0, 1
        j0, j1 = 0, 1
        terms = strassen_terms(i0, i1, j0, j1, k0, k1)
        for term in terms:
            U_cols.append(term['U'])
            V_cols.append(term['V'])
            W_cols.append(term['W'])
        for i in [i0, i1]:
            for j in [j0, j1]:
                for k in [k0, k1]:
                    strassen_covered.add((i,j,k))
    
    # Add schoolbook terms for uncovered triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i,j,k) not in strassen_covered:
                    U_cols.append({i*m+j: 1})
                    V_cols.append({j*p+k: 1})
                    W_cols.append({k*n+i: 1})
    
    rank = len(U_cols)
    
    # Build final matrices
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    for col in range(rank):
        for row_idx, coeff in U_cols[col].items():
            U[row_idx][col] = coeff
        for row_idx, coeff in V_cols[col].items():
            V[row_idx][col] = coeff
        for row_idx, coeff in W_cols[col].items():
            W[row_idx][col] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
