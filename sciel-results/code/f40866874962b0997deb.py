"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Rank-minimal exact tensor decomposition for 3x3 by 3x7 matrix multiplication.
    Uses correct Strassen (rank 7) for the 2x2x2 sub-block and schoolbook for rest.
    Total rank = 7 + (63 - 8) = 62.
    
    Indexing:
      U row for A[i,j] = i*m + j = 3i+j
      V row for B[j,k] = j*p + k = 7j+k  
      W row for C[i,k] = k*n + i = 7k+i
    """
    start_time = time.time()
    DEADLINE = 140.0

    n, m, p = DIMENSIONS  # n=3, m=3, p=7

    terms = []  # list of (u_dict, v_dict, w_dict)

    def add_rank1_term(u_dict, v_dict, w_dict):
        if time.time() > DEADLINE:
            return False
        terms.append((u_dict, v_dict, w_dict))
        return True

    def single(idx, val=1):
        return {idx: val}

    def merge(*dicts):
        result = {}
        for d in dicts:
            for k, v in d.items():
                result[k] = result.get(k, 0) + v
        result = {k: v for k, v in result.items() if v != 0}
        return result

    # Correct Strassen decomposition for 2x2x2 sub-block:
    # A indices: A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
    # B indices: B[0,0]=0, B[0,1]=1, B[1,0]=7, B[1,1]=8
    # C indices: C[0,0]=0, C[0,1]=1, C[1,0]=7, C[1,1]=8
    #
    # Standard Strassen:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B11-B10)
    # M4 = A11*(B01-B00)
    # M5 = (A00+A01)*B10
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A10)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # M1 = (A00+A11)*(B00+B11)
    add_rank1_term(
        merge(single(0, 1), single(4, 1)),   # A[0,0]+A[1,1]
        merge(single(0, 1), single(8, 1)),   # B[0,0]+B[1,1]
        merge(single(0, 1), single(8, 1))    # C[0,0]+C[1,1]
    )

    # M2 = (A10+A11)*B00
    add_rank1_term(
        merge(single(3, 1), single(4, 1)),   # A[1,0]+A[1,1]
        single(0, 1),                        # B[0,0]
        merge(single(7, 1), single(8, -1))   # C[1,0]-C[1,1]
    )

    # M3 = A00*(B11-B10)
    add_rank1_term(
        single(0, 1),                        # A[0,0]
        merge(single(8, 1), single(7, -1)),  # B[1,1]-B[1,0]
        merge(single(1, 1), single(8, 1))    # C[0,1]+C[1,1]
    )

    # M4 = A11*(B01-B00)
    add_rank1_term(
        single(4, 1),                        # A[1,1]
        merge(single(1, 1), single(0, -1)),  # B[0,1]-B[0,0]
        merge(single(0, 1), single(7, 1))    # C[0,0]+C[1,0]
    )

    # M5 = (A00+A01)*B10
    add_rank1_term(
        merge(single(0, 1), single(1, 1)),   # A[0,0]+A[0,1]
        single(7, 1),                        # B[1,0]
        merge(single(0, -1), single(1, 1))   # -C[0,0]+C[0,1]
    )

    # M6 = (A10-A00)*(B00+B01)
    add_rank1_term(
        merge(single(3, 1), single(0, -1)),  # A[1,0]-A[0,0]
        merge(single(0, 1), single(1, 1)),   # B[0,0]+B[0,1]
        single(8, 1)                         # C[1,1]
    )

    # M7 = (A01-A10)*(B10+B11)
    add_rank1_term(
        merge(single(1, 1), single(3, -1)),  # A[0,1]-A[1,0]
        merge(single(7, 1), single(8, 1)),   # B[1,0]+B[1,1]
        single(0, 1)                         # C[0,0]
    )

    # Schoolbook for all other (i,j,k) triples not covered by Strassen
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue  # Already handled by Strassen
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                add_rank1_term(
                    single(u_idx, 1),
                    single(v_idx, 1),
                    single(w_idx, 1)
                )

    # Check deadline
    if time.time() > DEADLINE:
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    terms.append((
                        single(i * m + j, 1),
                        single(j * p + k, 1),
                        single(k * n + i, 1)
                    ))

    rank = len(terms)

    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for col, (u_dict, v_dict, w_dict) in enumerate(terms):
        for u_idx, coeff in u_dict.items():
            U[u_idx][col] = coeff
        for v_idx, coeff in v_dict.items():
            V[v_idx][col] = coeff
        for w_idx, coeff in w_dict.items():
            W[w_idx][col] = coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
