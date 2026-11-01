"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a Strassen-based decomposition for 2x2 by 2x10 matrix multiplication.
    Rank = 35 (5 column pairs, each using rank-7 Strassen).
    """
    n, m, p = DIMENSIONS
    # 5 column pairs (k, k+5) for k=0..4
    num_pairs = p // 2  # 5
    rank = 7 * num_pairs  # 35

    # U: shape (n*m, rank) = (4, 35)
    U = [[0] * rank for _ in range(n * m)]
    # V: shape (m*p, rank) = (20, 35)
    V = [[0] * rank for _ in range(m * p)]
    # W: shape (p*n, rank) = (20, 35)
    W = [[0] * rank for _ in range(p * n)]

    # Strassen's 7 multiplications for 2x2 * 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B22-B21)
    # M4 = A22(B11-B12)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B12+B22)
    # M7 = (A12-A22)(B21+B11)
    #
    # Reconstruction:
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    #
    # In our notation:
    # A is 2x2: A[i][j] where i=0,1 (row), j=0,1 (col)
    # B is 2x2: B[j][k] where j=0,1 (row), k=0,1 (col)
    # C is 2x2: C[i][k] where i=0,1 (row), k=0,1 (col)
    #
    # U indices: i*m + j = i*2 + j
    # V indices: j*p + k = j*10 + k
    # W indices: k*n + i = k*2 + i
    #
    # For a column pair (k0, k0+1) where k0 is even (0,2,4,6,8):
    # B11 = B[0][k0], B12 = B[0][k0+1]
    # B21 = B[1][k0], B22 = B[1][k0+1]

    # Define the 7 Strassen terms for each column pair
    # Each term has coefficients for U, V, W
    # Term t: U coefficients for (i,j) pairs, V coefficients for (j,k) pairs, W coefficients for (k,i) pairs

    # For term M1 = (A11+A22)(B11+B22):
    # U: A11 -> (0,0)=1, A22 -> (1,1)=1
    # V: B11 -> (0,k0)=1, B22 -> (1,k0+1)=1
    # W: C11 gets +1, C22 gets +1
    # C11 = (0,0), C22 = (1,1) in (i,k) -> W index k*n+i

    # Let's build each term explicitly
    strassen_terms = []

    # M1 = (A11+A22)(B11+B22)
    # U: (0,0)=1, (1,1)=1
    # V: (0,k0)=1, (1,k0+1)=1
    # W: C11=(0,0) -> W[0*2+0]=W[0] += 1, C22=(1,1) -> W[1*2+1]=W[3] += 1
    strassen_terms.append({
        'U': {0: 1, 3: 1},  # (0,0) and (1,1)
        'V': {0: 1, 11: 1},  # (0,k0) and (1,k0+1) -> will be adjusted per pair
        'W': {0: 1, 3: 1},  # C11 and C22
    })

    # M2 = (A21+A22)B11
    # U: (1,0)=1, (1,1)=1
    # V: (0,k0)=1
    # W: C21=(1,0) -> W[0*2+1]=W[1] += 1
    strassen_terms.append({
        'U': {2: 1, 3: 1},  # (1,0) and (1,1)
        'V': {0: 1},  # (0,k0)
        'W': {1: 1},  # C21
    })

    # M3 = A11(B22-B21)
    # U: (0,0)=1
    # V: (1,k0+1)=1, (1,k0)=-1
    # W: C12=(0,1) -> W[1*2+0]=W[2] += 1
    strassen_terms.append({
        'U': {0: 1},  # (0,0)
        'V': {11: 1, 10: -1},  # (1,k0+1) and (1,k0)
        'W': {2: 1},  # C12
    })

    # M4 = A22(B11-B12)
    # U: (1,1)=1
    # V: (0,k0)=1, (0,k0+1)=-1
    # W: C11=(0,0) -> W[0] += 1, C21=(1,0) -> W[1] += 1
    strassen_terms.append({
        'U': {3: 1},  # (1,1)
        'V': {0: 1, 1: -1},  # (0,k0) and (0,k0+1)
        'W': {0: 1, 1: 1},  # C11 and C21
    })

    # M5 = (A11+A12)B22
    # U: (0,0)=1, (0,1)=1
    # V: (1,k0+1)=1
    # W: C11=(0,0) -> W[0] -= 1, C12=(0,1) -> W[2] += 1
    strassen_terms.append({
        'U': {0: 1, 1: 1},  # (0,0) and (0,1)
        'V': {11: 1},  # (1,k0+1)
        'W': {0: -1, 2: 1},  # -C11 and +C12
    })

    # M6 = (A21-A11)(B12+B22)
    # U: (1,0)=1, (0,0)=-1
    # V: (0,k0+1)=1, (1,k0+1)=1
    # W: C22=(1,1) -> W[3] += 1
    strassen_terms.append({
        'U': {2: 1, 0: -1},  # (1,0) and (0,0)
        'V': {1: 1, 11: 1},  # (0,k0+1) and (1,k0+1)
        'W': {3: 1},  # C22
    })

    # M7 = (A12-A22)(B21+B11)
    # U: (0,1)=1, (1,1)=-1
    # V: (1,k0)=1, (0,k0)=1
    # W: C11=(0,0) -> W[0] -= 1, C22=(1,1) -> W[3] += 1
    strassen_terms.append({
        'U': {1: 1, 3: -1},  # (0,1) and (1,1)
        'V': {10: 1, 0: 1},  # (1,k0) and (0,k0)
        'W': {0: -1, 3: 1},  # -C11 and +C22
    })

    # Now fill in the matrices for each column pair
    for pair_idx in range(num_pairs):
        k0 = pair_idx * 2  # 0, 2, 4, 6, 8
        k1 = k0 + 1  # 1, 3, 5, 7, 9

        for t_idx, term in enumerate(strassen_terms):
            col = pair_idx * 7 + t_idx

            # Fill U
            for idx, val in term['U'].items():
                U[idx][col] = val

            # Fill V - need to adjust indices for the specific column pair
            # V index = j * p + k
            # For term's V entries, we have base indices assuming k0=0, k0+1=1
            # We need to map: base index j*p + k_base -> j*p + (k_base + k0) if k_base < 2
            for base_idx, val in term['V'].items():
                j = base_idx // p
                k_base = base_idx % p
                if k_base == 0:
                    k_actual = k0
                elif k_base == 1:
                    k_actual = k1
                else:
                    # Should not happen for our 2x2 sub-problem
                    k_actual = k_base
                actual_idx = j * p + k_actual
                V[actual_idx][col] = val

            # Fill W
            for idx, val in term['W'].items():
                W[idx][col] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
