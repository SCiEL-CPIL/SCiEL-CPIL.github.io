"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # We are decomposing the tensor for multiplying a 2x2 matrix A by a 2x12 matrix B.
    # A is (n x m) = (2 x 2), B is (m x p) = (2 x 12), C is (n x p) = (2 x 12).
    # Indices: a = i*m + j (i in 0..1, j in 0..1), b = j*p + k (j in 0..1, k in 0..11), c = k*n + i (k in 0..11, i in 0..1)
    # The tensor T[a, b, c] = 1 iff there exist i, j, k such that a=i*m+j, b=j*p+k, c=k*n+i.
    # Since n=2, m=2:
    # i=0, j=0: a=0, b=k, c=2k -> for k=0..11
    # i=0, j=1: a=1, b=12+k, c=2k -> for k=0..11
    # i=1, j=0: a=2, b=k, c=2k+1 -> for k=0..11
    # i=1, j=1: a=3, b=12+k, c=2k+1 -> for k=0..11
    
    # Structure: We can view this as 4 independent 1x2 x 2x12 multiplications? No.
    # C[0, k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]  (i=0)
    # C[1, k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]  (i=1)
    # So for each k, we are computing a 2x1 vector = 2x2 matrix * 2x1 vector.
    # This is essentially 12 independent multiplications of a 2x2 matrix by a 2x1 vector.
    # The tensor for 2x2 x 2x1 has rank 4 (schoolbook: 2*2*1=4). Can it be lower?
    # The tensor T_2x2_x_2x1 has shape (2, 2, 2). Wait, indices: A(i,j), B(j,k), C(i,k) with k fixed.
    # This is the matrix multiplication tensor for m=2, n=2, p=1.
    # Rank of [2,2,1] is 4?
    # Let's check: T(i, j, k) where i=0,1; j=0,1; k=0,1.
    # T[0,0,0]=1, T[0,1,0]=1, T[1,0,0]=1, T[1,1,0]=1.
    # T[0,0,1]=0, T[0,1,1]=1, T[1,0,1]=0, T[1,1,1]=1.
    # Wait, for p=1, k is only 0. So C(i,0) = sum_j A(i,j) B(j,0).
    # So the tensor is T(i, j, 0) = 1 for all i,j.
    # This is a rank 1 tensor? No. T is (n, m, p) = (2, 2, 1).
    # T[i, j, 0] = 1.
    # This can be written as sum_{r=1}^4 u_r v_r w_r?
    # Actually, the "multiplication" of a 2x2 by a 2x1 vector is just 4 scalar multiplications and 2 additions?
    # No, tensor decomposition counts the scalar multiplications.
    # C_0 = A_00 B_0 + A_01 B_1
    # C_1 = A_10 B_0 + A_11 B_1
    # This requires 4 multiplications: A_00 B_0, A_01 B_1, A_10 B_0, A_11 B_1.
    # Can we do it in 3?
    # This is the rank of the [2,2,1] tensor.
    # It is known that the rank of [m, n, 1] is mn. So rank is 4.
    # So for 12 independent k's, if they were independent, rank would be 12 * 4 = 48.
    # But they are not independent? The structure is that A is the same for all k.
    # However, the tensor indices are a=(i,j), b=(j,k), c=(k,i).
    # Notice that k is in both b and c.
    # This is a "block diagonal" structure in some sense?
    # Let's try to exploit the small size of n=2, m=2.
    
    # Let's look at the Strassen-like decomposition for 2x2 x 2x2 (rank 7).
    # For 2x2 x 2x12, we can partition the 12 columns of B into 6 pairs.
    # For each pair (k, k+1), we multiply 2x2 A by a 2x2 submatrix of B.
    # This requires rank 7 per pair (Strassen).
    # 6 pairs * 7 = 42.
    # This matches the "inspiration" that got score 1.143 (48/42 = 1.1428...).
    # Wait, the inspiration said "reducing rank by 6" meaning 48 -> 42.
    # So rank 42 is achievable.
    
    # Can we do better than 42?
    # What if we partition into 3 groups of 4?
    # 2x2 x 2x4. What is the rank of [2,2,4]?
    # If we use Strassen for 2x2 x 2x2 twice, rank is 14.
    # Is there a better algorithm for 2x2 x 2x4?
    # Maybe a "3D" Strassen?
    # Or maybe we can exploit the structure more globally.
    
    # Let's try to construct the rank 42 decomposition first.
    # We partition k=0..11 into 6 pairs: (0,1), (2,3), (4,5), (6,7), (8,9), (10,11).
    # For each pair t (0..5), let the local k indices be k0=2t, k1=2t+1.
    # We use the 7 rank-1 terms for 2x2 x 2x2 multiplication.
    # The 2x2 x 2x2 tensor has indices (i, j, k) with i,j,k in {0,1}.
    # Strassen's algorithm for 2x2 x 2x2 uses 7 products.
    # The factors for 2x2 x 2x2 are:
    # Let A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]], C = [[c00, c01], [c10, c11]].
    # c00 = M1 + M4 - M5 - M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M3 + M6 + M7
    # where:
    # M1 = (a00 + a11) * (b00 + b11)
    # M2 = (a10 + a11) * b00
    # M3 = a00 * (b01 - b11)
    # M4 = a11 * (b10 - b00)
    # M5 = (a00 + a01) * b11
    # M6 = (a10 - a00) * (b00 + b01)
    # M7 = (a01 - a11) * (b10 + b11)
    
    # For each pair t, we define 7 rank-1 terms.
    # Total rank = 6 * 7 = 42.
    
    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Coefficients for A (U), B (V), C (W) for the 7 products.
    # Indices for A: i=0,1; j=0,1. So 4 entries.
    # Indices for B: j=0,1; k=0,1. So 4 entries.
    # Indices for C: k=0,1; i=0,1. So 4 entries.
    
    # Let's define the 7 sets of coefficients (u, v, w) where u, v, w are vectors of length 4.
    # We'll index them as (i,j) for u, (j,k) for v, (k,i) for w.
    
    def get_U(i, j):
        return i * m + j
    
    def get_V(j, k_local):
        return j * p + k_local  # k_local is 0 or 1 within the pair
    
    def get_W(k_local, i):
        return (2 * t + k_local) * n + i  # k = 2t + k_local
    
    # Wait, get_W depends on t. We'll handle that in the loop.
    
    # Coefficients for the 7 products:
    # Each product r has coefficients for A, B, C.
    # M1: A: a00+a11 -> u[0,0]=1, u[1,1]=1. B: b00+b11 -> v[0,0]=1, v[1,1]=1. C: c00+=1, c11+=1 -> w[0,0]=1, w[1,1]=1.
    # M2: A: a10+a11 -> u[1,0]=1, u[1,1]=1. B: b00 -> v[0,0]=1. C: c10+=1 -> w[0,1]=1.
    # M3: A: a00 -> u[0,0]=1. B: b01-b11 -> v[0,1]=1, v[1,1]=-1. C: c01+=1 -> w[1,0]=1.
    # M4: A: a11 -> u[1,1]=1. B: b10-b00 -> v[1,0]=1, v[0,0]=-1. C: c00+=1, c10+=1 -> w[0,0]=1, w[0,1]=1.
    # M5: A: a00+a01 -> u[0,0]=1, u[0,1]=1. B: b11 -> v[1,1]=1. C: c00-=1, c01+=1 -> w[0,0]=-1, w[1,0]=1.
    # M6: A: a10-a00 -> u[1,0]=1, u[0,0]=-1. B: b00+b01 -> v[0,0]=1, v[0,1]=1. C: c11+=1 -> w[1,1]=1.
    # M7: A: a01-a11 -> u[0,1]=1, u[1,1]=-1. B: b10+b11 -> v[1,0]=1, v[1,1]=1. C: c00-=1, c11+=1 -> w[0,0]=-1, w[1,1]=1.
    
    # Let's encode these as lists of (i,j), (j,k), (k,i) with coefficients.
    # For each of the 7 products, we have a set of (index, coeff) for U, V, W.
    
    products = [
        # M1
        {
            'U': {get_U(0,0): 1, get_U(1,1): 1},
            'V': {0*2+0: 1, 1*2+1: 1},  # j=0,k=0 and j=1,k=1
            'W': {0*2+0: 1, 1*2+1: 1}   # k=0,i=0 and k=1,i=1
        },
        # M2
        {
            'U': {get_U(1,0): 1, get_U(1,1): 1},
            'V': {0*2+0: 1},
            'W': {0*2+1: 1}             # k=0, i=1
        },
        # M3
        {
            'U': {get_U(0,0): 1},
            'V': {0*2+1: 1, 1*2+1: -1}, # j=0,k=1 and j=1,k=1
            'W': {1*2+0: 1}             # k=1, i=0
        },
        # M4
        {
            'U': {get_U(1,1): 1},
            'V': {1*2+0: 1, 0*2+0: -1}, # j=1,k=0 and j=0,k=0
            'W': {0*2+0: 1, 0*2+1: 1}   # k=0,i=0 and k=0,i=1
        },
        # M5
        {
            'U': {get_U(0,0): 1, get_U(0,1): 1},
            'V': {1*2+1: 1},
            'W': {0*2+0: -1, 1*2+0: 1}  # k=0,i=0 and k=1,i=0
        },
        # M6
        {
            'U': {get_U(1,0): 1, get_U(0,0): -1},
            'V': {0*2+0: 1, 0*2+1: 1},  # j=0,k=0 and j=0,k=1
            'W': {1*2+1: 1}             # k=1, i=1
        },
        # M7
        {
            'U': {get_U(0,1): 1, get_U(1,1): -1},
            'V': {1*2+0: 1, 1*2+1: 1},  # j=1,k=0 and j=1,k=1
            'W': {0*2+0: -1, 1*2+1: 1}  # k=0,i=0 and k=1,i=1
        }
    ]
    
    for t in range(6):
        for s in range(7):
            col = 7 * t + s
            # Fill U
            for idx, coeff in products[s]['U'].items():
                U[idx, col] = coeff
            # Fill V
            for idx, coeff in products[s]['V'].items():
                j = idx // 2
                k_local = idx % 2
                k = 2 * t + k_local
                V[j * p + k, col] = coeff
            # Fill W
            for idx, coeff in products[s]['W'].items():
                k_local = idx // 2
                i = idx % 2
                k = 2 * t + k_local
                W[k * n + i, col] = coeff
    
    # Verify the decomposition
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    T_rec = np.zeros((n * m, m * p, p * n))
    for col in range(rank):
        T_rec += np.einsum('i,j,k->ijk', U[:, col], V[:, col], W[:, col])
    
    err = np.linalg.norm(T_rec - T)
    
    if err < 1e-9:
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback: if Strassen verification fails, use schoolbook rank 48
        rank_fb = 48
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U_fb[a, col] = 1
                    V_fb[b, col] = 1
                    W_fb[c, col] = 1
                    col += 1
        return {
            "U": U_fb.tolist(),
            "V": V_fb.tolist(),
            "W": W_fb.tolist()
        }
# EVOLVE-BLOCK-END
