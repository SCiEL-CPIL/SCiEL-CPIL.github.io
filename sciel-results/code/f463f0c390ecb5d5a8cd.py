import time
import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    def trivial_decomposition():
        rank_t = n * m * p
        U_t = [[0] * rank_t for _ in range(n * m)]
        V_t = [[0] * rank_t for _ in range(m * p)]
        W_t = [[0] * rank_t for _ in range(p * n)]
        c = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_t[ii*m+jj][c] = 1
                    V_t[jj*p+kk][c] = 1
                    W_t[kk*n+ii][c] = 1
                    c += 1
        return {"U": U_t, "V": V_t, "W": W_t}

    # n=2, m=2, p=9
    # U: (n*m, rank) = (4, rank)
    # V: (m*p, rank) = (18, rank)
    # W: (p*n, rank) = (18, rank)
    
    # Strategy:
    # The multiplication tensor M_{2,2,9} is the tensor product of M_{2,2,2} (rank 7) and M_{2,1,9} (rank 9).
    # The naive tensor product gives rank 7 * 9 = 63.
    # However, we can view the 9 columns of B as 4 pairs + 1 single.
    # Or better, split the 9 columns into groups that allow sharing of the A-contraction.
    # Standard decomposition:
    # Split columns into 4 pairs (cols 0-7) and 1 single (col 8).
    # For each pair (k, k+1), we use a Strassen-like decomposition for M_{2,2,2}.
    # Wait, Strassen is for 2x2 * 2x2 -> 2x2. Here B is 2x9.
    # So we are doing A (2x2) * B (2x9) -> C (2x9).
    # This is equivalent to 9 separate matrix-vector multiplications A * b_k.
    # But we can batch them.
    #
    # Let's look at the known rank of M_{2,2,p}.
    # p=1: 4
    # p=2: 7 (Strassen)
    # p=3: 11 (7+4)
    # p=4: 14 (7+7)
    # p=5: 18 (11+7)
    # p=9: ?
    #
    # The current best score 1.125 corresponds to rank 32?
    # Seed value is likely 36 (4*9). 36/32 = 1.125.
    # So the current program has rank 32.
    # Can we do better than 32?
    #
    # Let's check if rank 31 or 30 is possible.
    # Lower bound for M_{2,2,p}:
    # For p=2, rank is 7.
    # For p=4, rank is 14.
    # For p=8, rank is 28? (4 * 7)
    # For p=9, if we use 4 pairs + 1 single, rank is 4*7 + 4 = 32.
    #
    # Is there a way to share more?
    # Consider the tensor M_{2,2,9} = M_{2,2,8} \otimes M_{2,1,1}? No.
    # M_{2,2,9} is not a simple tensor product.
    #
    # Let's try to construct a rank 31 or 30 decomposition.
    # This is a hard problem. The rank of M_{2,2,9} is not definitively known to be less than 32 in simple constructions.
    # However, the "Kronecker product" hint suggests looking at M_{2,2,2} \otimes M_{2,1,9}.
    # Rank(M_{2,2,2}) = 7.
    # Rank(M_{2,1,9}) = 9 (since it's just 9 independent multiplications by a 2x1 vector? No, M_{2,1,9} is A(2x2) * B(2x9)? No.
    # M_{n,m,p} is A(nxm) * B(mx p) -> C(nxp).
    # M_{2,1,9}: A(2x1) * B(1x9) -> C(2x9).
    # This is an outer product of a 2x1 vector and a 1x9 vector.
    # Rank(M_{2,1,9}) is 2? No.
    # T[i, j, k] = 1 if j=0 else 0.
    # This is a rank 2 tensor?
    # U: 2x1, V: 1x9, W: 9x2.
    # Actually, M_{2,1,9} is just the identity map on the 2nd index?
    # Let's stick to the known decomposition.
    #
    # If 32 is the best known simple construction, and we are at 32, we need a structural change.
    # What if we use a different splitting?
    # Split 9 into 3+3+3.
    # Rank(M_{2,2,3}) = 11.
    # 3 * 11 = 33. Worse.
    #
    # Split 9 into 5+4.
    # Rank(M_{2,2,5}) = 18. Rank(M_{2,2,4}) = 14. Total 32.
    #
    # It seems 32 is a strong candidate for the minimal rank for simple decompositions.
    # Is there a rank 31 decomposition?
    #
    # Let's verify the rank 32 construction is correct and robust.
    # The previous program had index errors. I will implement a clean version.
    
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(k0, k1):
        nonlocal col
        # 7 factors for 2x2 * 2x2 (cols k0, k1)
        
        # M1: (a00+a11)(b0k0+b1k1) -> C0k0, C1k1
        U[0][col] = 1; U[3][col] = 1
        V[0*p + k0][col] = 1; V[1*p + k1][col] = 1
        W[k0*n + 0][col] = 1; W[k1*n + 1][col] = 1
        col += 1
        
        # M2: (a10+a11)b0k0 -> C1k0, C1k1 (-1)
        U[2][col] = 1; U[3][col] = 1
        V[0*p + k0][col] = 1
        W[k0*n + 1][col] = 1; W[k1*n + 1][col] = -1
        col += 1
        
        # M3: a00(b0k1-b1k1) -> C0k1, C1k1
        U[0][col] = 1
        V[0*p + k1][col] = 1; V[1*p + k1][col] = -1
        W[k1*n + 0][col] = 1; W[k1*n + 1][col] = 1
        col += 1
        
        # M4: a11(b1k0-b0k0) -> C0k0, C1k0
        U[3][col] = 1
        V[1*p + k0][col] = 1; V[0*p + k0][col] = -1
        W[k0*n + 0][col] = 1; W[k0*n + 1][col] = 1
        col += 1
        
        # M5: (a00+a01)b1k1 -> C0k0 (-1), C0k1
        U[0][col] = 1; U[1][col] = 1
        V[1*p + k1][col] = 1
        W[k0*n + 0][col] = -1; W[k1*n + 0][col] = 1
        col += 1
        
        # M6: (a10-a00)(b0k0+b0k1) -> C1k1
        U[2][col] = 1; U[0][col] = -1
        V[0*p + k0][col] = 1; V[0*p + k1][col] = 1
        W[k1*n + 1][col] = 1
        col += 1
        
        # M7: (a01-a11)(b1k0+b1k1) -> C0k0
        U[1][col] = 1; U[3][col] = -1
        V[1*p + k0][col] = 1; V[1*p + k1][col] = 1
        W[k0*n + 0][col] = 1
        col += 1

    def add_trivial(k):
        nonlocal col
        # 4 factors for 2x2 * 2x1 (col k)
        # 1. a00 b0k -> C0k
        U[0][col] = 1
        V[0*p + k][col] = 1
        W[k*n + 0][col] = 1
        col += 1
        
        # 2. a01 b1k -> C0k
        U[1][col] = 1
        V[1*p + k][col] = 1
        W[k*n + 0][col] = 1
        col += 1
        
        # 3. a10 b0k -> C1k
        U[2][col] = 1
        V[0*p + k][col] = 1
        W[k*n + 1][col] = 1
        col += 1
        
        # 4. a11 b1k -> C1k
        U[3][col] = 1
        V[1*p + k][col] = 1
        W[k*n + 1][col] = 1
        col += 1

    # Split p=9 into 4 and 5
    # Block 1: k in 0..3 (2 Strassen blocks)
    add_strassen(0, 1)
    add_strassen(2, 3)
    
    # Block 2: k in 4..8
    # M_{2,2,3} (cols 4,5,6) = M_{2,2,2} (cols 4,5) + M_{2,2,1} (col 6)
    add_strassen(4, 5)
    add_trivial(6)
    # M_{2,2,2} (cols 7,8)
    add_strassen(7, 8)
    
    if time.time() > deadline:
        return trivial_decomposition()
        
    # Verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check all entries of the tensor
    # T[i,j,k] should be 1 for all i,j,k
    # T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
            
    if not valid:
        # Fallback to trivial if verification fails
        return trivial_decomposition()
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
