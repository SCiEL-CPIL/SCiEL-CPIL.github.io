"""Rewritten program for 4x5x6 matrix multiplication tensor rank minimization."""
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matmul tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    # Also check that all other entries are 0
    # The tensor has 1's at (i*m+j, j*p+k, k*n+i) and 0 elsewhere
    # So we just need to check that for all (a, b, c) not matching, the sum is 0
    # This is expensive; for small dimensions it's fine.
    # a = i*m+j, b = j'*p+k', c = k''*n+i''
    # The matmul tensor M[a,b,c] = 1 if a//m == b%p and a%m == b//p and c%n == a//m... 
    # Actually M[i*m+j, j*p+k, k*n+i] = 1, else 0.
    # So for a = i*m+j, b = j2*p+k2, c = k3*n+i3:
    # M = 1 iff j == j2 and i == i3 and k == k3 (and i = a//m, j = a%m, etc.)
    # i = a//m, j = a%m, j2 = b//p, k2 = b%p, k3 = c//n, i3 = c%n
    # So M=1 iff a%m == b//p and a//m == c%n and b%p == c//n
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                i = a // m
                j = a % m
                j2 = b // p
                k2 = b % p
                k3 = c // n
                i3 = c % n
                expected = 1 if (j == j2 and i == i3 and k2 == k3) else 0
                s = 0
                for col in range(rank):
                    s += U[a][col] * V[b][col] * W[c][col]
                if s != expected:
                    return False
    return True


def _try_rank_random(n, m, p, rank, timeout, seed=0):
    """Try to find a rank-rank decomposition using random search with rational refinement.
    Uses the approach: start with random rational matrices, then use linear algebra to
    project onto the feasible set.
    """
    import numpy as np
    deadline = time.time() + timeout
    
    # The matmul tensor M has entries M[i*m+j, j*p+k, k*n+i] = 1
    # We need U (nm x r), V (mp x r), W (pn x r) such that
    # U[a,c]*V[b,c]*W[c',c] summed over c = M[a,b,c']
    # This is equivalent to: for each c, U[:,c] ⊗ V[:,c] ⊗ W[:,c] sums to M
    
    # Strategy: parameterize as U, V, W with rational entries.
    # Use the fact that this is a 3-linear system.
    # 
    # A practical approach: use the "splitter" method.
    # Pick random U and V, then for each (a,b,c'), we need:
    # sum_d U[a,d]*V[b,d]*W[c',d] = M[a,b,c']
    # This is linear in W for fixed U,V.
    # So: W[c',:] is determined by solving for each c':
    # sum_d W[c',d] * (U[:,d] ⊗ V[:,d])_flattened_at_(a,b) = M[a,b,c']
    # But this is over-constrained in general.
    
    # Better: use bilinear approach.
    # Fix U. Then for each pair (b, c'), we need V[:,d]*W[c',d] to satisfy
    # sum_d U[a,d] * V[b,d] * W[c',d] = M[a,b,c'] for all a.
    # Let X_{b,c'} = V[b,:] * W[c',:] (element-wise, then we need U @ (V*W) structure)
    # Actually: for fixed b, c', define z_{b,c',d} = V[b,d]*W[c',d]
    # Then sum_d U[a,d] * z_{b,c',d} = M[a,b,c'] for all a
    # This means U @ z_{b,c',:}^T = M[:,b,c']
    # If U is full rank (nm x r with r >= nm), then z_{b,c',:}^T = (U^T U)^{-1} U^T M[:,b,c']
    # So z_{b,c'} is determined!
    # Then we need V[b,d] * W[c',d] = z_{b,c',d} for all d.
    # This is a rank-1 constraint on the V*W product.
    
    # If r = nm, then U is square. Pick U = identity.
    # Then z_{b,c',d} = M[d,b,c'] (since U is identity, z = M[:,b,c'])
    # So we need V[b,d] * W[c',d] = M[d,b,c'] for all b,c',d.
    # M[d,b,c'] = 1 if d//m == c'%n and d%m == b//p and b%p == c'//n
    # Let i=d//m, j=d%m, j2=b//p, k2=b%p, k3=c'//n, i3=c'%n
    # M=1 iff j==j2 and i==i3 and k2==k3
    
    # With U=I_{nm}, we need V[b,d]*W[c',d] = M[d,b,c']
    # V has shape (mp, nm), W has shape (pn, nm)
    # V[b,d] = V[j*p+k, i*m+j] for b=j*p+k, d=i*m+j
    # W[c',d] = W[k*n+i, i*m+j] for c'=k*n+i, d=i*m+j
    # M[d,b,c'] = 1 iff j==j and i==i and k==k -> always when consistent
    
    # So V[j*p+k, i*m+j] * W[k*n+i, i*m+j] = 1 if j==j and i==i and k==k (trivially true)
    # = 0 otherwise
    # But "otherwise" means when the indices don't match in the right way.
    # For a fixed (b, d, c'), the condition is specific.
    # Let's set V[j*p+k, i*m+j] = δ_{j, j_b} where j_b = b//p... no.
    
    # This approach with U=I gives rank = nm = 20, but V*W product must equal M which
    # is not separable in general. So it won't work directly.
    
    # Let me try a different known result.
    # For 4x5x6, the best known upper bound might be from Winograd or block methods.
    # Let me try to construct a decomposition using block Strassen.
    
    # Actually, let me just try to find ANY rank < 120 decomposition.
    # The simplest improvement: use the fact that some entries can share multiplications.
    # 
    # Let me try a systematic approach: use the "Lipschitz" method or just
    # try random rank r and solve via linear algebra.
    
    # For a given rank r, we can set up the equations:
    # sum_c U[a,c] V[b,c] W[c',c] = M[a,b,c']
    # This is nonlinear (trilinear). 
    # 
    # Linearize: fix two matrices, solve for the third.
    # If we fix U and V, then for each (b,c'):
    # sum_c W[c',c] * (U[:,c] ⊗ V[:,c])_{(a,b)} = M[a,b,c']
    # This is: (U ⊙ V)^T W[c',:]^T = M[:,b,c'] where U⊙V is nm*mp x r matrix
    # (U⊙V)[a*mp+b, c] = U[a,c]*V[b,c]
    # So W[c',:]^T = ((U⊙V)^T (U⊙V))^{-1} (U⊙V)^T M[:,b,c']
    # if (U⊙V) has full column rank r.
    
    # Then check if the resulting W gives the correct decomposition.
    
    np.random.seed(seed)
    best_rank = rank
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Random U, V with rational-like entries (use small integers)
        U_try = np.random.randint(-2, 3, size=(n*m, rank)).astype(float)
        V_try = np.random.randint(-2, 3, size=(m*p, rank)).astype(float)
        
        # Compute UV = kron-like product: UV[a*mp+b, c] = U[a,c]*V[b,c]
        UV = np.zeros((n*m*m*p, rank))
        for a in range(n*m):
            for b in range(m*p):
                UV[a*m*p + b, :] = U_try[a, :] * V_try[b, :]
        
        # For each c', solve for W[c',:]
        # (UV^T UV) W[c',:]^T = UV^T M[:,b,c']
        # But M depends on (a,b,c')... 
        # Actually for fixed c', we need:
        # sum_c W[c',c] UV[a*mp+b, c] = M[a,b,c'] for all a,b
        # This is a single linear system: UV^T W[c',:]^T = M_c' where M_c'[a*mp+b] = M[a,b,c']
        
        # Build M matrix
        M_flat = np.zeros(n*m*m*p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m+j
                    b = j*p+k
                    c_idx = k*n+i
                    M_flat[a*m*p + b] = 0  # will fill per c'
        
        # Actually M[a,b,c'] is 1 only when c' = (b%p)*n + a//m and a%m == b//p and b%p == c'//n
        # Let me just build it directly.
        
        # For each c', build the right-hand side
        A_matrix = UV.T @ UV  # r x r
        try:
            A_inv = np.linalg.inv(A_matrix)
        except:
            continue
        
        W_try = np.zeros((p*n, rank))
        valid = True
        for c_idx in range(p*n):
            rhs = np.zeros(n*m*m*p)
            for a in range(n*m):
                for b in range(m*p):
                    i = a//m
                    j = a%m
                    j2 = b//p
                    k2 = b%p
                    k3 = c_idx//n
                    i3 = c_idx%n
                    if j == j2 and i == i3 and k2 == k3:
                        rhs[a*m*p + b] = 1.0
            try:
                W_try[c_idx, :] = (A_inv @ (UV.T @ rhs)).T
            except:
                valid = False
                break
        
        if not valid:
            continue
        
        # Verify
        if _verify(U_try.tolist(), V_try.tolist(), W_try.tolist(), n, m, p):
            return U_try.tolist(), V_try.tolist(), W_try.tolist()
    
    return None


def _strassen_2x2x2():
    """Return the 7-rank Strassen decomposition for 2x2x2 matmul.
    Returns U, V, W each of shape (4, 7).
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    n = m = p = 2
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # Strassen formulas for C = A*B where A,B are 2x2:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # A index: i*2+j, B index: j*2+k, C index: k*2+i
    # U[i*2+j, c] = coefficient of A_{i,j} in M_c
    # V[j*2+k, c] = coefficient of B_{j,k} in M_c
    # W[k*2+i, c] = coefficient of M_c in C_{k*2+i}
    
    # M1: A: a11, a22; B: b11, b22
    U[0][0] = 1; U[3][0] = 1  # a11 (i=0,j=0), a22 (i=1,j=1) -> indices 0, 3
    V[0][0] = 1; V[3][0] = 1  # b11 (j=0,k=0), b22 (j=1,k=1) -> indices 0, 3
    W[0][0] = 1; W[3][0] = 1  # C11 (k=0,i=0), C22 (k=1,i=1) -> indices 0, 3
    
    # M2: A: a21, a22; B: b11
    U[2][1] = 1; U[3][1] = 1  # a21 (i=1,j=0)=idx2, a22=idx3
    V[0][1] = 1  # b11=idx0
    W[2][1] = 1; W[3][1] = -1  # C21 (k=0,i=1)=idx2, C22=idx3
    # Wait: C21 = M2+M4, C22 = M1-M2+M3+M6
    # So M2 contributes to C21 with +1 and C22 with -1
    # C21 is k=0,i=1 -> index k*2+i = 0*2+1 = 1? No.
    # W[k*n+i, c] with n=2: W[k*2+i, c]
    # C_{i,k} where i is row, k is col. C index = k*n + i = k*2 + i
    # C11: i=0,k=0 -> 0; C12: i=0,k=1 -> 1; C21: i=1,k=0 -> 2; C22: i=1,k=1 -> 3
    # So W[0] = C11, W[1] = C12, W[2] = C21, W[3] = C22
    
    # Redo:
    # C11 = M1+M4-M5+M7 -> W[0]: M1:+1, M4:+1, M5:-1, M7:+1
    # C12 = M3+M5 -> W[1]: M3:+1, M5:+1
    # C21 = M2+M4 -> W[2]: M2:+1, M4:+1
    # C22 = M1-M2+M3+M6 -> W[3]: M1:+1, M2:-1, M3:+1, M6:+1
    
    # Let me restart W:
    for c in range(7):
        W[0][c] = 0; W[1][c] = 0; W[2][c] = 0; W[3][c] = 0
    W[0][0] = 1; W[0][3] = 1; W[0][4] = -1; W[0][6] = 1  # M1,M4,M5,M7
    W[1][2] = 1; W[1][4] = 1  # M3,M5
    W[2][1] = 1; W[2][3] = 1  # M2,M4
    W[3][0] = 1; W[3][1] = -1; W[3][2] = 1; W[3][5] = 1  # M1,M2,M3,M6
    
    # U and V:
    # M1: (a11+a22)(b11+b22)
    U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1
    # M2: (a21+a22)b11
    U[2][1]=1; U[3][1]=1; V[0][1]=1
    # M3: a11(b12-b22)
    U[0][2]=1; V[1][2]=1; V[3][2]=-1
    # M4: a22(b21-b11)
    U[3][3]=1; V[2][3]=1; V[0][3]=-1
    # M5: (a11+a12)b22
    U[0][4]=1; U[1][4]=1; V[3][4]=1
    # M6: (a21-a11)(b11+b12)
    U[2][5]=1; U[0][5]=-1; V[0][5]=1; V[1][5]=1
    # M7: (a12-a22)(b21+b22)
    U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1
    
    return U, V, W


def _combine_blocks(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2):
    """Combine two block decompositions for a block diagonal structure.
    Block 1 handles rows 0..n1-1, cols 0..p2-1 of the result... 
    Actually for 4x5x6, let's split the middle dimension:
    A is 4x5, B is 5x6. Split B into B_left (5x3) and B_right (5x3).
    C_left = A * B_left (4x3), C_right = A * B_right (4x3).
    C = [C_left, C_right] (4x6).
    
    So we need decompositions for 4x5x3 and 4x5x3.
    Or split differently.
    """
    pass


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # Safety margin
    
    # First, try the schoolbook as fallback
    def schoolbook():
        rank = n*m*p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    
    # Try to find better decomposition
    # Strategy: use known low-rank blocks.
    # Split 4x5x6 into two 4x5x3 blocks (splitting p=6 into 3+3).
    # Each 4x5x3 block can potentially use lower rank.
    
    # For 4x5x3: schoolbook rank = 60. Total = 120.
    # Can we do better for 4x5x3?
    
    # Try random search for rank < 120
    for r in range(115, 119, -1):
        if time.time() > deadline:
            break
        result = _try_rank_random(n, m, p, r, timeout=5, seed=r)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            break
    
    return best_result

# EVOLVE-BLOCK-END
