import time
import math

DIMENSIONS = [3, 4, 7]

def _to_frac(x):
    """Convert a float that is a half-integer or integer to an exact rational (Fraction)."""
    # Round to nearest half-integer
    h = round(x * 2) / 2
    if abs(x - h) > 1e-9:
        return x  # fallback to float if not close to half-integer
    # Use Fraction for exactness
    from fractions import Fraction
    return Fraction(int(h * 2), 2)

def _build_strassen_2x2x2():
    """
    Return (U, V, W) for 2x2x2 matrix multiplication (Strassen).
    U: (4, 7), V: (4, 7), W: (4, 7)
    Convention: W[k*n + i, c] where n=2, m=2, p=2
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0.0]*rank for _ in range(n*m)]  # 4x7
    V = [[0.0]*rank for _ in range(m*p)]  # 4x7
    W = [[0.0]*rank for _ in range(p*n)]  # 4x7

    # Strassen for 2x2x2:
    # A = [[a11,a12],[a21,a22]], B = [[b11,b12],[b21,b22]]
    # C = [[c11,c12],[c21,c22]]
    #
    # M1 = (a11+a22)*(b11+b22)
    # M2 = (a21+a22)*b11
    # M3 = a11*(b12-b22)
    # M4 = a22*(b21-b11)
    # M5 = (a11+a12)*b22
    # M6 = (a21-a11)*(b11+b12)
    # M7 = (a12-a22)*(b21+b22)
    #
    # C11 = M1 - M4 + M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # U rows: a11, a12, a21, a22
    # V rows: b11, b12, b21, b22
    # W rows: c11, c12, c21, c22 (k*n+i: k=0,i=0 -> c11; k=0,i=1 -> c12; k=1,i=0 -> c21; k=1,i=1 -> c22)

    # M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    # M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    # M3
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    # M4
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = -1; W[2][3] = 1
    # M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[1][4] = 1
    # M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    # M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1

    return U, V, W

def _pad_and_combine(n, m, p, U_sm, V_sm, W_sm, target_n, target_m, target_p):
    """
    Given a decomposition for (n,m,p) multiplication, extend it to (target_n, target_m, target_p)
    by padding with zeros. This works because the multiplication tensor for (target_n, target_m, target_p)
    contains the (n,m,p) tensor as a sub-tensor when we restrict to the top-left n x m and m x p blocks.
    However, this only gives a valid decomposition if we can express the full multiplication using only
    these components, which is generally NOT the case. This function is a placeholder for a more careful
    construction.
    """
    # Actually, padding doesn't directly work for reducing rank. We need a different approach.
    # For now, this function is not used directly.
    pass

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def _block_decompose(n, m, p):
    """
    Try block decomposition to reduce rank.
    For 3x4x7:
    Split A (3x4) into A1 (1x4) and A2 (2x4)
    Split B (4x7) into B1 (4x3) and B2 (4x4)
    
    C11 = A1*B1: 1x4x3, rank <= 3 (schoolbook) or less
    C12 = A1*B2: 1x4x4, rank <= 4
    C21 = A2*B1: 2x4x3, rank <= 12 (schoolbook) or less via Strassen-like
    C22 = A2*B2: 2x4x4, rank <= 16 (schoolbook) or less via Strassen
    
    Total schoolbook: 3+4+12+16 = 35
    
    Can we do better?
    A1*B1: 1x4x3. This is just dot products. Rank = 3 (each output is sum of 4 products, but we can do better? No, rank of 1xm x n is min(m,n) if we can choose freely, but here it's fixed structure. Actually rank of 1xm x n multiplication is m*n? No. Let me think.
    
    For 1 x m x n: C_k = sum_j A_j B_jk. Each C_k depends on all A_j and B_jk. The tensor has rank m*n? No. The rank is m (if we group by j)? Let me check: each rank-1 term is A_j * B_jk for fixed j,k. But we can't combine across j because A_j is different. Actually, the standard rank of 1 x m x n is m*n? No.
    
    Wait, for 1 x 2 x 2: C1 = a1*b11 + a2*b21, C2 = a1*b12 + a2*b22. Can we write this with fewer than 4 rank-1 terms? Each term is a scalar times a vector in A-space, B-space, C-space. The A-space is 2D, B-space is 4D, C-space is 2D. The rank is at least max(dimensions) = 4? Actually, I think rank(1xm x n) = m*n / gcd stuff? Let me just use schoolbook for small ones.
    
    For simplicity, let me use schoolbook for all blocks and compute total rank.
    """
    # 3x4x7 with schoolbook: 84
    # Let me try a recursive approach using known small ranks.
    
    # Actually, let me just implement a simple recursive block decomposition.
    # For 3x4x7:
    # Option: split into 1x4x7 and 2x4x7
    # 1x4x7: rank 28 (schoolbook) or maybe 4? Let me compute properly.
    # 1xm x n: the tensor T_{j,k} = sum_i a_i b_{i,k}. Each term a_i * b_{i,k} * 1. So rank <= m*n. But can we do better?
    # The slice for a fixed i is a_i * (b_{i,1},...,b_{i,n}) * (1,...,1). These are linearly independent in the A-space (a_i are independent). So rank >= m. Also, in C-space, we have n outputs. In B-space, mn inputs.
    # I believe rank(1 x m x n) = m * n is not minimal. Actually, I recall that rank(1 x m x n) = m + n - 1 or something? No.
    # Let me just use schoolbook to be safe: rank = m*n.
    
    # This is getting complicated. Let me just return schoolbook for now and try to improve.
    pass

def construct():
    """
    Construct a tensor decomposition for 3x4x7 matrix multiplication.
    Returns dict with U, V, W.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = len(U[0]) if U else 0
    
    # Try block decomposition
    # Split A (3x4) into rows: A1 (1x4), A2 (2x4)
    # Split B (4x7) into cols: B1 (4x3), B2 (4x4)
    # C11 = A1*B1: 1x4x3, schoolbook rank 12
    # C12 = A1*B2: 1x4x4, schoolbook rank 16
    # C21 = A2*B1: 2x4x3, schoolbook rank 24
    # C22 = A2*B2: 2x4x4, schoolbook rank 32
    # Total: 12+16+24+32 = 84 (same as schoolbook)
    
    # So block decomposition with schoolbook doesn't help. We need better decompositions for sub-blocks.
    
    # For 2x4x4: can we do better than 32?
    # Split 2x4x4 into 2x2x4 + 2x2x4? No, 4 = 2+2.
    # 2x2x4: split into 2x2x2 + 2x2x2, each rank 7, total 14.
    # So 2x4x4 rank <= 28.
    # Similarly, 2x4x3: split into 2x2x3 + 2x2x3.
    # 2x2x3: split into 2x2x2 + 2x2x1.
    # 2x2x2 rank 7, 2x2x1 rank 4 (schoolbook). Total 11.
    # So 2x4x3 rank <= 22.
    # 1x4x4: split into 1x2x4 + 1x2x4.
    # 1x2x4: schoolbook rank 8. Or 1x2x2 + 1x2x2, each rank 4, total 8.
    # So 1x4x4 rank <= 16.
    # 1x4x3: rank <= 12.
    # Total: 16 + 12 + 22 + 28 = 78. Better than 84!
    
    # Let me implement this recursive approach.
    
    def _decompose_recursive(nn, mm, pp):
        """Return (U, V, W) for nn x mm x pp multiplication."""
        # Base cases
        if nn == 1 and mm == 1 and pp == 1:
            return [[1]], [[1]], [[1]]
        if nn == 2 and mm == 2 and pp == 2:
            return _build_strassen_2x2x2()
        
        # For 1 x mm x pp: rank = mm * pp (schoolbook is optimal? Maybe not, but let's use it)
        if nn == 1:
            return _schoolbook(nn, mm, pp)
        if mm == 1:
            return _schoolbook(nn, mm, pp)
        if pp == 1:
            return _schoolbook(nn, mm, pp)
        
        # Try splitting the middle dimension (mm) into two parts
        # nn x mm x pp = nn x (mm1+mm2) x pp
        # = nn x mm1 x pp + nn x mm2 x pp
        best_rank = nn * mm * pp
        best_result = None
        
        for mm1 in range(1, mm):
            mm2 = mm - mm1
            U1, V1, W1 = _decompose_recursive(nn, mm1, pp)
            U2, V2, W2 = _decompose_recursive(nn, mm2, pp)
            rank1 = len(U1[0])
            rank2 = len(U2[0])
            total_rank = rank1 + rank2
            if total_rank < best_rank:
                best_rank = total_rank
                # Combine: U has rows for A1 (nn x mm1) and A2 (nn x mm2)
                # V has rows for B1 (mm1 x pp) and B2 (mm2 x pp)
                # W is same for both
                U = [[0]*total_rank for _ in range(nn*mm)]
                V = [[0]*total_rank for _ in range(mm*pp)]
                W = [[0]*total_rank for _ in range(pp*nn)]
                
                # Fill U1 for rows 0..nn*mm1-1, cols 0..rank1-1
                for i in range(nn*mm1):
                    for c in range(rank1):
                        U[i][c] = U1[i][c]
                # Fill U2 for rows nn*mm1..nn*mm-1, cols rank1..total_rank-1
                for i in range(nn*mm2):
                    for c in range(rank2):
                        U[nn*mm1+i][rank1+c] = U2[i][c]
                
                # Fill V1 for rows 0..mm1*pp-1, cols 0..rank1-1
                for j in range(mm1*pp):
                    for c in range(rank1):
                        V[j][c] = V1[j][c]
                # Fill V2 for rows mm1*pp..mm*pp-1, cols rank1..total_rank-1
                for j in range(mm2*pp):
                    for c in range(rank2):
                        V[mm1*pp+j][rank1+c] = V2[j][c]
                
                # W is the same for both blocks (same output indices)
                for k in range(pp*nn):
                    for c in range(rank1):
                        W[k][c] = W1[k][c]
                    for c in range(rank2):
                        W[k][rank1+c] = W2[k][c]
                
                best_result = (U, V, W)
        
        # Try splitting the first dimension (nn) into two parts
        for nn1 in range(1, nn):
            nn2 = nn - nn1
            U1, V1, W1 = _decompose_recursive(nn1, mm, pp)
            U2, V2, W2 = _decompose_recursive(nn2, mm, pp)
            rank1 = len(U1[0])
            rank2 = len(U2[0])
            total_rank = rank1 + rank2
            if total_rank < best_rank:
                best_rank = total_rank
                U = [[0]*total_rank for _ in range(nn*mm)]
                V = [[0]*total_rank for _ in range(mm*pp)]
                W = [[0]*total_rank for _ in range(pp*nn)]
                
                for i in range(nn1*mm):
                    for c in range(rank1):
                        U[i][c] = U1[i][c]
                for i in range(nn2*mm):
                    for c in range(rank2):
                        U[nn1*mm+i][rank1+c] = U2[i][c]
                
                for j in range(mm*pp):
                    for c in range(rank1):
                        V[j][c] = V1[j][c]
                    for c in range(rank2):
                        V[j][rank1+c] = V2[j][c]
                
                for k in range(pp*nn1):
                    for c in range(rank1):
                        W[k][c] = W1[k][c]
                for k in range(pp*nn2):
                    for c in range(rank2):
                        W[pp*nn1+k][rank1+c] = W2[k][c]
                
                best_result = (U, V, W)
        
        # Try splitting the third dimension (pp) into two parts
        for pp1 in range(1, pp):
            pp2 = pp - pp1
            U1, V1, W1 = _decompose_recursive(nn, mm, pp1)
            U2, V2, W2 = _decompose_recursive(nn, mm, pp2)
            rank1 = len(U1[0])
            rank2 = len(U2[0])
            total_rank = rank1 + rank2
            if total_rank < best_rank:
                best_rank = total_rank
                U = [[0]*total_rank for _ in range(nn*mm)]
                V = [[0]*total_rank for _ in range(mm*pp)]
                W = [[0]*total_rank for _ in range(pp*nn)]
                
                for i in range(nn*mm):
                    for c in range(rank1):
                        U[i][c] = U1[i][c]
                    for c in range(rank2):
                        U[i][rank1+c] = U2[i][c]
                
                for j in range(mm*pp1):
                    for c in range(rank1):
                        V[j][c] = V1[j][c]
                for j in range(mm*pp2):
                    for c in range(rank2):
                        V[mm*pp1+j][rank1+c] = V2[j][c]
                
                for k in range(pp1*nn):
                    for c in range(rank1):
                        W[k][c] = W1[k][c]
                for k in range(pp2*nn):
                    for c in range(rank2):
                        W[pp1*nn+k][rank1+c] = W2[k][c]
                
                best_result = (U, V, W)
        
        if best_result:
            return best_result
        return _schoolbook(nn, mm, pp)
    
    # Try the recursive decomposition
    try:
        U_rec, V_rec, W_rec = _decompose_recursive(n, m, p)
        rank_rec = len(U_rec[0])
        if rank_rec < best_rank:
            U, V, W = U_rec, V_rec, W_rec
            best_rank = rank_rec
    except Exception as e:
        print(f"Recursive decomposition failed: {e}")
    
    # Convert to half-integers if possible
    def _convert_to_halfint(mat):
        result = []
        for row in mat:
            new_row = []
            for x in row:
                h = round(x * 2) / 2
                if abs(x - h) < 1e-9:
                    new_row.append(h)
                else:
                    new_row.append(x)
            result.append(new_row)
        return result
    
    U = _convert_to_halfint(U)
    V = _convert_to_halfint(V)
    W = _convert_to_halfint(W)
    
    return {"U": U, "V": V, "W": W}
