import numpy as np
import time

START_TIME = time.time()
TIME_LIMIT = 120.0

def _check_time():
    return (time.time() - START_TIME) > (TIME_LIMIT - 15.0)

def _verify_decomposition(U, V, W, n, m, p, trials=5):
    """Verify that the decomposition correctly computes matrix multiplication."""
    for _ in range(trials):
        A = np.random.randint(-3, 4, (n, m)).astype(float)
        B = np.random.randint(-3, 4, (m, p)).astype(float)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        for r in range(U.shape[1]):
            u_vec = U[:, r].reshape(n, m)
            v_vec = V[:, r].reshape(m, p)
            scalar = np.sum(u_vec * A) * np.sum(v_vec * B)
            if scalar != 0:
                w_mat = W[:, r].reshape(p, n).T
                C_computed += scalar * w_mat
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _strassen_2x2():
    """Strassen 2x2 decomposition (rank 7)."""
    n, m, p = 2, 2, 2
    U = np.zeros((n * m, 7))
    V = np.zeros((m * p, 7))
    W = np.zeros((p * n, 7))
    
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1
    U[0,2]=1; V[3,2]=1; V[1,2]=-1
    U[3,3]=1; V[2,3]=1; V[0,3]=-1
    U[0,4]=1; U[1,4]=1; V[3,4]=1
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1
    U[3,6]=1; U[2,6]=-1; V[2,6]=1; V[3,6]=1
    
    W[0,0]=1; W[0,3]=1; W[0,4]=-1; W[0,6]=1
    W[2,2]=1; W[2,4]=1
    W[1,1]=1; W[1,3]=1
    W[3,0]=1; W[3,1]=-1; W[3,2]=1; W[3,5]=1
    
    return U, V, W

def _laderman_3x3():
    """
    Laderman's 3x3 matrix multiplication (rank 23).
    Returns None if verification fails.
    """
    n, m, p = 3, 3, 3
    U = np.zeros((n * m, 23))
    V = np.zeros((m * p, 23))
    W = np.zeros((p * n, 23))
    
    # A = [[a,b,c],[d,e,f],[g,h,i]] indexed 0-8
    # B = [[w,x,y],[z,u,v],[p,q,r]] indexed 0-8
    # C = [[s,t,u],[v,w,x],[y,z,aa]] indexed 0-8 (W[k*3+i])
    
    # Laderman's 23 bilinear forms:
    # P1  = (a+e)(w+u)
    U[0,0]=1; U[4,0]=1
    V[0,0]=1; V[4,0]=1
    # P2  = (b+e)(x+u)
    U[1,1]=1; U[4,1]=1
    V[1,1]=1; V[4,1]=1
    # P3  = (d+g)(z+p)
    U[3,2]=1; U[6,2]=1
    V[3,2]=1; V[6,2]=1
    # P4  = (f+g)(v+p)
    U[5,3]=1; U[6,3]=1
    V[5,3]=1; V[6,3]=1
    # P5  = (a+i)(w+r)
    U[0,4]=1; U[8,4]=1
    V[0,4]=1; V[8,4]=1
    # P6  = (b+i)(x+r)
    U[1,5]=1; U[8,5]=1
    V[1,5]=1; V[8,5]=1
    # P7  = (c+i)(y+r)
    U[2,6]=1; U[8,6]=1
    V[2,6]=1; V[8,6]=1
    # P8  = (d+i)(z+r)
    U[3,7]=1; U[8,7]=1
    V[3,7]=1; V[8,7]=1
    # P9  = (e+i)(u+r)
    U[4,8]=1; U[8,8]=1
    V[4,8]=1; V[8,8]=1
    # P10 = (g+i)(p+r)
    U[6,9]=1; U[8,9]=1
    V[6,9]=1; V[8,9]=1
    # P11 = (a+c)(w+y)
    U[0,10]=1; U[2,10]=1
    V[0,10]=1; V[2,10]=1
    # P12 = (d+g)(z+p) -- wait, same as P3? No.
    # Let me use the standard Laderman formulation more carefully.
    
    # Standard Laderman 3x3 (from literature):
    # Let me reset and use a verified set.
    
    U[:] = 0
    V[:] = 0
    W[:] = 0
    
    # Using the formulation where we define 23 products explicitly
    # and solve for W via least squares.
    
    products = [
        # (A_indices, A_coeffs, B_indices, B_coeffs)
        ([0, 4], [1, 1], [0, 4], [1, 1]),    # (a00+a11)(b00+b11)
        ([1, 4], [1, 1], [0, 3], [1, 1]),    # (a01+a11)(b00+b10)
        ([3, 6], [1, 1], [0, 1], [1, 1]),    # (a10+a20)(b00+b01)
        ([5, 4], [1, 1], [3, 4], [1, 1]),    # (a21+a11)(b10+b11)
        ([0, 8], [1, 1], [0, 8], [1, 1]),    # (a00+a22)(b00+b22)
        ([1, 5], [1, 1], [2, 8], [1, 1]),    # (a01+a21)(b02+b22)
        ([2, 8], [1, 1], [0, 6], [1, 1]),    # (a02+a22)(b00+b20)
        ([3, 8], [1, 1], [2, 6], [1, 1]),    # (a10+a22)(b02+b20)
        ([4, 6], [1, 1], [0, 8], [1, 1]),    # (a12+a20)(b00+b22)
        ([6, 2], [1, 1], [4, 5], [1, 1]),    # (a20+a02)(b11+b12)
        ([5, 1], [1, 1], [3, 5], [1, 1]),    # (a21+a01)(b10+b12)
        ([0, 4], [1, 1], [1, 7], [1, 1]),    # (a00+a12)(b01+b21)
        ([4, 8], [1, 1], [0, 5], [1, 1]),    # (a11+a22)(b00+b12)
        ([1, 8], [1, 1], [0, 7], [1, 1]),    # (a01+a22)(b00+b21)
        ([3, 2], [1, 1], [4, 1], [1, 1]),    # (a10+a02)(b11+b01)
        ([6, 4], [1, 1], [3, 5], [1, 1]),    # (a20+a12)(b10+b12)
        ([2, 3], [1, 1], [4, 5], [1, 1]),    # (a02+a10)(b11+b12)
        ([4, 5], [1, 1], [0, 3], [1, 1]),    # (a12+a21)(b00+b10)
        ([0, 8], [1, 1], [1, 7], [1, 1]),    # (a00+a22)(b01+b21)
        ([3, 5], [1, 1], [0, 6], [1, 1]),    # (a10+a21)(b00+b20)
        ([1, 3], [1, 1], [4, 8], [1, 1]),    # (a01+a10)(b11+b22)
        ([6, 8], [1, 1], [3, 4], [1, 1]),    # (a20+a22)(b10+b11)
        ([2, 4], [1, 1], [7, 5], [1, 1]),    # (a02+a12)(b21+b22)
    ]
    
    for r, (u_idx, u_c, v_idx, v_c) in enumerate(products):
        for ii, cc in zip(u_idx, u_c):
            U[ii, r] = cc
        for ii, cc in zip(v_idx, v_c):
            V[ii, r] = cc
    
    # Compute S[i,k,r] = sum_j U[i,j,r] * V[j,k,r]
    S = np.zeros((3, 3, 23))
    for i in range(3):
        for k in range(3):
            for r in range(23):
                for j in range(3):
                    S[i, k, r] += U[i*3+j, r] * V[j*3+k, r]
    
    # Solve for W: C[i,k] = sum_r W[k*3+i, r] * S[i,k,r] should equal delta_{ik}
    # For each (i,k), we need W[k*3+i, :] such that W[k*3+i, :] . S[i,k,:] = delta_{ik}
    # Using minimum norm solution: W = S^T (S S^T)^{-1} delta
    # But S[i,k,:] is a row vector. We want w such that w . s = target.
    # Minimum norm: w = s * (target / ||s||^2)
    
    for i in range(3):
        for k in range(3):
            s = S[i, k, :]
            target = 1.0 if i == k else 0.0
            norm_sq = np.dot(s, s)
            if norm_sq > 1e-15:
                W[k*3+i, :] = s * (target / norm_sq)
            else:
                W[k*3+i, :] = 0
    
    if not _verify_decomposition(U, V, W, 3, 3, 3, trials=10):
        return None
    
    return U, V, W

def _kron_decompose(Ua, Va, Wa, Ub, Vb, Wb):
    """Kronecker product of two tensor decompositions."""
    import math
    n1 = int(math.isqrt(len(Ua)))
    m1 = int(math.isqrt(len(Va)))
    p1 = int(math.isqrt(len(Wa)))
    n2 = int(math.isqrt(len(Ub)))
    m2 = int(math.isqrt(len(Vb)))
    p2 = int(math.isqrt(len(Wb)))
    r1 = Ua.shape[1]
    r2 = Ub.shape[1]
    n, m, p = n1*n2, m1*m2, p1*p2
    r = r1 * r2
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    
    for r1i in range(r1):
        for r2i in range(r2):
            r_idx = r1i * r2 + r2i
            for i1 in range(n1):
                for j1 in range(m1):
                    a = Ua[i1*m1+j1, r1i]
                    if a == 0: continue
                    for i2 in range(n2):
                        for j2 in range(m2):
                            b = Ub[i2*m2+j2, r2i]
                            if b == 0: continue
                            i = i1*n2 + i2
                            j = j1*m2 + j2
                            U[i*m+j, r_idx] = a * b
            for j1 in range(m1):
                for k1 in range(p1):
                    a = Va[j1*p1+k1, r1i]
                    if a == 0: continue
                    for j2 in range(m2):
                        for k2 in range(p2):
                            b = Vb[j2*p2+k2, r2i]
                            if b == 0: continue
                            j = j1*m2 + j2
                            k = k1*p2 + k2
                            V[j*p+k, r_idx] = a * b
            for k1 in range(p1):
                for i1 in range(n1):
                    a = Wa[k1*n1+i1, r1i]
                    if a == 0: continue
                    for k2 in range(p2):
                        for i2 in range(n2):
                            b = Wb[k2*n2+i2, r2i]
                            if b == 0: continue
                            k = k1*p2 + k2
                            i = i1*n2 + i2
                            W[k*n+i, r_idx] = a * b
    return U, V, W

def _block_strassen_6x6(base_U, base_V, base_W, base_dim):
    """
    Block Strassen: 6x6 = 2x2 blocks of base_dim x base_dim.
    base_U, base_V, base_W: decomposition for base_dim x base_dim.
    Returns 6x6 decomposition.
    """
    n = m = p = 6
    r_base = base_U.shape[1]
    total_rank = 7 * r_base
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    block = base_dim
    
    # Strassen products for block matrices
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    products = [
        # (A_blocks, B_blocks, C_blocks)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    for s in range(7):
        a_blocks, b_blocks, c_blocks = products[s]
        for rb in range(r_base):
            r_idx = s * r_base + rb
            for (bi, bj), ca in a_blocks.items():
                for i2 in range(block):
                    for j2 in range(block):
                        a = base_U[(bi*block+i2)*block + (bj*block+j2), rb]
                        if a != 0:
                            U[(bi*block+i2)*6 + (bj*block+j2), r_idx] = ca * a
            for (bj, bk), cb in b_blocks.items():
                for j2 in range(block):
                    for k2 in range(block):
                        b = base_V[(bj*block+j2)*block + (bk*block+k2), rb]
                        if b != 0:
                            V[(bj*block+j2)*6 + (bk*block+k2), r_idx] = cb * b
            for (ci, ck), cc in c_blocks.items():
                for i2 in range(block):
                    for k2 in range(block):
                        c = base_W[(ci*block+i2)*block + (ck*block+k2), rb]
                        if c != 0:
                            W[(ck*block+k2)*6 + (ci*block+i2), r_idx] = cc * c
    
    return U, V, W

def construct():
    n, m, p = 6, 6, 6
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p  # 216 schoolbook
    
    # Strategy 1: Block Strassen with Laderman 3x3 base -> rank 7*23 = 161
    try:
        U3, V3, W3 = _laderman_3x3()
        if U3 is not None:
            U, V, W = _block_strassen_6x6(U3, V3, W3, 3)
            if _verify_decomposition(U, V, W, 6, 6, 6, trials=5):
                best_U, best_V, best_W = U, V, W
                best_rank = 161
                print(f"Block Strassen + Laderman 3x3: rank {best_rank}")
    except Exception as e:
        print(f"Block Strassen + Laderman failed: {e}")
    
    # Strategy 2: Block Strassen with schoolbook 3x3 base -> rank 7*27 = 189
    if best_rank > 189:
        try:
            U3, V3, W3 = _schoolbook(3, 3, 3)
            U, V, W = _block_strassen_6x6(U3, V3, W3, 3)
            if _verify_decomposition(U, V, W, 6, 6, 6, trials=5):
                best_U, best_V, best_W = U, V, W
                best_rank = 189
                print(f"Block Strassen + schoolbook 3x3: rank {best_rank}")
        except Exception as e:
            print(f"Block Strassen + schoolbook failed: {e}")
    
    # Strategy 3: Kronecker product Strassen 2x2 x Laderman 3x3 -> rank 7*23 = 161
    if best_rank > 161:
        try:
            U2, V2, W2 = _strassen_2x2()
            U3, V3, W3 = _laderman_3x3()
            if U3 is not None:
                U, V, W = _kron_decompose(U2, V2, W2, U3, V3, W3)
                if _verify_decomposition(U, V, W, 6, 6, 6, trials=5):
                    best_U, best_V, best_W = U, V, W
                    best_rank = 161
                    print(f"Kronecker Strassen2 x Laderman3: rank {best_rank}")
        except Exception as e:
            print(f"Kronecker Strassen2 x Laderman3 failed: {e}")
    
    # Fallback: schoolbook
    if best_U is None:
        U, V, W = _schoolbook(6, 6, 6)
        best_U, best_V, best_W = U, V, W
        best_rank = 216
        print(f"Schoolbook fallback: rank {best_rank}")
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
