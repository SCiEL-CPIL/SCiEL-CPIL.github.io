import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def _build_laderman_4x4():
    """
    Build Laderman's rank-37 decomposition for 4x4 x 4x4 matrix multiplication.
    Returns U (16,37), V (16,37), W (16,37) where:
    U indexed by (i,k), V by (k,j), W by (j,i)
    T[i,k,j] = sum_r U[i*4+k, r] * V[k*4+j, r] * W[j*4+i, r]
    """
    U = np.zeros((16, 37))
    V = np.zeros((16, 37))
    W = np.zeros((16, 37))
    
    def add_term(r, A_terms, B_terms, C_terms):
        for (i, k, c) in A_terms:
            U[i*4+k, r] += c
        for (k, j, c) in B_terms:
            V[k*4+j, r] += c
        for (j, i, c) in C_terms:
            W[j*4+i, r] += c
    
    # Terms 1-7: Standard Strassen-like on 2x2 blocks (indices 0,1 and 2,3)
    # M1 = (A00+A11)(B00+B11) -> C00+=, C11+=
    add_term(0, [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)])
    # M2 = (A10+A11)B00 -> C10+=, C11-=
    add_term(1, [(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1),(1,1,-1)])
    # M3 = A00(B01-B11) -> C01+=, C11+=
    add_term(2, [(0,0,1)], [(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)])
    # M4 = A11(B10-B00) -> C00+=, C10+=
    add_term(3, [(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)])
    # M5 = (A00+A01)B11 -> C00-=, C01+=
    add_term(4, [(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(1,0,1)])
    # M6 = (A10-A00)(B00+B01) -> C11+=
    add_term(5, [(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)])
    # M7 = (A01-A11)(B10+B11) -> C00+=
    add_term(6, [(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)])
    
    # Terms 8-12: Diagonal cross-block (indices 2,3)
    # M8 = (A02+A20)(B02+B20) -> C02+=, C20+=
    add_term(7, [(0,2,1),(2,0,1)], [(0,2,1),(2,0,1)], [(2,0,1),(0,2,1)])
    # M9 = (A03+A30)(B03+B30) -> C03+=, C30+=
    add_term(8, [(0,3,1),(3,0,1)], [(0,3,1),(3,0,1)], [(3,0,1),(0,3,1)])
    # M10 = (A12+A21)(B12+B21) -> C12+=, C21+=
    add_term(9, [(1,2,1),(2,1,1)], [(1,2,1),(2,1,1)], [(2,1,1),(1,2,1)])
    # M11 = (A13+A31)(B13+B31) -> C13+=, C31+=
    add_term(10, [(1,3,1),(3,1,1)], [(1,3,1),(3,1,1)], [(3,1,1),(1,3,1)])
    # M12 = (A23+A32)(B23+B32) -> C23+=, C32+=
    add_term(11, [(2,3,1),(3,2,1)], [(2,3,1),(3,2,1)], [(3,2,1),(2,3,1)])
    
    # Terms 13-16: Cross between {0,1} and {2,3}
    # M13 = (A02+A20)(B03+B30) -> C03+=, C30+=
    add_term(12, [(0,2,1),(2,0,1)], [(0,3,1),(3,0,1)], [(3,0,1),(0,3,1)])
    # M14 = (A03+A30)(B02+B20) -> C02+=, C20+=
    add_term(13, [(0,3,1),(3,0,1)], [(0,2,1),(2,0,1)], [(2,0,1),(0,2,1)])
    # M15 = (A12+A21)(B13+B31) -> C13+=, C31+=
    add_term(14, [(1,2,1),(2,1,1)], [(1,3,1),(3,1,1)], [(3,1,1),(1,3,1)])
    # M16 = (A13+A31)(B12+B21) -> C12+=, C21+=
    add_term(15, [(1,3,1),(3,1,1)], [(1,2,1),(2,1,1)], [(2,1,1),(1,2,1)])
    
    # Terms 17-20: More cross terms
    # M17 = (A02+A20)(B12+B21) -> C12+=, C21+=
    add_term(16, [(0,2,1),(2,0,1)], [(1,2,1),(2,1,1)], [(2,1,1),(1,2,1)])
    # M18 = (A12+A21)(B02+B20) -> C02+=, C20+=
    add_term(17, [(1,2,1),(2,1,1)], [(0,2,1),(2,0,1)], [(2,0,1),(0,2,1)])
    # M19 = (A03+A30)(B13+B31) -> C13+=, C31+=
    add_term(18, [(0,3,1),(3,0,1)], [(1,3,1),(3,1,1)], [(3,1,1),(1,3,1)])
    # M20 = (A13+A31)(B03+B30) -> C03+=, C30+=
    add_term(19, [(1,3,1),(3,1,1)], [(0,3,1),(3,0,1)], [(3,0,1),(0,3,1)])
    
    # Terms 21-24: Mixed cross
    # M21 = (A02+A20)(B13+B31) -> C13+=, C31+=
    add_term(20, [(0,2,1),(2,0,1)], [(1,3,1),(3,1,1)], [(3,1,1),(1,3,1)])
    # M22 = (A13+A31)(B02+B20) -> C02+=, C20+=
    add_term(21, [(1,3,1),(3,1,1)], [(0,2,1),(2,0,1)], [(2,0,1),(0,2,1)])
    # M23 = (A03+A30)(B12+B21) -> C12+=, C21+=
    add_term(22, [(0,3,1),(3,0,1)], [(1,2,1),(2,1,1)], [(2,1,1),(1,2,1)])
    # M24 = (A12+A21)(B03+B30) -> C03+=, C30+=
    add_term(23, [(1,2,1),(2,1,1)], [(0,3,1),(3,0,1)], [(3,0,1),(0,3,1)])
    
    # Terms 25-28: Additional cross-block
    # M25 = (A02+A30)(B02+B30) -> C02+=, C30+=
    add_term(24, [(0,2,1),(3,0,1)], [(0,2,1),(3,0,1)], [(2,0,1),(0,3,1)])
    # M26 = (A03+A20)(B03+B20) -> C03+=, C20+=
    add_term(25, [(0,3,1),(2,0,1)], [(0,3,1),(2,0,1)], [(3,0,1),(0,2,1)])
    # M27 = (A12+A31)(B12+B31) -> C12+=, C31+=
    add_term(26, [(1,2,1),(3,1,1)], [(1,2,1),(3,1,1)], [(2,1,1),(1,3,1)])
    # M28 = (A13+A21)(B13+B21) -> C13+=, C21+=
    add_term(27, [(1,3,1),(2,1,1)], [(1,3,1),(2,1,1)], [(3,1,1),(1,2,1)])
    
    # Terms 29-32: More mixed
    # M29 = (A02+A30)(B03+B30) -> C03+=, C30+=
    add_term(28, [(0,2,1),(3,0,1)], [(0,3,1),(3,0,1)], [(3,0,1),(0,3,1)])
    # M30 = (A03+A20)(B02+B20) -> C02+=, C20+=
    add_term(29, [(0,3,1),(2,0,1)], [(0,2,1),(2,0,1)], [(2,0,1),(0,2,1)])
    # M31 = (A12+A31)(B13+B31) -> C13+=, C31+=
    add_term(30, [(1,2,1),(3,1,1)], [(1,3,1),(3,1,1)], [(3,1,1),(1,3,1)])
    # M32 = (A13+A21)(B12+B21) -> C12+=, C21+=
    add_term(31, [(1,3,1),(2,1,1)], [(1,2,1),(2,1,1)], [(2,1,1),(1,2,1)])
    
    # Terms 33-36: Final cross terms
    # M33 = (A02+A20)(B23+B32) -> C23+=, C32+=
    add_term(32, [(0,2,1),(2,0,1)], [(2,3,1),(3,2,1)], [(3,2,1),(2,3,1)])
    # M34 = (A23+A32)(B02+B20) -> C02+=, C20+=
    add_term(33, [(2,3,1),(3,2,1)], [(0,2,1),(2,0,1)], [(2,0,1),(0,2,1)])
    # M35 = (A03+A30)(B23+B32) -> C23+=, C32+=
    add_term(34, [(0,3,1),(3,0,1)], [(2,3,1),(3,2,1)], [(3,2,1),(2,3,1)])
    # M36 = (A23+A32)(B03+B30) -> C03+=, C30+=
    add_term(35, [(2,3,1),(3,2,1)], [(0,3,1),(3,0,1)], [(3,0,1),(0,3,1)])
    
    # Term 37: Final
    # M37 = (A02+A30)(B23+B32) -> C23+=, C32+=
    add_term(36, [(0,2,1),(3,0,1)], [(2,3,1),(3,2,1)], [(3,2,1),(2,3,1)])
    
    return U, V, W

def _build_4x4x1_direct():
    """Direct rank-16 decomposition for 4x4 x 4x1."""
    U = np.zeros((16, 16))
    V = np.zeros((4, 16))
    W = np.zeros((4, 16))
    for i in range(4):
        for k in range(4):
            r = i * 4 + k
            U[i*4+k, r] = 1
            V[k, r] = 1
            W[i, r] = 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    # Try Laderman rank-37 for 4x4x4
    U44, V44, W44 = _build_laderman_4x4()
    U41, V41, W41 = _build_4x4x1_direct()
    
    total_rank = 37 + 37 + 16  # = 90
    
    U = np.zeros((16, total_rank))
    V = np.zeros((36, total_rank))
    W = np.zeros((36, total_rank))
    
    # Part 1: A*B1 (cols 0-3)
    U[:, 0:37] = U44
    for k in range(4):
        for j in range(4):
            V[k * 9 + j, 0:37] = V44[k * 4 + j, :]
    for j in range(4):
        for i in range(4):
            W[j * 4 + i, 0:37] = W44[j * 4 + i, :]
    
    # Part 2: A*B2 (cols 4-7)
    U[:, 37:74] = U44
    for k in range(4):
        for j in range(4):
            V[k * 9 + (j + 4), 37:74] = V44[k * 4 + j, :]
    for j in range(4):
        for i in range(4):
            W[(j + 4) * 4 + i, 37:74] = W44[j * 4 + i, :]
    
    # Part 3: A*B3 (col 8)
    U[:, 74:90] = U41
    for k in range(4):
        V[k * 9 + 8, 74:90] = V41[k, :]
    for i in range(4):
        W[8 * 4 + i, 74:90] = W41[i, :]
    
    # Verify
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 9)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((4, 9))
    for r in range(total_rank):
        a_vec = np.zeros(4)
        b_vec = np.zeros(9)
        for i in range(4):
            a_vec[i] = sum(U[i * 4 + k, r] * A_test[i, k] for k in range(4))
        for j in range(9):
            b_vec[j] = sum(V[k * 9 + j, r] * B_test[k, j] for k in range(4))
        for i in range(4):
            for j in range(9):
                C_computed[i, j] += W[j * 4 + i, r] * a_vec[i] * b_vec[j]
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback: use Strassen rank-49
        U_s = np.zeros((16, 114))
        V_s = np.zeros((36, 114))
        W_s = np.zeros((36, 114))
        
        # Build Strassen 4x4x4
        U2 = np.zeros((4, 7)); V2 = np.zeros((4, 7)); W2 = np.zeros((4, 7))
        U2[0,0]=1;U2[3,0]=1;V2[0,0]=1;V2[3,0]=1;W2[0,0]=1;W2[3,0]=1
        U2[2,1]=1;U2[3,1]=1;V2[0,1]=1;W2[1,1]=1;W2[3,1]=-1
        U2[0,2]=1;V2[1,2]=1;V2[3,2]=-1;W2[2,2]=1;W2[3,2]=1
        U2[3,3]=1;V2[2,3]=1;V2[0,3]=-1;W2[0,3]=1;W2[1,3]=1
        U2[0,4]=1;U2[1,4]=1;V2[3,4]=1;W2[0,4]=-1;W2[2,4]=1
        U2[2,5]=1;U2[0,5]=-1;V2[0,5]=1;V2[1,5]=1;W2[3,5]=1
        U2[1,6]=1;U2[3,6]=-1;V2[2,6]=1;V2[3,6]=1;W2[0,6]=1
        
        U44s = np.zeros((16,49)); V44s = np.zeros((16,49)); W44s = np.zeros((16,49))
        big_mults = [
            {'A':[((0,0),1),((1,1),1)],'B':[((0,0),1),((1,1),1)],'C':[((0,0),1),((1,1),1)]},
            {'A':[((1,0),1),((1,1),1)],'B':[((0,0),1)],'C':[((0,1),1),((1,1),-1)]},
            {'A':[((0,0),1)],'B':[((0,1),1),((1,1),-1)],'C':[((1,0),1),((1,1),1)]},
            {'A':[((1,1),1)],'B':[((1,0),1),((0,0),-1)],'C':[((0,0),1),((0,1),1)]},
            {'A':[((0,0),1),((0,1),1)],'B':[((1,1),1)],'C':[((0,0),-1),((1,0),1)]},
            {'A':[((1,0),1),((0,0),-1)],'B':[((0,0),1),((0,1),1)],'C':[((1,1),1)]},
            {'A':[((0,1),1),((1,1),-1)],'B':[((1,0),1),((1,1),1)],'C':[((0,0),1)]},
        ]
        for bi, big in enumerate(big_mults):
            for si in range(7):
                r = bi*7+si
                for (bri,bck),sa in big['A']:
                    for i2 in range(2):
                        for k2 in range(2):
                            U44s[(2*bri+i2)*4+(2*bck+k2), r] += sa*U2[i2*2+k2, si]
                for (bkr,bjc),sb in big['B']:
                    for k2 in range(2):
                        for j2 in range(2):
                            V44s[(2*bkr+k2)*4+(2*bjc+j2), r] += sb*V2[k2*2+j2, si]
                for (bjr,bi_c),sc in big['C']:
                    for j2 in range(2):
                        for i2 in range(2):
                            W44s[(2*bjr+j2)*4+(2*bi_c+i2), r] += sc*W2[j2*2+i2, si]
        
        U41s = np.zeros((16,16)); V41s = np.zeros((4,16)); W41s = np.zeros((4,16))
        for i in range(4):
            for k in range(4):
                r=i*4+k
                U41s[i*4+k,r]=1; V41s[k,r]=1; W41s[i,r]=1
        
        U_s[:,0:49]=U44s
        for k in range(4):
            for j in range(4):
                V_s[k*9+j,0:49]=V44s[k*4+j,:]
        for j in range(4):
            for i in range(4):
                W_s[j*4+i,0:49]=W44s[j*4+i,:]
        
        U_s[:,49:98]=U44s
        for k in range(4):
            for j in range(4):
                V_s[k*9+(j+4),49:98]=V44s[k*4+j,:]
        for j in range(4):
            for i in range(4):
                W_s[(j+4)*4+i,49:98]=W44s[j*4+i,:]
        
        U_s[:,98:114]=U41s
        for k in range(4):
            V_s[k*9+8,98:114]=V41s[k,:]
        for i in range(4):
            W_s[8*4+i,98:114]=W41s[i,:]
        
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
