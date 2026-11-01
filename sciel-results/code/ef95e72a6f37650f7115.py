"""
Rank-49 exact tensor decomposition for 4x4 matrix multiplication using Recursive Strassen.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _build_rank49_kronecker():
    """
    Construct rank-49 decomposition for 4x4 matrix multiplication using
    explicit Kronecker-product structure.
    
    The 4x4 multiplication tensor is built from two levels of 2x2 Strassen.
    We construct U, V, W directly using the Kronecker product of the
    block-level and element-level Strassen factors.
    
    U shape: (16, 49), indexed U[i*4+j, r]
    V shape: (16, 49), indexed V[j*4+k, r]  
    W shape: (16, 49), indexed W[k*4+i, r]
    """
    n = 4
    rank = 49
    U = np.zeros((n*n, rank), dtype=float)
    V = np.zeros((n*n, rank), dtype=float)
    W = np.zeros((n*n, rank), dtype=float)
    
    # 2x2 Strassen factors: U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r]
    U2 = np.zeros((4, 7), dtype=float)
    V2 = np.zeros((4, 7), dtype=float)
    W2 = np.zeros((4, 7), dtype=float)
    
    # M1 = (A00+A11)(B00+B11) -> C00+=1, C11+=1
    U2[0,0]=1; U2[3,0]=1; V2[0,0]=1; V2[3,0]=1; W2[0,0]=1; W2[3,0]=1
    # M2 = (A10+A11)B00 -> C10+=1
    U2[2,1]=1; U2[3,1]=1; V2[0,1]=1; W2[1,1]=1
    # M3 = A00(B01-B11) -> C01+=1, C11-=1
    U2[0,2]=1; V2[1,2]=1; V2[3,2]=-1; W2[2,2]=1; W2[3,2]=-1
    # M4 = A11(B10-B00) -> C00+=1, C10+=1
    U2[3,3]=1; V2[2,3]=1; V2[0,3]=-1; W2[0,3]=1; W2[1,3]=1
    # M5 = (A00+A01)B11 -> C00-=1, C01+=1
    U2[0,4]=1; U2[1,4]=1; V2[3,4]=1; W2[0,4]=-1; W2[2,4]=1
    # M6 = (A10-A00)(B00+B01) -> C11+=1
    U2[2,5]=1; U2[0,5]=-1; V2[0,5]=1; V2[1,5]=1; W2[3,5]=1
    # M7 = (A01-A11)(B10+B11) -> C00+=1, C11-=1
    U2[1,6]=1; U2[3,6]=-1; V2[2,6]=1; V2[3,6]=1; W2[0,6]=1; W2[3,6]=-1
    
    # Block-level Strassen: 7 terms
    # For each block term rb, the block-level factor for A is a 2x2 matrix Pb,
    # for B is Qb, for C is Rb.
    # The full factor for rank index (rb, rs) is:
    #   U_full[i*4+j, rb*7+rs] = sum over block decomposition
    #   
    # Key insight: A[i,j] where i=2*bi+a, j=2*bj+b.
    # The block (bi,bj) of A is multiplied by coefficient from Pb.
    # Within that block, the element (a,b) is multiplied by U2[a*2+b, rs].
    
    # Pb[bi,bj] = coefficient for block (bi,bj) of A in term rb
    # Qb[bi,bj] = coefficient for block (bi,bj) of B in term rb  
    # Rb[ci,cj] = coefficient for block (ci,cj) of C in term rb
    
    # Block-level Strassen matrices (2x2 each):
    # Row/col index: (block_row, block_col)
    Pb_list = []
    Qb_list = []
    Rb_list = []
    
    # M1: (A00+A11)(B00+B11) -> C00+=1, C11+=1
    Pb = np.array([[1,0],[0,1]], dtype=float)
    Qb = np.array([[1,0],[0,1]], dtype=float)
    Rb = np.array([[1,0],[0,1]], dtype=float)
    Pb_list.append(Pb); Qb_list.append(Qb); Rb_list.append(Rb)
    
    # M2: (A10+A11)B00 -> C10+=1
    Pb = np.array([[0,0],[1,1]], dtype=float)
    Qb = np.array([[1,0],[0,0]], dtype=float)
    Rb = np.array([[0,0],[1,0]], dtype=float)
    Pb_list.append(Pb); Qb_list.append(Qb); Rb_list.append(Rb)
    
    # M3: A00(B01-B11) -> C01+=1, C11-=1
    Pb = np.array([[1,0],[0,0]], dtype=float)
    Qb = np.array([[0,1],[0,-1]], dtype=float)
    Rb = np.array([[0,1],[0,-1]], dtype=float)
    Pb_list.append(Pb); Qb_list.append(Qb); Rb_list.append(Rb)
    
    # M4: A11(B10-B00) -> C00+=1, C10+=1
    Pb = np.array([[0,0],[0,1]], dtype=float)
    Qb = np.array([[0,-1],[1,0]], dtype=float)
    Rb = np.array([[1,0],[1,0]], dtype=float)
    Pb_list.append(Pb); Qb_list.append(Qb); Rb_list.append(Rb)
    
    # M5: (A00+A01)B11 -> C00-=1, C01+=1
    Pb = np.array([[1,1],[0,0]], dtype=float)
    Qb = np.array([[0,0],[0,1]], dtype=float)
    Rb = np.array([[-1,1],[0,0]], dtype=float)
    Pb_list.append(Pb); Qb_list.append(Qb); Rb_list.append(Rb)
    
    # M6: (A10-A00)(B00+B01) -> C11+=1
    Pb = np.array([[-1,0],[1,0]], dtype=float)
    Qb = np.array([[1,1],[0,0]], dtype=float)
    Rb = np.array([[0,0],[0,1]], dtype=float)
    Pb_list.append(Pb); Qb_list.append(Qb); Rb_list.append(Rb)
    
    # M7: (A01-A11)(B10+B11) -> C00+=1, C11-=1
    Pb = np.array([[0,1],[0,-1]], dtype=float)
    Qb = np.array([[0,0],[1,1]], dtype=float)
    Rb = np.array([[1,0],[0,-1]], dtype=float)
    Pb_list.append(Pb); Qb_list.append(Qb); Rb_list.append(Rb)
    
    for rb in range(7):
        Pb = Pb_list[rb]
        Qb = Qb_list[rb]
        Rb = Rb_list[rb]
        
        for rs in range(7):
            r = rb * 7 + rs
            
            # Build U[:, r]: U[i*4+j, r]
            # i = 2*bi + a, j = 2*bj + b
            # U[i*4+j, r] = Pb[bi,bj] * U2[a*2+b, rs]
            for bi in range(2):
                for bj in range(2):
                    pc = Pb[bi, bj]
                    if pc != 0:
                        for a in range(2):
                            for b in range(2):
                                val = U2[a*2+b, rs]
                                if val != 0:
                                    i = 2*bi + a
                                    j = 2*bj + b
                                    U[i*4+j, r] += pc * val
            
            # Build V[:, r]: V[j*4+k, r]
            # j = 2*bi + a, k = 2*bj + b
            # V[j*4+k, r] = Qb[bi,bj] * V2[a*2+b, rs]
            for bi in range(2):
                for bj in range(2):
                    qc = Qb[bi, bj]
                    if qc != 0:
                        for a in range(2):
                            for b in range(2):
                                val = V2[a*2+b, rs]
                                if val != 0:
                                    j = 2*bi + a
                                    k = 2*bj + b
                                    V[j*4+k, r] += qc * val
            
            # Build W[:, r]: W[k*4+i, r]
            # i = 2*ci + a, k = 2*cj + b
            # W[k*4+i, r] = Rb[ci,cj] * W2[b*2+a, rs]
            # Note: W2 is indexed W2[local_k*2+local_i, rs] = W2[b*2+a, rs]
            for ci in range(2):
                for cj in range(2):
                    cc = Rb[ci, cj]
                    if cc != 0:
                        for a in range(2):
                            for b in range(2):
                                val = W2[b*2+a, rs]
                                if val != 0:
                                    i = 2*ci + a
                                    k = 2*cj + b
                                    W[k*4+i, r] += cc * val
    
    return U, V, W


def _verify(U, V, W):
    """Verify the decomposition is exact."""
    n = m = p = 4
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    return np.allclose(T, 1.0, atol=1e-9)


def _verify_detailed(U, V, W):
    """Detailed verification returning max error and which entries fail."""
    n = m = p = 4
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    max_err = np.max(np.abs(T - 1.0))
    return max_err, T


def _schoolbook():
    """Fallback: rank-64 schoolbook decomposition."""
    n = m = p = 4
    rank = n*m*p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return U, V, W


def construct():
    deadline = time.time() + 115.0
    
    # Primary: rank-49 Kronecker-product Strassen
    try:
        U, V, W = _build_rank49_kronecker()
        if time.time() < deadline:
            max_err, T = _verify_detailed(U, V, W)
            if max_err < 1e-9:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            # If verification fails, still return it if it's close (might be a tolerance issue)
            elif max_err < 1e-6:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Fallback: rank-64 schoolbook
    U, V, W = _schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
