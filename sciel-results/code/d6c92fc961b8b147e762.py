"""Matrix multiplication 5x5x5 via recursive block partitioning with Strassen."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """
    Return (U, V, W) for 2x2x2 matrix multiplication tensor.
    U shape (4, 7), V shape (4, 7), W shape (4, 7).
    Indexing: U[i*m + j, c], V[j*p + k, c], W[k*n + i, c] with n=m=p=2.
    
    Strassen's 7 products:
    M1 = (A11+A22)*(B11+B22)
    M2 = (A21+A22)*B11
    M3 = A11*(B12-B22)
    M4 = A22*(B21-B11)
    M5 = (A11+A12)*B22
    M6 = (A21-A11)*(B11+B12)
    M7 = (A12-A22)*(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    U[i*m+j, c] corresponds to A[i,j] coefficient in product c
    V[j*p+k, c] corresponds to B[j,k] coefficient in product c
    W[k*n+i, c] corresponds to C[i,k]... wait, W[k*n+i, c] is the output factor.
    
    Let me re-derive. The tensor T[i,j,k] = delta_{i,k} for matrix mult C[i,k] = sum_j A[i,j]B[j,k].
    
    Actually the standard tensor decomposition: T = sum_c U_c ⊗ V_c ⊗ W_c
    where (U_c)[i,j], (V_c)[j,k], (W_c)[k,i]... 
    
    Let me use the convention from the task:
    U has shape (n*m, rank), indexed U[i*m+j, c]
    V has shape (m*p, rank), indexed V[j*p+k, c]  
    W has shape (p*n, rank), indexed W[k*n+i, c]
    
    The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    For matrix multiplication, T[i,j,k] = 1 if i==k else 0 (since C[i,k]=sum_j A[i,j]B[j,k])
    
    So we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = delta_{i,k}
    
    For Strassen on 2x2:
    n=m=p=2
    U indices: i*m+j where i,j in {0,1}, so 0,1,2,3
    V indices: j*p+k where j,k in {0,1}, so 0,1,2,3
    W indices: k*n+i where k,i in {0,1}, so 0,1,2,3
    
    Products:
    M1: (A00+A11)(B00+B11) -> U: A00=1,A11=1; V: B00=1,B11=1; W: contributes to C00 and C11
    M2: (A10+A11)B00 -> U: A10=1,A11=1; V: B00=1; W: contributes to C10 and C11
    M3: A00(B01-B11) -> U: A00=1; V: B01=1,B11=-1; W: contributes to C01 and C11
    M4: A11(B10-B00) -> U: A11=1; V: B10=1,B00=-1; W: contributes to C00 and C10
    M5: (A00+A01)B11 -> U: A00=1,A01=1; V: B11=1; W: contributes to C00 and C01
    M6: (A10-A00)(B00+B01) -> U: A10=1,A00=-1; V: B00=1,B01=1; W: contributes to C11
    M7: (A01-A11)(B10+B11) -> U: A01=1,A11=-1; V: B10=1,B11=1; W: contributes to C00 and C11
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    W[k*n+i, c]:
    c=0 (M1): W[0*2+0]=W[0]=1 (C00), W[1*2+1]=W[3]=1 (C11)
    c=1 (M2): W[0*2+1]=W[1]=1 (C10), W[1*2+1]=W[3]=-1 (C11)
    c=2 (M3): W[0*2+1]=W[1]=1 (C01), W[1*2+1]=W[3]=1 (C11)
    c=3 (M4): W[0*2+0]=W[0]=1 (C00), W[0*2+1]=W[1]=1 (C10)
    c=4 (M5): W[0*2+0]=W[0]=-1 (C00), W[0*2+1]=W[1]=1 (C01)
    c=5 (M6): W[1*2+1]=W[3]=1 (C11)
    c=6 (M7): W[0*2+0]=W[0]=1 (C00), W[1*2+1]=W[3]=1 (C11)
    
    Let me build these matrices:
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # U[i*2+j, c] for A[i,j]
    # c=0: A00=1, A11=1
    U[0, 0] = 1  # A00
    U[3, 0] = 1  # A11
    # c=1: A10=1, A11=1
    U[2, 1] = 1  # A10
    U[3, 1] = 1  # A11
    # c=2: A00=1
    U[0, 2] = 1  # A00
    # c=3: A11=1
    U[3, 3] = 1  # A11
    # c=4: A00=1, A01=1
    U[0, 4] = 1  # A00
    U[1, 4] = 1  # A01
    # c=5: A10=1, A00=-1
    U[2, 5] = 1  # A10
    U[0, 5] = -1  # A00
    # c=6: A01=1, A11=-1
    U[1, 6] = 1  # A01
    U[3, 6] = -1  # A11
    
    # V[j*2+k, c] for B[j,k]
    # c=0: B00=1, B11=1
    V[0, 0] = 1  # B00
    V[3, 0] = 1  # B11
    # c=1: B00=1
    V[0, 1] = 1  # B00
    # c=2: B01=1, B11=-1
    V[1, 2] = 1  # B01
    V[3, 2] = -1  # B11
    # c=3: B10=1, B00=-1
    V[2, 3] = 1  # B10
    V[0, 3] = -1  # B00
    # c=4: B11=1
    V[3, 4] = 1  # B11
    # c=5: B00=1, B01=1
    V[0, 5] = 1  # B00
    V[1, 5] = 1  # B01
    # c=6: B10=1, B11=1
    V[2, 6] = 1  # B10
    V[3, 6] = 1  # B11
    
    # W[k*2+i, c] for C[i,k] -> W[k*n+i, c]
    # c=0 (M1): C00+=1, C11+=1
    W[0, 0] = 1  # k=0,i=0 -> C00
    W[3, 0] = 1  # k=1,i=1 -> C11
    # c=1 (M2): C10+=1, C11+=-1
    W[1, 1] = 1  # k=0,i=1 -> C10
    W[3, 1] = -1  # k=1,i=1 -> C11
    # c=2 (M3): C01+=1, C11+=1
    W[1, 2] = 1  # k=0,i=1... wait W[k*n+i], C[i,k]
    # C[i,k] where i=row, k=col. W[k*n+i] so for C01: i=0,k=1 -> W[1*2+0]=W[2]
    # Let me redo. W[k*n+i, c] where the output is C[i,k].
    # For C00: i=0,k=0 -> W[0*2+0]=W[0]
    # For C01: i=0,k=1 -> W[1*2+0]=W[2]
    # For C10: i=1,k=0 -> W[0*2+1]=W[1]
    # For C11: i=1,k=1 -> W[1*2+1]=W[3]
    
    # c=0 (M1): C00+=1, C11+=1
    W[0, 0] = 1   # C00
    W[3, 0] = 1   # C11
    # c=1 (M2): C10+=1, C11+=-1
    W[1, 1] = 1   # C10
    W[3, 1] = -1  # C11
    # c=2 (M3): C01+=1, C11+=1
    W[2, 2] = 1   # C01
    W[3, 2] = 1   # C11
    # c=3 (M4): C00+=1, C10+=1
    W[0, 3] = 1   # C00
    W[1, 3] = 1   # C10
    # c=4 (M5): C00+=-1, C01+=1
    W[0, 4] = -1  # C00
    W[2, 4] = 1   # C01
    # c=5 (M6): C11+=1
    W[3, 5] = 1   # C11
    # c=6 (M7): C00+=1, C11+=1
    W[0, 6] = 1   # C00
    W[3, 6] = 1   # C11
    
    return U, V, W


def construct():
    """
    Block-partitioned decomposition: split 5x5x5 into [2,2,1] x [2,2,1] x [2,2,1].
    Blocks:
      (2,2,2) x 4 blocks -> Strassen rank 7 each = 28
      (1,1,1) x 1 block  -> rank 1
    Total rank = 29
    """
    n, m, p = DIMENSIONS
    
    # Partition sizes
    part = [2, 2, 1]
    # Verify
    assert sum(part) == n and sum(part) == m and sum(part) == p
    
    # Build the full U, V, W
    rank = 4 * 7 + 1  # 29
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Get Strassen factors for 2x2x2
    Us, Vs, Ws = _strassen_2x2x2()
    
    # Block offsets
    offsets = [0, 2, 4]  # cumulative sums of [2,2,1]
    
    col = 0
    for ia in range(3):
        for ib in range(3):
            for ic in range(3):
                sa, sb, sc = part[ia], part[ib], part[ic]
                oa, ob, oc = offsets[ia], offsets[ib], offsets[ic]
                
                if sa == 2 and sb == 2 and sc == 2:
                    # Strassen block
                    for r in range(7):
                        # U: A[i,j] where i in [oa, oa+2), j in [ob, ob+2)
                        for i_local in range(2):
                            for j_local in range(2):
                                i_glob = oa + i_local
                                j_glob = ob + j_local
                                U[i_glob * m + j_glob, col + r] = Us[i_local * 2 + j_local, r]
                        # V: B[j,k] where j in [ob, ob+2), k in [oc, oc+2)
                        for j_local in range(2):
                            for k_local in range(2):
                                j_glob = ob + j_local
                                k_glob = oc + k_local
                                V[j_glob * p + k_glob, col + r] = Vs[j_local * 2 + k_local, r]
                        # W: C[i,k] where k in [oc, oc+2), i in [oa, oa+2)
                        for k_local in range(2):
                            for i_local in range(2):
                                k_glob = oc + k_local
                                i_glob = oa + i_local
                                W[k_glob * n + i_glob, col + r] = Ws[k_local * 2 + i_local, r]
                    col += 7
                elif sa == 1 and sb == 1 and sc == 1:
                    # Trivial 1x1x1 block
                    i_glob = oa
                    j_glob = ob
                    k_glob = oc
                    U[i_glob * m + j_glob, col] = 1
                    V[j_glob * p + k_glob, col] = 1
                    W[k_glob * n + i_glob, col] = 1
                    col += 1
                else:
                    # Mixed block sizes - need to handle carefully
                    # For blocks where not all dimensions are equal, we need a valid decomposition
                    # A block of size sa x sb x sc for matrix mult where sa==sc (since n==m==p and we partition equally)
                    # Actually in our partition all parts are the same [2,2,1], so sa==sb==sc always
                    pass
    
    # Convert to lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
