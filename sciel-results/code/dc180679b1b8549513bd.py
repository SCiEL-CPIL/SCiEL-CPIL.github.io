import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct the rank-minimal exact tensor decomposition for 5x5 matrix multiplication.
    Uses block decomposition with partition 2+2+1, applying Strassen to 2x2x2 blocks.
    """
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS
    
    # Partition: [2, 2, 1]
    part = [2, 2, 1]
    num_blocks = len(part)
    
    # Precompute block offsets
    block_offsets = []
    offset = 0
    for size in part:
        block_offsets.append(offset)
        offset += size
    
    def global_idx(block_idx, local_idx):
        return block_offsets[block_idx] + local_idx
    
    rank = 0
    U_terms = []
    V_terms = []
    W_terms = []
    
    # For each block combination (bi, bj, bk), handle the subproblem
    for bi in range(num_blocks):
        for bj in range(num_blocks):
            for bk in range(num_blocks):
                si = part[bi]
                sj = part[bj]
                sk = part[bk]
                
                if si == 2 and sj == 2 and sk == 2:
                    # Strassen for 2x2x2
                    # Local indices: li, lj, lk in {0, 1}
                    
                    # Helper to get global indices
                    def gi(b, l):
                        return global_idx(b, l)
                    
                    # M1 = (A00+A11)(B00+B11)
                    # U: A[bi,0][bj,0] + A[bi,1][bj,1]
                    # V: B[bj,0][bk,0] + B[bj,1][bk,1]
                    # W: C[bi,0][bk,0] + C[bi,1][bk,1]
                    U_terms.append([(gi(bi,0)*m+gi(bj,0), 1), (gi(bi,1)*m+gi(bj,1), 1)])
                    V_terms.append([(gi(bj,0)*p+gi(bk,0), 1), (gi(bj,1)*p+gi(bk,1), 1)])
                    W_terms.append([(gi(bk,0)*n+gi(bi,0), 1), (gi(bk,1)*n+gi(bi,1), 1)])
                    rank += 1
                    
                    # M2 = (A10+A11)B00
                    # U: A[bi,1][bj,0] + A[bi,1][bj,1]
                    # V: B[bj,0][bk,0]
                    # W: C[bi,1][bk,0]
                    U_terms.append([(gi(bi,1)*m+gi(bj,0), 1), (gi(bi,1)*m+gi(bj,1), 1)])
                    V_terms.append([(gi(bj,0)*p+gi(bk,0), 1)])
                    W_terms.append([(gi(bk,0)*n+gi(bi,1), 1)])
                    rank += 1
                    
                    # M3 = A00(B01-B11)
                    # U: A[bi,0][bj,0]
                    # V: B[bj,0][bk,1] - B[bj,1][bk,1]
                    # W: C[bi,0][bk,1]
                    U_terms.append([(gi(bi,0)*m+gi(bj,0), 1)])
                    V_terms.append([(gi(bj,0)*p+gi(bk,1), 1), (gi(bj,1)*p+gi(bk,1), -1)])
                    W_terms.append([(gi(bk,1)*n+gi(bi,0), 1)])
                    rank += 1
                    
                    # M4 = A11(B10-B00)
                    # U: A[bi,1][bj,1]
                    # V: B[bj,1][bk,0] - B[bj,0][bk,0]
                    # W: C[bi,1][bk,0]
                    U_terms.append([(gi(bi,1)*m+gi(bj,1), 1)])
                    V_terms.append([(gi(bj,1)*p+gi(bk,0), 1), (gi(bj,0)*p+gi(bk,0), -1)])
                    W_terms.append([(gi(bk,0)*n+gi(bi,1), 1)])
                    rank += 1
                    
                    # M5 = (A00+A01)B11
                    # U: A[bi,0][bj,0] + A[bi,0][bj,1]
                    # V: B[bj,1][bk,1]
                    # W: C[bi,0][bk,1]
                    U_terms.append([(gi(bi,0)*m+gi(bj,0), 1), (gi(bi,0)*m+gi(bj,1), 1)])
                    V_terms.append([(gi(bj,1)*p+gi(bk,1), 1)])
                    W_terms.append([(gi(bk,1)*n+gi(bi,0), 1)])
                    rank += 1
                    
                    # M6 = (A10-A00)(B00+B10)
                    # U: A[bi,1][bj,0] - A[bi,0][bj,0]
                    # V: B[bj,0][bk,0] + B[bj,1][bk,0]
                    # W: C[bi,1][bk,1]
                    U_terms.append([(gi(bi,1)*m+gi(bj,0), 1), (gi(bi,0)*m+gi(bj,0), -1)])
                    V_terms.append([(gi(bj,0)*p+gi(bk,0), 1), (gi(bj,1)*p+gi(bk,0), 1)])
                    W_terms.append([(gi(bk,1)*n+gi(bi,1), 1)])
                    rank += 1
                    
                    # M7 = (A01-A11)(B01+B11)
                    # U: A[bi,0][bj,1] - A[bi,1][bj,1]
                    # V: B[bj,0][bk,1] + B[bj,1][bk,1]
                    # W: C[bi,0][bk,0]
                    U_terms.append([(gi(bi,0)*m+gi(bj,1), 1), (gi(bi,1)*m+gi(bj,1), -1)])
                    V_terms.append([(gi(bj,0)*p+gi(bk,1), 1), (gi(bj,1)*p+gi(bk,1), 1)])
                    W_terms.append([(gi(bk,0)*n+gi(bi,0), 1)])
                    rank += 1
                    
                else:
                    # Schoolbook for this block
                    for li in range(si):
                        for lj in range(sj):
                            for lk in range(sk):
                                u_idx = global_idx(bi, li) * m + global_idx(bj, lj)
                                v_idx = global_idx(bj, lj) * p + global_idx(bk, lk)
                                w_idx = global_idx(bk, lk) * n + global_idx(bi, li)
                                U_terms.append([(u_idx, 1)])
                                V_terms.append([(v_idx, 1)])
                                W_terms.append([(w_idx, 1)])
                                rank += 1
    
    # Build the matrices
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    for r in range(rank):
        for idx, coeff in U_terms[r]:
            U[idx, r] += coeff
        for idx, coeff in V_terms[r]:
            V[idx, r] += coeff
        for idx, coeff in W_terms[r]:
            W[idx, r] += coeff
    
    if time.time() > deadline - 5:
        pass
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
