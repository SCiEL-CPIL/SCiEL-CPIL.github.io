"""Matrix multiplication 3x5x6: rank-minimal exact tensor decomposition."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 5, 6]

def _strassen_2x2x2():
    """
    Return the 7 rank-1 components for 2x2x2 matrix multiplication.
    Each component is (u, v, w) where u is a 4-dim vector (flattened 2x2),
    v is a 4-dim vector, w is a 4-dim vector.
    The multiplication tensor is sum_c u_c ⊗ v_c ⊗ w_c.
    """
    # Strassen's 7 products for (A*B) where A, B are 2x2
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = A*B = [[c11, c12], [c21, c22]]
    #
    # Using standard Strassen:
    # M1 = (a11+a22)*(b11+b22)
    # M2 = (a21+a22)*b11
    # M3 = a11*(b12-b22)
    # M4 = a22*(b21-b11)
    # M5 = (a11+a12)*b22
    # M6 = (a21-a11)*(b11+b12)
    # M7 = (a12-a22)*(b21+b22)
    #
    # c11 = M1 - M4 + M5 - M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M7
    #
    # For each M_c, we need u_c (coefficients of A entries), v_c (coefficients of B entries),
    # and w_c (coefficients of C entries) such that M_c = sum u_c[i,j]*A[i,j] * sum v_c[i,j]*B[i,j]
    # and M_c contributes to C entries via w_c.
    #
    # The tensor T[i,j,k] = sum_c u_c[i,j] * v_c[j',k'] ... wait, let me be more careful.
    #
    # Actually for matrix multiplication: C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor is T[i,j,k] = 1 for all i,j,k (it's the identity in the middle index sense)
    # Actually T[i,j,k] = delta_{j, j} ... no.
    #
    # Let me think again. The matrix multiplication tensor for n x m * m x p -> n x p is:
    # T[i, j, k] where i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B multiply to contribute to (i,k) entry of C.
    # So T[i,j,k] = 1 for all valid i,j,k.
    #
    # A rank-1 decomposition: T = sum_c u_c ⊗ v_c ⊗ w_c
    # where u_c is in R^{n*m}, v_c in R^{m*p}, w_c in R^{n*p}
    # T[i,j,k] = sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i]
    #
    # For the 2x2x2 case: n=m=p=2
    # T[i,j,k] = 1 for all i,j,k in {0,1}
    #
    # Strassen gives 7 rank-1 terms. Let me write them out.
    # For M1 = (a11+a22)*(b11+b22):
    #   u1 = [1, 0, 0, 1] (a11 + a22)
    #   v1 = [1, 0, 0, 1] (b11 + b22)
    #   Contribution to C: c11 += 1, c12 += 1, c21 += 1, c22 += 1
    #   w1 = [1, 1, 1, 1] (c11 + c12 + c21 + c22)
    #
    # Wait, that's not right either. Let me be very explicit.
    #
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    #
    # Strassen:
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1])*B[0,0]
    # M3 = A[0,0]*(B[0,1]-B[1,1])
    # M4 = A[1,1]*(B[1,0]-B[0,0])
    # M5 = (A[0,0]+A[0,1])*B[1,1]
    # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    #
    # C[0,0] = M1 - M4 + M5 - M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M7
    #
    # So for each M_c, the rank-1 term is:
    # u_c[i*m+j] = coefficient of A[i,j] in M_c
    # v_c[j*p+k] = coefficient of B[j,k] in M_c
    # w_c[k*n+i] = coefficient of M_c in C[i,k]
    #
    # Note: the W index is k*n+i (trace convention W[k*n+i, r])
    
    n2, m2, p2 = 2, 2, 2
    components = []
    
    # M1 = (A00+A11)*(B00+B11)
    # u: A00=1, A01=0, A10=0, A11=1 -> [1,0,0,1]
    # v: B00=1, B01=0, B10=0, B11=1 -> [1,0,0,1]
    # w: C00+=1, C01+=1, C10+=1, C11+=1 -> w[k*n+i]: C[0,0]=w[0], C[0,1]=w[2], C[1,0]=w[1], C[1,1]=w[3]
    # w = [1, 1, 1, 1]
    components.append(([1,0,0,1], [1,0,0,1], [1,1,1,1]))
    
    # M2 = (A10+A11)*B00
    # u: A00=0, A01=0, A10=1, A11=1 -> [0,0,1,1]
    # v: B00=1, B01=0, B10=0, B11=0 -> [1,0,0,0]
    # w: C10+=1 -> w[1]=1, others 0 -> [0,1,0,0]
    components.append(([0,0,1,1], [1,0,0,0], [0,1,0,0]))
    
    # M3 = A00*(B01-B11)
    # u: A00=1 -> [1,0,0,0]
    # v: B01=1, B11=-1 -> [0,1,0,-1]
    # w: C01+=1 -> w[2]=1 -> [0,0,1,0]
    components.append(([1,0,0,0], [0,1,0,-1], [0,0,1,0]))
    
    # M4 = A11*(B10-B00)
    # u: A11=1 -> [0,0,0,1]
    # v: B10=1, B00=-1 -> [-1,0,1,0]
    # w: C00-=1, C10+=1 -> w[0]=-1, w[1]=1 -> [-1,1,0,0]
    components.append(([0,0,0,1], [-1,0,1,0], [-1,1,0,0]))
    
    # M5 = (A00+A01)*B11
    # u: A00=1, A01=1 -> [1,1,0,0]
    # v: B11=1 -> [0,0,0,1]
    # w: C00+=1, C01+=1 -> w[0]=1, w[2]=1 -> [1,0,1,0]
    components.append(([1,1,0,0], [0,0,0,1], [1,0,1,0]))
    
    # M6 = (A10-A00)*(B00+B01)
    # u: A10=1, A00=-1 -> [-1,0,1,0]
    # v: B00=1, B01=1 -> [1,1,0,0]
    # w: C11+=1 -> w[3]=1 -> [0,0,0,1]
    components.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
    
    # M7 = (A01-A11)*(B10+B11)
    # u: A01=1, A11=-1 -> [0,1,0,-1]
    # v: B10=1, B11=1 -> [0,0,1,1]
    # w: C00-=1, C11+=1 -> w[0]=-1, w[3]=1 -> [-1,0,0,1]
    components.append(([0,1,0,-1], [0,0,1,1], [-1,0,0,1]))
    
    return components

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c] equals the matmul tensor."""
    rank = len(U[0])
    # Reconstruct tensor
    T = np.zeros((n*m, m*p, p*n))
    for c in range(rank):
        u = np.array(U[:, c], dtype=float)
        v = np.array(V[:, c], dtype=float)
        w = np.array(W[:, c], dtype=float)
        T += np.einsum('a,b,c->abc', u, v, w)
    
    # Target tensor: T[i*m+j, j*p+k, k*n+i] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if T[i*m+j, j*p+k, k*n+i] != 1.0:
                    return False
                # Check that other entries are 0
    # Check all other entries are 0
    for idx in range(n*m):
        for idy in range(m*p):
            for idz in range(p*n):
                i = idx // m
                j = idx % m
                j2 = idy // p
                k = idy % p
                k2 = idz // n
                i2 = idz % n
                expected = 1.0 if (j == j2 and i == i2) else 0.0
                if abs(T[idx, idy, idz] - expected) > 1e-9:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Get Strassen components
    strassen = _strassen_2x2x2()
    
    # Try block decomposition: split n=3 as 2+1, m=5 as 2+2+1, p=6 as 2+2+2
    # For each pair of blocks (n_block, m_block, p_block), we need to multiply
    # n_block x m_block by m_block x p_block
    # If all are 2x2x2, we use Strassen (rank 7)
    # If any dimension is 1, we use schoolbook (rank = product)
    
    best_result = None
    best_rank = n * m * p  # Schoolbook rank = 90
    
    # Schoolbook as fallback
    U_school = [[0] * (n*m*p) for _ in range(n*m)]
    V_school = [[0] * (n*m*p) for _ in range(m*p)]
    W_school = [[0] * (n*m*p) for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    if _verify_decomposition(U_school, V_school, W_school, n, m, p):
        best_result = {"U": U_school, "V": V_school, "W": W_school}
        best_rank = n * m * p
    
    # Block decomposition approach
    # Split n=3 into [2,1], m=5 into [2,2,1], p=6 into [2,2,2]
    n_blocks = [2, 1]
    m_blocks = [2, 2, 1]
    p_blocks = [2, 2, 2]
    
    # For each combination of (n_block_idx, m_block_idx, p_block_idx)
    # where n_block has size nb, m_block has size mb, p_block has size pb
    # We compute nb x mb * mb x pb
    # If nb=mb=pb=2: use Strassen (7 terms)
    # Otherwise: schoolbook (nb*mb*pb terms)
    
    # We need to track which rows/cols of U, V, W are used
    # U has rows indexed by (i, j) where i in [0,n), j in [0,m)
    # V has rows indexed by (j, k) where j in [0,m), k in [0,p)
    # W has rows indexed by (k, i) where k in [0,p), i in [0,n)
    
    # For a block (i_start, i_end, j_start, j_end, k_start, k_end):
    # U rows: i in [i_start, i_end), j in [j_start, j_end)
    # V rows: j in [j_start, j_end), k in [k_start, k_end)
    # W rows: k in [k_start, k_end), i in [i_start, i_end)
    
    # Let's enumerate all possible block partitions
    # For simplicity, let's try the specific partition: n=[2,1], m=[2,2,1], p=[2,2,2]
    
    def try_partition(n_parts, m_parts, p_parts):
        """Try a specific partition and compute total rank."""
        # Compute starting offsets
        n_offsets = [0]
        for size in n_parts:
            n_offsets.append(n_offsets[-1] + size)
        m_offsets = [0]
        for size in m_parts:
            m_offsets.append(m_offsets[-1] + size)
        p_offsets = [0]
        for size in p_parts:
            p_offsets.append(p_offsets[-1] + size)
        
        # For each combination of block indices
        total_rank = 0
        for ni in range(len(n_parts)):
            for mi in range(len(m_parts)):
                for pi in range(len(p_parts)):
                    nb = n_parts[ni]
                    mb = m_parts[mi]
                    pb = p_parts[pi]
                    if nb == 2 and mb == 2 and pb == 2:
                        total_rank += 7
                    else:
                        total_rank += nb * mb * pb
        return total_rank
    
    # Try the known good partition
    rank_est = try_partition(n_blocks, m_blocks, p_blocks)
    if rank_est < best_rank:
        # Build the decomposition
        U = [[0] * rank_est for _ in range(n*m)]
        V = [[0] * rank_est for _ in range(m*p)]
        W = [[0] * rank_est for _ in range(p*n)]
        
        # Compute offsets
        n_offsets = [0]
        for size in n_blocks:
            n_offsets.append(n_offsets[-1] + size)
        m_offsets = [0]
        for size in m_blocks:
            m_offsets.append(m_offsets[-1] + size)
        p_offsets = [0]
        for size in p_blocks:
            p_offsets.append(p_offsets[-1] + size)
        
        col = 0
        for ni in range(len(n_blocks)):
            for mi in range(len(m_blocks)):
                for pi in range(len(p_blocks)):
                    nb = n_blocks[ni]
                    mb = m_blocks[mi]
                    pb = p_blocks[pi]
                    i_start = n_offsets[ni]
                    j_start = m_offsets[mi]
                    k_start = p_offsets[pi]
                    
                    if nb == 2 and mb == 2 and pb == 2:
                        # Use Strassen
                        for (u_comp, v_comp, w_comp) in strassen:
                            for a in range(nb):
                                for b in range(mb):
                                    u_idx = (i_start + a) * m + (j_start + b)
                                    U[u_idx][col] = u_comp[a*mb + b]
                            for a in range(mb):
                                for b in range(pb):
                                    v_idx = (j_start + a) * p + (k_start + b)
                                    V[v_idx][col] = v_comp[a*pb + b]
                            for a in range(pb):
                                for b in range(nb):
                                    w_idx = (k_start + a) * n + (i_start + b)
                                    W[w_idx][col] = w_comp[a*nb + b]
                            col += 1
                    else:
                        # Schoolbook
                        for a in range(nb):
                            for b in range(mb):
                                for c in range(pb):
                                    u_idx = (i_start + a) * m + (j_start + b)
                                    v_idx = (j_start + b) * p + (k_start + c)
                                    w_idx = (k_start + c) * n + (i_start + a)
                                    U[u_idx][col] = 1
                                    V[v_idx][col] = 1
                                    W[w_idx][col] = 1
                                    col += 1
        
        if col != rank_est:
            # Mismatch, skip
            pass
        elif _verify_decomposition(U, V, W, n, m, p):
            best_result = {"U": U, "V": V, "W": W}
            best_rank = rank_est
    
    # Try other partitions to see if we can do better
    # Generate all possible partitions of n=3, m=5, p=6
    def get_partitions(total, max_parts=None):
        """Generate all compositions of total into positive integers."""
        if max_parts is None:
            max_parts = total
        def gen(remaining, parts, current):
            if remaining == 0:
                yield list(current)
                return
            if len(current) >= max_parts:
                return
            for val in range(1, remaining + 1):
                current.append(val)
                yield from gen(remaining - val, parts, current)
                current.pop()
        return list(gen(total, [], []))
    
    n_partitions = get_partitions(3)
    m_partitions = get_partitions(5)
    p_partitions = get_partitions(6)
    
    # Try a subset of partitions (limited by time)
    tried = set()
    for npart in n_partitions:
        for mpart in m_partitions:
            for ppart in p_partitions:
                key = (tuple(npart), tuple(mpart), tuple(ppart))
                if key in tried:
                    continue
                tried.add(key)
                
                if time.time() > deadline - 10:
                    break
                
                rank_est = try_partition(npart, mpart, ppart)
                if rank_est < best_rank:
                    # Build and verify
                    U = [[0] * rank_est for _ in range(n*m)]
                    V = [[0] * rank_est for _ in range(m*p)]
                    W = [[0] * rank_est for _ in range(p*n)]
                    
                    n_offsets = [0]
                    for size in npart:
                        n_offsets.append(n_offsets[-1] + size)
                    m_offsets = [0]
                    for size in mpart:
                        m_offsets.append(m_offsets[-1] + size)
                    p_offsets = [0]
                    for size in ppart:
                        p_offsets.append(p_offsets[-1] + size)
                    
                    col = 0
                    for ni in range(len(npart)):
                        for mi in range(len(mpart)):
                            for pi in range(len(ppart)):
                                nb = npart[ni]
                                mb = mpart[mi]
                                pb = ppart[pi]
                                i_start = n_offsets[ni]
                                j_start = m_offsets[mi]
                                k_start = p_offsets[pi]
                                
                                if nb == 2 and mb == 2 and pb == 2:
                                    for (u_comp, v_comp, w_comp) in strassen:
                                        for a in range(nb):
                                            for b in range(mb):
                                                u_idx = (i_start + a) * m + (j_start + b)
                                                U[u_idx][col] = u_comp[a*mb + b]
                                        for a in range(mb):
                                            for b in range(pb):
                                                v_idx = (j_start + a) * p + (k_start + b)
                                                V[v_idx][col] = v_comp[a*pb + b]
                                        for a in range(pb):
                                            for b in range(nb):
                                                w_idx = (k_start + a) * n + (i_start + b)
                                                W[w_idx][col] = w_comp[a*nb + b]
                                        col += 1
                                else:
                                    for a in range(nb):
                                        for b in range(mb):
                                            for c in range(pb):
                                                u_idx = (i_start + a) * m + (j_start + b)
                                                v_idx = (j_start + b) * p + (k_start + c)
                                                w_idx = (k_start + c) * n + (i_start + a)
                                                U[u_idx][col] = 1
                                                V[v_idx][col] = 1
                                                W[w_idx][col] = 1
                                                col += 1
                    
                    if col == rank_est and _verify_decomposition(U, V, W, n, m, p):
                        best_result = {"U": U, "V": V, "W": W}
                        best_rank = rank_est
                
                if time.time() > deadline - 10:
                    break
            if time.time() > deadline - 10:
                break
        if time.time() > deadline - 10:
            break
    
    if best_result is None:
        # Fallback to schoolbook
        best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    return best_result
