import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = 5, 5, 5
    
    # Use 2+2+1 partition with Strassen for 2x2x2 blocks
    # This reliably gives rank 117
    return _construct_221()

def _construct_221():
    """2+2+1 partition with Strassen for 2x2x2 blocks. Rank = 117."""
    n, m, p = 5, 5, 5
    block_sizes = [2, 2, 1]
    offsets = [0]
    for s in block_sizes:
        offsets.append(offsets[-1] + s)
    
    # Strassen factors for 2x2x2 (rank 7)
    # Each factor is (u, v, w) where u,v,w are length-4 vectors (2x2 matrices flattened)
    # Convention: u[i*2+j], v[j*2+k], w[k*2+i]
    strassen_factors = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    # Calculate total rank
    total_rank = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                if ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * mj * pk
    
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ni = block_sizes[i]
                mj = block_sizes[j]
                pk = block_sizes[k]
                
                if ni == 2 and mj == 2 and pk == 2:
                    # Strassen for 2x2x2
                    for (uf, vf, wf) in strassen_factors:
                        for idx in range(4):
                            if uf[idx] != 0:
                                lr = idx // 2
                                lc = idx % 2
                                U[(offsets[i] + lr) * m + (offsets[j] + lc), col] += uf[idx]
                            if vf[idx] != 0:
                                lr = idx // 2
                                lc = idx % 2
                                V[(offsets[j] + lr) * p + (offsets[k] + lc), col] += vf[idx]
                            if wf[idx] != 0:
                                lr = idx // 2
                                lc = idx % 2
                                W[(offsets[k] + lc) * n + (offsets[i] + lr), col] += wf[idx]
                        col += 1
                else:
                    # Schoolbook for mixed-size blocks
                    for li in range(ni):
                        for lj in range(mj):
                            for lk in range(pk):
                                U[(offsets[i] + li) * m + (offsets[j] + lj), col] = 1.0
                                V[(offsets[j] + lj) * p + (offsets[k] + lk), col] = 1.0
                                W[(offsets[k] + lk) * n + (offsets[i] + li), col] = 1.0
                                col += 1
    
    assert col == total_rank, f"Column mismatch: {col} != {total_rank}"
    
    # Fast vectorized verification
    # Tensor decomposition: sum_r U[i*5+j, r] * V[j*5+k, r] * W[k*5+i, r] = 1 for all i,j,k
    # Equivalently, for matrices A, B: C[i,k] = sum_j A[i,j]*B[j,k]
    # The decomposition computes: C[i,k] = sum_r (sum_j U[i*5+j,r]*A[i,j]) * (sum_k' V[j*5+k',r]*B[j,k']) * W[k*5+i,r]
    # Wait, let me think about this more carefully.
    # The matmul tensor is T[i,j,k] = 1 (delta function on the "middle" index)
    # Decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # So: sum_r U[i*5+j, r] * V[j*5+k, r] * W[k*5+i, r] should equal 1 for all i,j,k
    
    # Verify using a random test
    rng = np.random.default_rng(42)
    A_test = rng.standard_normal((5, 5))
    B_test = rng.standard_normal((5, 5))
    C_expected = A_test @ B_test
    
    # Compute C using the tensor decomposition
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_j sum_r A[i,j] * U[i*5+j,r] * V[j*5+k,r] * W[k*5+i,r]
    # Let's define: for each r, we have matrices U_r (5x5), V_r (5x5), W_r (5x5)
    # Then the contribution is: sum_r (U_r^T @ A) * (B @ V_r) * W_r ... no that's not right either.
    
    # Actually the correct interpretation:
    # C[i,k] = sum_r (sum_j A[i,j] * U[i*5+j, r]) * (sum_k' V[j*5+k', r] * B[j,k']) ... 
    # This is getting confusing. Let me just verify directly.
    
    # Direct check: for each (i,j,k), sum_r U[i*5+j,r] * V[j*5+k,r] * W[k*5+i,r] == 1
    # We can check this efficiently using einsum
    # Build the tensor: T[i,j,k] = sum_r U[i*5+j,r] * V[j*5+k,r] * W[k*5+i,r]
    
    # Reshape U, V, W to 3D: U3[i,j,r], V3[j,k,r], W3[k,i,r]
    U3 = U.reshape(5, 5, total_rank)  # U3[i, j, r]
    V3 = V.reshape(5, 5, total_rank)  # V3[j, k, r]
    W3 = W.reshape(5, 5, total_rank)  # W3[k, i, r]
    
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # Use einsum: 'ijr,jkr,kir->ijk'
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3, optimize=True)
    
    if not np.allclose(T, 1.0, atol=1e-10):
        # This should never happen with correct construction
        # Fall back to schoolbook
        U_school = np.zeros((25, 125), dtype=np.float64)
        V_school = np.zeros((25, 125), dtype=np.float64)
        W_school = np.zeros((25, 125), dtype=np.float64)
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    r = i * 25 + j * 5 + k
                    U_school[i * 5 + j, r] = 1.0
                    V_school[j * 5 + k, r] = 1.0
                    W_school[k * 5 + i, r] = 1.0
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
