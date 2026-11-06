import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 5, 5, 5
    
    # Block sizes: 5 = 2 + 2 + 1
    block_sizes = [2, 2, 1]
    offsets = [0]
    for s in block_sizes:
        offsets.append(offsets[-1] + s)
    
    # Strassen factors for 2x2x2
    # U, V, W are flattened 2x2 matrices
    # Index mapping: [r0c0, r0c1, r1c0, r1c1]
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
                ni, mj, pk = block_sizes[i], block_sizes[j], block_sizes[k]
                if ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * mj * pk
    
    # Allocate
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ni, mj, pk = block_sizes[i], block_sizes[j], block_sizes[k]
                
                if ni == 2 and mj == 2 and pk == 2:
                    for (uf, vf, wf) in strassen_factors:
                        for idx in range(4):
                            if uf[idx] != 0:
                                lr, lc = idx // 2, idx % 2
                                U[(offsets[i] + lr) * m + (offsets[j] + lc), col] += uf[idx]
                            if vf[idx] != 0:
                                lr, lc = idx // 2, idx % 2
                                V[(offsets[j] + lr) * p + (offsets[k] + lc), col] += vf[idx]
                            if wf[idx] != 0:
                                lr, lc = idx // 2, idx % 2
                                W[(offsets[k] + lc) * n + (offsets[i] + lr), col] += wf[idx]
                        col += 1
                else:
                    for li in range(ni):
                        for lj in range(mj):
                            for lk in range(pk):
                                U[(offsets[i] + li) * m + (offsets[j] + lj), col] = 1.0
                                V[(offsets[j] + lj) * p + (offsets[k] + lk), col] = 1.0
                                W[(offsets[k] + lk) * n + (offsets[i] + li), col] = 1.0
                                col += 1
    
    if time.time() > deadline - 5.0:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
