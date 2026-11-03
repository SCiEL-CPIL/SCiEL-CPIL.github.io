import numpy as np
import time
import random

def construct():
    n, m, p = 2, 3, 8
    start_time = time.time()
    deadline = start_time + 140
    
    # Build the multiplication tensor
    tensor = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor[i*m+j, j*p+k, k*n+i] = 1
    
    # Known 44-rank decomposition
    U44 = np.zeros((n*m, 44), dtype=np.int64)
    V44 = np.zeros((m*p, 44), dtype=np.int64)
    W44 = np.zeros((p*n, 44), dtype=np.int64)
    col = 0
    for i in range(2):
        for k in range(8):
            U44[i*3+0, col] = 1
            V44[0*8+k, col] = 1
            W44[k*2+i, col] = 1
            col += 1
    for block in range(4):
        U44[1, col]=1; U44[5, col]=1; V44[8+block*2, col]=1; V44[16+block*2+1, col]=1; W44[(block*2)*2+0, col]=1; W44[(block*2+1)*2+1, col]=1; col+=1
        U44[4, col]=1; U44[5, col]=1; V44[8+block*2, col]=1; W44[(block*2)*2+1, col]=1; W44[(block*2+1)*2+1, col]=-1; col+=1
        U44[1, col]=1; V44[8+block*2+1, col]=1; V44[16+block*2+1, col]=-1; W44[(block*2+1)*2+0, col]=1; W44[(block*2+1)*2+1, col]=1; col+=1
        U44[5, col]=1; V44[16+block*2, col]=1; V44[8+block*2, col]=-1; W44[(block*2)*2+0, col]=1; W44[(block*2)*2+1, col]=1; col+=1
        U44[1, col]=1; U44[2, col]=1; V44[16+block*2+1, col]=1; W44[(block*2)*2+0, col]=-1; W44[(block*2+1)*2+0, col]=1; col+=1
        U44[4, col]=1; U44[1, col]=-1; V44[8+block*2, col]=1; V44[8+block*2+1, col]=1; W44[(block*2+1)*2+1, col]=1; col+=1
        U44[2, col]=1; U44[5, col]=-1; V44[16+block*2, col]=1; V44[16+block*2+1, col]=1; W44[(block*2)*2+0, col]=1; col+=1
    
    best_rank = 44
    best_U, best_V, best_W = U44, V44, W44
    
    # Try to find lower rank via random search
    random.seed(123)
    np.random.seed(123)
    
    for target_rank in [43, 42, 41, 40]:
        if time.time() > deadline - 10:
            break
        for attempt in range(500):
            if time.time() > deadline - 10:
                break
            U_r = np.random.randint(-2, 3, (n*m, target_rank))
            V_r = np.random.randint(-2, 3, (m*p, target_rank))
            W_r = np.random.randint(-2, 3, (p*n, target_rank))
            
            # Check if this decomposes the tensor
            ok = True
            for r in range(target_rank):
                pass  # We need to check the full sum
            
            # Efficient check: compute the tensor reconstruction
            # T[i,j,k] = sum_r U_r[i,r] * V_r[j,r] * W_r[k,r]
            recon = np.tensordot(U_r, V_r, axes=([1],[1]))  # shape (n*m, m*p, target_rank)
            recon = np.tensordot(recon, W_r, axes=([2],[1]))  # shape (n*m, m*p, p*n)
            
            if np.array_equal(recon, tensor):
                best_rank = target_rank
                best_U, best_V, best_W = U_r, V_r, W_r
                break
        if best_rank == target_rank:
            break
    
    # Convert to list format
    U_out = best_U.T.tolist()
    V_out = best_V.T.tolist()
    W_out = best_W.T.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
