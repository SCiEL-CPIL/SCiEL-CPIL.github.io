import numpy as np
import time
import random

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Generate all possible blocks
    blocks = []
    for I in [ {0,1}, {1,2}, {0,2} ]:
        for J in [ {j, j+1} for j in range(m-1) ] + [ {j, j+2} for j in range(m-2) ] + [ {j, j+3} for j in range(m-3) ] + [ {j, j+4} for j in range(m-4) ]:
            # Actually just all pairs
            pass
        # Simpler:
        for j1 in range(m):
            for j2 in range(j1+1, m):
                J = {j1, j2}
                for k1 in range(p):
                    for k2 in range(k1+1, p):
                        K = {k1, k2}
                        blocks.append((I, J, K))
                        
    # Remove duplicates? The generation above might have duplicates if not careful, but sets are unique.
    # Actually, the list comprehension for J above was messy. Let's just generate all pairs.
    blocks = []
    I_sets = [ {0,1}, {1,2}, {0,2} ]
    J_sets = [ {j1, j2} for j1 in range(m) for j2 in range(j1+1, m) ]
    K_sets = [ {k1, k2} for k1 in range(p) for k2 in range(k1+1, p) ]
    
    for I in I_sets:
        for J in J_sets:
            for K in K_sets:
                blocks.append((I, J, K))
                
    def are_disjoint(b1, b2):
        I1, J1, K1 = b1
        I2, J2, K2 = b2
        if I1 & I2:
            if (J1 & J2) and (K1 & K2):
                return False
        return True
        
    best_blocks = []
    
    # Greedy search with random restarts
    num_restarts = 50
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
        random.shuffle(blocks)
        current_blocks = []
        for b in blocks:
            if all(are_disjoint(b, cb) for cb in current_blocks):
                current_blocks.append(b)
        if len(current_blocks) > len(best_blocks):
            best_blocks = current_blocks
            if len(best_blocks) >= 10: # Good enough?
                break
                
    # Construct decomposition
    rank = n * m * p - len(best_blocks)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    covered = set()
    
    for I, J, K in best_blocks:
        i0, i1 = sorted(I)
        j0, j1 = sorted(J)
        k0, k1 = sorted(K)
        
        # Mark covered
        for i in I:
            for j in J:
                for k in K:
                    covered.add((i, j, k))
                    
        # Add Strassen terms
        # 1
        U[i0*m + j0, col] += 1
        U[i1*m + j1, col] += 1
        V[j0*p + k0, col] += 1
        V[j1*p + k1, col] += 1
        W[k0*n + i0, col] += 1
        W[k1*n + i1, col] += 1
        col += 1
        
        # 2
        U[i1*m + j0, col] += 1
        U[i1*m + j1, col] += 1
        V[j0*p + k0, col] += 1
        W[k0*n + i1, col] += 1
        W[k1*n + i1, col] -= 1
        col += 1
        
        # 3
        U[i0*m + j0, col] += 1
        V[j0*p + k1, col] += 1
        V[j1*p + k1, col] -= 1
        W[k1*n + i0, col] += 1
        W[k1*n + i1, col] += 1
        col += 1
        
        # 4
        U[i1*m + j1, col] += 1
        V[j1*p + k0, col] += 1
        V[j0*p + k0, col] -= 1
        W[k0*n + i0, col] += 1
        W[k0*n + i1, col] += 1
        col += 1
        
        # 5
        U[i0*m + j0, col] += 1
        U[i0*m + j1, col] += 1
        V[j1*p + k1, col] += 1
        W[k1*n + i0, col] += 1
        W[k0*n + i0, col] -= 1
        col += 1
        
        # 6
        U[i1*m + j0, col] += 1
        U[i0*m + j0, col] -= 1
        V[j0*p + k0, col] += 1
        V[j0*p + k1, col] += 1
        W[k1*n + i1, col] += 1
        col += 1
        
        # 7
        U[i0*m + j1, col] += 1
        U[i1*m + j1, col] -= 1
        V[j1*p + k0, col] += 1
        V[j1*p + k1, col] += 1
        W[k0*n + i0, col] += 1
        col += 1
        
        if time.time() > deadline:
            break
            
    # Add remaining schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    if time.time() > deadline:
                        break
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1

    if col < U.shape[1]:
        U = U[:, :col]
        V = V[:, :col]
        W = W[:, :col]
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
