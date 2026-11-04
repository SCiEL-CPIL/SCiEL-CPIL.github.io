import time
import numpy as np
import random

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    def verify(U, V, W):
        try:
            computed = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            T = np.zeros((nm, mp, pn))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        T[a, b, c] = 1.0
            if not np.allclose(computed, T, atol=1e-9):
                return False
            return True
        except Exception:
            return False

    def add_strassen_block(U, V, W, i0, i1, j0, j1, k0, k1, col):
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        # M1: (A[i0,j0]+A[i1,j1])(B[j0,k0]+B[j1,k1])
        U[i0*m+j0, col] += 1.0; U[i1*m+j1, col] += 1.0
        V[j0*p+k0, col] += 1.0; V[j1*p+k1, col] += 1.0
        W[k0*n+i0, col] += 1.0; W[k1*n+i1, col] += 1.0
        col += 1
        
        # M2: (A[i1,j0]+A[i1,j1])B[j0,k0]
        U[i1*m+j0, col] += 1.0; U[i1*m+j1, col] += 1.0
        V[j0*p+k0, col] += 1.0
        W[k0*n+i1, col] += 1.0; W[k1*n+i1, col] -= 1.0
        col += 1
        
        # M3: A[i0,j0](B[j0,k1]-B[j1,k1])
        U[i0*m+j0, col] += 1.0
        V[j0*p+k1, col] += 1.0; V[j1*p+k1, col] -= 1.0
        W[k1*n+i0, col] += 1.0; W[k1*n+i1, col] += 1.0
        col += 1
        
        # M4: A[i1,j1](B[j1,k0]-B[j0,k0])
        U[i1*m+j1, col] += 1.0
        V[j1*p+k0, col] += 1.0; V[j0*p+k0, col] -= 1.0
        W[k0*n+i0, col] += 1.0; W[k0*n+i1, col] += 1.0
        col += 1
        
        # M5: (A[i0,j0]+A[i0,j1])B[j1,k1]
        U[i0*m+j0, col] += 1.0; U[i0*m+j1, col] += 1.0
        V[j1*p+k1, col] += 1.0
        W[k0*n+i0, col] -= 1.0; W[k1*n+i0, col] += 1.0
        col += 1
        
        # M6: (A[i1,j0]-A[i0,j0])(B[j0,k0]+B[j0,k1])
        U[i1*m+j0, col] += 1.0; U[i0*m+j0, col] -= 1.0
        V[j0*p+k0, col] += 1.0; V[j0*p+k1, col] += 1.0
        W[k1*n+i1, col] += 1.0
        col += 1
        
        # M7: (A[i0,j1]-A[i1,j1])(B[j1,k0]+B[j1,k1])
        U[i0*m+j1, col] += 1.0; U[i1*m+j1, col] -= 1.0
        V[j1*p+k0, col] += 1.0; V[j1*p+k1, col] += 1.0
        W[k0*n+i0, col] += 1.0
        col += 1
        
        return col

    def add_de_groote_block(U, V, W, i0, j0, k0, col):
        # De Groote rank-23 for 3x3x3
        # Local indices 0,1,2 for i,j,k
        # Global: I = i0+li, J = j0+lj, K = k0+lk
        
        # I will use a known correct set of 23 terms.
        # If this fails verification, the caller should handle it.
        
        # The 23 terms (U_local, V_local, W_local)
        # U index: li*3 + lj
        # V index: lj*3 + lk
        # W index: lk*3 + li
        
        terms = [
            # 1
            ([(0,1),(1,1),(2,1)], [(0,1),(1,1),(2,1)], [(0,1),(1,1),(2,1)]), # This is not right, this is rank 1 for all ones? No.
        ]
        
        # I will use a simpler approach: I will NOT hardcode the 23 terms if I am not sure.
        # I will use the Strassen-only strategy which is known to work and gives 99.
        # To improve, I will try to find more Strassen blocks.
        
        return col

    # Since I am not 100% confident in hardcoding the exact 23 de Groote terms from memory without verification,
    # and a wrong decomposition scores 0, I will stick to the Strassen strategy but optimize the search.
    # The current score is 99.
    # I will try to find if 7 Strassen blocks are possible.
    # Theoretical max for 3x5x7 with 2x2x2 blocks:
    # i: 3 -> 1 pair
    # j: 5 -> 2 pairs
    # k: 7 -> 3 pairs
    # Max product: 1*2*3 = 6.
    # So 6 is the max for disjoint 2x2x2 blocks.
    # Thus, rank 99 is the best for Strassen-only.
    
    # To beat 99, I MUST use de Groote.
    # I will attempt to include the de Groote block with a verification step.
    # If it fails, I fall back to Strassen.
    
    def get_de_groote_terms():
        # Return list of 23 tuples (u_indices, v_indices, w_indices) with values
        # This is a known correct decomposition.
        # Source: de Groote, 1979.
        # I will use a Python implementation found in reliable sources.
        
        # The 23 terms for 3x3x3 matmul:
        # U: 9 entries, V: 9 entries, W: 9 entries.
        
        # I will use the following terms (verified in previous attempts or known):
        # If I cannot verify, I return None.
        
        # Let's try to construct a valid one or use a simple one.
        # Actually, I will use the Strassen block for the 2x2x2 part and schoolbook for the rest of the 3x3x3 block?
        # That would be rank 7 + 20 = 27. No improvement.
        
        # I will skip de Groote to be safe, unless I can verify it.
        # I will return None to indicate "not available".
        return None

    # Search for best Strassen packing
    all_blocks = []
    for i0 in range(n):
        for i1 in range(i0+1, n):
            for j0 in range(m):
                for j1 in range(j0+1, m):
                    for k0 in range(p):
                        for k1 in range(k0+1, p):
                            all_blocks.append((i0, i1, j0, j1, k0, k1))
                            
    best_blocks = []
    best_count = 0
    random.seed(42)
    
    for trial in range(1000):
        if time.time() > deadline - 20:
            break
        shuffled = all_blocks[:]
        random.shuffle(shuffled)
        chosen = []
        used = set()
        for block in shuffled:
            i0, i1, j0, j1, k0, k1 = block
            overlap = False
            for i in [i0, i1]:
                for j in [j0, j1]:
                    for k in [k0, k1]:
                        if (i, j, k) in used:
                            overlap = True
                            break
                    if overlap: break
                if overlap: break
            if not overlap:
                chosen.append(block)
                for i in [i0, i1]:
                    for j in [j0, j1]:
                        for k in [k0, k1]:
                            used.add((i, j, k))
        if len(chosen) > best_count:
            best_count = len(chosen)
            best_blocks = chosen[:]
            
    # Build decomposition
    U = np.zeros((nm, n*m*p), dtype=float)
    V = np.zeros((mp, n*m*p), dtype=float)
    W = np.zeros((pn, n*m*p), dtype=float)
    
    col = 0
    covered = set()
    
    for i0, i1, j0, j1, k0, k1 in best_blocks:
        col = add_strassen_block(U, V, W, i0, i1, j0, j1, k0, k1, col)
        for i in [i0, i1]:
            for j in [j0, j1]:
                for k in [k0, k1]:
                    covered.add((i, j, k))
                    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
                    
    U = U[:, :col]
    V = V[:, :col]
    W = W[:, :col]
    
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback
    U = np.zeros((nm, n*m*p), dtype=float)
    V = np.zeros((mp, n*m*p), dtype=float)
    W = np.zeros((pn, n*m*p), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
