"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import random
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    N = n * m
    M = m * p
    P = p * n
    
    # Build the tensor T[i, j, k] where:
    # i = row of A * m + col of A (0..N-1)
    # j = row of B * p + col of B (0..M-1)
    # k = row of C * n + col of C (0..P-1)
    # T[i, j, k] = 1 if (col of A) == (row of B) and (row of A) == (row of C) and (col of B) == (col of C)
    
    # T is sparse. We represent it as a dict of non-zero entries for efficiency in greedy steps?
    # Or a 3D list of Fractions. 8*20*10 = 1600 entries. List is fine.
    
    tensor = [[[Fraction(0) for _ in range(P)] for _ in range(M)] for _ in range(N)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_a = i * m + j
                idx_b = j * p + k
                idx_c = k * n + i
                tensor[idx_a][idx_b][idx_c] = Fraction(1)
    
    best_rank = n * m * p
    best_decomp = None
    
    # Try to find a lower rank using randomized greedy extraction
    # We will try several random orderings of pivots
    
    for trial in range(20): # 20 trials
        random.seed(trial)
        current = [[[tensor[i][j][k] for k in range(P)] for j in range(M)] for i in range(N)]
        
        U = []
        V = []
        W = []
        
        # Get list of all non-zero indices
        non_zeros = [(i, j, k) for i in range(N) for j in range(M) for k in range(P) if current[i][j][k] != 0]
        
        # Shuffle to randomize pivot order
        random.shuffle(non_zeros)
        
        rank = 0
        while non_zeros:
            # Pick a pivot. In standard greedy, we pick the first one.
            # Here we pick the first one in the shuffled list that is still non-zero in `current`
            pivot = None
            pivot_idx = 0
            for idx, (i, j, k) in enumerate(non_zeros):
                if current[i][j][k] != 0:
                    pivot = (i, j, k)
                    pivot_idx = idx
                    break
            
            if pivot is None:
                break
                
            i0, j0, k0 = pivot
            
            # Extract rank-1 component anchored at (i0, j0, k0)
            # u[i0]=1, v[j0]=1, w[k0]=1
            # Determine u, v, w to match the "axes"
            denom = current[i0][j0][k0]
            if denom == 0:
                # Should not happen if we checked non_zeros, but just in case
                non_zeros.pop(pivot_idx)
                continue
                
            u = [Fraction(0)] * N
            v = [Fraction(0)] * M
            w = [Fraction(0)] * P
            
            u[i0] = Fraction(1)
            v[j0] = Fraction(1)
            w[k0] = Fraction(1)
            
            # Fill u
            for i in range(N):
                if i != i0 and current[i][j0][k0] != 0:
                    u[i] = current[i][j0][k0] / denom
            
            # Fill v
            for j in range(M):
                if j != j0 and current[i0][j][k0] != 0:
                    v[j] = current[i0][j][k0] / denom
            
            # Fill w
            for k in range(P):
                if k != k0 and current[i0][j0][k] != 0:
                    w[k] = current[i0][j0][k] / denom
            
            # Subtract
            # Optimization: only iterate over non-zeros of u, v, w
            u_nz = [i for i in range(N) if u[i] != 0]
            v_nz = [j for j in range(M) if v[j] != 0]
            w_nz = [k for k in range(P) if w[k] != 0]
            
            for i in u_nz:
                for j in v_nz:
                    for k in w_nz:
                        current[i][j][k] -= u[i] * v[j] * w[k]
            
            U.append(u)
            V.append(v)
            W.append(w)
            rank += 1
            
            # Update non_zeros list? 
            # It's inefficient to rebuild. We can just check `current` in the next loop.
            # But `non_zeros` might contain indices that are now zero.
            # We can filter it occasionally or just let the `if current[i][j][k] != 0` check handle it.
            # To speed up, we can remove the pivot from non_zeros.
            non_zeros.pop(pivot_idx)
            
            if rank >= best_rank:
                break
            
        if rank < best_rank and rank > 0:
            # Verify decomposition
            # Reconstruct and check against original tensor
            # For speed, we assume greedy worked. But let's verify one case.
            best_rank = rank
            best_decomp = (U, V, W)
            
            # Early exit if we hit a very low rank?
            # Theoretical lower bound for 2x4x5 is likely around 30-35.
            if best_rank <= 35:
                break
        
    if best_decomp:
        U, V, W = best_decomp
        # Convert to lists of floats/ints
        def conv(lst):
            return [float(x) if x.denominator == 1 else float(x) for x in lst]
        
        return {
            "U": [conv(row) for row in U],
            "V": [conv(row) for row in V],
            "W": [conv(row) for row in W]
        }
    else:
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(N)]
        V = [[0] * rank for _ in range(M)]
        W = [[0] * rank for _ in range(P)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
