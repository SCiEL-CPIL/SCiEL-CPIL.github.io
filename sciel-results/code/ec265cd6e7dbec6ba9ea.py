import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 13]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 26, 26

    # Build the rank-46 baseline (6 Strassen blocks + 1 schoolbook block)
    R = 46
    U = np.zeros((nm, R), dtype=float)
    V = np.zeros((mp, R), dtype=float)
    W = np.zeros((pn, R), dtype=float)

    def A_idx(i, j): return i * m + j
    def B_idx(j, k): return j * p + k
    def C_idx(i, k): return k * n + i

    def add_strassen_block(k0, col_offset):
        prods = [
            ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
            ({(1,0):1,(1,1):1}, {(0,0):1}),
            ({(0,0):1}, {(0,1):1,(1,1):-1}),
            ({(1,1):1}, {(1,0):1,(0,0):-1}),
            ({(0,0):1,(0,1):1}, {(1,1):1}),
            ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}),
            ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}),
        ]
        for t, (a_coeff, b_coeff) in enumerate(prods):
            col = col_offset + t
            for (i, j), c in a_coeff.items():
                U[A_idx(i, j), col] += c
            for (j, kl), c in b_coeff.items():
                V[B_idx(j, k0 + kl), col] += c
        c_combos = [
            (0, k0,     [(0,1),(3,1),(4,-1),(6,1)]),
            (0, k0+1,   [(2,1),(4,1)]),
            (1, k0,     [(1,1),(3,1)]),
            (1, k0+1,   [(0,1),(1,-1),(2,1),(5,1)]),
        ]
        for (i, k, terms) in c_combos:
            for (t, coeff) in terms:
                W[C_idx(i, k), col_offset + t] += coeff

    def add_single_block(k, col_offset):
        prods = [
            ((0,0),(0,k)),
            ((0,1),(1,k)),
            ((1,0),(0,k)),
            ((1,1),(1,k)),
        ]
        for t, ((i,j),(jj,kk)) in enumerate(prods):
            col = col_offset + t
            U[A_idx(i, j), col] += 1
            V[B_idx(jj, kk), col] += 1
        W[C_idx(0, k), col_offset+0] += 1
        W[C_idx(0, k), col_offset+1] += 1
        W[C_idx(1, k), col_offset+2] += 1
        W[C_idx(1, k), col_offset+3] += 1

    col = 0
    for k0 in range(0, 12, 2):
        add_strassen_block(k0, col)
        col += 7
    add_single_block(12, col)
    col += 4
    assert col == R, f"col={col}, R={R}"

    # Verify baseline
    def verify(U, V, W, tol=1e-9):
        T = np.einsum('ir,jr,kr->ijk', U, V, W)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target_val = T[A_idx(i,j), B_idx(j,k), C_idx(i,k)]
                    if abs(target_val - 1.0) > tol:
                        return False
        for i in range(nm):
            for j in range(mp):
                for k in range(pn):
                    a = i // m; b = i % m
                    b2 = j // p; k2 = j % p
                    k3 = k // n; a2 = k % n
                    if a == a2 and b == b2 and k2 == k3:
                        continue
                    if abs(T[i, j, k]) > tol:
                        return False
        return True

    if not verify(U, V, W):
        # Fallback to schoolbook
        R0 = n * m * p
        U0 = np.zeros((nm, R0), dtype=float)
        V0 = np.zeros((mp, R0), dtype=float)
        W0 = np.zeros((pn, R0), dtype=float)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U0[i*m+j, c] = 1.0
                    V0[j*p+k, c] = 1.0
                    W0[k*n+i, c] = 1.0
                    c += 1
        return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}

    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = R

    # Attempt rank reduction via column dependency elimination
    # A rank-r decomposition can be reduced to rank r-1 if the columns are linearly dependent
    # in a way that preserves the tensor structure.
    
    # Strategy: Try to find if any column can be expressed as a linear combination of others
    # such that the tensor equation still holds.
    
    # For each pair of columns (a, b), check if column b can be replaced by column a + alpha * (some other column)
    # This is a structural search.
    
    # A more direct approach: check if the matrix of "effective" column vectors has rank < R
    # The effective constraint is that for all (i,j,k):
    # sum_c U[i,c]*V[j,c]*W[k,c] = delta(i,j,k)
    
    # We can try to perturb the columns slightly to see if we can reduce rank.
    # But a cleaner method: check if any column is redundant.
    
    # For a column c to be redundant, there must exist a linear combination of other columns
    # that produces the same tensor contribution.
    
    # Let's try a simpler heuristic: check if the last column (schoolbook) can be merged
    # with a Strassen column by adjusting coefficients.
    
    # Actually, the most promising structural change: 
    # Use a different grouping. Instead of 6 Strassen + 1 schoolbook, 
    # try 5 Strassen + 1 "extended" block for the last 3 columns.
    
    # The rank of 2x2x3 matrix multiplication is known to be 8 (not 12 schoolbook).
    # So 5*7 + 8 = 35 + 8 = 43? Let's check if we can build a rank-8 2x2x3 decomposition.
    
    # Known: M_{2,2,3} has rank 8.
    # If we use 5 Strassen blocks (rank 7 each) for columns 0-9, and a rank-8 block for columns 10-12,
    # total rank = 5*7 + 8 = 43.
    
    # Let's try to construct a rank-8 decomposition for 2x2x3.
    
    def _build_2x2x3_rank8(offset):
        # This is a known result. The 2x2x3 tensor has rank 8.
        # We need to find the explicit decomposition.
        # This is complex, so let's first try the simpler approach of checking
        # if we can reduce the rank by 1 from 46 via column dependencies.
        
        # For now, return None to indicate this needs more work.
        return None
    
    # Alternative: Try to find a rank-45 decomposition by checking column dependencies.
    # Build the tensor contribution matrix and check its rank.
    
    # For each column c, define the tensor T_c[i,j,k] = U[i,c]*V[j,c]*W[k,c]
    # The sum of all T_c must equal the target tensor T_target.
    # If sum_c T_c = T_target, and the T_c are linearly dependent, we might reduce rank.
    
    # Let's check if the columns are linearly independent by forming a matrix
    # where each column c is represented by the vector (U[:,c] ⊗ V[:,c] ⊗ W[:,c])
    # This is a 4*26*26 = 2704-dimensional vector.
    
    # Check rank of the matrix with these vectors as columns.
    
    if time.time() < deadline - 30:
        # Form the matrix M where M[:,c] = vec(U[:,c] ⊗ V[:,c] ⊗ W[:,c])
        # This is expensive but doable for R=46.
        
        # Actually, we can check linear dependencies more efficiently.
        # If column c is a linear combination of other columns, then:
        # U[:,c] = sum_{c'} alpha_{c,c'} U[:,c']  (not exactly, because V and W also change)
        
        # The correct condition is:
        # sum_c' alpha_{c,c'} (U[:,c'] ⊗ V[:,c'] ⊗ W[:,c']) = U[:,c] ⊗ V[:,c] ⊗ W[:,c]
        
        # This is a nonlinear condition. Let's skip this for now and try a different approach.
        
        # Try to use a different block structure: 5 Strassen + 1 rank-8 block
        # We need to find a rank-8 decomposition for 2x2x3.
        
        # Known rank-8 decomposition for 2x2x3:
        # This requires careful construction. Let's try to find it.
        
        # For now, let's just return the rank-46 solution.
        pass
    
    # Convert to lists for the return interface
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    print(f"U shape: {len(result['U'])}x{len(result['U'][0])}")
    print(f"V shape: {len(result['V'])}x{len(result['V'][0])}")
    print(f"W shape: {len(result['W'])}x{len(result['W'][0])}")
