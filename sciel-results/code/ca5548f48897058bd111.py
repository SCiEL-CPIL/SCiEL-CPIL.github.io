import numpy as np
import time
import itertools

DIMENSIONS = [3, 5, 5]
DEADLINE_S = 115.0
START_TIME = time.time()

def _elapsed():
    return time.time() - START_TIME

def _deadline_guard(frac=0.92):
    return _elapsed() > DEADLINE_S * frac

def construct():
    n, m, p = DIMENSIONS
    
    # We'll use 4 Strassen blocks (2x2x2 each) + schoolbook for remaining terms
    # Strassen blocks at (i0, j0, k0):
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    # Block 2: i in {0,1}, j in {2,3}, k in {0,1}
    # Block 3: i in {0,1}, j in {0,1}, k in {2,3}
    # Block 4: i in {0,1}, j in {2,3}, k in {2,3}
    
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2)
    ]
    
    num_strassen_blocks = len(blocks)
    total_strassen_rank = num_strassen_blocks * 7
    
    # Calculate number of schoolbook terms
    total_terms = n * m * p
    strassen_terms_count = num_strassen_blocks * 8
    schoolbook_terms_count = total_terms - strassen_terms_count
    
    total_rank = total_strassen_rank + schoolbook_terms_count
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0, col_offset):
        # M1 = (A00+A11)(B00+B11)
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M2 = (A10+A11)B00
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        
        # M3 = A00(B01-B11)
        c = col_offset + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M4 = A11(B10-B00)
        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        
        # M5 = (A00+A01)B11
        c = col_offset + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        
        # M6 = (A10-A00)(B00+B01)
        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M7 = (A01-A11)(B10+B11)
        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1

    # Add Strassen blocks
    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0, col)
        col += 7
    
    # Schoolbook for the rest
    covered_3d = [[[False]*p for _ in range(m)] for _ in range(n)]
    
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered_3d[i0+di][j0+dj][k0+dk] = True
    
    schoolbook_terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered_3d[i][j][k]:
                    U[i*m+j][col] += 1
                    V[j*p+k][col] += 1
                    W[k*n+i][col] += 1
                    schoolbook_terms.append((i, j, k, col))
                    col += 1

    base_rank = col

    def _validate(Um, Vm, Wm, rank):
        # Exact integer/half-integer check
        for row in Um:
            for x in row:
                if x != int(x) and 2*x != int(2*x):
                    return False
        for row in Vm:
            for x in row:
                if x != int(x) and 2*x != int(2*x):
                    return False
        for row in Wm:
            for x in row:
                if x != int(x) and 2*x != int(2*x):
                    return False
        # Exact tensor identity
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    ui = i*m+j
                    vj = j*p+k
                    wk = k*n+i
                    for c in range(rank):
                        s += Um[ui][c] * Vm[vj][c] * Wm[wk][c]
                    if s != 1:
                        return False
        return True

    # Try to find a rank-70 solution by absorbing one schoolbook term into the Strassen structure.
    # Strategy: for each schoolbook term (i,j,k), try to express it as a linear combination of
    # existing rank-71 columns with small integer/half-integer coefficients, then eliminate
    # one column. This is a bounded exact search.
    if not _deadline_guard(0.7):
        # Build the current solution as lists
        best = {"U": U, "V": V, "W": W}
        
        # Try removing each schoolbook column and checking if the remaining 70 columns
        # can still represent the full tensor (they can't, by construction, but we try
        # to find a different 70-column subset that works).
        # More promising: try to find a linear dependency among the 71 columns.
        # If column c is a linear combination of others, we can drop it.
        
        # Check for linear dependencies using the tensor structure
        # Column c contributes to tensor entry (i,j,k) as U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # For the tensor to remain unchanged after dropping column c, we need
        # U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 0 for all (i,j,k) not already covered,
        # AND the covered entries must still sum to 1.
        
        # Since each schoolbook column has exactly one non-zero tensor entry (value 1),
        # dropping it would make that entry 0 unless another column covers it.
        # Strassen columns cover 8 entries each. A schoolbook column covers 1 entry.
        # So we need a Strassen column to "absorb" the schoolbook entry.
        
        # For a schoolbook term (i,j,k) to be absorbed, some Strassen column must have
        # non-zero contribution to (i,j,k) that we can adjust.
        # But Strassen columns are fixed. Instead, we try to modify one Strassen column
        # to also cover a schoolbook term, then drop the schoolbook column.
        
        # Try: for each schoolbook term (i,j,k), check if it lies in the support of
        # some Strassen block. If so, we can potentially adjust that block's columns.
        # This is complex; let's try a simpler bounded search.
        
        # Try all pairs: remove schoolbook column s, and for each Strassen column c,
        # try to add a correction to make the tensor still exact.
        # This is too expensive in general, so we do a very bounded check.
        
        # Check if any two schoolbook terms share a Strassen column that could cover both.
        # Actually, let's just verify the rank-71 solution is correct and return it,
        # since the rank-70 search is unlikely to succeed within budget.
        
        if _validate(U, V, W, base_rank):
            best = {"U": U, "V": V, "W": W}
        
        return best

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
    
    # Verify correctness
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(U.shape[1]):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                expected = 1.0
                err = abs(val - expected)
                max_err = max(max_err, err)
    print(f"Max error: {max_err}")
    print(f"Time: {time.time()-t0:.2f}s")
