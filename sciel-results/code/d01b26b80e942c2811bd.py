import time
import sys
import numpy as np
from itertools import combinations, product

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = 145.0

    n, m, p = DIMENSIONS  # 2, 4, 6
    nm, mp, pn = n * m, m * p, p * n  # 8, 24, 12

    # Build the matrix multiplication tensor T of shape (nm, mp, pn)
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify(U_mat, V_mat, W_mat, tol=1e-8):
        r = U_mat.shape[1]
        # Compute the tensor from the decomposition
        # T_decomp[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Efficiently: for each r, add outer(U[:,r], outer(V[:,r], W[:,r]))
        T_decomp = np.zeros((nm, mp, pn))
        for r in range(r):
            T_decomp += np.einsum('a,b,c->abc', U_mat[:, r], V_mat[:, r], W_mat[:, r])
        if np.max(np.abs(T_decomp - T)) > tol:
            return False
        return True

    # Build the Strassen rank-42 decomposition
    # Split into 2x2x2 blocks: i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5}
    # But Strassen works on 2x2x2. We have 2x4x6.
    # We can tile: 1 block in i (size 2), 2 blocks in j (size 2 each), 3 blocks in k (size 2 each)
    # Each 2x2x2 block uses rank 7 Strassen
    # Total: 1 * 2 * 3 * 7 = 42

    rank42 = 42
    U42 = np.zeros((nm, rank42), dtype=float)
    V42 = np.zeros((mp, rank42), dtype=float)
    W42 = np.zeros((pn, rank42), dtype=float)

    # Standard Strassen 7 multiplications for 2x2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6

    def add_strassen_block(i_start, j_start, k_start, offset):
        # A indices
        a00 = i_start * m + j_start
        a01 = i_start * m + j_start + 1
        a10 = (i_start + 1) * m + j_start
        a11 = (i_start + 1) * m + j_start + 1
        # B indices
        b00 = j_start * p + k_start
        b01 = j_start * p + k_start + 1
        b10 = (j_start + 1) * p + k_start
        b11 = (j_start + 1) * p + k_start + 1
        # C indices
        c00 = k_start * n + i_start
        c01 = (k_start + 1) * n + i_start
        c10 = k_start * n + i_start + 1
        c11 = (k_start + 1) * n + i_start + 1

        # M1 = (A00+A11)(B00+B11) -> C00 += 1, C11 -= 1
        U42[a00, offset] += 1; U42[a11, offset] += 1
        V42[b00, offset] += 1; V42[b11, offset] += 1
        W42[c00, offset] += 1; W42[c11, offset] -= 1

        # M2 = (A10+A11)B00 -> C10 += 1, C11 -= 1
        U42[a10, offset+1] += 1; U42[a11, offset+1] += 1
        V42[b00, offset+1] += 1
        W42[c10, offset+1] += 1; W42[c11, offset+1] -= 1

        # M3 = A00(B01-B11) -> C01 += 1, C11 += 1
        U42[a00, offset+2] += 1
        V42[b01, offset+2] += 1; V42[b11, offset+2] -= 1
        W42[c01, offset+2] += 1; W42[c11, offset+2] += 1

        # M4 = A11(B10-B00) -> C00 += 1, C10 += 1
        U42[a11, offset+3] += 1
        V42[b10, offset+3] += 1; V42[b00, offset+3] -= 1
        W42[c00, offset+3] += 1; W42[c10, offset+3] += 1

        # M5 = (A00+A01)B11 -> C00 -= 1, C01 += 1
        U42[a00, offset+4] += 1; U42[a01, offset+4] += 1
        V42[b11, offset+4] += 1
        W42[c00, offset+4] -= 1; W42[c01, offset+4] += 1

        # M6 = (A10-A00)(B00+B01) -> C11 += 1
        U42[a10, offset+5] += 1; U42[a00, offset+5] -= 1
        V42[b00, offset+5] += 1; V42[b01, offset+5] += 1
        W42[c11, offset+5] += 1

        # M7 = (A01-A11)(B10+B11) -> C00 += 1
        U42[a01, offset+6] += 1; U42[a11, offset+6] -= 1
        V42[b10, offset+6] += 1; V42[b11, offset+6] += 1
        W42[c00, offset+6] += 1

    offset = 0
    for i_start in range(0, n, 2):
        for j_start in range(0, m, 2):
            for k_start in range(0, p, 2):
                add_strassen_block(i_start, j_start, k_start, offset)
                offset += 7

    if not verify(U42, V42, W42):
        # Fallback to schoolbook
        rank0 = n * m * p
        U = np.zeros((nm, rank0), dtype=float)
        V = np.zeros((mp, rank0), dtype=float)
        W = np.zeros((pn, rank0), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i * (m * p) + j * p + k
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_U, best_V, best_W = U42.copy(), V42.copy(), W42.copy()
    best_rank = 42

    # Try to reduce rank by finding linear dependencies
    # Flatten each term and check if any column is a linear combination of others
    
    def try_reduce_rank(U_mat, V_mat, W_mat, max_remove=2):
        r = U_mat.shape[1]
        # For each subset of columns to potentially remove, check if the remaining
        # columns can still span the tensor
        
        # First, check if any single column is redundant
        for r_idx in range(r):
            if time.time() > deadline - 5:
                break
            # Try to express term r_idx as combination of others
            # This is hard in general, but we can check if removing it still gives a valid decomposition
            # by solving for adjusted coefficients - but that's complex.
            # Simpler: just check if the column is all zeros or proportional to another
            pass
        
        # Try pairwise merging
        def try_merge(U_mat, V_mat, W_mat, tol=1e-10):
            r = U_mat.shape[1]
            keep = [True] * r
            merged = False
            
            for i in range(r):
                if not keep[i]:
                    continue
                for j in range(i+1, r):
                    if not keep[j]:
                        continue
                    # Check if U_i ∝ U_j and V_i ∝ V_j
                    nz_u_i = np.nonzero(np.abs(U_mat[:, i]) > tol)[0]
                    nz_u_j = np.nonzero(np.abs(U_mat[:, j]) > tol)[0]
                    if len(nz_u_i) == 0 or len(nz_u_j) == 0:
                        continue
                    if not np.array_equal(nz_u_i, nz_u_j):
                        continue
                    ratios_u = U_mat[nz_u_i, i] / U_mat[nz_u_j, j]
                    if np.std(ratios_u) > tol:
                        continue
                    su = ratios_u[0]
                    if abs(su) < tol:
                        continue
                    
                    nz_v_i = np.nonzero(np.abs(V_mat[:, i]) > tol)[0]
                    nz_v_j = np.nonzero(np.abs(V_mat[:, j]) > tol)[0]
                    if len(nz_v_i) == 0 or len(nz_v_j) == 0:
                        continue
                    if not np.array_equal(nz_v_i, nz_v_j):
                        continue
                    ratios_v = V_mat[nz_v_i, i] / V_mat[nz_v_j, j]
                    if np.std(ratios_v) > tol:
                        continue
                    sv = ratios_v[0]
                    if abs(sv) < tol:
                        continue
                    
                    scale = 1.0 / (su * sv)
                    W_mat[:, i] = W_mat[:, i] + scale * W_mat[:, j]
                    keep[j] = False
                    merged = True
            
            if not merged:
                return U_mat, V_mat, W_mat, r
            kept = [i for i in range(r) if keep[i]]
            return U_mat[:, kept], V_mat[:, kept], W_mat[:, kept], len(kept)
        
        changed = True
        while changed and time.time() < deadline - 5:
            new_U, new_V, new_W, new_rank = try_merge(U_mat, V_mat, W_mat)
            if new_rank < U_mat.shape[1]:
                if verify(new_U, new_V, new_W):
                    U_mat, V_mat, W_mat = new_U, new_V, new_W
                    changed = True
                else:
                    changed = False
            else:
                changed = False
        return U_mat, V_mat, W_mat, U_mat.shape[1]

    best_U, best_V, best_W, best_rank = try_reduce_rank(best_U, best_V, best_W)

    # Try to find a rank-40 or lower decomposition by searching for sparse solutions
    # This is the key improvement: try to remove 2 columns from the Strassen decomposition
    # and solve for the remaining coefficients
    
    # For rank r, we have r terms. The tensor equation is:
    # sum_{r_idx=0}^{r-1} U[:,r_idx] ⊗ V[:,r_idx] ⊗ W[:,r_idx] = T
    # This is a system of nm*mp*pn equations in r*(nm+mp+pn) unknowns.
    # For r=42, this is 2304 equations in 42*44=1848 unknowns - underdetermined.
    # We can fix the supports (which entries are nonzero) and solve.
    
    # Try removing pairs of columns and see if we can adjust the rest
    # This is a non-linear problem in general, but if we fix the supports,
    # it becomes a system of polynomial equations.
    
    # A simpler approach: use the fact that the Strassen decomposition has a specific structure.
    # Each 2x2x2 block is independent. We can try to merge blocks.
    
    # For now, return the best we have
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

if __name__ == "__main__":
    result = construct()
    print("Constructed", file=sys.stderr)
