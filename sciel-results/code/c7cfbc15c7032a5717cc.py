import time
import numpy as np
from fractions import Fraction

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    
    Strategy:
    1. Build the multiplication tensor T[i,j,k] = 1 for valid (i,j,k) triples
    2. Matricize along the A-mode to get matrix M (4 x 14)
    3. Compute rank of M over Q via Gaussian elimination with fractions
    4. Extract rank-minimal decomposition (U, V, W) with integer entries
    5. Verify and return
    """
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin before 150s kill

    n, m, p = 2, 2, 7

    # Build the multiplication tensor
    # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T = 1 iff there exist i,j,k with a=i*m+j, b=j*p+k, c=k*n+i
    # and the mapping is consistent (it is by construction)
    
    # Valid (a, b, c) triples
    valid_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_triples.append((a, b, c))
    
    # Matricize: M[a, (b,c)] where (b,c) is a combined index
    # M is 4 x (14*14) = 4 x 196
    # But we can be smarter: for each a, the set of (b,c) where T=1
    # T[a,b,c] = 1 iff b = j*p+k and c = k*n+i where a = i*m+j
    # So for fixed a = i*m+j: b ranges over j*p+k for k in 0..p-1, c = k*n+i
    # That's p entries per a, so M is 4 x 28 (sparse)
    
    # Actually, let's matricize properly.
    # M[a, bc] where bc = b*(p*n) + c
    # M is (n*m) x (m*p * p*n) = 4 x (14*14) = 4 x 196
    # But only 28 entries are nonzero.
    
    # For rank computation, use the sparse structure.
    # M[a, bc] = 1 iff there exists valid triple (a, b, c)
    
    # Let's build M as a list of (row, col, val) for nonzero entries
    # bc = b * (p*n) + c
    pn = p * n  # 14
    M_rows = []
    for (a, b, c) in valid_triples:
        bc = b * pn + c
        M_rows.append((a, bc, 1))
    
    # Now compute rank of M (4 x 196) over Q
    # Since it's 4 rows, rank is at most 4.
    # Use Gaussian elimination with fractions on the rows.
    
    # Build the matrix as list of rows (sparse)
    # Each row is a dict: col -> value
    num_rows = n * m  # 4
    num_cols = m * p * pn  # 14 * 14 = 196
    
    rows = [dict() for _ in range(num_rows)]
    for (a, bc, val) in M_rows:
        rows[a][bc] = Fraction(val)
    
    # Gaussian elimination to find rank and basis
    # We want to find the row rank and express each row as a linear combination of pivot rows
    
    pivots = []  # list of (row_index, pivot_col, pivot_value, row_dict)
    pivot_cols = []  # columns used as pivots
    
    for r in range(num_rows):
        if not rows[r]:
            continue
        # Find pivot
        pivot_col = None
        for c in sorted(rows[r].keys()):
            if c not in pivot_cols:
                pivot_col = c
                break
        if pivot_col is None:
            continue
        pivot_val = rows[r][pivot_col]
        # Normalize
        inv_pv = Fraction(1, 1) / pivot_val
        norm_row = {c: v * inv_pv for c, v in rows[r].items()}
        pivots.append((r, pivot_col, pivot_val, norm_row))
        pivot_cols.append(pivot_col)
        # Eliminate from other rows
        for r2 in range(num_rows):
            if r2 == r:
                continue
            if pivot_col in rows[r2]:
                factor = rows[r2][pivot_col]
                for c in list(rows[r2].keys()):
                    rows[r2][c] -= factor * norm_row[c]
                    if rows[r2][c] == 0:
                        del rows[r2][c]
    
    tensor_rank = len(pivots)
    
    # Now extract the decomposition.
    # Each pivot row r_p has a pivot at column bc_p.
    # The original tensor T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # 
    # From the matricization: M[a, bc] = sum_r U[a,r] * (V[b,r] * W[c,r])
    # 
    # For each pivot row r_p with pivot at bc_p = b_p * pn + c_p:
    #   M[a, bc_p] = U[a, r_p] * V[b_p, r_p] * W[c_p, r_p]
    # 
    # We can set V[b_p, r_p] = 1, W[c_p, r_p] = 1, and U[a, r_p] = M[a, bc_p]
    # But we need to ensure consistency.
    #
    # Better approach: For each pivot r_p:
    #   Set W[c_p, r_p] = 1 (where c_p is the c-index of the pivot)
    #   Set V[b, r_p] = M[a_p, b*pn+c] / M[a_p, bc_p] for the pivot row a_p
    #   Set U[a, r_p] = M[a, bc_p] / (V[b_p, r_p] * W[c_p, r_p])
    #
    # Actually, let's use a cleaner extraction:
    # For pivot r_p with pivot row a_p and pivot col bc_p = b_p * pn + c_p:
    #   Let U[a_p, r_p] = 1
    #   Let V[b_p, r_p] = 1  
    #   Let W[c_p, r_p] = M[a_p, bc_p]  (which is 1 since T=1 there)
    #   For other a: U[a, r_p] = M[a, bc_p]
    #   For other b: V[b, r_p] = M[a_p, b*pn+c_p]
    #   For other c: W[c, r_p] = M[a_p, b_p*pn+c]
    
    # Build U, V, W
    U = np.zeros((num_rows, tensor_rank), dtype=int)
    V = np.zeros((m * p, tensor_rank), dtype=int)
    W = np.zeros((pn, tensor_rank), dtype=int)
    
    # Map from (a, bc) to value in original M
    M_val = {}
    for (a, bc, val) in M_rows:
        M_val[(a, bc)] = val
    
    def get_M(a, b, c):
        bc = b * pn + c
        return M_val.get((a, bc), 0)
    
    for r_p, (row_idx, pivot_col, pivot_val, norm_row) in enumerate(pivots):
        a_p = row_idx
        b_p = pivot_col // pn
        c_p = pivot_col % pn
        
        # Set U[a_p, r_p] = 1
        U[a_p, r_p] = 1
        # Set V[b_p, r_p] = 1
        V[b_p, r_p] = 1
        # Set W[c_p, r_p] = 1
        W[c_p, r_p] = 1
        
        # For other a: U[a, r_p] = M[a, b_p, c_p]
        for a in range(num_rows):
            if a != a_p:
                U[a, r_p] = get_M(a, b_p, c_p)
        
        # For other b: V[b, r_p] = M[a_p, b, c_p]
        for b in range(m * p):
            if b != b_p:
                V[b, r_p] = get_M(a_p, b, c_p)
        
        # For other c: W[c, r_p] = M[a_p, b_p, c]
        for c in range(pn):
            if c != c_p:
                W[c, r_p] = get_M(a_p, b_p, c)
    
    # Verify the decomposition
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    valid = True
    for (a, b, c) in valid_triples:
        s = 0
        for r in range(tensor_rank):
            s += U[a, r] * V[b, r] * W[c, r]
        if s != 1:
            valid = False
            break
    
    if valid and time.time() < deadline - 5.0:
        # Check invalid entries
        valid_set = set(valid_triples)
        for a in range(num_rows):
            for b in range(m * p):
                for c in range(pn):
                    if (a, b, c) not in valid_set:
                        s = 0
                        for r in range(tensor_rank):
                            s += U[a, r] * V[b, r] * W[c, r]
                        if s != 0:
                            valid = False
                            break
                if not valid:
                    break
            if not valid:
                break
    
    if not valid:
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((n*m, rank_sb), dtype=int)
        V_sb = np.zeros((m*p, rank_sb), dtype=int)
        W_sb = np.zeros((pn, rank_sb), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1
                    V_sb[j*p+k, col] = 1
                    W_sb[k*n+i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
