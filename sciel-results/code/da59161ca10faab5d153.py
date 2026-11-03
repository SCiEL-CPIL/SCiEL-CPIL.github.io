import time
import numpy as np

DIMENSIONS = [3, 3, 8]

def _strassen_2x2_2():
    """Return Strassen rank-7 decomposition for 2x2x2 matrix multiplication.
    Returns lists of (u, v, w) coefficient triples for the 7 rank-1 terms.
    Each is a tuple of 4 entries for U (2x2), 4 for V (2x2), 4 for W (2x2).
    """
    # Strassen's 7 products for 2x2 matrix multiplication
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    # i = p1 + p4 - p5 + p7
    # j = p3 + p5
    # k = p2 + p4
    # l = p1 - p2 + p3 + p6
    
    # U factors (for A): 7 vectors in R^4 (flattened 2x2)
    # V factors (for B): 7 vectors in R^4 (flattened 2x2)
    # W factors (for C): 7 vectors in R^4 (flattened 2x2)
    
    # p1 = (a+d)(e+h): U=[1,0,0,1], V=[1,1,0,0], W=[1,0,0,-1]
    # p2 = (c+d)e: U=[0,0,1,1], V=[1,0,0,0], W=[0,0,1,0]
    # p3 = a(f-h): U=[1,0,0,0], V=[0,1,0,-1], W=[0,1,0,0]
    # p4 = d(g-e): U=[0,0,0,1], V=[0,0,1,-1], W=[1,0,0,0]
    # p5 = (a+b)h: U=[1,1,0,0], V=[0,0,0,1], W=[0,-1,0,0]
    # p6 = (c-a)(e+f): U=[-1,0,1,0], V=[1,1,0,0], W=[0,0,0,1]
    # p7 = (b-d)(g+h): U=[0,1,0,-1], V=[0,0,1,1], W=[1,0,0,0]
    
    U_s = [
        [1, 0, 0, 1],  # p1
        [0, 0, 1, 1],  # p2
        [1, 0, 0, 0],  # p3
        [0, 0, 0, 1],  # p4
        [1, 1, 0, 0],  # p5
        [-1, 0, 1, 0], # p6
        [0, 1, 0, -1], # p7
    ]
    V_s = [
        [1, 1, 0, 0],  # p1
        [1, 0, 0, 0],  # p2
        [0, 1, 0, -1], # p3
        [0, 0, 1, -1], # p4
        [0, 0, 0, 1],  # p5
        [1, 1, 0, 0],  # p6
        [0, 0, 1, 1],  # p7
    ]
    W_s = [
        [1, 0, 0, -1], # p1
        [0, 0, 1, 0],  # p2
        [0, 1, 0, 0],  # p3
        [1, 0, 0, 0],  # p4
        [0, -1, 0, 0], # p5
        [0, 0, 0, 1],  # p6
        [1, 0, 0, 0],  # p7
    ]
    
    return U_s, V_s, W_s

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W give a valid decomposition of M_{n,m,p}."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reconstruct tensor
    # T[i,j,k] should equal sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    # But we need to check that this equals the multiplication tensor.
    # The multiplication tensor has T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # Actually, the tensor is defined such that for basis vectors, the contraction gives the product.
    
    # Let's verify by checking: for all i,j,k, sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] 
    # should equal 1 if this is a valid "multiplication" term, else 0.
    # Actually, the tensor M_{n,m,p} has entries: T_{(i,j),(j,k),(i,k)} = 1 for all valid i,j,k
    # and 0 otherwise. But in the flattened form:
    # U index: i*m + j (range n*m)
    # V index: j*p + k (range m*p)  
    # W index: k*n + i (range p*n)
    
    # The tensor T has T[a,b,c] = 1 iff there exist i,j,k such that a=i*m+j, b=j*p+k, c=k*n+i
    # and this is consistent (same i,j,k).
    
    # Check: for each (i,j,k), the term U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] summed over t should be 1
    # And for any other combination, it should be 0.
    
    # More precisely, the bilinear map defined by U,V,W should satisfy:
    # C[i,k] = sum_j A[i,j] B[j,k]
    
    # Let's test with random matrices
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    # Compute using decomposition
    C_computed = np.zeros((n, p))
    for t in range(rank):
        u_col = U_arr[:, t]
        v_col = V_arr[:, t]
        w_col = W_arr[:, t]
        
        A_t = u_col.reshape(n, m)
        B_t = v_col.reshape(m, p)
        C_t = w_col.reshape(p, n)
        
        # The contribution is A_t * B_t * C_t^T? No.
        # The decomposition means: C = sum_t (A * U_t) * (V_t * B) * W_t? 
        # Actually, the standard form: C_{ik} = sum_t (sum_j A_{ij} U_{ij,t}) (sum_j V_{jk,t} B_{jk}) W_{ik,t}
        # Wait, let me re-read the convention.
        
        # W[k*n+i, r] - so W is indexed by (k, i) where k is column of C, i is row of C.
        # So W_{k*n+i, t} is the coefficient for C[i,k] from term t.
        
        # The decomposition: C[i,k] = sum_t (sum_j A[i,j] * U[i*m+j, t]) * (sum_j B[j,k] * V[j*p+k, t]) * W[k*n+i, t]
        
        # Let me compute this properly
        for i in range(n):
            for k in range(p):
                u_val = sum(A[i,j] * U[i*m+j, t] for j in range(m))
                v_val = sum(B[j,k] * V[j*p+k, t] for j in range(m))
                C_computed[i,k] += u_val * v_val * W[k*n+i, t]
    
    if not np.allclose(C_computed, C_expected, atol=tol):
        return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _build_strassen_embedding(n, m, p):
    """Build a rank-68 decomposition by embedding 4 Strassen blocks."""
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    rank = len(U[0])
    
    # We'll try to replace 8 schoolbook terms with 7 Strassen terms for each of 4 blocks.
    # The 4 blocks use 2x2 submatrices of A.
    # Block (r1,r2,c1,c2): uses A entries (r1,c1),(r1,c2),(r2,c1),(r2,c2)
    # For each such block, the relevant schoolbook terms are those where i in {r1,r2} and j in {c1,c2}
    # and for all k in {0,...,p-1}.
    # That's 4*p terms per block. But we only want to replace the 2x2x2 part (p=2 worth).
    # Actually, for 3x3x8, each 2x2 block of A interacts with all 8 columns of B.
    # So we can't directly apply 2x2x2 Strassen.
    
    # Alternative: think of it as 2x2x8. Can we decompose 2x2x8 with rank 7*4=28?
    # 2x2x8 schoolbook is 2*2*8=32. Strassen on 2x2x2 gives 7 for the 2x2 part, 
    # but the 8 columns complicate things.
    
    # Let me try a different approach: explicitly construct the rank-68 decomposition
    # by identifying which terms can be compressed.
    
    # Actually, let me try: for each pair of adjacent rows in A and each pair of adjacent columns in A,
    # if we can find a 2x2x2 sub-tensor that is "isolated", we can apply Strassen.
    
    # For 3x3x8, consider the sub-tensor involving A[0:2, 0:2] (rows 0,1; cols 0,1).
    # This involves B rows 0,1 and C rows 0,1.
    # The terms are: for i in {0,1}, j in {0,1}, k in {0,...,7}
    # That's 4*8=32 terms.
    # But we can decompose the 2x2x8 part.
    # 2x2x8 = sum over k of 2x2x1.
    # Rank of 2x2x1 is 4 (schoolbook). So 2x2x8 has rank 4*8=32 schoolbook.
    # Can we do better? 2x2x8 is like 8 independent 2x2x1 problems.
    # Rank of 2x2x1: is it 4? I believe so, since it's a full 2x2 matrix times a vector.
    
    # Hmm, this isn't leading to a clear reduction.
    
    # Let me try yet another approach: use the fact that we can find 4 disjoint 2x2x2 sub-tensors.
    # In 3x3x8, can we find 4 disjoint sets of (i,j,k) triples that each form a 2x2x2 tensor?
    # A 2x2x2 sub-tensor needs 2 distinct i values, 2 distinct j values, 2 distinct k values.
    # And the 8 (i,j,k) combinations must all be present.
    
    # For example, sub-tensor 1: i in {0,1}, j in {0,1}, k in {0,1}
    # Sub-tensor 2: i in {0,1}, j in {1,2}, k in {2,3}
    # Sub-tensor 3: i in {1,2}, j in {0,1}, k in {4,5}
    # Sub-tensor 4: i in {1,2}, j in {1,2}, k in {6,7}
    
    # Check if these are disjoint:
    # ST1: (0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)
    # ST2: (0,1,2),(0,1,3),(0,2,2),(0,2,3),(1,1,2),(1,1,3),(1,2,2),(1,2,3)
    # ST3: (1,0,4),(1,0,5),(1,1,4),(1,1,5),(2,0,4),(2,0,5),(2,1,4),(2,1,5)
    # ST4: (1,1,6),(1,1,7),(1,2,6),(1,2,7),(2,1,6),(2,1,7),(2,2,6),(2,2,7)
    
    # Check overlaps:
    # ST1 and ST2: ST1 has j in {0,1}, k in {0,1}. ST2 has j in {1,2}, k in {2,3}. 
    # Overlap at j=1, but k values are disjoint ({0,1} vs {2,3}). So no overlap.
    # ST1 and ST3: ST1 has i in {0,1}, k in {0,1}. ST3 has i in {1,2}, k in {4,5}. 
    # Overlap at i=1, but k disjoint. No overlap.
    # ST1 and ST4: i overlap at 1, k disjoint. No overlap.
    # ST2 and ST3: ST2 has i in {0,1}, k in {2,3}. ST3 has i in {1,2}, k in {4,5}. 
    # i overlap at 1, k disjoint. No overlap.
    # ST2 and ST4: ST2 has i in {0,1}, j in {1,2}, k in {2,3}. ST4 has i in {1,2}, j in {1,2}, k in {6,7}.
    # i overlap at 1, j overlap at 1,2, but k disjoint. No overlap.
    # ST3 and ST4: ST3 has i in {1,2}, j in {0,1}, k in {4,5}. ST4 has i in {1,2}, j in {1,2}, k in {6,7}.
    # i overlap, j overlap at 1, k disjoint. No overlap.
    
    # Great! These 4 sub-tensors are disjoint.
    # Each is a 2x2x2 tensor, so we can apply Strassen to each, reducing rank from 8 to 7.
    # Total savings: 4 * 1 = 4 ranks.
    # New rank: 72 - 4 = 68.
    
    # Now let's construct this.
    # Start with schoolbook (rank 72).
    # For each of the 4 sub-tensors, remove the 8 schoolbook terms and add 7 Strassen terms.
    
    U, V, W = _schoolbook(n, m, p)
    
    # Define the 4 sub-tensors
    # Each sub-tensor is defined by (i_vals, j_vals, k_vals)
    sub_tensors = [
        ({0, 1}, {0, 1}, {0, 1}),
        ({0, 1}, {1, 2}, {2, 3}),
        ({1, 2}, {0, 1}, {4, 5}),
        ({1, 2}, {1, 2}, {6, 7}),
    ]
    
    # For each sub-tensor, find the schoolbook terms and replace with Strassen.
    # Schoolbook term for (i,j,k) is at column index: i*m*j + j*p*k + k*n*i? 
    # No, the column index in schoolbook is determined by the order:
    # col = 0
    # for i in range(n):
    #   for j in range(m):
    #     for k in range(p):
    #       col for (i,j,k) is the current value
    
    # Let me build a mapping from (i,j,k) to column index.
    col_map = {}
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col_map[(i, j, k)] = col
                col += 1
    
    # For each sub-tensor, get the 8 schoolbook columns
    # and replace them with 7 Strassen columns.
    
    # We'll build new U, V, W by modifying the schoolbook.
    # First, identify which columns to remove and which to add.
    
    columns_to_remove = []
    for i_vals, j_vals, k_vals in sub_tensors:
        for i in i_vals:
            for j in j_vals:
                for k in k_vals:
                    columns_to_remove.append(col_map[(i, j, k)])
    
    # columns_to_remove should have 32 entries (4 sub-tensors * 8 terms each)
    # We'll remove these 32 columns and add 28 new Strassen columns (4 * 7).
    # Net change: -4 columns. Final rank: 72 - 32 + 28 = 68.
    
    # Build the new matrices
    # Keep all columns not in columns_to_remove
    keep_cols = [c for c in range(n*m*p) if c not in set(columns_to_remove)]
    
    new_U = [row[c] for row in U for c in keep_cols]  # This is wrong, need to be careful
    # Actually, U has n*m rows and rank columns.
    # new_U[i, c] = U[i, c] for c in keep_cols
    
    new_U = [[U[i][c] for c in keep_cols] for i in range(n*m)]
    new_V = [[V[i][c] for c in keep_cols] for i in range(m*p)]
    new_W = [[W[i][c] for c in keep_cols] for i in range(p*n)]
    
    # Now add the 28 Strassen columns
    U_s, V_s, W_s = _strassen_2x2_2()
    
    for st_idx, (i_vals, j_vals, k_vals) in enumerate(sub_tensors):
        i_list = sorted(i_vals)
        j_list = sorted(j_vals)
        k_list = sorted(k_vals)
        
        for s in range(7):
            # Add new column
            new_col_idx = len(keep_cols) + st_idx * 7 + s
            
            # Extend U, V, W
            for i in range(n*m):
                new_U[i].append(0)
            for i in range(m*p):
                new_V[i].append(0)
            for i in range(p*n):
                new_W[i].append(0)
            
            # Fill in the Strassen coefficients
            # U_s[s] is a 4-element vector for the 2x2 A submatrix
            # Map to full U: U[i*m+j, new_col_idx]
            for local_i, i in enumerate(i_list):
                for local_j, j in enumerate(j_list):
                    new_U[i*m + j][new_col_idx] = U_s[s][local_i * 2 + local_j]
            
            # V_s[s] is a 4-element vector for the 2x2 B submatrix
            # B submatrix uses rows j_list (which are the columns of A)
            # V[j*p+k, new_col_idx]
            for local_j, j in enumerate(j_list):
                for local_k, k in enumerate(k_list):
                    new_V[j*p + k][new_col_idx] = V_s[s][local_j * 2 + local_k]
            
            # W_s[s] is a 4-element vector for the 2x2 C submatrix
            # C submatrix uses rows i_list and columns k_list
            # W[k*n+i, new_col_idx]
            for local_k, k in enumerate(k_list):
                for local_i, i in enumerate(i_list):
                    new_W[k*n + i][new_col_idx] = W_s[s][local_k * 2 + local_i]
    
    return new_U, new_V, new_W

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = DIMENSIONS  # n=3, m=3, p=8
    
    # Try to build the rank-68 decomposition
    try:
        U, V, W = _build_strassen_embedding(n, m, p)
        
        # Verify
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        else:
            # Fallback to schoolbook
            U, V, W = _schoolbook(n, m, p)
            return {"U": U, "V": V, "W": W}
    except Exception:
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
