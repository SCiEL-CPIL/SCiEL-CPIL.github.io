"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a fixed rank decomposition
    R = 43
    
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    r = 0 # current rank index
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        """
        Adds a rank-1 term to the decomposition.
        u_idx: index in U (i*m + j)
        u_val: coefficient
        v_idx: index in V (j*p + k)
        v_val: coefficient
        w_idx: index in W (k*n + i)
        w_val: coefficient
        """
        nonlocal r
        U[u_idx, r] = u_val
        V[v_idx, r] = v_val
        W[w_idx, r] = w_val
        r += 1
        
    def add_strassen_222(i_off, j_off, k_off):
        """
        Adds Strassen decomposition for a 2x2x2 block.
        i_off, j_off, k_off are the starting indices for i, j, k.
        """
        nonlocal r
        # Indices:
        # A: i in {i_off, i_off+1}, j in {j_off, j_off+1}
        # B: j in {j_off, j_off+1}, k in {k_off, k_off+1}
        # C: k in {k_off, k_off+1}, i in {i_off, i_off+1}
        
        a00_idx = (i_off + 0) * m + (j_off + 0)
        a01_idx = (i_off + 0) * m + (j_off + 1)
        a10_idx = (i_off + 1) * m + (j_off + 0)
        a11_idx = (i_off + 1) * m + (j_off + 1)
        
        b00_idx = (j_off + 0) * p + (k_off + 0)
        b01_idx = (j_off + 0) * p + (k_off + 1)
        b10_idx = (j_off + 1) * p + (k_off + 0)
        b11_idx = (j_off + 1) * p + (k_off + 1)
        
        c00_idx = (k_off + 0) * n + (i_off + 0)
        c01_idx = (k_off + 0) * n + (i_off + 1)
        c10_idx = (k_off + 1) * n + (i_off + 0)
        c11_idx = (k_off + 1) * n + (i_off + 1)
        
        # M1 = (A00+A11)(B00+B11)
        U[a00_idx, r] = 1; V[b00_idx, r] = 1; W[c00_idx, r] = 1; r+=1
        U[a11_idx, r] = 1; V[b11_idx, r] = 1; W[c00_idx, r] = 1; r+=1
        U[a00_idx, r] = 1; V[b11_idx, r] = 1; W[c00_idx, r] = 1; r+=1
        U[a11_idx, r] = 1; V[b00_idx, r] = 1; W[c00_idx, r] = 1; r+=1
        
        # M2 = (A10+A11)B00
        U[a10_idx, r] = 1; V[b00_idx, r] = 1; W[c10_idx, r] = 1; r+=1
        U[a11_idx, r] = 1; V[b00_idx, r] = 1; W[c10_idx, r] = 1; r+=1
        
        # M3 = A01(B11-B01)
        U[a01_idx, r] = 1; V[b11_idx, r] = 1; W[c01_idx, r] = 1; r+=1
        U[a01_idx, r] = 1; V[b01_idx, r] = -1; W[c01_idx, r] = 1; r+=1
        
        # M4 = A10(B10-B00)
        U[a10_idx, r] = 1; V[b10_idx, r] = 1; W[c10_idx, r] = 1; r+=1
        U[a10_idx, r] = 1; V[b00_idx, r] = -1; W[c10_idx, r] = 1; r+=1
        
        # M5 = (A00+A01)B11
        U[a00_idx, r] = 1; V[b11_idx, r] = 1; W[c01_idx, r] = 1; r+=1
        U[a01_idx, r] = 1; V[b11_idx, r] = 1; W[c01_idx, r] = 1; r+=1
        
        # M6 = (A10-A00)(B00+B01)
        U[a10_idx, r] = 1; V[b00_idx, r] = 1; W[c11_idx, r] = 1; r+=1
        U[a00_idx, r] = -1; V[b00_idx, r] = 1; W[c11_idx, r] = 1; r+=1
        U[a10_idx, r] = 1; V[b01_idx, r] = 1; W[c11_idx, r] = 1; r+=1
        U[a00_idx, r] = -1; V[b01_idx, r] = 1; W[c11_idx, r] = 1; r+=1
        
        # M7 = (A01-A11)(B10+B11)
        U[a01_idx, r] = 1; V[b10_idx, r] = 1; W[c00_idx, r] = 1; r+=1
        U[a11_idx, r] = -1; V[b10_idx, r] = 1; W[c00_idx, r] = 1; r+=1
        U[a01_idx, r] = 1; V[b11_idx, r] = 1; W[c00_idx, r] = 1; r+=1
        U[a11_idx, r] = -1; V[b11_idx, r] = 1; W[c00_idx, r] = 1; r+=1
        
        # Now we have added 14 terms? No, Strassen has 7 products.
        # My implementation above expanded the products into sum of rank-1 terms for the factors?
        # Wait. Strassen reduces the number of *multiplications*.
        # The factor matrices U, V, W represent the *multiplication*.
        # If we use Strassen, we define 7 intermediate values M1..M7.
        # Each M is a product of a linear combo of A and a linear combo of B.
        # So for each M, we have one rank-1 term in the tensor decomposition?
        # No.
        # The tensor decomposition rank is the number of multiplications.
        # If we replace 8 multiplications with 7, the rank drops by 1.
        # How do we encode Strassen in U, V, W?
        #
        # Let's look at the standard rank-1 term:
        # U[i,j,r]=1, V[j,k,r]=1, W[k,i,r]=1.
        # This corresponds to the product A[i,j] * B[j,k] contributing to C[i,k].
        #
        # Strassen calculates C[i,k] as a linear combination of M's.
        # M_s = (sum u_i A[i,j]) * (sum v_j B[j,k]).
        # This is NOT a single rank-1 term in the U,V,W sense unless the sums are single elements.
        #
        # Ah, the definition of tensor rank decomposition allows *any* values in U, V, W.
        # It does not restrict them to 0 or 1.
        # So for a Strassen product M1 = (A00+A11)(B00+B11):
        # We cannot just say "U has 1 at A00 and A11". That would be a sum of two rank-1 terms.
        # But Strassen *is* the decomposition into 7 terms.
        # Each term s=1..7 corresponds to a multiplication of (linear combo of A) and (linear combo of B).
        # So for each s, we add ONE rank index r.
        # In that rank r:
        # U[i,j,r] should be the coefficient of A[i,j] in the first factor of M_s.
        # V[j,k,r] should be the coefficient of B[j,k] in the second factor of M_s.
        # W[k,i,r] should be the coefficient of C[k,i] in the result of M_s?
        #
        # Wait. The decomposition is T = sum_r U_r \otimes V_r \otimes W_r.
        # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r].
        #
        # If we use Strassen, we are effectively defining a new basis for the multiplication.
        # Let the 7 Strassen products be S_1...S_7.
        # Each S_s is a product of two linear forms.
        # This product contributes to the output C.
        # The contribution of S_s to C[i,k] is coeff_s[i,k] * S_s.
        #
        # So, for rank r corresponding to Strassen product s:
        # U[i,j,r] = coeff of A[i,j] in the first linear form of S_s.
        # V[j,k,r] = coeff of B[j,k] in the second linear form of S_s.
        # W[k,i,r] = coeff of C[i,k] in the sum of S_s contributions?
        #
        # No, W[k,i,r] must be a constant for that rank such that the identity holds.
        # Identity: A[i,j] B[j,k] C[k,i] ...
        #
        # Let's verify with a simple case.
        # If we have a term where U is a vector u, V is a vector v, W is a vector w.
        # The term is u \otimes v \otimes w.
        # The contraction is sum_{i,j,k} u_i v_j w_k A_i B_j C_k? No.
        #
        # The tensor T is the multiplication map.
        # T(A,B) = A B.
        # T = sum_r U_r \otimes V_r \otimes W_r.
        # This means T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r].
        # And T[i,j,k] is 1 if the indices match for the "middle" index j?
        # No, T is the tensor such that <T, A \otimes B> = C.
        # So T[i,j,k] is the coefficient of A[i,j] B[j,k] in C[i,k].
        # T[i,j,k] = 1. (For the specific i,j,k involved in the product).
        # Actually, T is a 3-way tensor of size n x m x p? No.
        # The indices are (i, j, k) where i is row of A/col of C, j is col of A/row of B, k is col of B/row of C?
        # Prompt: U (n*m, R), V (m*p, R), W (p*n, R).
        # U indexed by (i,j). V by (j,k). W by (k,i).
        # Condition: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k?
        # Yes, this is the condition for the standard basis of the multiplication tensor.
        #
        # So, if we use Strassen, we replace the 8 basis terms (which would have U,V,W as unit vectors)
        # with 7 terms where U, V, W are linear combinations.
        #
        # Example: M1 = (A00+A11)(B00+B11).
        # This contributes to C00, C01, C10, C11.
        # In the standard decomposition, C00 gets A00B00 + A01B10 + ...
        # In Strassen, C00 = M1 + M4 - M5 + M7.
        #
        # So for rank r=1 (corresponding to M1):
        # First factor L_A = A00 + A11.
        # U[0,0,1] = 1 (coeff of A00)
        # U[1,1,1] = 1 (coeff of A11)
        # Second factor L_B = B00 + B11.
        # V[0,0,1] = 1 (coeff of B00)
        # V[1,1,1] = 1 (coeff of B11)
        #
        # Now, what is W[?, ?, 1]?
        # The term in the sum is U * V * W.
        # We need sum_r ... = T[i,j,k].
        # This form of decomposition is tricky because W is coupled to the output indices.
        #
        # Actually, the standard way to write tensor rank is:
        # C[i,k] = sum_r (sum_j U[i,j,r] A[i,j]) (sum_k V[j,k,r] B[j,k]) W[k,i,r] ??
        # No, that's not right.
        #
        # Let's look at the definition again.
        # "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
        # This implies the tensor is a sum of rank-1 tensors $u_r \otimes v_r \otimes w_r$.
        # $T = \sum_{r} u_r \otimes v_r \otimes w_r$.
        # $T[i,j,k] = \sum_r u_r[i,j] v_r[j,k] w_r[k,i]$.
        # And we require $T[i,j,k] = 1$ for the "multiplication" entries?
        # No, the multiplication tensor $T_{mul}$ is defined by $T_{mul}[i,j,k] = 1$ if we view it as the map $A \otimes B \to C$?
        # Actually, $T_{mul}$ is the tensor such that $\sum_{i,j,k} T_{mul}[i,j,k] A[i,j] B[j,k] = C[i,k]$?
        # No, $C[i,k] = \sum_j A[i,j] B[j,k]$.
        # So the tensor $T$ has entries $T[i,j,k]$.
        # If we contract $T$ with $A$ and $B$, we get $C$.
        # $(T \cdot A \cdot B)[i,k] = \sum_{j,l} T[i,j,l] A[i,j] B[l,k]$? No.
        #
        # Let's assume the standard definition:
        # The tensor $T$ of matrix multiplication has rank R.
        # $T = \sum_{r=1}^R \mathbf{u}_r \otimes \mathbf{v}_r \otimes \mathbf{w}_r$.
        # Where $\mathbf{u}_r$ is in $\mathbb{R}^{n \times m}$, $\mathbf{v}_r$ in $\mathbb{R}^{m \times p}$, $\mathbf{w}_r$ in $\mathbb{R}^{p \times n}$.
        # The condition is that for all $A, B$:
        # $AB = \sum_r (\mathbf{u}_r \cdot A) (\mathbf{v}_r \cdot B) \mathbf{w}_r$?
        # No, $\mathbf{w}_r$ must be a matrix?
        # If $\mathbf{w}_r$ is a vector, then the output would be a vector.
        # So $\mathbf{w}_r$ must be in $\mathbb{R}^{n \times p}$?
        # But the prompt says W is $(p \cdot n, \text{rank})$.
        # So $W$ is a matrix.
        #
        # Let's check the indices in the prompt:
        # $W[k \cdot n + i, r]$.
        # This suggests $W$ is indexed by $k$ and $i$.
        # If $W$ is $(p \cdot n, R)$, and we flatten it, $w_r$ is a vector of length $p \cdot n$.
        # If we reshape $w_r$ to $(p, n)$, let's call it $W_r$.
        # Then the term is $(\mathbf{u}_r \cdot A) (\mathbf{v}_r \cdot B) W_r$.
        # Here $\mathbf{u}_r \cdot A$ is a scalar: $\sum_{i,j} u_r[i,j] A[i,j]$.
        # $\mathbf{v}_r \cdot B$ is a scalar: $\sum_{j,k} v_r[j,k] B[j,k]$.
        # So the contribution to $C$ is a scalar times the matrix $W_r$.
        # $C = \sum_r (\sum_{i,j} u_r[i,j] A[i,j]) (\sum_{j,k} v_r[j,k] B[j,k]) W_r$.
        #
        # We want this to equal $\sum_j A[i,j] B[j,k]$ for all $i,k$.
        #
        # Let's test this with a rank-1 schoolbook term.
        # Term for $A[i_0, j_0] B[j_0, k_0]$ contributing to $C[i_0, k_0]$.
        # $u_r[i_0, j_0] = 1$, others 0.
        # $v_r[j_0, k_0] = 1$, others 0.
        # $W_r$ should be a matrix with 1 at $[k_0, i_0]$?
        # The term is $(A[i_0, j_0]) (B[j_0, k_0]) W_r$.
        # We want this to contribute $A[i_0, j_0] B[j_0, k_0]$ to $C[i_0, k_0]$.
        # So $W_r[i_0, k_0]$ should be 1.
        # But the index in the prompt is $W[k \cdot n + i, r]$.
        # This is $W_r[k, i]$.
        # So $W_r[k_0, i_0] = 1$.
        # This matches.
        
        # So, for Strassen:
        # Each Strassen product $M_s$ corresponds to a rank term $r$.
        # $u_r$ is the vector of coefficients for $A$ in the first linear form of $M_s$.
        # $v_r$ is the vector of coefficients for $B$ in the second linear form of $M_s$.
        # $W_r$ is the matrix such that $M_s$ contributes to $C$ via $W_r$.
        # Specifically, $C = \sum_s (L_{A,s} \cdot A) (L_{B,s} \cdot B) W_s$.
        # We know $C_{i,k} = \sum_s c_{s}[i,k] M_s$.
        # So $W_s$ must have entries $W_s[k,i] = c_s[i,k]$.
        
        # Let's implement this.
        
    def add_strassen_222_correct(i_off, j_off, k_off):
        nonlocal r
        # Strassen products M1..M7
        # i in {0,1}+off, j in {0,1}+off, k in {0,1}+off
        
        # M1 = (A00+A11)(B00+B11)
        # u: A00=1, A11=1
        # v: B00=1, B11=1
        # C contribution: M1 contributes to C00, C11 with +1, and C01, C10 with 0?
        # Wait, Strassen formulas:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        #
        # So for M1:
        # W1[0,0] = 1, W1[1,1] = 1. Others 0.
        # Note W is indexed [k,i].
        # W1[k0, i0] = 1 => k=0, i=0.
        # W1[k1, i1] = 1 => k=1, i=1.
        
        # u
        u_00 = (i_off+0)*m + (j_off+0)
        u_11 = (i_off+1)*m + (j_off+1)
        U[u_00, r] = 1
        U[u_11, r] = 1
        # v
        v_00 = (j_off+0)*p + (k_off+0)
        v_11 = (j_off+1)*p + (k_off+1)
        V[v_00, r] = 1
        V[v_11, r] = 1
        # w
        w_00 = (k_off+0)*n + (i_off+0)
        w_11 = (k_off+1)*n + (i_off+1)
        W[w_00, r] = 1
        W[w_11, r] = 1
        r += 1
        
        # M2 = (A10+A11)B00
        u_10 = (i_off+1)*m + (j_off+0)
        U[u_10, r] = 1
        U[u_11, r] = 1
        V[v_00, r] = 1
        # C10 += M2
        w_10 = (k_off+0)*n + (i_off+1)
        W[w_10, r] = 1
        r += 1
        
        # M3 = A01(B11-B01)
        u_01 = (i_off+0)*m + (j_off+1)
        U[u_01, r] = 1
        v_11 = (j_off+1)*p + (k_off+1)
        v_01 = (j_off+0)*p + (k_off+1)
        V[v_11, r] = 1
        V[v_01, r] = -1
        # C01 += M3
        w_01 = (k_off+1)*n + (i_off+0)
        W[w_01, r] = 1
        r += 1
        
        # M4 = A10(B10-B00)
        U[u_10, r] = 1
        v_10 = (j_off+1)*p + (k_off+0)
        V[v_10, r] = 1
        V[v_00, r] = -1
        # C10 += M4
        W[w_10, r] = 1
        r += 1
        
        # M5 = (A00+A01)B11
        U[u_00, r] = 1
        U[u_01, r] = 1
        V[v_11, r] = 1
        # C01 += M5
        W[w_01, r] = 1
        # C00 -= M5
        W[w_00, r] = -1
        r += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[u_10, r] = 1
        U[u_00, r] = -1
        V[v_00, r] = 1
        V[v_01, r] = 1
        # C11 += M6
        W[w_11, r] = 1
        r += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[u_01, r] = 1
        U[u_11, r] = -1
        V[v_10, r] = 1
        V[v_11, r] = 1
        # C00 += M7
        W[w_00, r] = 1
        r += 1

    def add_school_221(i_off, j_off, k_val):
        """
        Schoolbook for 2x2 A by 2x1 B.
        i in {0,1}+off, j in {0,1}+off, k=k_val.
        """
        nonlocal r
        for i_l in range(2):
            for j_l in range(2):
                u_idx = (i_off+i_l)*m + (j_off+j_l)
                v_idx = (j_off+j_l)*p + k_val
                w_idx = k_val*n + (i_off+i_l)
                U[u_idx, r] = 1
                V[v_idx, r] = 1
                W[w_idx, r] = 1
                r += 1

    def add_school_215(i_off, j_val, k_start, k_end):
        """
        Schoolbook for 2x1 A by 1x5 B.
        i in {0,1}+off, j=j_val, k in [k_start, k_end).
        """
        nonlocal r
        for i_l in range(2):
            for k in range(k_start, k_end):
                u_idx = (i_off+i_l)*m + j_val
                v_idx = j_val*p + k
                w_idx = k*n + (i_off+i_l)
                U[u_idx, r] = 1
                V[v_idx, r] = 1
                W[w_idx, r] = 1
                r += 1

    def add_school_125(i_val, j_start, j_end, k_start, k_end):
        """
        Schoolbook for 1x2 A by 2x5 B.
        i=i_val, j in [j_start, j_end), k in [k_start, k_end).
        """
        nonlocal r
        for j in range(j_start, j_end):
            for k in range(k_start, k_end):
                u_idx = i_val*m + j
                v_idx = j*p + k
                w_idx = k*n + i_val
                U[u_idx, r] = 1
                V[v_idx, r] = 1
                W[w_idx, r] = 1
                r += 1

    def add_school_115(i_val, j_val, k_start, k_end):
        """
        Schoolbook for 1x1 A by 1x5 B.
        """
        nonlocal r
        for k in range(k_start, k_end):
            u_idx = i_val*m + j_val
            v_idx = j_val*p + k
            w_idx = k*n + i_val
            U[u_idx, r] = 1
            V[v_idx, r] = 1
            W[w_idx, r] = 1
            r += 1

    # Execute decomposition
    # 1. Top-Left 2x2 by Top-Left 2x5
    # Split 2x5 into 2x3 and 2x2
    # 2x3 split into 2x2 and 2x1
    
    # Part 1.1: 2x2 x 2x2 (k=0,1)
    add_strassen_222_correct(0, 0, 0) # i=0,1, j=0,1, k=0,1
    
    # Part 1.2: 2x2 x 2x1 (k=2)
    add_school_221(0, 0, 2) # i=0,1, j=0,1, k=2
    
    # Part 1.3: 2x2 x 2x2 (k=3,4)
    add_strassen_222_correct(0, 0, 3) # i=0,1, j=0,1, k=3,4
    
    # 2. Top-Right 2x1 by Bottom-Row 1x5 (j=2)
    # i=0,1, j=2, k=0..4
    add_school_215(0, 2, 0, 5)
    
    # 3. Bottom-Row 1x2 by Top-Left 2x5 (i=2, j=0,1)
    # i=2, j=0,1, k=0..4
    add_school_125(2, 0, 2, 0, 5)
    
    # 4. Bottom-Right 1x1 by Bottom-Row 1x5 (i=2, j=2)
    # i=2, j=2, k=0..4
    add_school_115(2, 2, 0, 5)
    
    # Verify r == 43
    if r != 43:
        print(f"Warning: Rank is {r}, expected 43")
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
