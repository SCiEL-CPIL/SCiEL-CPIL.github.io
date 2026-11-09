"""Tensor decomposition for 4x5x5 matrix multiplication."""
import time
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2x2(I_start, J_start, K_start, U, V, W, offset):
    """
    Add Strassen decomposition for a 2x2x2 sub-problem.
    I indices: I_start, I_start+1
    J indices: J_start, J_start+1
    K indices: K_start, K_start+1
    """
    n, m, p = DIMENSIONS
    def U_idx(i, j): return i * m + j
    def V_idx(j, k): return j * p + k
    def W_idx(k, i): return k * n + i
    
    i0, i1 = I_start, I_start + 1
    j0, j1 = J_start, J_start + 1
    k0, k1 = K_start, K_start + 1
    
    # Strassen's 7 multiplications for 2x2x2:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each rank-1 term, we need (U_row, V_row, W_row, coeff)
    # U is indexed by (i,j), V by (j,k), W by (k,i)
    
    terms = []
    
    # M1 = (a+d)(e+h): a=A[i0,j0], d=A[i1,j1], e=B[j0,k0], h=B[j1,k1]
    # (a+d) involves U rows: (i0,j0) and (i1,j1)
    # (e+h) involves V rows: (j0,k0) and (j1,k1)
    # We can split this into rank-1 terms:
    # (a+d)(e+h) = a*e + a*h + d*e + d*h
    # But that's 4 terms, not 1. Wait, Strassen uses the product (a+d)(e+h) as ONE rank-1 term.
    # In tensor decomposition, each "multiplication" corresponds to a rank-1 term.
    # So (a+d)(e+h) means: U has a+d pattern, V has e+h pattern, W connects them.
    # Actually, the way Strassen works is that M1 is a scalar computed from linear combinations.
    # In the tensor decomposition framework, each rank-1 term c has:
    #   U[., c] = vector u, V[., c] = vector v, W[., c] = vector w
    #   such that for all (i,j,k): tensor[i,j,k] = sum_c u[i,j] * v[j,k] * w[k,i]
    
    # For (a+d)(e+h) to be one rank-1 term, we need:
    # u[i0,j0] = 1, u[i1,j1] = 1, others 0
    # v[j0,k0] = 1, v[j1,k1] = 1, others 0
    # But then the contribution to C is:
    # C[i0,j0,k0] gets u[i0,j0]*v[j0,k0]*w[k0,i0] = 1*1*w[k0,i0]
    # C[i0,j0,k1] gets u[i0,j0]*v[j0,k1]*w[k1,i0] = 1*0*... = 0
    # C[i0,j1,k0] gets u[i0,j1]*v[j1,k0]*w[k0,i0] = 0*... = 0
    # C[i0,j1,k1] gets u[i0,j1]*v[j1,k1]*w[k1,i0] = 0*... = 0
    # C[i1,j0,k0] gets u[i1,j0]*v[j0,k0]*w[k0,i1] = 0*... = 0
    # ...
    # This doesn't work directly because a single rank-1 term with sparse U and V can't produce
    # the full (a+d)(e+h) pattern in C.
    
    # Actually, I think the standard way to represent Strassen in U,V,W form is:
    # Each of the 7 "multiplications" M_s corresponds to a rank-1 term where:
    # U_s is the vector of coefficients for A, V_s is the vector of coefficients for B,
    # and W_s distributes M_s to the appropriate C entries.
    
    # Let me reconsider. The matrix multiplication tensor T[i,j,k] = delta[i,j,k] (1 if it's a valid (i,j,k) triple, 0 otherwise... no wait.
    # Actually T[i,j,k] = 1 for all (i,j,k) where i in [0,n), j in [0,m), k in [0,p), and the tensor is such that
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = 1 for all valid (i,j,k).
    
    # For Strassen on 2x2x2, the 7 rank-1 terms are defined by their U, V, W patterns.
    # Let me look up the explicit U,V,W for Strassen.
    
    # Actually, let me just implement it by building the rank-1 terms directly.
    # For each of the 7 Strassen products, I create a rank-1 term.
    
    # M1 = (a+d)(e+h)
    # U pattern: u[i0,j0]=1, u[i1,j1]=1
    # V pattern: v[j0,k0]=1, v[j1,k1]=1
    # W pattern: w[k0,i0]=1, w[k1,i1]=1  (to distribute M1 to C[i0,k0] and C[i1,k1])
    # But this gives: C[i0,k0] += u[i0,j0]*v[j0,k0]*w[k0,i0] = 1 (for the a*e part)
    # But also C[i0,k1] would get contributions if there's overlap...
    
    # Hmm, I think the issue is that a single rank-1 term in the U,V,W formulation
    # corresponds to u ⊗ v ⊗ w, which gives a specific pattern in T.
    
    # Let me think differently. I'll construct the 7 rank-1 terms for Strassen explicitly.
    
    # The 7 rank-1 decompositions for 2x2x2 Strassen:
    # Each term s has vectors u_s (length 4, indexed by (i,j) pairs), v_s (length 4, indexed by (j,k) pairs), w_s (length 4, indexed by (k,i) pairs)
    
    # Term 1: M1 = (a+d)(e+h)
    # u1: [1, 0, 0, 1]  (a=A[0,0], d=A[1,1])
    # v1: [1, 0, 0, 1]  (e=B[0,0], h=B[1,1])
    # w1: [1, 0, 0, 1]  (contributes to C[0,0] and C[1,1])
    # But wait, w is indexed by (k,i), so w[0,0] corresponds to (k=0,i=0), w[1,1] to (k=1,i=1)
    # The contribution to T[i,j,k] from this term is u[i,j]*v[j,k]*w[k,i]
    # T[0,0,0] = u[0,0]*v[0,0]*w[0,0] = 1*1*1 = 1
    # T[0,0,1] = u[0,0]*v[0,1]*w[1,0] = 1*0*0 = 0
    # T[0,1,0] = u[0,1]*v[1,0]*w[0,0] = 0*0*1 = 0
    # T[0,1,1] = u[0,1]*v[1,1]*w[1,0] = 0*1*0 = 0
    # T[1,0,0] = u[1,0]*v[0,0]*w[0,1] = 0*1*0 = 0
    # T[1,0,1] = u[1,0]*v[0,1]*w[1,1] = 0*0*1 = 0
    # T[1,1,0] = u[1,1]*v[1,0]*w[0,1] = 1*0*0 = 0
    # T[1,1,1] = u[1,1]*v[1,1]*w[1,1] = 1*1*1 = 1
    # So this term only gives T[0,0,0]=1 and T[1,1,1]=1. That's not enough for M1.
    
    # I see the problem. The rank-1 decomposition of the tensor T (which is all 1s for 2x2x2)
    # is not simply given by Strassen's 7 terms in the U,V,W form I'm thinking of.
    
    # Let me look at this from the perspective of the matrix multiplication tensor.
    # For 2x2x2, T[i,j,k] = 1 for all i,j,k in {0,1}.
    # We want to write T = sum_s u_s ⊗ v_s ⊗ w_s where u_s is 2x2, v_s is 2x2, w_s is 2x2.
    
    # Strassen shows that 7 such terms suffice. Let me find the explicit u_s, v_s, w_s.
    
    # From the literature, the 7 rank-1 terms for 2x2x2 Strassen are:
    # s=1: u=[[1,0],[0,1]], v=[[1,0],[0,1]], w=[[1,0],[0,1]]  -- no, that's just the identity, gives T[0,0,0]=T[1,1,1]=1
    # That's not right either.
    
    # Let me just implement it by constructing the full 2x2x2 tensor and decomposing it.
    # Actually, for this specific problem, let me take a different approach.
    
    # I'll implement the Strassen decomposition by tracking the coefficients.
    # For a 2x2x2 block, the 7 rank-1 terms have the following U,V,W patterns (in local coords):
    
    # I'll define them explicitly.
    
    return 0  # placeholder

def construct():
    """
    Construct a rank-64 tensor decomposition for 4x5x5 matrix multiplication.
    Uses Strassen for the 4x4x4 core and schoolbook for the remainders.
    """
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Partition
    I_all = list(range(n))  # [0,1,2,3]
    J1 = list(range(4))     # [0,1,2,3]
    J2 = [4]
    K1 = list(range(4))     # [0,1,2,3]
    K2 = [4]
    
    # We'll build U, V, W as lists of lists (or use arrays)
    # Start with a list of rank-1 terms, each as (U_vec, V_vec, W_vec)
    # U_vec is a dict or list indexed by (i,j)
    # V_vec is a dict or list indexed by (j,k)
    # W_vec is a dict or list indexed by (k,i)
    
    # For efficiency, I'll use dense arrays of length n*m, m*p, p*n
    # But since we're building up terms, let's use a list of terms first.
    
    terms = []  # Each term: (U_array, V_array, W_array) where arrays are dicts {(i,j): val}, etc.
    
    def add_term(U_dict, V_dict, W_dict):
        terms.append((U_dict, V_dict, W_dict))
    
    # Strassen 2x2x2 decomposition
    # For local indices (li, lj, lk) in {0,1}^3, the 7 rank-1 terms are:
    # I need to find the explicit U,V,W for each of the 7 Strassen products.
    
    # Let me derive them. For 2x2x2, T[li,lj,lk] = 1 for all li,lj,lk.
    # We want T = sum_{s=1}^7 u_s ⊗ v_s ⊗ w_s
    # where (u_s)_{li,lj}, (v_s)_{lj,lk}, (w_s)_{lk,li}
    
    # From Strassen's algorithm:
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    # where M1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1])B[0,0]
    # M3 = A[0,0](B[0,1]-B[1,1])
    # M4 = A[1,1](B[1,0]-B[0,0])
    # M5 = (A[0,0]+A[0,1])B[1,1]
    # M6 = (A[1,0]-A[0,0])(B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1])(B[1,0]+B[1,1])
    
    # In the tensor decomposition, each M_s corresponds to a rank-1 term.
    # The contribution of term s to T[li,lj,lk] is (u_s)[li,lj] * (v_s)[lj,lk] * (w_s)[lk,li]
    
    # For M1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1]):
    # u1[0,0] = 1, u1[1,1] = 1, others 0
    # v1[0,0] = 1, v1[1,1] = 1, others 0
    # w1: we need w1 such that the term contributes correctly to C.
    # The product u1 ⊗ v1 ⊗ w1 has non-zero entries at:
    # (0,0,0): u1[0,0]*v1[0,0]*w1[0,0] = w1[0,0]
    # (0,0,1): u1[0,0]*v1[0,1]*w1[1,0] = 0
    # (0,1,0): u1[0,1]*v1[1,0]*w1[0,0] = 0
    # (0,1,1): u1[0,1]*v1[1,1]*w1[1,0] = 0
    # (1,0,0): u1[1,0]*v1[0,0]*w1[0,1] = 0
    # (1,0,1): u1[1,0]*v1[0,1]*w1[1,1] = 0
    # (1,1,0): u1[1,1]*v1[1,0]*w1[0,1] = 0
    # (1,1,1): u1[1,1]*v1[1,1]*w1[1,1] = w1[1,1]
    
    # So this term only contributes to T[0,0,0] and T[1,1,1].
    # But T[0,0,0] should be 1 (from the full tensor), and this term contributes w1[0,0].
    # Similarly T[1,1,1] should be 1, and this term contributes w1[1,1].
    # But we also need contributions to other (li,lj,lk) from other terms.
    
    # This approach is getting complicated. Let me just hardcode the 7 rank-1 terms
    # for Strassen 2x2x2.
    
    # After research, the 7 rank-1 terms for 2x2x2 Strassen (in U,V,W form) are:
    
    def strassen_terms_2x2x2():
        """Return 7 rank-1 terms for 2x2x2 Strassen decomposition."""
        # Each term is (U, V, W) where U is 2x2, V is 2x2, W is 2x2 (as dicts)
        
        # I'll define them based on the standard Strassen construction.
        # The key insight is that each term s has:
        # U_s[i,j] = coefficient of A[i,j] in the i-th input to M_s
        # V_s[j,k] = coefficient of B[j,k] in the k-th input to M_s  
        # W_s[k,i] = coefficient of M_s in C[i,k]
        
        # M1 = (A00+A11)(B00+B11)
        # U1: A00->1, A11->1
        # V1: B00->1, B11->1
        # W1: C00->1, C11->1
        t1 = (
            {(0,0): 1, (1,1): 1},  # U
            {(0,0): 1, (1,1): 1},  # V
            {(0,0): 1, (1,1): 1},  # W  (W[k,i]: k=0,i=0 -> C[0,0]; k=1,i=1 -> C[1,1])
        )
        
        # M2 = (A10+A11)B00
        # U2: A10->1, A11->1
        # V2: B00->1
        # W2: C10->1, C11->(-1)  (from C10=M2+M4, C11=M1-M2+M3+M6)
        t2 = (
            {(1,0): 1, (1,1): 1},  # U
            {(0,0): 1},            # V
            {(0,1): 1, (1,1): -1}, # W  (W[k,i]: k=0,i=1 -> C[1,0]; k=1,i=1 -> C[1,1])
        )
        
        # M3 = A00(B01-B11)
        # U3: A00->1
        # V3: B01->1, B11->-1
        # W3: C01->1, C11->1
        t3 = (
            {(0,0): 1},            # U
            {(0,1): 1, (1,1): -1}, # V
            {(1,0): 1, (1,1): 1},  # W  (W[k,i]: k=1,i=0 -> C[0,1]; k=1,i=1 -> C[1,1])
        )
        
        # M4 = A11(B10-B00)
        # U4: A11->1
        # V4: B10->1, B00->-1
        # W4: C00->1, C10->1
        t4 = (
            {(1,1): 1},            # U
            {(1,0): 1, (0,0): -1}, # V
            {(0,0): 1, (0,1): 1},  # W  (W[k,i]: k=0,i=0 -> C[0,0]; k=0,i=1 -> C[1,0])
        )
        
        # M5 = (A00+A01)B11
        # U5: A00->1, A01->1
        # V5: B11->1
        # W5: C00->-1, C01->1
        t5 = (
            {(0,0): 1, (0,1): 1},  # U
            {(1,1): 1},            # V
            {(1,0): -1, (1,1): 1}, # W  (W[k,i]: k=1,i=0 -> C[0,1]; k=1,i=1 -> C[1,1])
            # Wait, C00 = M1+M4-M5+M7, so M5 contributes -1 to C00
            # C01 = M3+M5, so M5 contributes +1 to C01
        )
        # Fix W5:
        t5 = (
            {(0,0): 1, (0,1): 1},  # U
            {(1,1): 1},            # V
            {(0,0): -1, (1,0): 1}, # W  (W[k,i]: k=0,i=0 -> C[0,0] coeff -1; k=1,i=0 -> C[0,1] coeff 1)
            # But W is indexed by (k,i), so W[0,0] is for (k=0,i=0) which is C[0,0]
            # W[1,0] is for (k=1,i=0) which is C[0,1]
        )
        
        # M6 = (A10-A00)(B00+B01)
        # U6: A10->1, A00->-1
        # V6: B00->1, B01->1
        # W6: C11->1
        t6 = (
            {(1,0): 1, (0,0): -1}, # U
            {(0,0): 1, (0,1): 1},  # V
            {(1,1): 1},            # W  (W[k,i]: k=1,i=1 -> C[1,1])
        )
        
        # M7 = (A01-A11)(B10+B11)
        # U7: A01->1, A11->-1
        # V7: B10->1, B11->1
        # W7: C00->1
        t7 = (
            {(0,1): 1, (1,1): -1}, # U
            {(1,0): 1, (1,1): 1},  # V
            {(0,0): 1},            # W  (W[k,i]: k=0,i=0 -> C[0,0])
        )
        
        return [t1, t2, t3, t4, t5, t6, t7]
    
    # Verify Strassen decomposition for 2x2x2
    # T[i,j,k] = sum_s U_s[i,j] * V_s[j,k] * W_s[k,i]
    # Should equal 1 for all i,j,k in {0,1}
    
    terms_2x2 = strassen_terms_2x2x2()
    for i in range(2):
        for j in range(2):
            for k in range(2):
                total = 0
                for U_s, V_s, W_s in terms_2x2:
                    u_val = U_s.get((i,j), 0)
                    v_val = V_s.get((j,k), 0)
                    w_val = W_s.get((k,i), 0)
                    total += u_val * v_val * w_val
                assert total == 1, f"Strassen verification failed at ({i},{j},{k}): got {total}"
    
    # Now build the full 4x5x5 decomposition
    # Partition:
    # I: [0,1,2,3] (all 4 rows)
    # J: J1=[0,1,2,3], J2=[4]
    # K: K1=[0,1,2,3], K2=[4]
    
    # Sub-problems:
    # 1. I=[0,1,2,3], J=J1, K=K1: 4x4x4 -> 4 Strassen 2x2x2 blocks
    # 2. I=[0,1,2,3], J=J1, K=K2: 4x4x1 -> schoolbook, rank 16
    # 3. I=[0,1,2,3], J=J2, K=K1: 4x1x4 -> schoolbook, rank 16
    # 4. I=[0,1,2,3], J=J2, K=K2: 4x1x1 -> schoolbook, rank 4
    
    # For each sub-problem, add the corresponding rank-1 terms to the global U,V,W.
    
    all_terms = []  # List of (U_dict, V_dict, W_dict) with global indices
    
    # Sub-problem 1: 4x4x4 with 4 Strassen blocks
    I_blocks = [[0,1], [2,3]]
    J_blocks = [[0,1], [2,3]]
    K_blocks = [[0,1], [2,3]]
    
    for Ib in I_blocks:
        for Jb in J_blocks:
            for Kb in K_blocks:
                i0, i1 = Ib
                j0, j1 = Jb
                k0, k1 = Kb
                for U_s, V_s, W_s in terms_2x2:
                    # Map local to global
                    U_g = {}
                    for (li, lj), val in U_s.items():
                        U_g[(i0 + li, j0 + lj)] = val
                    V_g = {}
                    for (lj, lk), val in V_s.items():
                        V_g[(j0 + lj, k0 + lk)] = val
                    W_g = {}
                    for (lk, li), val in W_s.items():
                        W_g[(k0 + lk, i0 + li)] = val
                    all_terms.append((U_g, V_g, W_g))
    
    # Sub-problem 2: 4x4x1 schoolbook
    # I=[0,1,2,3], J=[0,1,2,3], K=[4]
    # For each (i,j,k=4): T[i,j,4] = 1
    # Rank-1 term: U[(i,j)]=1, V[(j,4)]=1, W[(4,i)]=1
    for i in range(4):
        for j in range(4):
            U_g = {(i, j): 1}
            V_g = {(j, 4): 1}
            W_g = {(4, i): 1}
            all_terms.append((U_g, V_g, W_g))
    
    # Sub-problem 3: 4x1x4 schoolbook
    # I=[0,1,2,3], J=[4], K=[0,1,2,3]
    for i in range(4):
        for k in range(4):
            U_g = {(i, 4): 1}
            V_g = {(4, k): 1}
            W_g = {(k, i): 1}
            all_terms.append((U_g, V_g, W_g))
    
    # Sub-problem 4: 4x1x1 schoolbook
    # I=[0,1,2,3], J=[4], K=[4]
    for i in range(4):
        U_g = {(i, 4): 1}
        V_g = {(4, 4): 1}
        W_g = {(4, i): 1}
        all_terms.append((U_g, V_g, W_g))
    
    rank = len(all_terms)
    assert rank == 64, f"Expected rank 64, got {rank}"
    
    # Convert to dense arrays
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, (U_dict, V_dict, W_dict) in enumerate(all_terms):
        for (i, j), val in U_dict.items():
            U[i * m + j][col] = val
        for (j, k), val in V_dict.items():
            V[j * p + k][col] = val
        for (k, i), val in W_dict.items():
            W[k * n + i][col] = val
    
    # Verify the decomposition
    # Check that sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] == 1 for all valid (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                assert total == 1, f"Verification failed at ({i},{j},{k}): got {total}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
