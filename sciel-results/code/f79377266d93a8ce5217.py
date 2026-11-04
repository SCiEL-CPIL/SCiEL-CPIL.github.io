"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

_start_time = time.time()
_DEADLINE = 145  # seconds, safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Strassen 2x2x2 tensor: rank 7
    # U_strassen: 4x7, V_strassen: 4x7, W_strassen: 4x7
    # C = A*B for 2x2 matrices
    # Standard Strassen:
    # M1 = (A11-A22)*(B22-B11)
    # M2 = (A11+A22)*B22
    # M3 = A11*(B11-B22)
    # M4 = A22*(B11+B22)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A22)*(B11+B12)
    # M7 = (A12-A22)*(B21-B22)
    # C11 = M1+M4+M5-M7
    # C12 = M3+M7
    # C21 = M1+M6
    # C22 = M2+M4-M1-M5
    
    # Indexing: A[i,j] with i,j in {0,1} -> flat index i*2+j
    # B[j,k] with j,k in {0,1} -> flat index j*2+k
    # C[i,k] with i,k in {0,1} -> flat index i*2+k
    
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # M1 = (A00-A11)*(B11-B00)
    # A indices: A00=0, A11=3
    # B indices: B11=3, B00=0
    # C contributions: C00 += 1, C11 += 1, C10 += 1, C11 += -1 (wait, let me be careful)
    # C11 = M1+M4+M5-M7 -> C[0,0] += M1
    # C22 = M2+M4-M1-M5 -> C[1,1] += -M1
    # C21 = M1+M6 -> C[1,0] += M1
    # C12 = M3+M7 -> no M1
    
    # Let me use 0-indexed: C[i,k], i in {0,1}, k in {0,1}
    # C[0,0] = M1+M4+M5-M7
    # C[0,1] = M3+M7
    # C[1,0] = M1+M6
    # C[1,1] = M2+M4-M1-M5
    
    # M1: A00=1, A11=-1; B11=1, B00=-1; C00+=1, C10+=1, C11+=-1
    U_s[0, 0] = 1; U_s[3, 0] = -1
    V_s[3, 0] = 1; V_s[0, 0] = -1
    W_s[0, 0] = 1; W_s[2, 0] = 1; W_s[3, 0] = -1  # C[0,0]=idx0, C[1,0]=idx2, C[1,1]=idx3
    
    # M2 = (A00+A11)*B11
    U_s[0, 1] = 1; U_s[3, 1] = 1
    V_s[3, 1] = 1
    W_s[3, 1] = 1  # C[1,1] += M2
    
    # M3 = A00*(B00-B11)
    U_s[0, 2] = 1
    V_s[0, 2] = 1; V_s[3, 2] = -1
    W_s[1, 2] = 1  # C[0,1] += M3
    
    # M4 = A11*(B00+B11)
    U_s[3, 3] = 1
    V_s[0, 3] = 1; V_s[3, 3] = 1
    W_s[0, 3] = 1; W_s[3, 3] = 1  # C[0,0] += M4, C[1,1] += M4
    
    # M5 = (A00+A01)*B11
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    W_s[0, 4] = 1; W_s[3, 4] = -1  # C[0,0] += M5, C[1,1] -= M5
    
    # M6 = (A10-A11)*(B00+B01)
    U_s[2, 5] = 1; U_s[3, 5] = -1
    V_s[0, 5] = 1; V_s[1, 5] = 1
    W_s[2, 5] = 1  # C[1,0] += M6
    
    # M7 = (A01-A11)*(B10-B11)
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[2, 6] = 1; V_s[3, 6] = -1
    W_s[0, 6] = -1; W_s[1, 6] = 1  # C[0,0] -= M7, C[0,1] += M7
    
    # Verify Strassen
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = np.sum(U_s[i*2+j] * V_s[j*2+k] * W_s[k*2+i])
                if abs(val - 1.0) > 1e-9:
                    print(f"Strassen error at ({i},{j},{k}): {val}")
    
    best_rank = n * m * p  # 128 schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    def try_decomposition(U, V, W):
        """Verify and return True if valid."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def build_kronecker(n_blocks, m_blocks, p_blocks, base_U, base_V, base_W):
        """
        Build a Kronecker product decomposition.
        n_blocks: number of 2x2 blocks along n dimension
        m_blocks: number of 2x2 blocks along m dimension  
        p_blocks: number of 2x2 blocks along p dimension
        
        For 4x4x8: n=4, m=4, p=8
        Possible: 2x2 blocks: n_blocks=2, m_blocks=2, p_blocks=4
        Or: 4x4 blocks: n_blocks=1, m_blocks=1, p_blocks=2 (but 4x4 needs its own decomposition)
        """
        # This approach: treat as (n_blocks x m_blocks x p_blocks) independent 2x2x2 problems
        # Each uses Strassen rank 7
        # Total rank = n_blocks * m_blocks * p_blocks * 7
        # But we need to handle the actual dimensions
        
        # For 4x4x8:
        # Option 1: 2x2 blocks: 2*2*4 = 16 subproblems, each rank 7 -> total 112
        # Option 2: Use 4x4 base for n,m and 2x2 for p
        # Option 3: Mixed hierarchies
        
        # General: if we can factor n = a*2^n1, m = b*2^m1, p = c*2^p1
        # where a, b, c are base sizes handled by some base tensor
        
        pass
    
    # Strategy: Enumerate all ways to partition 4x4x8 into block structures
    # using 2x2 Strassen as the atomic operation
    
    # 4 = 2*2, 4 = 2*2, 8 = 2*2*2
    # So we have 2 levels of 2x2 blocks in n, 2 in m, 3 in p
    # Total subproblems: 2*2*2*2*2*2 = 64? No...
    
    # Actually, let's think of it as:
    # The tensor T_{n,m,p} can be written as a Kronecker product of smaller tensors
    # if n, m, p are all powers of 2 (or have a common factorization)
    
    # 4 = 2^2, 4 = 2^2, 8 = 2^3
    # T_{4,4,8} = T_{2,2,2} ⊗ T_{2,2,2} ⊗ T_{2,2,2}? No, that gives 2^3=8 in each dim.
    
    # More carefully:
    # T_{4,4,8}: n=4, m=4, p=8
    # We can write 4 = 2*2, 4 = 2*2, 8 = 2*4 = 2*2*2
    # So: T_{4,4,8} = T_{2,2,2} ⊗ T_{2,2,2} ⊗ T_{2,2,2}? 
    # That would be n=2*2*2=8, m=2*2*2=8, p=2*2*2=8. Not right.
    
    # Correct: T_{n1*n2, m1*m2, p1*p2} = T_{n1,m1,p1} ⊗ T_{n2,m2,p2}
    # So T_{4,4,8} = T_{2,2,2} ⊗ T_{2,2,4}
    # And T_{2,2,4} = T_{2,2,2} ⊗ T_{2,2,2}? No, that gives T_{4,4,4}.
    
    # T_{2,2,4}: n=2, m=2, p=4. 4=2*2.
    # T_{2,2,4} = T_{2,2,2} ⊗ T_{1,1,2}? T_{1,1,2} is trivial: rank 1.
    # Actually T_{1,1,p} has rank 1 for any p (just A[0,0]*B[0,k]*W[k,0])
    # Wait no: T_{1,1,p}: C[0,k] = A[0,0]*B[0,k]. This is rank 1: 
    # U[0,0]=1, V[k,0]=1 for all k... no, that's not right either.
    # T_{1,1,p}: sum over j (only j=0): C[0,k] = A[0,0] * B[0,k]
    # This is already a product, so rank 1: U[0,0]=1, V[0,k]=1 for all k? No.
    # In tensor form: T_{i,j,k} = delta_{i,0} * delta_{j,0} * delta_{k,k}... 
    # T_{0,0,k} = 1 for all k. This is a sum of p rank-1 terms? No, it's one tensor.
    # The rank of T_{1,1,p}: we need to write sum_r U[0,r]*V[0,r]*W[k,r] = 1 for all k.
    # So V[0,r]*W[k,r] must sum to 1 for each k. We can set W[k,0]=1, V[0,0]=1, U[0,0]=1, rest 0. Rank 1.
    
    # So T_{2,2,4} = T_{2,2,2} ⊗ T_{1,1,2}
    # T_{1,1,2} rank 1.
    # T_{2,2,2} rank 7 (Strassen).
    # So T_{2,2,4} rank = 7*1 = 7.
    
    # Then T_{4,4,8} = T_{2,2,2} ⊗ T_{2,2,4} = T_{2,2,2} ⊗ (T_{2,2,2} ⊗ T_{1,1,2})
    # Rank = 7 * 7 * 1 = 49? 
    
    # Wait, let me verify the Kronecker product structure more carefully.
    # T_{n1*n2, m1*m2, p1*p2} with indices:
    # i = i1*n2 + i2, j = j1*m2 + j2, k = k1*p2 + k2
    # T[i,j,k] = delta_{i,j,k} (1 if it's a valid matmul term)
    # = sum_{j} delta... 
    # For matrix multiplication: T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # Actually T_{i,j,k} is the coefficient in the bilinear form.
    # T[i,j,k] = 1 always? No.
    
    # The matrix multiplication tensor T_{n,m,p} has entries T[i,j,k] where:
    # i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 if j is a valid index connecting A and B, which is always true.
    # Wait, no. The tensor is defined such that:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor decomposition: C[i,k] = sum_r U[i*m+j... 
    
    # Let me re-read the interface:
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # The decomposition is: for all i,j,k:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # (since T[i,j,k] = 1 for all valid i,j,k in matrix multiplication)
    
    # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # This is just the all-ones tensor! The "structure" comes from how the indices are grouped.
    
    # Wait, that can't be right. If T is all ones, then the rank is 1.
    # Let me re-read...
    
    # "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor T_{n,m,p} has T[i,j,k] = 1 if the (i,j) entry of A 
    # and (j,k) entry of B contribute to (i,k) entry of C.
    # For standard matrix multiplication, EVERY (i,j,k) triple with valid indices contributes.
    # So T[i,j,k] = 1 for all i,j,k.
    
    # But then the rank would be 1, which is absurd. Let me re-read the problem.
    
    # "Find a rank-minimal exact tensor decomposition for multiplying a 4 by 4 matrix by a 4 by 8 matrix."
    # The output factor follows the official trace convention W[k*n+i,r].
    
    # I think the tensor is NOT all ones. The matrix multiplication tensor is:
    # T[i,j,k] = 1 if j is the "middle" index, which it always is.
    # But the constraint is that the SAME j must be used in A[i,j] and B[j,k].
    # In the tensor decomposition, the index j appears in BOTH U and V.
    # So the "tensor" is really a 3-tensor with the constraint that the j-index in U 
    # and the j-index in V must match.
    
    # Actually, I think I was overcomplicating this. The standard definition:
    # The matrix multiplication tensor M_{n,m,p} is a tensor in C^{n*m} ⊗ C^{m*p} ⊗ C^{p*n}
    # defined by M_{n,m,p} = sum_{i,j,k} e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i}
    # where the sum is over all valid i,j,k.
    # So M[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k.
    
    # This IS the all-ones tensor in the appropriate basis! And its rank is indeed n*m*p 
    # for the trivial decomposition, but can be lower with clever decompositions.
    
    # Wait, no. The tensor M has entries:
    # M[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # M[a,b,c] = 1 if there exist i,j,k such that a=i*m+j, b=j*p+k, c=k*n+i
    # Otherwise M[a,b,c] = 0.
    
    # So it's NOT all ones. It's a sparse tensor where only specific triples (a,b,c) are 1.
    # The number of 1s is n*m*p = 4*4*8 = 128.
    
    # Now, the Kronecker product structure:
    # If n = n1*n2, m = m1*m2, p = p1*p2, then:
    # a = i*m+j = (i1*n2+i2)*(m1*m2) + (j1*m2+j2)
    #   = i1*n2*m1*m2 + i2*m1*m2 + j1*m2 + j2
    #   = (i1*m1 + j1/m2... ) this is getting messy.
    
    # Let me use a cleaner indexing. Let i = i1*n2 + i2, j = j1*m2 + j2, k = k1*p2 + k2.
    # Then:
    # a = i*m + j = (i1*n2+i2)*(m1*m2) + (j1*m2+j2)
    #   = i1*n2*m1*m2 + i2*m1*m2 + j1*m2 + j2
    # b = j*p + k = (j1*m2+j2)*(p1*p2) + (k1*p2+k2)
    #   = j1*m2*p1*p2 + j2*p1*p2 + k1*p2 + k2
    # c = k*n + i = (k1*p2+k2)*(n1*n2) + (i1*n2+i2)
    #   = k1*p2*n1*n2 + k2*n1*n2 + i1*n2 + i2
    
    # For the Kronecker product M_{n1,m1,p1} ⊗ M_{n2,m2,p2}:
    # The (a1,a2) index of U corresponds to (i1*m1+j1, i2*m2+j2) in the two factors.
    # But our a = i*m+j is a single index, not a pair.
    
    # I think the correct statement is:
    # M_{n1*n2, m1*m2, p1*p2} = M_{n1,m1,p1} ⊗ M_{n2,m2,p2}
    # where the tensor product is taken with appropriate index reshuffling.
    
    # Let me just try to construct it directly.
    
    # For 4x4x8:
    # Factor as 2*2 x 2*2 x 2*4 or 2*2 x 2*2 x 4*2 or 2*2 x 4*1 x 2*4 etc.
    
    # Let me try: n=2*2, m=2*2, p=2*4
    # M_{4,4,8} = M_{2,2,2} ⊗ M_{2,2,4}
    # And M_{2,2,4} = M_{2,2,2} ⊗ M_{2,2,2}? No, that gives M_{4,4,4}.
    # M_{2,2,4}: n=2, m=2, p=4=2*2.
    # M_{2,2,4} = M_{2,2,2} ⊗ M_{1,1,2}? 
    # M_{1,1,2}: n=1, m=1, p=2.
    # M_{1,1,2}[a,b,c] where a=i*1+j=i+j, b=j*2+k, c=k*1+i=k+i.
    # i=0, j=0, k=0: a=0, b=0, c=0
    # i=0, j=0, k=1: a=0, b=1, c=1
    # So M_{1,1,2} has M[0,0,0]=1 and M[0,1,1]=1. Rank 1? 
    # We need sum_r U[0,r]*V[b,r]*W[c,r] = 1 when (b,c) in {(0,0),(1,1)}, else 0.
    # Set U[0,0]=1, V[0,0]=1, V[1,0]=0... no.
    # V[b,r] and W[c,r]: for r=0: V[0,0]=1, W[0,0]=1, V[1,0]=1, W[1,0]=1.
    # Then sum_r U[0,r]*V[b,r]*W[c,r] = 1*1*1 = 1 for b=0,c=0 and b=1,c=1. 
    # But also for b=0,c=1: 1*1*1=1, which should be 0. So rank 1 doesn't work.
    
    # M_{1,1,2} rank: we need V[0,r]*W[0,r] + ... = 1 for (b,c)=(0,0)
    # V[1,r]*W[1,r] = 1 for (b,c)=(1,1)  
    # V[0,r]*W[1,r] = 0 for (b,c)=(0,1)
    # V[1,r]*W[0,r] = 0 for (b,c)=(1,0)
    
    # Let r=0: V[0,0]=1, W[0,0]=1, V[1,0]=0, W[1,0]=0
    # r=1: V[0,1]=0, W[0,1]=0, V[1,1]=1, W[1,1]=1
    # U[0,0]=1, U[0,1]=1
    # Check: (b,c)=(0,0): 1*1*1 + 1*0*0 = 1 ✓
    # (b,c)=(1,1): 1*0*0 + 1*1*1 = 1 ✓
    # (b,c)=(0,1): 1*1*0 + 1*0*1 = 0 ✓
    # (b,c)=(1,0): 1*0*1 + 1*1*0 = 0 ✓
    # So M_{1,1,2} has rank 2.
    
    # Hmm, so M_{2,2,4} = M_{2,2,2} ⊗ M_{1,1,2} would have rank 7*2=14.
    # Then M_{4,4,8} = M_{2,2,2} ⊗ M_{2,2,4} would have rank 7*14=98.
    
    # That matches the "rank-98 Strassen-Kronecker" from the cell findings!
    
    # Let me construct this explicitly.
    
    # M_{4,4,8} = M_{2,2,2} ⊗ M_{2,2,4}
    # where M_{2,2,4} = M_{2,2,2} ⊗ M_{1,1,2}
    
    # Let me define the tensors explicitly.
    
    # M_{1,1,2}: U (1*1=1 x rank), V (1*2=2 x rank), W (2*1=2 x rank)
    U_112 = np.array([[1, 1]])  # 1x2
    V_112 = np.array([[1, 0],  # b=0
                       [0, 1]])  # b=1
    W_112 = np.array([[1, 0],  # c=0
                       [0, 1]])  # c=1
    
    # Verify M_{1,1,2}
    for b in range(2):
        for c in range(2):
            val = np.sum(U_112[0] * V_112[b] * W_112[c])
            expected = 1 if b == c else 0
            if abs(val - expected) > 1e-9:
                print(f"M_112 error at ({b},{c}): {val} vs {expected}")
    
    # M_{2,2,2} (Strassen): already defined as U_s, V_s, W_s (4x7 each)
    
    # M_{2,2,4} = M_{2,2,2} ⊗ M_{1,1,2}
    # n=2*1=2, m=2*1=2, p=2*2=4
    # U: (n*m=4, rank=7*2=14)
    # V: (m*p=8, rank=14)
    # W: (p*n=8, rank=14)
    
    # Index mapping for Kronecker product:
    # i = i1*1 + i2 = i2 (i1 always 0 since n1=2, n2=1)
    # j = j1*1 + j2 = j2 (j1 always 0 since m1=2, m2=1)
    # k = k1*2 + k2 (k1 in {0,1}, k2 in {0,1})
    
    # a = i*m + j = i2*2 + j2 (for M_{2,2,4})
    # But in the Kronecker product, the index is (a1, a2) where a1 is for M_{2,2,2} and a2 for M_{1,1,2}
    # a1 = i1*m1 + j1 = i*2 + j (for 2x2)
    # a2 = i2*m2 + j2 = 0*1 + 0 = 0 (for 1x1, always 0)
    # Combined a = a1 * (m2*p2) + a2? Or some other mapping.
    
    # This is getting complicated. Let me just directly construct M_{2,2,4} by combining Strassen with the 1x1x2 tensor.
    
    # For M_{2,2,4}: C[i,k] = sum_j A[i,j] * B[j,k] where i in {0,1}, j in {0,1}, k in {0,1,2,3}
    # We can write k = k_hi*2 + k_lo where k_hi in {0,1}, k_lo in {0,1}
    # C[i, k_hi*2+k_lo] = sum_j A[i,j] * B[j, k_hi*2+k_lo]
    
    # If we define B_hi[j, k_hi] = B[j, k_hi*2] and B_lo[j, k_hi] = B[j, k_hi*2+1],
    # then C[i, k_hi*2] = sum_j A[i,j] * B_hi[j, k_hi] and C[i, k_hi*2+1] = sum_j A[i,j] * B_lo[j, k_hi]
    
    # So M_{2,2,4} is like two independent M_{2,2,2} problems sharing the same A!
    # C_hi[i, k_hi] = sum_j A[i,j] * B_hi[j, k_hi] (2x2 * 2x2)
    # C_lo[i, k_hi] = sum_j A[i,j] * B_lo[j, k_hi] (2x2 * 2x2)
    
    # Each uses Strassen with rank 7, but they share A.
    # Total rank = 7 + 7 = 14 (since A is shared, we can reuse the same U factors).
    
    # Actually, in the tensor decomposition, U is indexed by (i*m+j) = (i*2+j).
    # For both C_hi and C_lo, the U factors are the same (same A indices).
    # So we can have:
    # For r = 0..6 (Strassen for hi part):
    #   U[i*2+j, r] = U_s[i*2+j, r]
    #   V[j*4 + k_hi*2, r] = V_s[j*2+k_hi, r]  (B_hi)
    #   W[(k_hi*2)*2 + i, r] = W_s[k_hi*2+i, r]  (C_hi, output index k_hi*2)
    #   V[j*4 + k_hi*2+1, r] = 0
    #   W[(k_hi*2+1)*2 + i, r] = 0
    # For r = 7..13 (Strassen for lo part):
    #   U[i*2+j, r] = U_s[i*2+j, r-7]
    #   V[j*4 + k_hi*2+1, r] = V_s[j*2+k_hi, r-7]  (B_lo)
    #   W[(k_hi*2+1)*2 + i, r] = W_s[k_hi*2+i, r-7]  (C_lo, output index k_hi*2+1)
    #   V[j*4 + k_hi*2, r] = 0
    #   W[(k_hi*2)*2 + i, r] = 0
    
    # Let me construct M_{2,2,4} this way.
    U_224 = np.zeros((4, 14))
    V_224 = np.zeros((8, 14))
    W_224 = np.zeros((8, 14))
    
    for i in range(2):
        for j in range(2):
            for k_hi in range(2):
                # Hi part
                for r in range(7):
                    U_224[i*2+j, r] = U_s[i*2+j, r]
                    V_224[j*4 + k_hi*2, r] = V_s[j*2+k_hi, r]
                    W_224[(k_hi*2)*2 + i, r] = W_s[k_hi*2+i, r]
                # Lo part
                for r in range(7, 14):
                    U_224[i*2+j, r] = U_s[i*2+j, r-7]
                    V_224[j*4 + k_hi*2+1, r] = V_s[j*2+k_hi, r-7]
                    W_224[(k_hi*2+1)*2 + i, r] = W_s[k_hi*2+i, r-7]
    
    # Verify M_{2,2,4}
    for i in range(2):
        for j in range(2):
            for k in range(4):
                val = np.sum(U_224[i*2+j] * V_224[j*4+k] * W_224[k*2+i])
                if abs(val - 1.0) > 1e-9:
                    print(f"M_224 error at ({i},{j},{k}): {val}")
    
    # Now M_{4,4,8} = M_{2,2,2} ⊗ M_{2,2,4}
    # n=2*2=4, m=2*2=4, p=2*4=8
    # i = i1*2 + i2, j = j1*2 + j2, k = k1*4 + k2
    # where i1,j1,k1 in {0,1}, i2,j2 in {0,1}, k2 in {0,1,2,3}
    
    # a = i*m + j = (i1*2+i2)*4 + (j1*2+j2) = i1*8 + i2*4 + j1*2 + j2
    # b = j*p + k = (j1*2+j2)*8 + (k1*4+k2) = j1*16 + j2*8 + k1*4 + k2
    # c = k*n + i = (k1*4+k2)*4 + (i1*2+i2) = k1*16 + k2*4 + i1*2 + i2
    
    # Kronecker product:
    # M_{4,4,8}[a,b,c] = M_{2,2,2}[a1,b1,c1] * M_{2,2,4}[a2,b2,c2]
    # where (a1,b1,c1) corresponds to the "outer" 2x2x2 and (a2,b2,c2) to the "inner" 2x2x4.
    
    # a1 = i1*2 + j1 (for 2x2 outer)
    # a2 = i2*2 + j2 (for 2x2 inner, but inner has m=2, so a2 = i2*2+j2)
    # Combined a = a1 * 4 + a2? Let's check: a1*4 + a2 = (i1*2+j1)*4 + i2*2+j2 = i1*8+j1*4+i2*2+j2. 
    # But we need i1*8+i2*4+j1*2+j2. So the mapping is different.
    
    # The correct mapping for Kronecker product of tensors:
    # If T1 has indices (i1,j1,k1) with i1 in [0,n1), j1 in [0,m1), k1 in [0,p1)
    # and T2 has indices (i2,j2,k2) with i2 in [0,n2), j2 in [0,m2), k2 in [0,p2)
    # Then T = T1 ⊗ T2 has:
    # i = i1*n2 + i2, j = j1*m2 + j2, k = k1*p2 + k2
    # a = i*m + j = (i1*n2+i2)*(m1*m2) + (j1*m2+j2)
    #   = i1*n2*m1*m2 + i2*m1*m2 + j1*m2 + j2
    # b = j*p + k = (j1*m2+j2)*(p1*p2) + (k1*p2+k2)
    #   = j1*m2*p1*p2 + j2*p1*p2 + k1*p2 + k2
    # c = k*n + i = (k1*p2+k2)*(n1*n2) + (i1*n2+i2)
    #   = k1*p2*n1*n2 + k2*n1*n2 + i1*n2 + i2
    
    # For T1 = M_{2,2,2} (n1=2,m1=2,p1=2) and T2 = M_{2,2,4} (n2=2,m2=2,p2=4):
    # a = i1*2*2*2 + i2*2*2 + j1*2 + j2 = i1*8 + i2*4 + j1*2 + j2
    # b = j1*2*2*4 + j2*2*4 + k1*4 + k2 = j1*16 + j2*8 + k1*4 + k2
    # c = k1*4*2*2 + k2*2*2 + i1*2 + i2 = k1*16 + k2*4 + i1*2 + i2
    
    # Now, a1 for T1: a1 = i1*m1 + j1 = i1*2 + j1
    # a2 for T2: a2 = i2*m2 + j2 = i2*2 + j2
    # b1 for T1: b1 = j1*p1 + k1 = j1*2 + k1
    # b2 for T2: b2 = j2*p2 + k2 = j2*4 + k2
    # c1 for T1: c1 = k1*n1 + i1 = k1*2 + i1
    # c2 for T2: c2 = k2*n2 + i2 = k2*2 + i2
    
    # Check: a = i1*8 + i2*4 + j1*2 + j2 = (i1*2+j1)*4 + (i2*2+j2) = a1*4 + a2 ✓
    # b = j1*16 + j2*8 + k1*4 + k2 = (j1*2+k1)*8 + (j2*4+k2) = b1*8 + b2 ✓
    # c = k1*16 + k2*4 + i1*2 + i2 = (k1*2+i1)*8 + (k2*2+i2) = c1*8 + c2 ✓
    
    # So the Kronecker product is:
    # U[a, r] = U1[a1, r1] * U2[a2, r2] where r = r1*rank2 + r2
    # V[b, r] = V1[b1, r1] * V2[b2, r2]
    # W[c, r] = W1[c1, r1] * W2[c2, r2]
    
    rank_448 = 7 * 14  # 98
    U_448 = np.zeros((16, rank_448))
    V_448 = np.zeros((32, rank_448))
    W_448 = np.zeros((32, rank_448))
    
    for i1 in range(2):
        for j1 in range(2):
            for k1 in range(2):
                for i2 in range(2):
                    for j2 in range(2):
                        for k2 in range(4):
                            a1 = i1*2 + j1
                            b1 = j1*2 + k1
                            c1 = k1*2 + i1
                            a2 = i2*2 + j2
                            b2 = j2*4 + k2
                            c2 = k2*2 + i2
                            
                            a = a1*4 + a2
                            b = b1*8 + b2
                            c = c1*8 + c2
                            
                            for r1 in range(7):
                                for r2 in range(14):
                                    r = r1*14 + r2
                                    U_448[a, r] = U_s[a1, r1] * U_224[a2, r2]
                                    V_448[b, r] = V_s[b1, r1] * V_224[b2, r2]
                                    W_448[c, r] = W_s[c1, r1] * W_224[c2, r2]
    
    # Verify M_{4,4,8}
    valid = True
    for i in range(4):
        for j in range(4):
            for k in range(8):
                val = np.sum(U_448[i*4+j] * V_448[j*8+k] * W_448[k*4+i])
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    print(f"M_448 error at ({i},{j},{k}): {val}")
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        # Convert to list format
        U_list = U_448.tolist()
        V_list = V_448.tolist()
        W_list = W_448.tolist()
        # Round to nearest integer or half-integer
        U_list = [[round(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in U_list]
        V_list = [[round(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in V_list]
        W_list = [[round(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in W_list]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
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
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
