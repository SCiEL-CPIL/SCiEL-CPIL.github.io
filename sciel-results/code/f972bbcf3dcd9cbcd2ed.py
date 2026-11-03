"""Rank-minimal exact tensor decomposition for 3x4x4 matrix multiplication."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 4, 4]
DEADLINE = time.time() + 140  # safety margin before 150s kill

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    dim_u = n * m  # 12
    dim_v = m * p  # 16
    dim_w = p * n  # 12
    
    # Build target tensor
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Strategy: Coppersmith-Winograd style recursive decomposition
    # Split 3 = 2 + 1, 4 = 2 + 2
    # Block structure:
    #   (i1,j1,k1) where i1 in {0,1}, j1 in {0,1}, k1 in {0,1} -> 2x2x2 Strassen (rank 7)
    #   (i1,j1,k2) where i1 in {0,1}, j1 in {0,1}, k2 in {2,3} -> 2x2x2 schoolbook (rank 8)
    #   (i1,j2,k1) where i1 in {0,1}, j2 in {2,3}, k1 in {0,1} -> 2x2x2 schoolbook (rank 8)
    #   (i1,j2,k2) where i1 in {0,1}, j2 in {2,3}, k2 in {2,3} -> 2x2x2 schoolbook (rank 8)
    #   (i2,*,*)  where i2=2, j in {0,1,2,3}, k in {0,1,2,3} -> 1x4x4 schoolbook (rank 16)
    # Total: 7 + 8 + 8 + 8 + 16 = 47
    # 
    # But we can do better! Let's use Strassen for all 2x2x2 blocks:
    #   (i1,j1,k1) -> Strassen (7)
    #   (i1,j1,k2) -> Strassen (7) 
    #   (i1,j2,k1) -> Strassen (7)
    #   (i1,j2,k2) -> Strassen (7)
    #   (i2,*,*) -> schoolbook (16)
    # Total: 7*4 + 16 = 44
    #
    # Even better: use Strassen for the 1x4x4 block by splitting it differently
    # Actually, let's try a different decomposition.
    #
    # Alternative: 3x4x4 = (2+1)x(2+2)x(2+2)
    # Use Strassen for all 2x2x2 sub-blocks:
    # 4 blocks of 2x2x2: 4*7 = 28
    # 1 block of 1x4x4: 16
    # Total: 44
    #
    # Can we do better on the 1x4x4? 
    # 1x4x4 matrix multiplication: C = A * B where A is 1x4, B is 4x4
    # This is just a linear combination of rows of B weighted by A.
    # Rank is exactly 4 (each output element is a dot product).
    # Wait, no. The tensor for 1x4x4 matmul has rank 4.
    # Because C[i,k] = sum_j A[0,j] * B[j,k] for i=0.
    # This is rank 4: for each k, C[0,k] = sum_j A[0,j]*B[j,k]
    # Actually the rank of the 1x4x4 matmul tensor is 4.
    # Let me verify: the tensor T[i,j,k] = 1 for i=0, all j,k.
    # In flat form: U has 1*4=4 rows, V has 4*4=16 rows, W has 4*1=4 rows.
    # T[0*m+j, j*p+k, k*n+0] = 1 for j=0..3, k=0..3
    # So 16 non-zero entries.
    # Rank is at least 4 (since we need to represent 4 independent linear forms).
    # Actually rank = 4: for each k, define u_k[j] = delta_{j,k}, v_k[j,k'] = delta_{k,k'}, w_k[k'] = 1
    # Hmm, let me think more carefully.
    # The tensor is T[j, j*p+k, k] = 1 for j,k in 0..3 (with i=0 fixed)
    # Wait, I need to be more careful with indexing.
    # For 1x4x4: n=1, m=4, p=4
    # U has shape (1*4, R) = (4, R)
    # V has shape (4*4, R) = (16, R)  
    # W has shape (4*1, R) = (4, R)
    # T[i*m+j, j*p+k, k*n+i] = T[0*4+j, j*4+k, k*1+0] = T[j, 4j+k, k] = 1 for all j,k in 0..3
    # So 16 non-zero entries.
    # The rank is 4: we can write this as sum over r=0..3 of:
    #   U[j,r] = delta_{j,r}, V[4j+k,r] = delta_{k,r}, W[k,r] = 1
    # Check: sum_r U[j,r]*V[4j+k,r]*W[k,r] = sum_r delta_{j,r} * delta_{k,r} * 1 = delta_{j,k}
    # But we need T[j,4j+k,k] = 1 for ALL j,k, not just j=k.
    # So that's wrong.
    # 
    # Let me reconsider. The 1x4x4 matmul tensor:
    # C[0,k] = sum_j A[0,j] * B[j,k]
    # In tensor form with indices (i,j,k) where i=0:
    # T[i,j,k] = 1 for all j,k (since the coefficient of A[0,j]*B[j,k] in C[0,k] is 1)
    # So the tensor has T[0,j,k] = 1 for all j,k.
    # In flat indexing: U row = i*m+j = 0*4+j = j, V row = j*p+k = j*4+k, W row = k*n+i = k*1+0 = k
    # So T[j, 4j+k, k] = 1 for j,k in 0..3.
    # 
    # To decompose: we need sum_r U[j,r] * V[4j+k,r] * W[k,r] = 1 for all j,k.
    # One way: R=4, with U[j,r] = delta_{j,r}, V[4j+k,r] = 1 if j=r else 0, W[k,r] = 1
    # Check: sum_r delta_{j,r} * (1 if j=r else 0) * 1 = 1 if there exists r=j, which is always true.
    # Wait: V[4j+k,r] should depend on both j and k. Let me redefine.
    # U[j,r] = delta_{j,r}
    # V[4j+k,r] = 1 if j=r, else 0  (i.e., V[4r+k,r] = 1 for all k, V[4j+k,r] = 0 for j≠r)
    # W[k,r] = 1 for all k,r
    # Then sum_r U[j,r]*V[4j+k,r]*W[k,r] = sum_r delta_{j,r} * (1 if j=r else 0) * 1
    # = 1 (since r=j is the only non-zero term, and it equals 1)
    # Yes! So rank 4 works for 1x4x4.
    
    # So total rank: 4*7 + 4 = 32
    # Wait, I have 4 blocks of 2x2x2 and 1 block of 1x4x4.
    # 4*7 + 4 = 32.
    
    # Actually, let me reconsider the block structure more carefully.
    # 3x4x4: i in 0..2, j in 0..3, k in 0..3
    # Split: i = i1 + i2 where i1 in {0,1}, i2 in {0,1} (but 3=2+1, so i2 is just {2})
    # j = j1 + j2 where j1 in {0,1}, j2 in {0,1} (4=2+2)
    # k = k1 + k2 where k1 in {0,1}, k2 in {0,1} (4=2+2)
    
    # Blocks:
    # 1. i1 in {0,1}, j1 in {0,1}, k1 in {0,1}: 2x2x2 -> Strassen rank 7
    # 2. i1 in {0,1}, j1 in {0,1}, k2 in {0,1}: 2x2x2 -> Strassen rank 7
    # 3. i1 in {0,1}, j2 in {0,1}, k1 in {0,1}: 2x2x2 -> Strassen rank 7
    # 4. i1 in {0,1}, j2 in {0,1}, k2 in {0,1}: 2x2x2 -> Strassen rank 7
    # 5. i2=2, j in {0,1,2,3}, k in {0,1,2,3}: 1x4x4 -> rank 4
    
    # Total: 7+7+7+7+4 = 32
    
    # Let's construct this decomposition explicitly.
    
    def strassen_decomposition():
        """Return Strassen's 7-rank decomposition for 2x2x2 matmul.
        Returns U, V, W each of shape (4, 7) for the 2x2x2 case.
        U indexed by (i,j) -> i*2+j, V indexed by (j,k) -> j*2+k, W indexed by (k,i) -> k*2+i
        """
        # Strassen's algorithm:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # For the tensor decomposition, we need:
        # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k in {0,1}
        
        # Each Strassen term r corresponds to a product of two linear forms.
        # Term r: U[:,r] * V[:,r] * W[:,r]
        
        # Let me define the 7 terms:
        # r=0: M1 = (A00+A11)(B00+B11)
        #   U: A00+A11 -> U[0,0]=1, U[3,0]=1 (i=0,j=0 and i=1,j=1)
        #   V: B00+B11 -> V[0,0]=1, V[3,0]=1 (j=0,k=0 and j=1,k=1)
        #   W: contributes to C00 (+1) and C11 (+1) -> W[0,0]=1, W[3,0]=1 (k=0,i=0 and k=1,i=1)
        
        # r=1: M2 = (A10+A11)B00
        #   U: A10+A11 -> U[2,1]=1, U[3,1]=1 (i=1,j=0 and i=1,j=1)
        #   V: B00 -> V[0,1]=1 (j=0,k=0)
        #   W: contributes to C10 (+1) and C11 (-1) -> W[0,1]=1, W[3,1]=-1 (k=0,i=0 and k=1,i=1)
        
        # r=2: M3 = A00(B10-B11)
        #   U: A00 -> U[0,2]=1 (i=0,j=0)
        #   V: B10-B11 -> V[2,2]=1, V[3,2]=-1 (j=1,k=0 and j=1,k=1)
        #   W: contributes to C01 (+1) and C11 (+1) -> W[1,2]=1, W[3,2]=1 (k=1,i=0 and k=1,i=1)
        
        # r=3: M4 = A11(B01-B10)
        #   U: A11 -> U[3,3]=1 (i=1,j=1)
        #   V: B01-B10 -> V[1,3]=1, V[2,3]=-1 (j=0,k=1 and j=1,k=0)
        #   W: contributes to C00 (+1) and C10 (+1) -> W[0,3]=1, W[2,3]=1 (k=0,i=0 and k=0,i=1)
        
        # r=4: M5 = (A00+A01)B11
        #   U: A00+A01 -> U[0,4]=1, U[1,4]=1 (i=0,j=0 and i=0,j=1)
        #   V: B11 -> V[3,4]=1 (j=1,k=1)
        #   W: contributes to C00 (-1) and C01 (+1) -> W[0,4]=-1, W[1,4]=1 (k=0,i=0 and k=1,i=0)
        
        # r=5: M6 = (A10-A00)(B00+B10)
        #   U: A10-A00 -> U[2,5]=1, U[0,5]=-1 (i=1,j=0 and i=0,j=0)
        #   V: B00+B10 -> V[0,5]=1, V[2,5]=1 (j=0,k=0 and j=1,k=0)
        #   W: contributes to C11 (+1) -> W[3,5]=1 (k=1,i=1)
        
        # r=6: M7 = (A01-A11)(B01+B11)
        #   U: A01-A11 -> U[1,6]=1, U[3,6]=-1 (i=0,j=1 and i=1,j=1)
        #   V: B01+B11 -> V[1,6]=1, V[3,6]=1 (j=0,k=1 and j=1,k=1)
        #   W: contributes to C00 (+1) -> W[0,6]=1 (k=0,i=0)
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # r=0
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[3,0] = 1
        W[0,0] = 1; W[3,0] = 1
        
        # r=1
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        W[0,1] = 1; W[3,1] = -1
        
        # r=2
        U[0,2] = 1
        V[2,2] = 1; V[3,2] = -1
        W[1,2] = 1; W[3,2] = 1
        
        # r=3
        U[3,3] = 1
        V[1,3] = 1; V[2,3] = -1
        W[0,3] = 1; W[2,3] = 1
        
        # r=4
        U[0,4] = 1; U[1,4] = 1
        V[3,4] = 1
        W[0,4] = -1; W[1,4] = 1
        
        # r=5
        U[2,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[2,5] = 1
        W[3,5] = 1
        
        # r=6
        U[1,6] = 1; U[3,6] = -1
        V[1,6] = 1; V[3,6] = 1
        W[0,6] = 1
        
        return U, V, W
    
    def verify_strassen():
        """Verify Strassen decomposition is correct for 2x2x2."""
        U, V, W = strassen_decomposition()
        # Check: sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] = 1 for all i,j,k
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    val = sum(U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] for r in range(7))
                    if abs(val - 1) > 1e-10:
                        print(f"Strassen error at i={i},j={j},k={k}: got {val}")
                        return False
        # Also check zero positions
        for a in range(4):
            for b in range(4):
                for c in range(4):
                    # Check if this is a valid (i,j,k) position
                    # a = i*2+j, b = j*2+k, c = k*2+i
                    # Find i,j,k: i = a//2, j = a%2, but need consistency
                    # a = i*2+j -> i=a//2, j=a%2
                    # b = j*2+k -> k = b%2, and j should = b//2
                    # c = k*2+i -> i should = c%2, and k should = c//2
                    i_a = a // 2
                    j_a = a % 2
                    j_b = b // 2
                    k_b = b % 2
                    k_c = c // 2
                    i_c = c % 2
                    if i_a == i_c and j_a == j_b and k_b == k_c:
                        continue  # This is a non-zero position, already checked
                    val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(7))
                    if abs(val) > 1e-10:
                        print(f"Strassen zero-position error at a={a},b={b},c={c}: got {val}")
                        return False
        return True
    
    # Verify Strassen
    if not verify_strassen():
        print("Strassen verification failed!")
        # Fall back to schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((dim_u, rank_fb))
        V_fb = np.zeros((dim_v, rank_fb))
        W_fb = np.zeros((dim_w, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # Now construct the full 3x4x4 decomposition
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1} -> Strassen
    # Block 2: i in {0,1}, j in {0,1}, k in {2,3} -> Strassen (shifted)
    # Block 3: i in {0,1}, j in {2,3}, k in {0,1} -> Strassen (shifted)
    # Block 4: i in {0,1}, j in {2,3}, k in {2,3} -> Strassen (shifted)
    # Block 5: i=2, j in {0,1,2,3}, k in {0,1,2,3} -> rank 4
    
    # For each 2x2x2 block, we need to map the local (i,j,k) indices to global indices.
    # Block 1: local (i,j,k) -> global (i, j, k) [i in 0,1; j in 0,1; k in 0,1]
    # Block 2: local (i,j,k) -> global (i, j, k+2) [i in 0,1; j in 0,1; k in 0,1]
    # Block 3: local (i,j,k) -> global (i, j+2, k) [i in 0,1; j in 0,1; k in 0,1]
    # Block 4: local (i,j,k) -> global (i, j+2, k+2) [i in 0,1; j in 0,1; k in 0,1]
    
    # For the 1x4x4 block (i=2):
    # We need rank 4 decomposition.
    # Local: i=0 (fixed), j in 0..3, k in 0..3
    # Global: i=2, j in 0..3, k in 0..3
    # U row: i*m+j = 2*4+j = 8+j
    # V row: j*p+k = j*4+k
    # W row: k*n+i = k*3+2
    
    # Rank 4 decomposition for 1x4x4:
    # For r in 0..3:
    #   U[8+j, r] = 1 if j==r else 0
    #   V[j*4+k, r] = 1 if j==r else 0
    #   W[k*3+2, r] = 1 for all k
    # Check: sum_r U[8+j,r]*V[j*4+k,r]*W[k*3+2,r] = sum_r (1 if j==r else 0)*(1 if j==r else 0)*1
    # = 1 (since r=j is the only non-zero term)
    
    # Let's build the full decomposition
    total_rank = 4 * 7 + 4  # 32
    U_full = np.zeros((dim_u, total_rank))
    V_full = np.zeros((dim_v, total_rank))
    W_full = np.zeros((dim_w, total_rank))
    
    U_str, V_str, W_str = strassen_decomposition()
    
    col = 0
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U_full[i*4+j, col] = U_str[i*2+j, r]
        for j in range(2):
            for k in range(2):
                V_full[j*4+k, col] = V_str[j*2+k, r]
        for k in range(2):
            for i in range(2):
                W_full[k*3+i, col] = W_str[k*2+i, r]
        col += 1
    
    # Block 2: i in {0,1}, j in {0,1}, k in {2,3} -> global k = k_local + 2
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U_full[i*4+j, col] = U_str[i*2+j, r]
        for j in range(2):
            for k_local in range(2):
                k_global = k_local + 2
                V_full[j*4+k_global, col] = V_str[j*2+k_local, r]
        for k_local in range(2):
            k_global = k_local + 2
            for i in range(2):
                W_full[k_global*3+i, col] = W_str[k_local*2+i, r]
        col += 1
    
    # Block 3: i in {0,1}, j in {2,3}, k in {0,1} -> global j = j_local + 2
    for r in range(7):
        for i in range(2):
            for j_local in range(2):
                j_global = j_local + 2
                U_full[i*4+j_global, col] = U_str[i*2+j_local, r]
        for j_local in range(2):
            j_global = j_local + 2
            for k in range(2):
                V_full[j_global*4+k, col] = V_str[j_local*2+k, r]
        for k in range(2):
            for i in range(2):
                W_full[k*3+i, col] = W_str[k*2+i, r]
        col += 1
    
    # Block 4: i in {0,1}, j in {2,3}, k in {2,3} -> global j = j_local + 2, k = k_local + 2
    for r in range(7):
        for i in range(2):
            for j_local in range(2):
                j_global = j_local + 2
                U_full[i*4+j_global, col] = U_str[i*2+j_local, r]
        for j_local in range(2):
            j_global = j_local + 2
            for k_local in range(2):
                k_global = k_local + 2
                V_full[j_global*4+k_global, col] = V_str[j_local*2+k_local, r]
        for k_local in range(2):
            k_global = k_local + 2
            for i in range(2):
                W_full[k_global*3+i, col] = W_str[k_local*2+i, r]
        col += 1
    
    # Block 5: i=2, j in 0..3, k in 0..3 -> rank 4
    for r in range(4):
        for j in range(4):
            U_full[2*4+j, col] = 1.0 if j == r else 0.0
        for j in range(4):
            for k in range(4):
                V_full[j*4+k, col] = 1.0 if j == r else 0.0
        for k in range(4):
            W_full[k*3+2, col] = 1.0
        col += 1
    
    # Verify the full decomposition
    def check_full(U, V, W, tol=1e-9):
        T_rec = np.zeros((dim_u, dim_v, dim_w))
        for r in range(U.shape[1]):
            T_rec += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        return np.allclose(T_rec, T, atol=tol)
    
    if check_full(U_full, V_full, W_full, tol=1e-8):
        # Great! We have a rank-32 decomposition.
        # Check if entries are integers/half-integers
        def to_half_int_array(arr):
            return np.round(arr * 2) / 2
        
        U_h = to_half_int_array(U_full)
        V_h = to_half_int_array(V_full)
        W_h = to_half_int_array(W_full)
        
        if check_full(U_h, V_h, W_h, tol=1e-8):
            U_out, V_out, W_out = U_h, V_h, W_h
        else:
            U_out, V_out, W_out = U_full, V_full, W_full
        
        return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
    else:
        print("Full decomposition verification failed!")
        # Try to debug
        T_rec = np.zeros((dim_u, dim_v, dim_w))
        for r in range(U_full.shape[1]):
            T_rec += np.einsum('a,b,c->abc', U_full[:,r], V_full[:,r], W_full[:,r])
        diff = T_rec - T
        print(f"Max diff: {np.max(np.abs(diff))}")
        print(f"Num non-zero diffs: {np.sum(np.abs(diff) > 1e-8)}")
        
        # Fall back to schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((dim_u, rank_fb))
        V_fb = np.zeros((dim_v, rank_fb))
        W_fb = np.zeros((dim_w, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
