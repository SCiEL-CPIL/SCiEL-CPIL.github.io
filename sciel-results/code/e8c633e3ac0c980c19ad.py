import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = 2, 3, 8
    dim_U = n * m  # 6
    dim_V = m * p  # 24
    dim_W = p * n  # 16
    
    t_start = time.time()
    deadline = t_start + 130.0  # 20s safety margin
    
    # Strategy: Direct sum of four independent 2x3x2 sub-tensors
    # Split p=8 into 4 pairs: (0,1), (2,3), (4,5), (6,7)
    # Each sub-tensor is 2x3x2 (n=2, m=3, p=2)
    # Total rank = 4 * rank(2x3x2)
    
    # For 2x3x2, we can use a rank-7 decomposition (schoolbook is 12)
    # Actually, let's think about this more carefully.
    # The 2x3x2 tensor: T[i,j,k] where i in {0,1}, j in {0,1,2}, k in {0,1}
    # Schoolbook rank = 2*3*2 = 12
    # Can we do better?
    
    # Let's try to find a rank-7 or rank-8 decomposition for 2x3x2
    # using Strassen-like techniques.
    
    # For 2x3x2, we can view it as:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    
    # We can use Strassen for the 2x2 part (rows 0,1 of A and columns 0,1 of B)
    # and handle the j=2 row separately.
    
    # Strassen for 2x2x2 (using 7 multiplications):
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1])*B[0,0]
    # M3 = A[0,0]*(B[0,1]-B[1,1])
    # M4 = A[1,1]*(B[1,0]-B[0,0])
    # M5 = (A[0,0]+A[0,1])*B[1,1]
    # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    
    # C[0,0] = M1 - M4 + M6
    # C[0,1] = M3 + M5 + M7
    # C[1,0] = M2 + M4 - M6
    # C[1,1] = M1 - M3 - M5
    
    # For 2x3x2, we need to add the j=2 terms:
    # C[0,0] += A[0,2]*B[2,0]
    # C[0,1] += A[0,2]*B[2,1]
    # C[1,0] += A[1,2]*B[2,0]
    # C[1,1] += A[1,2]*B[2,1]
    
    # We can add 4 more multiplications for the j=2 part:
    # M8 = A[0,2]*B[2,0]
    # M9 = A[0,2]*B[2,1]
    # M10 = A[1,2]*B[2,0]
    # M11 = A[1,2]*B[2,1]
    
    # Total rank = 7 + 4 = 11 for 2x3x2
    # But can we do better?
    
    # Let's try to combine the j=2 terms with the Strassen terms.
    # Actually, let's just use rank 11 for now: total rank = 4*11 = 44 (same as before)
    
    # Let's try to find a better decomposition for 2x3x2.
    # What if we use rank 10? Or rank 9?
    
    # For now, let's implement the rank-11 decomposition for 2x3x2
    # and see if we can improve it.
    
    def build_2x3x2_rank11():
        """Build a rank-11 decomposition for 2x3x2 matmul."""
        rank = 11
        U = [[0]*rank for _ in range(6)]  # 2*3 = 6
        V = [[0]*rank for _ in range(6)]  # 3*2 = 6
        W = [[0]*rank for _ in range(4)]  # 2*2 = 4
        
        col = 0
        
        # Strassen terms for 2x2x2 (j=0,1)
        # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        U[0*3+0][col] = 1
        U[1*3+1][col] = 1
        V[0*2+0][col] = 1
        V[1*2+1][col] = 1
        W[0*2+0][col] = 1   # C[0,0]
        W[1*2+1][col] = 1   # C[1,1]
        col += 1
        
        # M2 = (A[1,0]+A[1,1])*B[0,0]
        U[1*3+0][col] = 1
        U[1*3+1][col] = 1
        V[0*2+0][col] = 1
        W[1*2+0][col] = 1   # C[1,0]
        col += 1
        
        # M3 = A[0,0]*(B[0,1]-B[1,1])
        U[0*3+0][col] = 1
        V[0*2+1][col] = 1
        V[1*2+1][col] = -1
        W[0*2+1][col] = 1   # C[0,1]
        W[1*2+1][col] = -1  # C[1,1]
        col += 1
        
        # M4 = A[1,1]*(B[1,0]-B[0,0])
        U[1*3+1][col] = 1
        V[1*2+0][col] = 1
        V[0*2+0][col] = -1
        W[0*2+0][col] = -1  # C[0,0]
        W[1*2+0][col] = 1   # C[1,0]
        col += 1
        
        # M5 = (A[0,0]+A[0,1])*B[1,1]
        U[0*3+0][col] = 1
        U[0*3+1][col] = 1
        V[1*2+1][col] = 1
        W[0*2+1][col] = 1   # C[0,1]
        W[1*2+1][col] = -1  # C[1,1]
        col += 1
        
        # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
        U[1*3+0][col] = 1
        U[0*3+0][col] = -1
        V[0*2+0][col] = 1
        V[0*2+1][col] = 1
        W[0*2+0][col] = -1  # C[0,0]
        W[1*2+0][col] = 1   # C[1,0]
        col += 1
        
        # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
        U[0*3+1][col] = 1
        U[1*3+1][col] = -1
        V[1*2+0][col] = 1
        V[1*2+1][col] = 1
        W[0*2+1][col] = 1   # C[0,1]
        col += 1
        
        # j=2 terms
        # M8 = A[0,2]*B[2,0]
        U[0*3+2][col] = 1
        V[2*2+0][col] = 1
        W[0*2+0][col] = 1   # C[0,0]
        col += 1
        
        # M9 = A[0,2]*B[2,1]
        U[0*3+2][col] = 1
        V[2*2+1][col] = 1
        W[0*2+1][col] = 1   # C[0,1]
        col += 1
        
        # M10 = A[1,2]*B[2,0]
        U[1*3+2][col] = 1
        V[2*2+0][col] = 1
        W[1*2+0][col] = 1   # C[1,0]
        col += 1
        
        # M11 = A[1,2]*B[2,1]
        U[1*3+2][col] = 1
        V[2*2+1][col] = 1
        W[1*2+1][col] = 1   # C[1,1]
        col += 1
        
        return U, V, W, rank
    
    # Build the full 2x3x8 decomposition using 4 copies of 2x3x2
    def build_full_rank44():
        U_sub, V_sub, W_sub, rank_sub = build_2x3x2_rank11()
        rank = 4 * rank_sub  # 44
        
        U = [[0]*rank for _ in range(dim_U)]
        V = [[0]*rank for _ in range(dim_V)]
        W = [[0]*rank for _ in range(dim_W)]
        
        for b in range(4):
            k0 = 2 * b
            k1 = 2 * b + 1
            offset = b * rank_sub
            
            for r in range(rank_sub):
                for i in range(6):
                    U[i][offset + r] = U_sub[i][r]
                
                for j in range(3):
                    for k in range(2):
                        # V_sub[j*2+k] maps to V[j*p + (k0+k)]
                        V[j * p + (k0 + k)][offset + r] = V_sub[j * 2 + k][r]
                
                for i in range(2):
                    for k in range(2):
                        # W_sub[k*2+i] maps to W[(k0+k)*n + i]
                        W[(k0 + k) * n + i][offset + r] = W_sub[k * 2 + i][r]
        
        return U, V, W, rank
    
    U, V, W, rank = build_full_rank44()
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank
    
    # Try to find a lower-rank decomposition
    # For 2x3x2, can we do better than rank 11?
    # Let's try rank 10, 9, 8, 7
    
    def build_2x3x2_rank_r(r):
        """Try to build a rank-r decomposition for 2x3x2."""
        U = [[0]*r for _ in range(6)]
        V = [[0]*r for _ in range(6)]
        W = [[0]*r for _ in range(4)]
        
        # Use the Strassen terms for j=0,1 (7 terms)
        # and try to handle j=2 with fewer terms
        
        col = 0
        
        # First 7 terms: Strassen for 2x2x2
        # M1
        U[0][col] = 1; U[4][col] = 1
        V[0][col] = 1; V[3][col] = 1
        W[0][col] = 1; W[3][col] = 1
        col += 1
        
        # M2
        U[3][col] = 1; U[4][col] = 1
        V[0][col] = 1
        W[2][col] = 1
        col += 1
        
        # M3
        U[0][col] = 1
        V[1][col] = 1; V[3][col] = -1
        W[1][col] = 1; W[3][col] = -1
        col += 1
        
        # M4
        U[4][col] = 1
        V[2][col] = 1; V[0][col] = -1
        W[0][col] = -1; W[2][col] = 1
        col += 1
        
        # M5
        U[0][col] = 1; U[1][col] = 1
        V[3][col] = 1
        W[1][col] = 1; W[3][col] = -1
        col += 1
        
        # M6
        U[3][col] = 1; U[0][col] = -1
        V[0][col] = 1; V[1][col] = 1
        W[0][col] = -1; W[2][col] = 1
        col += 1
        
        # M7
        U[1][col] = 1; U[4][col] = -1
        V[2][col] = 1; V[3][col] = 1
        W[1][col] = 1
        col += 1
        
        # Remaining terms for j=2
        remaining = r - 7
        if remaining < 0:
            return None
        
        # We need to handle:
        # C[0,0] += A[0,2]*B[2,0]  -> U[2]*V[4]*W[0]
        # C[0,1] += A[0,2]*B[2,1]  -> U[2]*V[5]*W[1]
        # C[1,0] += A[1,2]*B[2,0]  -> U[5]*V[4]*W[2]
        # C[1,1] += A[1,2]*B[2,1]  -> U[5]*V[5]*W[3]
        
        # With 4 terms, we can do it directly.
        # With 3 terms, can we combine?
        # With 2 terms?
        # With 1 term?
        
        if remaining >= 4:
            # Direct terms
            U[2][col] = 1; V[4][col] = 1; W[0][col] = 1
            col += 1
            U[2][col] = 1; V[5][col] = 1; W[1][col] = 1
            col += 1
            U[5][col] = 1; V[4][col] = 1; W[2][col] = 1
            col += 1
            U[5][col] = 1; V[5][col] = 1; W[3][col] = 1
            col += 1
            return U, V, W, r
        elif remaining == 3:
            # Try to combine two terms
            # M8 = (A[0,2]+A[1,2])*(B[2,0]+B[2,1])
            # M9 = (A[0,2]-A[1,2])*B[2,0]
            # M10 = A[0,2]*(B[2,1]-B[2,0])
            # Let's check if this works...
            # Actually, let's just return None for now
            return None
        else:
            return None
    
    # Try lower ranks
    for target_rank_sub in [10, 9, 8, 7]:
        if time.time() > deadline:
            break
        
        result = build_2x3x2_rank_r(target_rank_sub)
        if result is not None:
            U_sub, V_sub, W_sub, rank_sub = result
            # Verify the sub-decomposition
            valid = True
            for i0 in range(2):
                for j0 in range(3):
                    for k0 in range(2):
                        for i in range(2):
                            for k in range(2):
                                val = 0
                                for r in range(rank_sub):
                                    if i == i0 and k == k0:
                                        val += W_sub[k*2+i][r] * U_sub[i*3+j0][r] * V_sub[j0*2+k][r]
                                expected = 1 if (i == i0 and k == k0) else 0
                                if val != expected:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                # Build the full decomposition
                rank = 4 * rank_sub
                U = [[0]*rank for _ in range(dim_U)]
                V = [[0]*rank for _ in range(dim_V)]
                W = [[0]*rank for _ in range(dim_W)]
                
                for b in range(4):
                    k0 = 2 * b
                    k1 = 2 * b + 1
                    offset = b * rank_sub
                    
                    for r in range(rank_sub):
                        for i in range(6):
                            U[i][offset + r] = U_sub[i][r]
                        
                        for j in range(3):
                            for k in range(2):
                                V[j * p + (k0 + k)][offset + r] = V_sub[j * 2 + k][r]
                        
                        for i in range(2):
                            for k in range(2):
                                W[(k0 + k) * n + i][offset + r] = W_sub[k * 2 + i][r]
                
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                break
    
    return best_result

def _verify(U, V, W):
    """Check that the decomposition is exact."""
    n, m, p = 2, 3, 8
    R = len(U[0]) if U else 0
    
    for i0 in range(n):
        for j0 in range(m):
            for k0 in range(p):
                for i in range(n):
                    for k in range(p):
                        val = 0
                        for r in range(R):
                            if i == i0 and k == k0:
                                val += W[k * n + i][r] * U[i * m + j0][r] * V[j0 * p + k][r]
                        
                        expected_val = 1 if (i == i0 and k == k0) else 0
                        if val != expected_val:
                            return False
    return True

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
    if _verify(res["U"], res["V"], res["W"]):
        print("Verification PASSED")
    else:
        print("Verification FAILED")
