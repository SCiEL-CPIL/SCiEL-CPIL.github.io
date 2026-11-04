import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    n, m, p = 3, 4, 6
    
    def build_target_tensor():
        """Build the target matrix multiplication tensor."""
        target = np.zeros((n * m, m * p, p * n), dtype=float)
        for c_r in range(n):
            for b in range(m):
                for c_c in range(p):
                    i = c_r * m + b
                    k = b * p + c_c
                    l = c_c * n + c_r
                    target[i, k, l] = 1.0
        return target
    
    def verify(U, V, W, target):
        """Verify that the decomposition reconstructs the target tensor."""
        if not U or not V or not W:
            return False
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        rank = Ua.shape[1]
        R = np.einsum('ic,kc,lc->ikl', Ua, Va, Wa)
        return np.allclose(R, target, atol=1e-9)
    
    def get_strassen_66():
        """Build the 66-rank decomposition using block structure."""
        rank = 66
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
            """Add Strassen 2x2*2x2 block (rank 7) at given offsets."""
            nonlocal col
            A00 = a_r * m + a_c
            A01 = a_r * m + a_c + 1
            A10 = (a_r + 1) * m + a_c
            A11 = (a_r + 1) * m + a_c + 1
            
            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1
            
            C00 = c_c * n + c_r
            C01 = (c_c + 1) * n + c_r
            C10 = c_c * n + (c_r + 1)
            C11 = (c_c + 1) * n + (c_r + 1)
            
            # M1 = (A00+A11)(B00+B11)
            U[A00][col] += 1; U[A11][col] += 1
            V[B00][col] += 1; V[B11][col] += 1
            W[C00][col] += 1; W[C11][col] += 1
            col += 1
            # M2 = (A10+A11)B00
            U[A10][col] += 1; U[A11][col] += 1
            V[B00][col] += 1
            W[C10][col] += 1; W[C11][col] -= 1
            col += 1
            # M3 = A00(B01-B11)
            U[A00][col] += 1
            V[B01][col] += 1; V[B11][col] -= 1
            W[C01][col] += 1; W[C11][col] += 1
            col += 1
            # M4 = A11(B10-B00)
            U[A11][col] += 1
            V[B10][col] += 1; V[B00][col] -= 1
            W[C00][col] += 1; W[C10][col] += 1
            col += 1
            # M5 = (A00+A01)B11
            U[A00][col] += 1; U[A01][col] += 1
            V[B11][col] += 1
            W[C00][col] -= 1; W[C01][col] += 1
            col += 1
            # M6 = (A10-A00)(B00+B01)
            U[A10][col] += 1; U[A00][col] -= 1
            V[B00][col] += 1; V[B01][col] += 1
            W[C11][col] += 1
            col += 1
            # M7 = (A01-A11)(B10+B11)
            U[A01][col] += 1; U[A11][col] -= 1
            V[B10][col] += 1; V[B11][col] += 1
            W[C00][col] += 1
            col += 1
        
        def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
            """Add 1x2*2x2 block (rank 4) at given offsets."""
            nonlocal col
            A0 = a_r * m + a_c
            A1 = a_r * m + a_c + 1
            
            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1
            
            C0 = c_c * n + c_r
            C1 = (c_c + 1) * n + c_r
            
            # M1: A0*B00
            U[A0][col] += 1; V[B00][col] += 1; W[C0][col] += 1
            col += 1
            # M2: A1*B10
            U[A1][col] += 1; V[B10][col] += 1; W[C0][col] += 1
            col += 1
            # M3: A0*B01
            U[A0][col] += 1; V[B01][col] += 1; W[C1][col] += 1
            col += 1
            # M4: A1*B11
            U[A1][col] += 1; V[B11][col] += 1; W[C1][col] += 1
            col += 1
        
        # Build decomposition:
        # C[0,0:2] = A[0,0:2]*B[0:2,0:2] + A[0,2:4]*B[2:4,0:2]
        add_1x2_2x2(0, 0, 0, 0, 0, 0)
        add_1x2_2x2(0, 2, 2, 0, 0, 0)
        
        # C[0,2:4] = A[0,0:2]*B[0:2,2:4] + A[0,2:4]*B[2:4,2:4]
        add_1x2_2x2(0, 0, 0, 2, 0, 2)
        add_1x2_2x2(0, 2, 2, 2, 0, 2)
        
        # C[0,4:6] = A[0,0:2]*B[0:2,4:6] + A[0,2:4]*B[2:4,4:6]
        add_1x2_2x2(0, 0, 0, 4, 0, 4)
        add_1x2_2x2(0, 2, 2, 4, 0, 4)
        
        # C[1:3,0:2] = A[1:3,0:2]*B[0:2,0:2] + A[1:3,2:4]*B[2:4,0:2]
        add_strassen(1, 0, 0, 0, 1, 0)
        add_strassen(1, 2, 2, 0, 1, 0)
        
        # C[1:3,2:4] = A[1:3,0:2]*B[0:2,2:4] + A[1:3,2:4]*B[2:4,2:4]
        add_strassen(1, 0, 0, 2, 1, 2)
        add_strassen(1, 2, 2, 2, 1, 2)
        
        # C[1:3,4:6] = A[1:3,0:2]*B[0:2,4:6] + A[1:3,2:4]*B[2:4,4:6]
        add_strassen(1, 0, 0, 4, 1, 4)
        add_strassen(1, 2, 2, 4, 1, 4)
        
        assert col == 66, f"Expected 66 columns, got {col}"
        
        return U, V, W
    
    def prune_decomposition(U, V, W, target, max_iter=50):
        """Try to prune the decomposition by removing redundant components."""
        current_U, current_V, current_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
        current_rank = len(U[0])
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            if current_rank <= 0:
                break
            
            # Build the tensor from current decomposition
            Ua = np.array(current_U, dtype=float)
            Va = np.array(current_V, dtype=float)
            Wa = np.array(current_W, dtype=float)
            R = np.einsum('ic,kc,lc->ikl', Ua, Va, Wa)
            
            # For each column, check if it's redundant
            removed_any = False
            for c in range(current_rank - 1, -1, -1):
                if time.time() > deadline - 5.0:
                    break
                
                # Try removing column c
                U_new = [row[:c] + row[c+1:] for row in current_U]
                V_new = [row[:c] + row[c+1:] for row in current_V]
                W_new = [row[:c] + row[c+1:] for row in current_W]
                
                # Quick check: is the column actually non-zero?
                if not (any(Ua[:, c]) or any(Va[:, c]) or any(Wa[:, c])):
                    current_U, current_V, current_W = U_new, V_new, W_new
                    current_rank -= 1
                    removed_any = True
                    break
                
                # Verify the new decomposition
                if len(U_new) == 0 or not U_new[0]:
                    continue
                    
                Ua_new = np.array(U_new, dtype=float)
                Va_new = np.array(V_new, dtype=float)
                Wa_new = np.array(W_new, dtype=float)
                R_new = np.einsum('ic,kc,lc->ikl', Ua_new, Va_new, Wa_new)
                
                if np.allclose(R_new, target, atol=1e-9):
                    current_U, current_V, current_W = U_new, V_new, W_new
                    current_rank -= 1
                    removed_any = True
                    break
            
            if not removed_any:
                break
        
        return current_U, current_V, current_W, current_rank
    
    # Build the target tensor
    target = build_target_tensor()
    
    # Get the 66-rank solution
    U66, V66, W66 = get_strassen_66()
    
    # Verify the initial solution
    if not verify(U66, V66, W66, target):
        # Fallback: return schoolbook if 66 doesn't verify
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for c_r in range(n):
            for b in range(m):
                for c_c in range(p):
                    i = c_r * m + b
                    k = b * p + c_c
                    l = c_c * n + c_r
                    U[i][col] = 1
                    V[k][col] = 1
                    W[l][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to prune
    U_p, V_p, W_p, r_p = prune_decomposition(U66, V66, W66, target)
    
    # Verify the pruned solution
    if verify(U_p, V_p, W_p, target) and r_p < 66:
        return {"U": U_p, "V": V_p, "W": W_p}
    
    # Return the 66-rank solution
    return {"U": U66, "V": V66, "W": W66}
