import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = 2, 3, 5
    nm, mp, pn = n * m, m * p, p * n  # 6, 15, 10

    start_time = time.time()
    deadline = start_time + 110.0  # Leave safety margin before 150s kill

    def verify(U, V, W):
        r = len(U[0]) if U else 0
        if not U or not V or not W:
            return False
        if len(V[0]) != r or len(W[0]) != r:
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def strassen_28():
        """
        Build a rank-28 decomposition:
        - Use Strassen's 7 multiplications for the 2x2x2 sub-problem (rows 0,1 of A; cols 0,1 of B)
        - Use 5 scalar multiplications for the remaining cross terms involving col 2 of B and rows 0,1 of A
        Total: 7 + 5 = 12? No wait.
        Let's think carefully.
        A is 2x3, B is 3x5.
        A = [A00 A01 A02; A10 A11 A12]
        B = [B00 B01 B02 B03 B04; B10 B11 B12 B13 B14; B20 B21 B22 B23 B24]
        
        C = AB is 2x5.
        C00 = A00*B00 + A01*B10 + A02*B20
        C01 = A00*B01 + A01*B11 + A02*B21
        C02 = A00*B02 + A01*B12 + A02*B22
        C03 = A00*B03 + A01*B13 + A02*B23
        C04 = A00*B04 + A01*B14 + A02*B24
        ... similar for row 1.
        
        We can view this as:
        C = [A_left * B_full] + [A_right * B_full] where A_left is 2x2 (cols 0,1), A_right is 2x1 (col 2).
        
        For the 2x2 * 3x5 part:
        Let A2x2 = A[:, 0:2], B3x5 = B.
        We can use Strassen for the 2x2 * 2x2 sub-problem (using B[:, 0:2]), and then add the terms involving B[:, 2].
        
        Strassen for 2x2 * 2x2 (A2x2 * B2x2 where B2x2 = B[:, 0:2]):
        M1 = (A00 + A11) * (B00 + B11)
        M2 = (A10 + A11) * B00
        M3 = A00 * (B11 - B00)
        M4 = A11 * (B00 - B10)
        M5 = (A00 + A01) * B10
        M6 = (A10 - A00) * (B00 + B01)
        M7 = (A01 - A11) * (B10 + B11)
        
        These give us 7 terms for C[:, 0:2].
        
        For the remaining columns of B (col 2), we need:
        C02 = A00*B02 + A01*B12 + A02*B22
        C03 = A00*B03 + A01*B13 + A02*B23
        C04 = A00*B04 + A01*B14 + A02*B24
        And similarly for row 1.
        
        Wait, Strassen gives C for 2x2 * 2x2. The remaining part is 2x1 * 3x3? No.
        Actually, it's easier to think of it as:
        C = A * B
        We can use the standard decomposition but apply Strassen to the 2x2 block of A and the corresponding 2x5 block of B?
        
        Let's try a different approach.
        Use Strassen for the 2x2 part of A and the first 2 columns of B.
        This gives us the first 2 columns of C.
        Then we need the remaining 3 columns of C, which involve all of A and the last 3 columns of B.
        That's a 2x3 * 3x3 problem. The rank of that is at most 18.
        Total rank = 7 + 18 = 25?
        
        Or:
        Use Strassen for the 2x2 part of A and the first 2 columns of B -> 7 terms for C[:, 0:2].
        Then for C[:, 2:5], we have a 2x3 * 3x3 multiplication.
        Schoolbook for 2x3*3x3 is 18 terms.
        Total 25.
        
        Can we do better?
        Maybe use Strassen on the 2x3 * 3x5 directly?
        Pad A to 2x4 and B to 4x5? No, that increases size.
        
        Let's try the 7 + 18 = 25 construction.
        
        Steps:
        1. Strassen for A[:, 0:2] * B[:, 0:2] -> C[:, 0:2] (7 terms)
        2. Schoolbook for A[:, 0:2:5] * B[:, 2:5] -> C[:, 2:5] (18 terms)
        
        Wait, C[:, 2:5] depends on all of A, not just A[:, 0:2].
        So step 2 should be: A (2x3) * B[:, 2:5] (3x3) -> C[:, 2:5] (2x3).
        Schoolbook rank for 2x3*3x3 is 18.
        
        Total rank = 7 + 18 = 25.
        
        Let's implement this.
        """
        U_cols = []
        V_cols = []
        W_cols = []
        
        # Part 1: Strassen for A[:, 0:2] * B[:, 0:2] -> C[:, 0:2]
        # A indices: i in {0,1}, j in {0,1}
        # B indices: j in {0,1}, k in {0,1}
        # C indices: i in {0,1}, k in {0,1}
        
        # M1 = (A00 + A11) * (B00 + B11)
        # U: A00 -> +1, A11 -> +1
        # V: B00 -> +1, B11 -> +1
        # W: C00 -> +1, C11 -> +1
        u1 = [1, 0, 0, 1, 0, 0]  # i=0,j=0:1; i=1,j=1:1
        v1 = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # j=0,k=0:1; j=1,k=1:1
        w1 = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0]  # k=0,i=0:1; k=1,i=1:1
        U_cols.append(u1); V_cols.append(v1); W_cols.append(w1)
        
        # M2 = (A10 + A11) * B00
        # U: A10 -> +1, A11 -> +1
        # V: B00 -> +1
        # W: C10 -> +1
        u2 = [0, 0, 0, 1, 1, 0]
        v2 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        w2 = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  # k=0, i=1
        U_cols.append(u2); V_cols.append(v2); W_cols.append(w2)
        
        # M3 = A00 * (B11 - B00)
        # U: A00 -> +1
        # V: B11 -> +1, B00 -> -1
        # W: C01 -> +1
        u3 = [1, 0, 0, 0, 0, 0]
        v3 = [0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0]  # j=1,k=1:1; j=0,k=0:-1
        w3 = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  # k=1, i=0
        U_cols.append(u3); V_cols.append(v3); W_cols.append(w3)
        
        # M4 = A11 * (B00 - B10)
        # U: A11 -> +1
        # V: B00 -> +1, B10 -> -1
        # W: C10 -> +1
        u4 = [0, 0, 0, 0, 1, 0]
        v4 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # j=0,k=0:1
        # Wait, B10 is j=1,k=0
        v4 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # j=0,k=0
        # Need to subtract B10
        v4 = [1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0]  # j=0,k=0:1; j=1,k=0:-1
        w4 = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  # k=0, i=1
        U_cols.append(u4); V_cols.append(v4); W_cols.append(w4)
        
        # M5 = (A00 + A01) * B10
        # U: A00 -> +1, A01 -> +1
        # V: B10 -> +1
        # W: C01 -> +1
        u5 = [1, 1, 0, 0, 0, 0]
        v5 = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  # j=1,k=0
        w5 = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]  # k=1, i=0
        U_cols.append(u5); V_cols.append(v5); W_cols.append(w5)
        
        # M6 = (A10 - A00) * (B00 + B01)
        # U: A10 -> +1, A00 -> -1
        # V: B00 -> +1, B01 -> +1
        # W: C10 -> +1, C11 -> +1
        u6 = [-1, 0, 0, 1, 0, 0]  # A00:-1, A10:+1
        v6 = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # j=0,k=0:1; j=0,k=1:1
        w6 = [1, 0, 0, 0, 0, 0, 0, 1, 0, 0]  # k=0,i=1:1; k=1,i=1:1
        U_cols.append(u6); V_cols.append(v6); W_cols.append(w6)
        
        # M7 = (A01 - A11) * (B10 + B11)
        # U: A01 -> +1, A11 -> -1
        # V: B10 -> +1, B11 -> +1
        # W: C01 -> +1, C00 -> -1? 
        # Strassen: C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # So M7 contributes to C00 with +1 and C01 with ? 
        # Actually, M7 = (A01 - A11)(B10 + B11)
        # In standard Strassen, M7 is used in C00 and C01?
        # Let's check:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # So M7 only appears in C00.
        # Wait, is that right?
        # M7 = (A01 - A11)(B10 + B11)
        # This expands to A01*B10 + A01*B11 - A11*B10 - A11*B11.
        # C00 = A00*B00 + A01*B10.
        # C01 = A00*B01 + A01*B11.
        # C10 = A10*B00 + A11*B10.
        # C11 = A10*B01 + A11*B11.
        # Let's verify the Strassen formulas:
        # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
        # M4 = A11(B00-B10) = A11B00 - A11B10
        # M5 = (A00+A01)B10 = A00B10 + A01B10
        # M7 = (A01-A11)(B10+B11) = A01B10 + A01B11 - A11B10 - A11B11
        # C00 = M1 + M4 - M5 + M7
        # = (A00B00 + A00B11 + A11B00 + A11B11) + (A11B00 - A11B10) - (A00B10 + A01B10) + (A01B10 + A01B11 - A11B10 - A11B11)
        # = A00B00 + A00B11 + 2A11B00 + A11B11 - A11B10 - A00B10 + A01B10 + A01B11 - A11B10 - A11B11
        # = A00B00 + A00B11 + 2A11B00 - 2A11B10 - A00B10 + A01B10 + A01B11
        # This does not look like C00 = A00B00 + A01B10.
        
        # I think I mixed up the Strassen formulas.
        # Standard Strassen:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B11 - B00)
        # M4 = A11 * (B00 - B10)
        # M5 = (A00 + A01) * B10
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's re-verify C00:
        # M1 = A00B00 + A00B11 + A11B00 + A11B11
        # M4 = A11B00 - A11B10
        # M5 = A00B10 + A01B10
        # M7 = A01B10 + A01B11 - A11B10 - A11B11
        
        # C00 = M1 + M4 - M5 + M7
        # = A00B00 + A00B11 + A11B00 + A11B11 + A11B00 - A11B10 - A00B10 - A01B10 + A01B10 + A01B11 - A11B10 - A11B11
        # = A00B00 + A00B11 + 2A11B00 - 2A11B10 - A00B10 + A01B11
        
        # This is not A00B00 + A01B10.
        
        # I think the standard Strassen is for C = AB where A is 2x2, B is 2x2.
        # C00 = A00B00 + A01B10
        # C01 = A00B01 + A01B11
        # C10 = A10B00 + A11B10
        # C11 = A10B01 + A11B11
        
        # The formulas I used are correct for this.
        # Let's check C00 again.
        # Maybe I made an arithmetic error.
        # M1 = A00B00 + A00B11 + A11B00 + A11B11
        # M4 = A11B00 - A11B10
        # M5 = A00B10 + A01B10
        # M7 = A01B10 + A01B11 - A11B10 - A11B11
        
        # Sum:
        # A00B00: 1
        # A00B11: 1
        # A11B00: 1 + 1 = 2
        # A11B11: 1 - 1 = 0
        # A11B10: -1 - 1 = -2
        # A00B10: -1
        # A01B10: -1 + 1 = 0
        # A01B11: 1
        
        # So C00 = A00B00 + A00B11 + 2A11B00 - 2A11B10 - A00B10 + A01B11
        # This is not C00.
        
        # I think the issue is that the standard Strassen formulas assume a different indexing or I am misremembering them.
        # Let's use a known correct set of Strassen terms.
        # Alternatively, since implementing Strassen correctly is error-prone, let's just use schoolbook for the whole thing and see if we can find a better decomposition via search.
        # But the inspiration said rank 28 is possible.
        # Let's try to construct rank 28 by using Strassen for 2x2x2 and then adding 5 terms for the remaining part.
        # Actually, let's just return the schoolbook solution to be safe, since the previous program scored 1.0.
        # The goal is to improve, but if I can't verify the new solution, I should stick to a known valid one.
        # However, the task is to find a rank-minimal decomposition.
        # Let's try to implement a simple search for rank 28.
        
        return None  # Disable Strassen for now to avoid errors

    # Since Strassen is complex to implement correctly in this context without extensive verification,
    # and the previous program scored 1.0 with schoolbook, let's try to improve the search strategy.
    # The current program uses random masks and least squares.
    # Let's try a more structured search.
    
    def try_rank_structured(rank):
        if rank < 20:
            return None
        
        max_restarts = 20
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
                
            U = np.zeros((6, rank), dtype=float)
            V = np.zeros((15, rank), dtype=float)
            
            # Sparse random initialization
            density_u = 0.4
            density_v = 0.5
            
            mask_u = np.random.rand(6, rank) < density_u
            mask_v = np.random.rand(15, rank) < density_v
            
            U_vals = np.random.choice([-1, 1], size=(6, rank))
            V_vals = np.random.choice([-1, 1], size=(15, rank))
            
            U[mask_u] = U_vals[mask_u]
            V[mask_v] = V_vals[mask_v]
            
            # Build linear system for W
            A_mat = np.zeros((30, rank * 10))
            b = np.ones(30)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_row = U[i*m+j, :]
                        v_row = V[j*p+k, :]
                        coeffs = u_row * v_row
                        start_idx = (k*n+i) * rank
                        A_mat[idx, start_idx:start_idx+rank] = coeffs
                        idx += 1
            
            try:
                x, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b, rcond=None)
            except:
                continue
                
            W_sol = x.reshape(10, rank)
            
            # Check if W is close to integers or half-integers
            W_int = np.round(W_sol)
            if np.allclose(W_int, W_sol, atol=1e-4):
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W_int.tolist()
                if verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
                    
            W_half = np.round(W_sol * 2) / 2
            if np.allclose(W_half, W_sol, atol=1e-4):
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W_half.tolist()
                if verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
            
        return None

    # Try ranks from low to high
    for r in range(20, 30):
        if time.time() > deadline:
            break
        res = try_rank_structured(r)
        if res:
            return res

    return schoolbook()
