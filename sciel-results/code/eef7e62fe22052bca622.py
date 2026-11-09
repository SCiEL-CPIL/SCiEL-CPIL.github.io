"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    def time_left():
        return deadline - time.time()
    
    # Schoolbook solution (always valid)
    def make_schoolbook():
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
    
    best_result = make_schoolbook()
    best_rank = n * m * p
    
    # Verification function
    def verify(U, V, W):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    ui = U[i * m + j]
                    vj = V[j * p + k]
                    wk = W[k * n + i]
                    for r in range(rank):
                        s += ui[r] * vj[r] * wk[r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Convert to numpy for faster verification
    def verify_np(U, V, W):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = U_np.shape[1]
        # Check all (i,j,k) entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    ui = U_np[i * m + j]
                    vj = V_np[j * p + k]
                    wk = W_np[k * n + i]
                    s = np.dot(ui * vj, wk)
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Structural initialization: split middle dimension 4 = 2+2
    # A (3x4) = [A1 | A2], B (4x6) = [B1; B2]
    # C = A1*B1 + A2*B2
    # Each 3x2 * 2x6 product uses schoolbook rank 3*2*6 = 36
    # Total: 72 (same as full schoolbook, but structured)
    def make_structured_init():
        rank = 72
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        # Block 1: A1 (3x2) * B1 (2x6)
        for i in range(n):
            for j in range(2):  # middle indices 0,1
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        # Block 2: A2 (3x2) * B2 (2x6)
        for i in range(n):
            for j in range(2, 4):  # middle indices 2,3
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank by removing columns that are linearly dependent
    # or by merging columns with algebraic cancellations
    def try_reduce_rank(U, V, W, max_reduction=10):
        """Try to reduce rank by testing if any column can be removed or merged."""
        U = [row[:] for row in U]
        V = [row[:] for row in V]
        W = [row[:] for row in W]
        rank = len(U[0])
        
        for attempt in range(max_reduction):
            if time_left() < 5:
                break
            # Try removing each column one at a time
            found = False
            for c in range(rank):
                if time_left() < 5:
                    break
                # Save column
                u_col = [U[i][c] for i in range(len(U))]
                v_col = [V[i][c] for i in range(len(V))]
                w_col = [W[i][c] for i in range(len(W))]
                
                # Remove column
                for i in range(len(U)):
                    U[i].pop(c)
                for i in range(len(V)):
                    V[i].pop(c)
                for i in range(len(W)):
                    W[i].pop(c)
                
                rank -= 1
                if verify_np(U, V, W):
                    found = True
                    break
                else:
                    # Restore
                    for i in range(len(U)):
                        U[i].insert(c, u_col[i])
                    for i in range(len(V)):
                        V[i].insert(c, v_col[i])
                    for i in range(len(W)):
                        W[i].insert(c, w_col[i])
                    rank += 1
            
            if not found:
                break
        
        return U, V, W, rank
    
    # Greedy column addition: try to build up from a lower-rank base
    # by adding columns that fix violations
    def greedy_build():
        """Start from rank 0 and greedily add columns to satisfy constraints."""
        if time_left() < 10:
            return None
        
        target_constraints = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target_constraints.append((i, j, k))
        
        U = [[0] * 0 for _ in range(n * m)]
        V = [[0] * 0 for _ in range(m * p)]
        W = [[0] * 0 for _ in range(p * n)]
        
        # This approach is too slow for full greedy; skip
        return None
    
    # Try structured reduction from schoolbook
    structured = make_structured_init()
    U_red, V_red, W_red, rank_red = try_reduce_rank(
        structured["U"], structured["V"], structured["W"], max_reduction=15
    )
    if rank_red < best_rank and verify_np(U_red, V_red, W_red):
        best_rank = rank_red
        best_result = {"U": U_red, "V": V_red, "W": W_red}
    
    # Random search for lower ranks
    if time_left() > 20:
        for target_rank in range(best_rank - 1, max(0, best_rank - 30)):
            if time_left() < 15:
                break
            
            attempts = 0
            max_attempts = 200
            while time_left() > 10 and attempts < max_attempts:
                attempts += 1
                
                # Generate random integer/half-integer matrices
                scale = random.choice([0.5, 1.0, 1.5, 2.0])
                choices = [0, 1, -1, 2, -2, 3, -3]
                U = [[random.choice(choices) * scale for _ in range(target_rank)] for _ in range(n * m)]
                V = [[random.choice(choices) * scale for _ in range(target_rank)] for _ in range(m * p)]
                W = [[random.choice(choices) * scale for _ in range(target_rank)] for _ in range(p * n)]
                
                if verify_np(U, V, W):
                    best_rank = target_rank
                    best_result = {"U": U, "V": V, "W": W}
                    break
    
    # Final verification
    if not verify_np(best_result["U"], best_result["V"], best_result["W"]):
        best_result = make_schoolbook()
    
    return best_result
# EVOLVE-BLOCK-END
