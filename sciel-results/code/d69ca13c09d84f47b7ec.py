"""Program for rank-minimal tensor decomposition of 2x3x9 matrix multiplication."""
import random
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank as fallback
    schoolbook_rank = n * m * p

    def verify(U, V, W, n, m, p):
        """Check if U, V, W exactly decompose the tensor."""
        rank = len(U[0]) if U else 0
        if rank == 0:
            return False
        # Check all active entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    d = k * n + i
                    s = 0
                    for c in range(rank):
                        s += U[a][c] * V[b][c] * W[d][c]
                    if s != 1:
                        return False
        # Check all inactive entries (full check is 2916, which is fast)
        for a in range(n * m):
            for b in range(m * p):
                for d in range(p * n):
                    # Check if this is an active entry
                    i1 = a // m
                    j1 = a % m
                    j2 = b // p
                    k2 = b % p
                    k1 = d // n
                    i2 = d % n
                    if i1 == i2 and j1 == j2 and k1 == k2:
                        continue  # active, already checked
                    s = 0
                    for c in range(rank):
                        s += U[a][c] * V[b][c] * W[d][c]
                    if s != 0:
                        return False
        return True

    # Try structured decompositions using "Fix U, V and solve for W"
    best_rank = schoolbook_rank
    best_result = None

    random.seed(42)
    # Try ranks from a reasonable lower bound to schoolbook_rank
    # Lower bound estimate: The tensor has 54 non-zero entries.
    # Each term in the sum covers 1 active entry.
    # But we can have cancellations.
    # Let's try from 20 to 54.
    for r in range(20, schoolbook_rank + 1):
        # Try a few random U, V pairs
        for attempt in range(100):
            # Generate random U, V with integer entries in {-1, 0, 1}
            U = [[random.randint(-1, 1) for _ in range(r)] for _ in range(n * m)]
            V = [[random.randint(-1, 1) for _ in range(r)] for _ in range(m * p)]
            
            # Convert to numpy for linear algebra
            U_np = np.array(U)
            V_np = np.array(V)
            
            W_np = np.zeros((p * n, r))
            success = True
            
            for d in range(p * n):
                # Construct M_d and t_d
                # M_d has rows for each (a, b)
                # t_d has values T[a, b, d]
                M_d_rows = []
                t_d_vals = []
                
                for a in range(n * m):
                    for b in range(m * p):
                        # Calculate M[a, b] = U[a] * V[b] (element-wise)
                        M_ab = U_np[a] * V_np[b]
                        M_d_rows.append(M_ab)
                        
                        # Calculate T[a, b, d]
                        i1 = a // m
                        j1 = a % m
                        j2 = b // p
                        k2 = b % p
                        k1 = d // n
                        i2 = d % n
                        
                        if i1 == i2 and j1 == j2 and k1 == k2:
                            t_d_vals.append(1)
                        else:
                            t_d_vals.append(0)
                
                M_d = np.array(M_d_rows)
                t_d = np.array(t_d_vals)
                
                # Solve M_d W_d = t_d
                # Use least squares, but check if exact solution exists
                # Since M_d is 162 x r, and r < 162, it's overdetermined.
                # We check if the residual is zero.
                try:
                    W_d, residuals, rank_d, sv = np.linalg.lstsq(M_d, t_d, rcond=None)
                    # Check if the solution is exact
                    if np.allclose(M_d @ W_d, t_d, atol=1e-9):
                        # Check if W_d is close to integers (or half-integers?)
                        # The problem allows integer/half-integer.
                        # Let's check if it's close to integers first.
                        W_d_int = np.round(W_d)
                        if np.allclose(W_d, W_d_int, atol=1e-9):
                            W_np[d] = W_d_int
                        else:
                            # Try half-integers?
                            W_d_half = np.round(2 * W_d) / 2
                            if np.allclose(W_d, W_d_half, atol=1e-9):
                                W_np[d] = W_d_half
                            else:
                                # If not integer/half-integer, we might still use it if we verify later?
                                # No, the contract says integer/half-integer entries.
                                # So we must reject if not close to such values.
                                success = False
                                break
                    else:
                        success = False
                        break
                except Exception:
                    success = False
                    break
            
            if success:
                # Convert W_np to list of lists
                W = [[int(x) if x == int(x) else float(x) for x in row] for row in W_np]
                # Ensure types are consistent (int or float)
                # The contract says integer/half-integer.
                # Let's verify the full tensor with these exact values.
                if verify(U, V, W, n, m, p):
                    best_rank = r
                    best_result = {"U": U, "V": V, "W": W}
                    break
        
        if best_result is not None:
            break

    if best_result is not None:
        return best_result

    # Fallback to schoolbook
    rank = schoolbook_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
