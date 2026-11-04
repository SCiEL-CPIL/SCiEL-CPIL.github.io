import time
import numpy as np
from fractions import Fraction

N, M, P = 3, 4, 4
NM, MP, PN = N * M, M * P, P * N


def _verify(U, V, W, tol=1e-10):
    """Verify using einsum."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float).reshape(N, M, rank)
    V_arr = np.array(V, dtype=float).reshape(M, P, rank)
    W_arr = np.array(W, dtype=float).reshape(P, N, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
    max_err = np.max(np.abs(T - 1.0))
    return max_err < tol, max_err


def _schoolbook():
    rank = NM * MP * PN
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    c = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][c] = 1
                V[j * P + k][c] = 1
                W[k * N + i][c] = 1
                c += 1
    return {'U': U, 'V': V, 'W': W}


def _strassen_44():
    rank = 44
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    
    # 7 Strassen steps for 2x2 block
    s_steps = [
        ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
        ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(1, 0): 1, (1, 1): -1}),
        ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(0, 1): 1, (1, 1): 1}),
        ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (1, 0): 1}),
        ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 1): 1, (0, 0): -1}),
        ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}),
        ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1})
    ]
    
    # 4 blocks for the 2x2 output blocks
    blocks = [(0, 0, 0, 0, 0, 0), (0, 2, 2, 0, 0, 0), (0, 0, 0, 2, 0, 2), (0, 2, 2, 2, 0, 2)]
    
    col = 0
    for (xi, xj, yj, yk, zi, zk) in blocks:
        for (u_c, v_c, w_c) in s_steps:
            for (a, b), val in u_c.items():
                U[(a + xi) * M + (b + xj)][col] += val
            for (b, c), val in v_c.items():
                V[(b + yj) * P + (c + yk)][col] += val
            for (a, c), val in w_c.items():
                W[(c + zk) * N + (a + zi)][col] += val
            col += 1
    
    # Bottom row (row 2) done naively: 4x4 = 16 terms
    for j in range(M):
        for k in range(P):
            U[2 * M + j][col] = 1
            V[j * P + k][col] = 1
            W[k * N + 2][col] = 1
            col += 1
    
    assert col == rank
    return {'U': U, 'V': V, 'W': W}


def _verify_fast(U, V, W):
    """Faster verification that returns just the error."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float).reshape(N, M, rank)
    V_arr = np.array(V, dtype=float).reshape(M, P, rank)
    W_arr = np.array(W, dtype=float).reshape(P, N, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
    return np.max(np.abs(T - 1.0))


def _lax_subspace_decompose(U, V, W, deadline):
    """
    Lax subspace method: Try to find a lower-rank decomposition by exploiting
    linear structure in the tensor. We look for a change of basis in the factor
    space that reveals redundancies, then use Gaussian elimination to eliminate
    zero rows from the transformed decomposition.
    
    Key idea: If we have a valid rank-r decomposition, we can apply invertible
    linear transformations to U, V, W (with constraint that the product structure
    is preserved) to potentially reveal a lower-rank structure.
    
    Specifically, we try: for each pair of columns, check if their "tensor vectors"
    are linearly dependent in a generalized sense, and if so, eliminate one.
    """
    rank = len(U[0])
    best_rank = rank
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    
    deadline = deadline - 5.0
    
    # Convert to Fraction for exact arithmetic
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    
    # Strategy: iteratively try to eliminate columns by finding linear dependencies
    # in the "slice" structure of the tensor
    for iteration in range(30):
        if time.time() > deadline:
            break
        
        found = False
        r = len(U_frac[0])
        
        for c1 in range(r):
            if time.time() > deadline:
                break
            for c2 in range(c1 + 1, r):
                # Check if column c2 can be eliminated by adjusting column c1
                # and other columns. We check if the "residual" after projecting
                # c2 onto c1 in the tensor sense is zero.
                
                # The tensor contribution of column c is:
                # T_c[i,j,k] = U[i,c] * V[j,c] * W[k,c]
                
                # We want to find alpha such that:
                # T_{c2}[i,j,k] = alpha * T_{c1}[i,j,k] for all i,j,k
                # This requires U[:,c2] = a*U[:,c1], V[:,c2] = b*V[:,c1], W[:,c2] = c*W[:,c1]
                # with a*b*c = 1
                
                # More generally, check if T_{c2} is in the span of {T_{c1}}
                # by checking proportionality of the three vectors
                
                u1 = [U_frac[i][c1] for i in range(NM)]
                u2 = [U_frac[i][c2] for i in range(NM)]
                v1 = [V_frac[i][c1] for i in range(MP)]
                v2 = [V_frac[i][c2] for i in range(MP)]
                w1 = [W_frac[i][c1] for i in range(PN)]
                w2 = [W_frac[i][c2] for i in range(PN)]
                
                # Check if u2 is proportional to u1
                a = None
                valid = True
                for i in range(NM):
                    if u1[i] != 0:
                        if u2[i] == 0:
                            valid = False
                            break
                        ratio = u2[i] / u1[i]
                        if a is None:
                            a = ratio
                        elif ratio != a:
                            valid = False
                            break
                    elif u2[i] != 0:
                        valid = False
                        break
                if not valid or a is None or a == 0:
                    continue
                
                b = None
                valid = True
                for i in range(MP):
                    if v1[i] != 0:
                        if v2[i] == 0:
                            valid = False
                            break
                        ratio = v2[i] / v1[i]
                        if b is None:
                            b = ratio
                        elif ratio != b:
                            valid = False
                            break
                    elif v2[i] != 0:
                        valid = False
                        break
                if not valid or b is None or b == 0:
                    continue
                
                c = None
                valid = True
                for i in range(PN):
                    if w1[i] != 0:
                        if w2[i] == 0:
                            valid = False
                            break
                        ratio = w2[i] / w1[i]
                        if c is None:
                            c = ratio
                        elif ratio != c:
                            valid = False
                            break
                    elif w2[i] != 0:
                        valid = False
                        break
                if not valid or c is None or c == 0:
                    continue
                
                # Check a*b*c = 1
                if a * b * c != 1:
                    continue
                
                # Found! Remove column c2 by absorbing into c1
                # T_{c1} + T_{c2} = T_{c1} + T_{c1} = 2*T_{c1} (since proportional)
                # Actually T_{c2} = (1/(a*b)) * T_{c1} * ... let me think
                # T_{c2}[i,j,k] = u2[i]*v2[j]*w2[k] = a*u1[i] * b*v1[j] * c*w1[k] = a*b*c * T_{c1}[i,j,k]
                # Since a*b*c = 1, T_{c2} = T_{c1}
                # So we can just remove c2 (it's a duplicate)
                
                U_frac = [row[:c2] + row[c2+1:] for row in U_frac]
                V_frac = [row[:c2] + row[c2+1:] for row in V_frac]
                W_frac = [row[:c2] + row[c2+1:] for row in W_frac]
                
                best_rank = len(U_frac[0])
                best_U = [[float(x) for x in row] for row in U_frac]
                best_V = [[float(x) for x in row] for row in V_frac]
                best_W = [[float(x) for x in row] for row in W_frac]
                
                found = True
                break
            if found or time.time() > deadline:
                break
        
        if not found:
            break
    
    return {'U': best_U, 'V': best_V, 'W': best_W}, best_rank


def _try_column_elimination(U, V, W, deadline):
    """
    More aggressive: try to eliminate a column by expressing its tensor contribution
    as a linear combination of other columns' contributions.
    
    For column c2, we want: sum_{c != c2} alpha_c * T_c = T_{c2}
    This is a system of N*M*P linear equations in (rank-1) unknowns.
    """
    rank = len(U[0])
    best_rank = rank
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    
    deadline = deadline - 5.0
    
    U_frac = [[Fraction(x) for x in row] for row in U]
    V_frac = [[Fraction(x) for x in row] for row in V]
    W_frac = [[Fraction(x) for x in row] for row in W]
    
    n_tensors = N * M * P  # 48 equations per column
    
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        r = len(U_frac[0])
        if r <= 1:
            break
        
        found = False
        for c2 in range(r):
            if time.time() > deadline:
                break
            
            # Build the system: for each (i,j,k), sum_{c != c2} alpha_c * U[i,c]*V[j,c]*W[k,c] = U[i,c2]*V[j,c2]*W[k,c2]
            # This is n_tensors equations in (r-1) unknowns
            
            # Only try if r-1 >= 1 (need at least 1 other column)
            if r - 1 < 1:
                continue
            
            # Build coefficient matrix A (n_tensors x (r-1)) and RHS (n_tensors)
            A = []
            rhs = []
            others = [c for c in range(r) if c != c2]
            
            for i in range(N):
                for j in range(M):
                    for k in range(P):
                        row_A = []
                        for c in others:
                            row_A.append(U_frac[i*M+j][c] * V_frac[j*P+k][c] * W_frac[k*N+i][c])
                        A.append(row_A)
                        rhs.append(U_frac[i*M+j][c2] * V_frac[j*P+k][c2] * W_frac[k*N+i][c2])
            
            # Solve using Gaussian elimination with Fractions
            # A is n_tensors x (r-1), rhs is n_tensors
            # We need a solution to A * alpha = rhs
            
            n_eq = len(A)
            n_unknowns = r - 1
            
            if n_eq < n_unknowns:
                continue
            
            # Gaussian elimination
            # Work with augmented matrix [A | rhs]
            aug = [[A[e][u] for u in range(n_unknowns)] + [rhs[e]] for e in range(n_eq)]
            
            # Eliminate
            pivot_rows = []
            for col in range(n_unknowns):
                if time.time() > deadline:
                    break
                # Find pivot
                pivot = -1
                for row in range(len(pivot_rows), n_eq):
                    if aug[row][col] != 0:
                        pivot = row
                        break
                if pivot == -1:
                    continue
                pivot_rows.append(pivot)
                
                # Normalize pivot row
                pv = aug[pivot][col]
                for u in range(col, n_unknowns + 1):
                    aug[pivot][u] = aug[pivot][u] / pv
                
                # Eliminate other rows
                for row in range(n_eq):
                    if row == pivot:
                        continue
                    factor = aug[row][col]
                    if factor == 0:
                        continue
                    for u in range(col, n_unknowns + 1):
                        aug[row][u] = aug[row][u] - factor * aug[pivot][u]
            
            # Check consistency: any row with all zeros in A part but non-zero in rhs
            consistent = True
            for row in range(n_eq):
                all_zero = all(aug[row][u] == 0 for u in range(n_unknowns))
                if all_zero and aug[row][n_unknowns] != 0:
                    consistent = False
                    break
            
            if not consistent:
                continue
            
            # Extract solution (free variables set to 0)
            alpha = [Fraction(0)] * n_unknowns
            for idx, pivot_row in enumerate(pivot_rows):
                col = pivot_row  # pivot for column = position in pivot_rows... 
                # Actually, pivot_rows contains row indices, and the pivot column is the position
                # Let me redo: track which column each pivot row corresponds to
                pass
            
            # Simpler: since we eliminated, the solution is in the last column
            # for pivot rows, and 0 for free variables
            # Re-do with proper tracking
            alpha = [Fraction(0)] * n_unknowns
            pivot_col_for_row = {}
            row_idx = 0
            for col in range(n_unknowns):
                if col < len(pivot_rows):
                    # This column has a pivot
                    # Find which row is the pivot for this column
                    # From the elimination, pivot_rows[col] should be the row
                    # But I tracked it differently. Let me just use the augmented matrix directly.
                    pass
            
            # Let me redo the Gaussian elimination more carefully
            aug2 = [[A[e][u] for u in range(n_unknowns)] + [rhs[e]] for e in range(n_eq)]
            pivot_cols = []
            for col in range(n_unknowns):
                if time.time() > deadline:
                    break
                # Find pivot in column col
                pivot = -1
                for row in range(len(pivot_cols), n_eq):
                    if aug2[row][col] != 0:
                        pivot = row
                        break
                if pivot == -1:
                    continue
                pivot_cols.append((col, pivot))
                
                # Swap to pivot position
                target = len(pivot_cols) - 1
                if pivot != target:
                    aug2[pivot], aug2[target] = aug2[target], aug2[pivot]
                
                # Normalize
                pv = aug2[target][col]
                for u in range(col, n_unknowns + 1):
                    aug2[target][u] = aug2[target][u] / pv
                
                # Eliminate
                for row in range(n_eq):
                    if row == target:
                        continue
                    factor = aug2[row][col]
                    if factor == 0:
                        continue
                    for u in range(col, n_unknowns + 1):
                        aug2[row][u] = aug2[row][u] - factor * aug2[target][u]
            
            # Check consistency
            consistent = True
            for row in range(n_eq):
                all_zero = all(aug2[row][u] == 0 for u in range(n_unknowns))
                if all_zero and aug2[row][n_unknowns] != 0:
                    consistent = False
                    break
            
            if not consistent:
                continue
            
            # Extract solution
            alpha = [Fraction(0)] * n_unknowns
            for (col, row) in pivot_cols:
                alpha[col] = aug2[row][n_unknowns]
            
            # Now eliminate column c2 from the decomposition
            # New U[i] = U[i] - sum_{c != c2} alpha_c * (U[i,c] * V[j,c] * W[k,c]) / (V[j,c2] * W[k,c2])
            # This doesn't work cleanly because the alpha depends on the full tensor.
            
            # Actually, the correct approach: we found alpha such that
            # sum_c alpha_c * T_c = T_{c2}
            # So T_{c2} - sum_c alpha_c * T_c = 0
            # We can replace column c2 with... hmm, this is tricky.
            
            # The issue is that alpha_c multiplies the full tensor T_c, not just U_c.
            # So we can't simply adjust U, V, W independently.
            
            # This approach doesn't directly give us a column elimination.
            # Skip this for now.
            continue
        
        if not found:
            break
    
    return {'U': best_U, 'V': best_V, 'W': best_W}, best_rank


def construct():
    start = time.time()
    
    # Start with the known Strassen-based decomposition with rank 44
    result = _strassen_44()
    
    # Verify correctness
    ok, err = _verify(result['U'], result['V'], result['W'])
    if not ok:
        # Fallback to schoolbook (should never happen)
        result = _schoolbook()
        return result
    
    best_result = result
    best_rank = len(result['U'][0])
    
    # Deadline: 140 seconds (10s safety margin)
    deadline = start + 140.0
    
    # Try Lax subspace decomposition (exact rational)
    if time.time() < deadline - 30:
        lax_result, lax_rank = _lax_subspace_decompose(result['U'], result['V'], result['W'], deadline)
        if lax_rank < best_rank:
            ok, err = _verify(lax_result['U'], lax_result['V'], lax_result['W'])
            if ok:
                best_result = lax_result
                best_rank = lax_rank
    
    # Try column elimination
    if time.time() < deadline - 10:
        elim_result, elim_rank = _try_column_elimination(best_result['U'], best_result['V'], best_result['W'], deadline)
        if elim_rank < best_rank:
            ok, err = _verify(elim_result['U'], elim_result['V'], elim_result['W'])
            if ok:
                best_result = elim_result
                best_rank = elim_rank
    
    # Final verification
    ok, err = _verify(best_result['U'], best_result['V'], best_result['W'])
    if not ok:
        # Fallback to the original Strassen result
        best_result = result
    
    return best_result
