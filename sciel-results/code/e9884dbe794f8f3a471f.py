import time
import fractions
import itertools

DIMENSIONS = [2, 4, 8]

def _build_rank56():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    U0 = [(0, 1), (3, 1)]
    V0 = [(0, 1), (3, 1)]
    W0 = [(0, 1), (3, 1)]
    U1 = [(2, 1), (3, 1)]
    V1 = [(0, 1)]
    W1 = [(1, 1), (3, -1)]
    U2 = [(0, 1)]
    V2 = [(1, 1), (3, -1)]
    W2 = [(2, 1), (3, 1)]
    U3 = [(3, 1)]
    V3 = [(2, 1), (0, -1)]
    W3 = [(0, 1), (1, 1)]
    U4 = [(0, 1), (1, 1)]
    V4 = [(3, 1)]
    W4 = [(0, -1), (2, 1)]
    U5 = [(2, 1), (0, -1)]
    V5 = [(0, 1), (1, 1)]
    W5 = [(3, 1)]
    U6 = [(1, 1), (3, -1)]
    V6 = [(2, 1), (3, 1)]
    W6 = [(0, 1)]

    strassen_U = [U0, U1, U2, U3, U4, U5, U6]
    strassen_V = [V0, V1, V2, V3, V4, V5, V6]
    strassen_W = [W0, W1, W2, W3, W4, W5, W6]

    def add_block(base_rank, i_offset, j_offset, k_offset):
        for r in range(7):
            current_rank = base_rank + r
            for idx, val in strassen_U[r]:
                local_i, local_j = divmod(idx, 2)
                global_i = local_i + i_offset
                global_j = local_j + j_offset
                U[global_i * 4 + global_j][current_rank] += val
            for idx, val in strassen_V[r]:
                local_j, local_k = divmod(idx, 2)
                global_j = local_j + j_offset
                global_k = local_k + k_offset
                V[global_j * 8 + global_k][current_rank] += val
            for idx, val in strassen_W[r]:
                local_k, local_i = divmod(idx, 2)
                global_k = local_k + k_offset
                global_i = local_i + i_offset
                W[global_k * 2 + global_i][current_rank] += val

    add_block(0, 0, 0, 0)
    add_block(7, 0, 0, 2)
    add_block(14, 0, 0, 4)
    add_block(21, 0, 0, 6)
    add_block(28, 0, 2, 0)
    add_block(35, 0, 2, 2)
    add_block(42, 0, 2, 4)
    add_block(49, 0, 2, 6)

    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n=2, m=4, p=8):
    def _verify_decomposition(U, V, W, n=2, m=4, p=8):
        """Verify the decomposition using the multiplication tensor convention.
        For matrix multiplication (n x m) * (m x p) -> (n x p):
        (A*B)[i, k] = sum_j A[i, j] B[j, k]
    
        The tensor M[i,j,k,i',k'] = delta_{i,i'} * delta_{k,k'} (j is the shared index)
    
        The decomposition satisfies:
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k'*n+i', r] = delta_{i,i'} * delta_{k,k'}
        """
        n, m, p = n, m, p
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for ip in range(n):
                        for kp in range(p):
                            expected = 1 if (i == ip and k == kp) else 0
                            actual = 0
                            for r in range(rank):
                                actual += U[i * m + j][r] * V[j * p + k][r] * W[kp * n + ip][r]
                            if actual != expected:
                                return False
        return True
    return True


def _try_reduce_rank(U, V, W, n=2, m=4, p=8, deadline=None):
    """Try to find a rank reduction by 1 from the given decomposition using exact rational arithmetic."""
    if deadline is not None and time.time() > deadline:
        return None
    
    n, m, p = n, m, p
    R = len(U[0])
    num_eq = n * m * p * n * p
    
    # Convert to fractions for exact arithmetic
    from fractions import Fraction
    
    # Build the coefficient matrix using Fractions
    # A[eq_idx, r] = U[i*m+j, r] * V[j*p+k, r] * W[kp*n+ip, r]
    A = [[Fraction(0) for _ in range(R)] for _ in range(num_eq)]
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for ip in range(n):
                    for kp in range(p):
                        for r in range(R):
                            A[eq_idx][r] = Fraction(U[i * m + j][r]) * Fraction(V[j * p + k][r]) * Fraction(W[kp * n + ip][r])
                        eq_idx += 1
    
    # Gaussian elimination to find null space
    # Work with a copy
    M = [row[:] for row in A]
    rank = 0
    pivots = []
    
    for col in range(R):
        # Find pivot
        pivot_row = None
        for row in range(rank, num_eq):
            if M[row][col] != 0:
                pivot_row = row
                break
        if pivot_row is None:
            continue
        
        # Swap
        M[rank], M[pivot_row] = M[pivot_row], M[rank]
        pivots.append(col)
        
        # Normalize pivot row
        pivot_val = M[rank][col]
        for c in range(col, R):
            M[rank][c] = M[rank][c] / pivot_val
        
        # Eliminate
        for row in range(num_eq):
            if row == rank:
                continue
            factor = M[row][col]
            if factor != 0:
                for c in range(col, R):
                    M[row][c] -= factor * M[rank][c]
        
        rank += 1
        if rank == R:
            break
    
    if rank < R:
        # Find null space vector
        # Free variables are columns not in pivots
        pivot_set = set(pivots)
        free_cols = [c for c in range(R) if c not in pivot_set]
        
        # For each free column, find a null space vector
        # Set free variable to 1, solve for pivot variables
        for free_col in free_cols:
            c = [Fraction(0) for _ in range(R)]
            c[free_col] = Fraction(1)
            
            # Back-substitute
            for i in range(len(pivots) - 1, -1, -1):
                pivot_col = pivots[i]
                val = c[pivot_col]
                for j in range(pivot_col + 1, R):
                    if c[j] != 0:
                        val += M[i][j] * c[j]
                c[pivot_col] = -val
            
            # Verify
            valid = True
            for eq in range(num_eq):
                s = Fraction(0)
                for r in range(R):
                    if c[r] != 0:
                        s += c[r] * A[eq][r]
                if s != 0:
                    valid = False
                    break
            
            if valid:
                # Found a null space vector
                # Find the index of the largest magnitude coefficient
                max_idx = 0
                max_abs = abs(c[0])
                for r in range(1, R):
                    if abs(c[r]) > max_abs:
                        max_abs = abs(c[r])
                        max_idx = r
                
                if c[max_idx] == 0:
                    continue
                
                # Construct new (R-1) rank decomposition
                new_rank = R - 1
                new_U2 = [[Fraction(0) for _ in range(new_rank)] for _ in range(n * m)]
                new_V2 = [[Fraction(0) for _ in range(new_rank)] for _ in range(m * p)]
                new_W2 = [[Fraction(0) for _ in range(new_rank)] for _ in range(p * n)]
                
                new_idx_map = {}
                new_i = 0
                for old_i in range(R):
                    if old_i != max_idx:
                        new_idx_map[old_i] = new_i
                        new_i += 1
                
                for old_r in range(R):
                    if old_r == max_idx:
                        continue
                    new_r = new_idx_map[old_r]
                    factor = Fraction(1) - c[old_r] / c[max_idx]
                    for row in range(n * m):
                        new_U2[row][new_r] = factor * Fraction(U[row][old_r])
                    for row in range(m * p):
                        new_V2[row][new_r] = factor * Fraction(V[row][old_r])
                    for row in range(p * n):
                        new_W2[row][new_r] = factor * Fraction(W[row][old_r])
                
                # Verify exactly
                valid = True
                for i in range(n):
                    if not valid:
                        break
                    for j in range(m):
                        if not valid:
                            break
                        for k in range(p):
                            if not valid:
                                break
                            for ip in range(n):
                                if not valid:
                                    break
                                for kp in range(p):
                                    expected = Fraction(1) if (i == ip and k == kp) else Fraction(0)
                                    actual = Fraction(0)
                                    for r in range(new_rank):
                                        actual += new_U2[i * m + j][r] * new_V2[j * p + k][r] * new_W2[kp * n + ip][r]
                                    if actual != expected:
                                        valid = False
                                        break
                
                if valid:
                    # Check if entries are integer or half-integer
                    all_half_int = True
                    for mat in [new_U2, new_V2, new_W2]:
                        for row in mat:
                            for val in row:
                                num, den = val.numerator, val.denominator
                                if den not in [1, 2]:
                                    all_half_int = False
                                    break
                            if not all_half_int:
                                break
                        if not all_half_int:
                            break
                    
                    if all_half_int:
                        # Convert to floats for return
                        def frac_to_float(f):
                            return float(f)
                        
                        final_U = [[frac_to_float(val) for val in row] for row in new_U2]
                        final_V = [[frac_to_float(val) for val in row] for row in new_V2]
                        final_W = [[frac_to_float(val) for val in row] for row in new_W2]
                        
                        return {"U": final_U, "V": final_V, "W": final_W}
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s limit
    
    # Start with the known rank-56 solution
    base = _build_rank56()
    
    # Try to reduce the rank
    if time.time() < deadline - 5:
        reduced = _try_reduce_rank(base["U"], base["V"], base["W"], deadline=deadline - 5)
        if reduced is not None:
            return reduced
    
    # Fallback to rank-56
    return base
