"""Matrix multiplication 3x3 by 3x7 - rank minimization."""
DIMENSIONS = [3, 3, 7]

import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 125  # 25s safety margin before 150s limit
    
    # Schoolbook decomposition (rank = n*m*p = 63)
    rank_schoolbook = n * m * p
    U_schoolbook = np.zeros((n * m, rank_schoolbook))
    V_schoolbook = np.zeros((m * p, rank_schoolbook))
    W_schoolbook = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j, col] = 1
                V_schoolbook[j * p + k, col] = 1
                W_schoolbook[k * n + i, col] = 1
                col += 1
    
    best_U = U_schoolbook.tolist()
    best_V = V_schoolbook.tolist()
    best_W = W_schoolbook.tolist()
    best_rank = rank_schoolbook
    
    def is_half_integer(x, tol=1e-9):
        """Check if all entries are integer or half-integer."""
        return np.allclose(2 * x, np.round(2 * x), atol=tol)
    
    def snap_to_half_integer(x, tol=1e-8):
        """Snap values to nearest half-integer if close enough."""
        snapped = np.round(2 * x) / 2.0
        if np.allclose(x, snapped, atol=tol):
            return snapped
        return None
    
    def check_decomposition(U, V, W, n, m, p, tol=1e-9):
        """Verify the tensor decomposition is exact."""
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        val = np.dot(W[k * n + i, :], U[i * m + j1, :] * V[j2 * p + k, :])
                        expected = 1.0 if j1 == j2 else 0.0
                        if not np.isclose(val, expected, atol=tol):
                            return False
        return True
    
    def solve_for_W(U, V, n, m, p, r):
        """Solve for W given U and V. Returns None if no valid solution."""
        W_solution = np.zeros((p * n, r))
        
        for i in range(n):
            for k in range(p):
                A_ik = np.zeros((m * m, r))
                b_ik = np.zeros(m * m)
                idx = 0
                for j1 in range(m):
                    for j2 in range(m):
                        A_ik[idx, :] = U[i * m + j1, :] * V[j2 * p + k, :]
                        b_ik[idx] = 1.0 if j1 == j2 else 0.0
                        idx += 1
                
                try:
                    W_ik, residuals, rank_ik, sv = np.linalg.lstsq(A_ik, b_ik, rcond=1e-10)
                    W_solution[k * n + i, :] = W_ik
                except np.linalg.LinAlgError:
                    return None
        
        # Verify exactness
        for i in range(n):
            for k in range(p):
                A_ik = np.zeros((m * m, r))
                b_ik = np.zeros(m * m)
                idx = 0
                for j1 in range(m):
                    for j2 in range(m):
                        A_ik[idx, :] = U[i * m + j1, :] * V[j2 * p + k, :]
                        b_ik[idx] = 1.0 if j1 == j2 else 0.0
                        idx += 1
                residual = np.linalg.norm(A_ik @ W_solution[k * n + i, :] - b_ik)
                if residual > 1e-8:
                    return None
        
        return W_solution
    
    def try_candidate(U, V, n, m, p, r):
        """Try a candidate (U, V) pair: solve for W, check constraints, return result or None."""
        W_solution = solve_for_W(U, V, n, m, p, r)
        if W_solution is None:
            return None
        
        # Try to snap to half-integers
        W_snapped = snap_to_half_integer(W_solution, tol=1e-6)
        if W_snapped is not None:
            W_solution = W_snapped
        
        # Check half-integer constraint
        if not is_half_integer(W_solution, tol=1e-6):
            # Try snapping with looser tolerance
            W_snapped = snap_to_half_integer(W_solution, tol=1e-4)
            if W_snapped is not None:
                W_solution = W_snapped
                if not is_half_integer(W_solution, tol=1e-3):
                    return None
            else:
                return None
        
        # Also check U and V are half-integers (they should be by construction)
        if not is_half_integer(U, tol=1e-6):
            U_snapped = snap_to_half_integer(U, tol=1e-4)
            if U_snapped is None or not is_half_integer(U_snapped, tol=1e-3):
                return None
            U = U_snapped
        if not is_half_integer(V, tol=1e-6):
            V_snapped = snap_to_half_integer(V, tol=1e-4)
            if V_snapped is None or not is_half_integer(V_snapped, tol=1e-3):
                return None
            V = V_snapped
        
        # Final verification
        if check_decomposition(U, V, W_solution, n, m, p, tol=1e-6):
            return (U, V, W_solution)
        return None
    
    def generate_structured_candidates(n, m, p, r, attempt, rng):
        """Generate structured candidate U and V matrices."""
        strategy = attempt % 8
        
        if strategy == 0:
            # Block-diagonal structure: exploit 3x3 substructure
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            # For each output column k, we need 3 multiplications (for 3x3)
            # Try to share some across k values
            for k in range(p):
                base = k * 3 if k * 3 + 3 <= r else 0
                for i in range(n):
                    for j in range(m):
                        if base + j < r:
                            U[i * m + j, base + j] = 1.0
                        if base + j < r:
                            V[j * p + k, base + j] = 1.0
        elif strategy == 1:
            # Dense random from {-1, -0.5, 0, 0.5, 1}
            U = rng.choice([-1, -0.5, 0, 0.5, 1], size=(n * m, r))
            V = rng.choice([-1, -0.5, 0, 0.5, 1], size=(m * p, r))
        elif strategy == 2:
            # Sparse: mostly zeros with some ±1
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            for _ in range(int(n * m * r * 0.4)):
                U[rng.randint(n * m), rng.randint(r)] = rng.choice([-1, 1])
            for _ in range(int(m * p * r * 0.4)):
                V[rng.randint(m * p), rng.randint(r)] = rng.choice([-1, 1])
        elif strategy == 3:
            # Binary ±1
            U = rng.choice([-1, 1], size=(n * m, r))
            V = rng.choice([-1, 1], size=(m * p, r))
        elif strategy == 4:
            # Half-integer sparse
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            for _ in range(int(n * m * r * 0.5)):
                U[rng.randint(n * m), rng.randint(r)] = rng.choice([-0.5, 0.5])
            for _ in range(int(m * p * r * 0.5)):
                V[rng.randint(m * p), rng.randint(r)] = rng.choice([-0.5, 0.5])
        elif strategy == 5:
            # Exploit symmetry: group k indices
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            # Use fewer columns by sharing across k
            cols_per_group = max(1, r // p)
            for k in range(p):
                group_start = (k * r) // p
                group_end = ((k + 1) * r) // p
                for i in range(n):
                    for j in range(m):
                        col = group_start + j
                        if col < r:
                            U[i * m + j, col] = 1.0
                            V[j * p + k, col] = 1.0
        elif strategy == 6:
            # Random with structure: rows correlated
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            for c in range(r):
                # Pick a random (i,j) pattern for U
                u_vals = rng.choice([0, 0.5, 1], size=n * m)
                U[:, c] = u_vals
                v_vals = rng.choice([0, 0.5, 1], size=m * p)
                V[:, c] = v_vals
        else:
            # Mixed: some structured, some random
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            half_r = r // 2
            for c in range(half_r):
                i = c % n
                j = c // n % m
                k = c % p
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
            for c in range(half_r, r):
                U[:, c] = rng.choice([-0.5, 0, 0.5], size=n * m)
                V[:, c] = rng.choice([-0.5, 0, 0.5], size=m * p)
        
        return U, V
    
    # Try ranks from best_rank-1 down to a reasonable minimum
    min_rank_to_try = max(18, best_rank - 45)
    
    for r in range(best_rank - 1, min_rank_to_try - 1, -1):
        if time.time() > deadline:
            break
        
        # More attempts for lower ranks where solutions are rarer
        num_attempts = 80 if r < 35 else 50
        
        for attempt in range(num_attempts):
            if time.time() > deadline:
                break
            
            rng = np.random.RandomState(42 + r * 1000 + attempt)
            U, V = generate_structured_candidates(n, m, p, r, attempt, rng)
            
            result = try_candidate(U, V, n, m, p, r)
            if result is not None:
                U_sol, V_sol, W_sol = result
                best_U = U_sol.tolist()
                best_V = V_sol.tolist()
                best_W = W_sol.tolist()
                best_rank = r
                break  # Found valid decomposition for this rank, try lower
        
        if time.time() > deadline:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
