"""Combined program for matrix multiplication 2x3x7 rank minimization."""
import numpy as np
import random
import time
import sys
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    deadline = time.time() + 130  # safety margin before 150s kill

    # Schoolbook as baseline
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

    best = schoolbook()
    best_rank = n * m * p

    def compute_error(U, V, W, rank):
        """Compute the error tensor T where T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] - 1."""
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)
        U3 = U_np.reshape(n, m, rank)
        V3 = V_np.reshape(m, p, rank)
        W3 = W_np.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        return T - 1

    def verify_fast(U, V, W, rank):
        """Vectorized verification using numpy."""
        err = compute_error(U, V, W, rank)
        return np.all(err == 0)

    def count_errors(U, V, W, rank):
        """Count how many entries are wrong."""
        err = compute_error(U, V, W, rank)
        return int(np.count_nonzero(err))

    def local_search_error_guided(U, V, W, target_rank, max_iters=100):
        """Error-guided local search: modify entries that contribute most to errors."""
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)
        U3 = U_np.reshape(n, m, target_rank)
        V3 = V_np.reshape(m, p, target_rank)
        W3 = W_np.reshape(p, n, target_rank)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            err = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3) - 1
            if np.all(err == 0):
                # Convert back to list format
                U = U_np.reshape(nm, target_rank).tolist()
                V = V_np.reshape(mp, target_rank).tolist()
                W = W_np.reshape(pn, target_rank).tolist()
                return True, U, V, W
            
            # Compute gradient-like signals: for each entry, how much does it affect error?
            # dT[i,j,k]/dU[i,j,c] = V[j,k,c] * W[k,i,c]
            # dT[i,j,k]/dV[j,k,c] = U[i,j,c] * W[k,i,c]
            # dT[i,j,k]/dW[k,i,c] = U[i,j,c] * V[j,k,c]
            
            # Pick the matrix and entry with highest expected improvement
            mat_choice = random.random()
            if mat_choice < 0.33:
                # Modify U[i,j,c]
                i, j, c = random.randint(0, n-1), random.randint(0, m-1), random.randint(0, target_rank-1)
                # Compute contribution: sum over k of err[i,j,k] * V[j,k,c] * W[k,i,c]
                contrib = np.sum(err[i,j,:] * V3[j,:,c] * W3[:,i,c])
                # Try changing U[i,j,c]
                old = U3[i,j,c]
                for delta in [-2, -1, 1, 2]:
                    new_val = old + delta
                    if new_val < -3 or new_val > 3:
                        continue
                    U3[i,j,c] = new_val
                    new_err = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3) - 1
                    if np.count_nonzero(new_err) < np.count_nonzero(err):
                        err = new_err
                        break
                    U3[i,j,c] = old
            elif mat_choice < 0.66:
                # Modify V[j,k,c]
                j, k, c = random.randint(0, m-1), random.randint(0, p-1), random.randint(0, target_rank-1)
                old = V3[j,k,c]
                for delta in [-2, -1, 1, 2]:
                    new_val = old + delta
                    if new_val < -3 or new_val > 3:
                        continue
                    V3[j,k,c] = new_val
                    new_err = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3) - 1
                    if np.count_nonzero(new_err) < np.count_nonzero(err):
                        err = new_err
                        break
                    V3[j,k,c] = old
            else:
                # Modify W[k,i,c]
                k, i, c = random.randint(0, p-1), random.randint(0, n-1), random.randint(0, target_rank-1)
                old = W3[k,i,c]
                for delta in [-2, -1, 1, 2]:
                    new_val = old + delta
                    if new_val < -3 or new_val > 3:
                        continue
                    W3[k,i,c] = new_val
                    new_err = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3) - 1
                    if np.count_nonzero(new_err) < np.count_nonzero(err):
                        err = new_err
                        break
                    W3[k,i,c] = old
        
        # Final check
        final_err = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3) - 1
        if np.all(final_err == 0):
            U = U_np.reshape(nm, target_rank).tolist()
            V = V_np.reshape(mp, target_rank).tolist()
            W = W_np.reshape(pn, target_rank).tolist()
            return True, U, V, W
        return False, None, None, None

    def structured_init(target_rank):
        """Initialize with partial structure: some entries set to create near-valid patterns."""
        U = [[0] * target_rank for _ in range(nm)]
        V = [[0] * target_rank for _ in range(mp)]
        W = [[0] * target_rank for _ in range(pn)]
        
        # Start with a few schoolbook-like terms
        used_cols = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if len(used_cols) >= target_rank:
                        break
                    if random.random() < 0.3:  # Include 30% of schoolbook terms
                        col = len(used_cols)
                        if col < target_rank:
                            U[i*m+j][col] = 1
                            V[j*p+k][col] = 1
                            W[k*n+i][col] = 1
                            used_cols.add(col)
                if len(used_cols) >= target_rank:
                    break
            if len(used_cols) >= target_rank:
                break
        
        # Fill remaining columns with small random values
        for col in range(len(used_cols), target_rank):
            for row in range(nm):
                if random.random() < 0.3:
                    U[row][col] = random.choice([-1, 1])
            for row in range(mp):
                if random.random() < 0.3:
                    V[row][col] = random.choice([-1, 1])
            for row in range(pn):
                if random.random() < 0.3:
                    W[row][col] = random.choice([-1, 1])
        
        return U, V, W

    def try_rank_structured(target_rank, max_attempts=50):
        """Try structured initialization + error-guided local search."""
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                return None
            
            U, V, W = structured_init(target_rank)
            success, U, V, W = local_search_error_guided(U, V, W, target_rank, max_iters=50)
            if success:
                return {"U": U, "V": V, "W": W}
        return None

    def try_rank_pure_random(target_rank, max_attempts=30, val_range=(-1, 1)):
        """Pure random search with fast verification."""
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                return None
            U = [[random.randint(val_range[0], val_range[1]) for _ in range(target_rank)] for _ in range(nm)]
            V = [[random.randint(val_range[0], val_range[1]) for _ in range(target_rank)] for _ in range(mp)]
            W = [[random.randint(val_range[0], val_range[1]) for _ in range(target_rank)] for _ in range(pn)]
            if verify_fast(U, V, W, target_rank):
                return {"U": U, "V": V, "W": W}
        return None

    # Try ranks from 21 upward (theoretical lower bound is 21)
    for target_rank in range(21, best_rank):
        if time.time() > deadline - 15:
            break
        
        # First try structured initialization with error-guided local search
        result = try_rank_structured(target_rank, max_attempts=20)
        if result:
            best = result
            best_rank = target_rank
            continue
        
        # Try pure random with small values
        result = try_rank_pure_random(target_rank, max_attempts=15, val_range=(-1, 1))
        if result:
            best = result
            best_rank = target_rank
            continue
    
    return best
# EVOLVE-BLOCK-END
