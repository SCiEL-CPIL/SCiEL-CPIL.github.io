"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    def time_left():
        return deadline - time.time()
    
    # Precompute all (i,j,k) triples
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    num_triples = len(triples)
    
    # Schoolbook solution
    def make_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = make_schoolbook()
    best_rank = n * m * p
    
    # Fast vectorized verification
    def verify(U, V, W):
        """U, V, W are lists of lists or numpy arrays"""
        try:
            U_np = np.array(U, dtype=float)
            V_np = np.array(V, dtype=float)
            W_np = np.array(W, dtype=float)
            rank = U_np.shape[1]
            
            # For each triple (i,j,k), compute sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            # Vectorize over triples
            results = np.zeros(num_triples)
            idx = 0
            for (i, j, k) in triples:
                ui = U_np[i * m + j]
                vj = V_np[j * p + k]
                wk = W_np[k * n + i]
                results[idx] = np.sum(ui * vj * wk)
                idx += 1
            
            return np.all(np.abs(results - 1.0) < 1e-9)
        except Exception:
            return False
    
    # Try to remove a single column
    def try_remove_column(U, V, W):
        """Try removing each column, return (success, new_U, new_V, new_W, col_index)"""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = U_np.shape[1]
        
        for c in range(rank):
            U_test = np.delete(U_np, c, axis=1)
            V_test = np.delete(V_np, c, axis=1)
            W_test = np.delete(W_np, c, axis=1)
            
            if verify(U_test, V_test, W_test):
                return True, U_test.tolist(), V_test.tolist(), W_test.tolist(), c
        return False, None, None, None, -1
    
    # Try to merge two columns: find a, b such that replacing cols c1,c2 with a*col_c1 + b*col_c2 works
    # Actually, we want to remove one column by expressing it as a combination of others
    # Simpler: try setting one column to zero and see if it still works (i.e., that column was redundant)
    
    # Greedy: iteratively remove columns
    def greedy_reduce(U, V, W, max_iters=100):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        for _ in range(max_iters):
            if time_left() < 5:
                break
            
            removed = False
            for c in range(U_np.shape[1]):
                if time_left() < 3:
                    break
                
                U_test = np.delete(U_np, c, axis=1)
                V_test = np.delete(V_np, c, axis=1)
                W_test = np.delete(W_np, c, axis=1)
                
                if verify(U_test, V_test, W_test):
                    U_np = U_test
                    V_np = V_test
                    W_np = W_test
                    removed = True
                    break
            
            if not removed:
                break
        
        return U_np.tolist(), V_np.tolist(), W_np.tolist(), U_np.shape[1]
    
    # Try adding a new column with small integer values to enable removal of others
    def try_add_and_remove(U, V, W, max_attempts=50):
        """Try adding a new column with random small integers, then greedily reduce"""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        rank = U_np.shape[1]
        
        for _ in range(max_attempts):
            if time_left() < 10:
                break
            
            # Create a new column with random small integers
            new_U = np.zeros(U_np.shape[0])
            new_V = np.zeros(V_np.shape[0])
            new_W = np.zeros(W_np.shape[0])
            
            # Sparse: only a few non-zero entries
            for _ in range(random.randint(3, 8)):
                idx = random.randint(0, len(new_U) - 1)
                new_U[idx] = random.choice([-2, -1, 1, 2])
            
            for _ in range(random.randint(3, 8)):
                idx = random.randint(0, len(new_V) - 1)
                new_V[idx] = random.choice([-2, -1, 1, 2])
            
            for _ in range(random.randint(3, 8)):
                idx = random.randint(0, len(new_W) - 1)
                new_W[idx] = random.choice([-2, -1, 1, 2])
            
            U_new = np.column_stack([U_np, new_U])
            V_new = np.column_stack([V_np, new_V])
            W_new = np.column_stack([W_np, new_W])
            
            # Now greedily reduce
            U_r, V_r, W_r, r = greedy_reduce(U_new, V_new, W_new, max_iters=30)
            
            if r < rank:
                return U_r, V_r, W_r, r
        
        return None, None, None, rank
    
    # Random search for low-rank solutions
    def random_search(target_rank, max_attempts=100):
        """Try random sparse matrices of given rank"""
        for _ in range(max_attempts):
            if time_left() < 10:
                break
            
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Fill with sparse random values
            density = 0.3
            for r in range(target_rank):
                # Each column has some non-zero entries
                num_nonzero_U = max(1, int(n * m * density))
                num_nonzero_V = max(1, int(m * p * density))
                num_nonzero_W = max(1, int(p * n * density))
                
                U_indices = random.sample(range(n * m), min(num_nonzero_U, n * m))
                V_indices = random.sample(range(m * p), min(num_nonzero_V, m * p))
                W_indices = random.sample(range(p * n), min(num_nonzero_W, p * n))
                
                for idx in U_indices:
                    U[idx, r] = random.choice([-2, -1, 1, 2])
                for idx in V_indices:
                    V[idx, r] = random.choice([-2, -1, 1, 2])
                for idx in W_indices:
                    W[idx, r] = random.choice([-2, -1, 1, 2])
            
            if verify(U, V, W):
                return U.tolist(), V.tolist(), W.tolist(), target_rank
        
        return None, None, None, None
    
    # Start with schoolbook and reduce
    U_cur = np.array(best_result["U"])
    V_cur = np.array(best_result["V"])
    W_cur = np.array(best_result["W"])
    
    # Greedy reduction from schoolbook
    U_r, V_r, W_r, r = greedy_reduce(U_cur, V_cur, W_cur, max_iters=200)
    if r < best_rank:
        best_rank = r
        best_result = {"U": U_r, "V": V_r, "W": W_r}
    
    # Try add-and-remove
    if time_left() > 30:
        U_cur = np.array(best_result["U"])
        V_cur = np.array(best_result["V"])
        W_cur = np.array(best_result["W"])
        U_a, V_a, W_a, r_a = try_add_and_remove(U_cur, V_cur, W_cur, max_attempts=100)
        if r_a is not None and r_a < best_rank:
            best_rank = r_a
            best_result = {"U": U_a, "V": V_a, "W": W_a}
    
    # Random search for even lower ranks
    if time_left() > 20:
        for target_rank in range(max(20, best_rank - 30), best_rank):
            if time_left() < 15:
                break
            
            U_rs, V_rs, W_rs, r_rs = random_search(target_rank, max_attempts=50)
            if r_rs is not None:
                best_rank = r_rs
                best_result = {"U": U_rs, "V": V_rs, "W": W_rs}
                break
    
    # Final verification
    if not verify(best_result["U"], best_result["V"], best_result["W"]):
        best_result = make_schoolbook()
    
    return best_result
# EVOLVE-BLOCK-END
