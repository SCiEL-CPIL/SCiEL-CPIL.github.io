"""Matrix multiplication 2x3x4 rank-minimal tensor decomposition."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 6, 12, 8
    
    # Build the target tensor entries
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                tensor_entries.append((u_idx, v_idx, w_idx))
    
    num_entries = len(tensor_entries)  # 24
    
    # Schoolbook fallback (rank 24)
    best_rank = n * m * p
    U_sb = [[0] * best_rank for _ in range(nm)]
    V_sb = [[0] * best_rank for _ in range(mp)]
    W_sb = [[0] * best_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    def verify_solution(U, V, W, rank):
        """Check if the decomposition is exact."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check tensor entries
        for (u, v, w) in tensor_entries:
            val = np.dot(U_arr[u], V_arr[v] * W_arr[w])
            if abs(val - 1.0) > 1e-9:
                return False
        
        # Check all non-entries
        entry_set = set(tensor_entries)
        for u in range(nm):
            for v in range(mp):
                for w in range(pn):
                    if (u, v, w) not in entry_set:
                        val = np.dot(U_arr[u], V_arr[v] * W_arr[w])
                        if abs(val) > 1e-9:
                            return False
        return True

    # Strategy: Use finite field F_5 arithmetic to search for solutions
    # Half-integers can be represented as elements of F_5: {0, 1, 2, 3, 4}
    # where 2 maps to 1/2, 3 maps to -1/2, etc.
    # We search for rank-r decompositions over F_5, then lift to rationals.
    
    # The tensor equation over F_5: sum_r U[u,r] * V[v,r] * W[w,r] = T[u,v,w] mod 5
    # where T[u,v,w] = 1 if (u,v,w) is an entry, 0 otherwise.
    
    # We'll use a randomized approach:
    # 1. Pick random U, V over F_5
    # 2. Solve for W over F_5
    # 3. Check if the solution lifts to a valid rational solution
    
    def solve_over_F5(target_rank):
        """Try to find a rank-target_rank solution over F_5."""
        # Build the system: for each entry (u,v,w), sum_r U[u,r]*V[v,r]*W[w,r] = 1
        # Unknowns: W[w,r] for w in 0..pn-1, r in 0..target_rank-1
        # Total unknowns: pn * target_rank
        
        num_unknowns = pn * target_rank
        
        # We'll use a different approach: iterate over possible W values
        # and check consistency. But that's too expensive.
        
        # Instead, use Gaussian elimination over F_5.
        # Build the coefficient matrix A where A[e, (w,r)] = U[u_e, r] * V[v_e, r]
        # and b[e] = 1 for entries.
        
        # But we also need to ensure non-entries give 0.
        # This is a system of num_entries + (nm*mp*pn - num_entries) equations.
        
        # For efficiency, let's use a smarter approach:
        # Fix U and V, then the constraint on W is linear.
        
        # Let's try a direct search with random U, V and solve for W.
        
        for attempt in range(2000):
            if time.time() > deadline - 5:
                return None
            
            # Generate random U, V over F_5
            U_f5 = np.random.randint(0, 5, size=(nm, target_rank))
            V_f5 = np.random.randint(0, 5, size=(mp, target_rank))
            
            # Build system for W
            # For each entry (u,v,w): sum_r U[u,r]*V[v,r]*W[w,r] = 1
            A = np.zeros((num_entries, num_unknowns), dtype=int)
            b = np.ones(num_entries, dtype=int)
            
            for e_idx, (u_idx, v_idx, w_idx) in enumerate(tensor_entries):
                for r in range(target_rank):
                    col = w_idx * target_rank + r
                    A[e_idx, col] = (U_f5[u_idx, r] * V_f5[v_idx, r]) % 5
            
            # Also add constraints for non-entries (sample some for speed)
            # But this makes the system too large. Let's just solve the entry constraints
            # and then verify.
            
            # Solve A * W_flat = b over F_5 using Gaussian elimination
            W_flat = gaussian_elimination_f5(A, b, target_rank)
            if W_flat is None:
                continue
            
            W_f5 = W_flat.reshape(pn, target_rank)
            
            # Now verify over the rationals by lifting
            # Map F_5 values to rationals: 0->0, 1->1, 2->1/2, 3->-1/2, 4->-1
            def f5_to_rat(x):
                if x == 0: return 0.0
                if x == 1: return 1.0
                if x == 2: return 0.5
                if x == 3: return -0.5
                if x == 4: return -1.0
                return 0.0
            
            U_rat = np.array([[f5_to_rat(x) for x in row] for row in U_f5])
            V_rat = np.array([[f5_to_rat(x) for x in row] for row in V_f5])
            W_rat = np.array([[f5_to_rat(x) for x in row] for row in W_f5])
            
            # Verify
            if verify_solution(U_rat.tolist(), V_rat.tolist(), W_rat.tolist(), target_rank):
                return {"U": U_rat.tolist(), "V": V_rat.tolist(), "W": W_rat.tolist()}
        
        return None
    
    def gaussian_elimination_f5(A, b, target_rank):
        """Solve A*x = b over F_5 using Gaussian elimination."""
        n_rows = A.shape[0]
        n_cols = A.shape[1]
        
        # Augmented matrix
        M = np.hstack([A.copy(), b.reshape(-1, 1)])
        
        # Gaussian elimination
        rank = 0
        for col in range(n_cols):
            if time.time() > deadline - 5:
                return None
            
            # Find pivot
            pivot_row = -1
            for row in range(rank, n_rows):
                if M[row, col] % 5 != 0:
                    pivot_row = row
                    break
            
            if pivot_row == -1:
                continue
            
            # Swap rows
            if pivot_row != rank:
                M[[rank, pivot_row]] = M[[pivot_row, rank]]
            
            # Normalize pivot row
            pivot_val = M[rank, col] % 5
            inv_pivot = pow(pivot_val, -1, 5)  # Modular inverse
            M[rank] = (M[rank] * inv_pivot) % 5
            
            # Eliminate other rows
            for row in range(n_rows):
                if row == rank:
                    continue
                factor = M[row, col] % 5
                if factor != 0:
                    M[row] = (M[row] - factor * M[rank]) % 5
            
            rank += 1
        
        # Check for consistency
        for row in range(rank, n_rows):
            if M[row, n_cols] % 5 != 0:
                return None
        
        # Back substitution (or just extract solution if square)
        # For simplicity, assume we have a unique solution or pick one
        x = np.zeros(n_cols, dtype=int)
        for row in range(rank - 1, -1, -1):
            # Find the leading column in this row
            lead_col = -1
            for col in range(n_cols):
                if M[row, col] % 5 != 0:
                    lead_col = col
                    break
            if lead_col == -1:
                continue
            
            val = M[row, n_cols] % 5
            for col in range(lead_col + 1, n_cols):
                val = (val - M[row, col] * x[col]) % 5
            
            inv_lead = pow(M[row, lead_col] % 5, -1, 5)
            x[lead_col] = (val * inv_lead) % 5
        
        return x
    
    # Search for minimal rank using F_5 approach
    for target_rank in range(18, best_rank):
        if time.time() > deadline - 5:
            break
        
        np.random.seed(42 + target_rank * 1000)
        result = solve_over_F5(target_rank)
        
        if result is not None:
            best_rank = target_rank
            best_result = result
            break
    
    # If F_5 search didn't find anything, try the original random approach as fallback
    if best_rank == n * m * p:
        for target_rank in range(18, best_rank):
            if time.time() > deadline - 5:
                break
            
            found = False
            np.random.seed(42 + target_rank)
            
            for attempt in range(3000):
                if time.time() > deadline - 2:
                    break
                
                U_init = np.random.choice([-1, 0, 1], size=(nm, target_rank))
                V_init = np.random.choice([-1, 0, 1], size=(mp, target_rank))
                
                if np.all(U_init == 0) or np.all(V_init == 0):
                    continue
                
                A_W = np.zeros((num_entries, pn * target_rank))
                b_W = np.ones(num_entries)
                
                for e_idx, (u_idx, v_idx, w_idx) in enumerate(tensor_entries):
                    for r in range(target_rank):
                        col = w_idx * target_rank + r
                        A_W[e_idx, col] = U_init[u_idx, r] * V_init[v_idx, r]
                
                try:
                    W_flat, residuals, rank_A, s = np.linalg.lstsq(A_W, b_W, rcond=None)
                except np.linalg.LinAlgError:
                    continue
                    
                W_curr = W_flat.reshape(pn, target_rank)
                W_check = np.round(W_curr * 2) / 2.0
                
                error = 0.0
                for (u, v, w) in tensor_entries:
                    val = sum(U_init[u, r] * V_init[v, r] * W_check[w, r] for r in range(target_rank))
                    error += abs(val - 1.0)
                    
                if error > 1e-6:
                    continue
                    
                if verify_solution(U_init.tolist(), V_init.tolist(), W_check.tolist(), target_rank):
                    best_rank = target_rank
                    best_result = {
                        "U": U_init.tolist(),
                        "V": V_init.tolist(),
                        "W": W_check.tolist()
                    }
                    found = True
                    break
            
            if found:
                break
    
    return best_result
# EVOLVE-BLOCK-END

