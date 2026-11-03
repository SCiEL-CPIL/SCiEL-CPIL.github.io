"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomp(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct using numpy."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for l in range(m):
            for k in range(p):
                u_row = U_arr[i*m+l, :]
                v_row = V_arr[l*p+k, :]
                w_row = W_arr[k*n+i, :]
                s = np.dot(u_row, v_row * w_row)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _build_strassen_decomp(n, m, p, j_pairs, k_pairs):
    """Build a Strassen-based decomposition with given pairings."""
    rank = 7 * len(j_pairs) * len(k_pairs) + 2 * (m - 2*len(j_pairs)) * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            # P1 = (A[0,j0]+A[1,j1]) * (B[j0,k0]+B[j1,k1])
            U[0*m + j0][col] += 1
            U[1*m + j1][col] += 1
            V[j0*p + k0][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # P2 = (A[1,j0]+A[1,j1]) * B[j0,k0]
            U[1*m + j0][col] += 1
            U[1*m + j1][col] += 1
            V[j0*p + k0][col] += 1
            W[k0*n + 1][col] += 1
            W[k1*n + 1][col] -= 1
            col += 1
            
            # P3 = A[0,j0] * (B[j0,k1]-B[j1,k1])
            U[0*m + j0][col] += 1
            V[j0*p + k1][col] += 1
            V[j1*p + k1][col] -= 1
            W[k1*n + 0][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # P4 = A[1,j1] * (B[j1,k0]-B[j0,k0])
            U[1*m + j1][col] += 1
            V[j1*p + k0][col] += 1
            V[j0*p + k0][col] -= 1
            W[k0*n + 0][col] += 1
            W[k0*n + 1][col] += 1
            col += 1
            
            # P5 = (A[0,j0]+A[0,j1]) * B[j1,k1]
            U[0*m + j0][col] += 1
            U[0*m + j1][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] -= 1
            W[k1*n + 0][col] += 1
            col += 1
            
            # P6 = (A[1,j0]-A[0,j0]) * (B[j0,k0]+B[j0,k1])
            U[1*m + j0][col] += 1
            U[0*m + j0][col] -= 1
            V[j0*p + k0][col] += 1
            V[j0*p + k1][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # P7 = (A[0,j1]-A[1,j1]) * (B[j1,k0]+B[j1,k1])
            U[0*m + j1][col] += 1
            U[1*m + j1][col] -= 1
            V[j1*p + k0][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] += 1
            col += 1
    
    # Remaining j indices: standard products
    used_j = set()
    for j0, j1 in j_pairs:
        used_j.add(j0)
        used_j.add(j1)
    
    for j in range(m):
        if j in used_j:
            continue
        for k in range(p):
            U[0*m + j][col] += 1
            V[j*p + k][col] += 1
            W[k*n + 0][col] += 1
            col += 1
            
            U[1*m + j][col] += 1
            V[j*p + k][col] += 1
            W[k*n + 1][col] += 1
            col += 1
    
    return U, V, W

def _try_eliminate_term(U, V, W, n, m, p, rank, term_idx):
    """
    Try to eliminate term term_idx by redistributing its contribution
    to other terms. Returns (success, new_U, new_V, new_W, new_rank).
    
    Idea: If we remove term r, the residual is:
      residual[i,l,k] = -U[i,l,r]*V[l,k,r]*W[k,i,r]
    We need to find deltas dU, dV, dW for other terms such that
      sum_{r' != r} dU[i,l,r']*V[l,k,r']*W[k,i,r'] + U[i,l,r']*dV[l,k,r']*W[k,i,r'] + ... = residual
    
    This is complex. Simpler: check if term r's contribution is already
    captured (i.e., removing it still gives correct result).
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Compute tensor with term r removed
    tensor = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for l in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    if r == term_idx:
                        continue
                    s += U_arr[i*m+l, r] * V_arr[l*p+k, r] * W_arr[k*n+i, r]
                tensor[i*m+l, l*p+k, k*n+i] = s
    
    # Check if tensor equals identity
    max_err = 0.0
    for i in range(n):
        for l in range(m):
            for k in range(p):
                expected = 1.0 if (i == 0 and l == 0 and k == 0) or (i == 1 and l == 0 and k == 0) else 0.0
                # Actually the identity is: T[i,l,k] = 1 if this is a valid product entry
                # For matrix mult: C[i,k] = sum_l A[i,l]*B[l,k]
                # So T[i*m+l, l*p+k, k*n+i] = 1 for all valid (i,l,k)
                # Wait, that's not right. Let me re-examine.
                # The tensor T[a,b,c] where a=i*m+l, b=l*p+k, c=k*n+i
                # should equal 1 for all (i,l,k) since each product contributes exactly 1
                # to its corresponding output entry.
                pass
    
    # Simpler check: just verify the full decomposition without term r
    U_new = [row[:] for row in U]
    V_new = [row[:] for row in V]
    W_new = [row[:] for row in W]
    
    # Remove term r by setting all its entries to 0
    for i in range(n*m):
        U_new[i][term_idx] = 0
    for l in range(m*p):
        V_new[l][term_idx] = 0
    for k in range(p*n):
        W_new[k][term_idx] = 0
    
    if _verify_decomp(U_new, V_new, W_new, n, m, p, rank):
        # Term r is redundant! We can drop it.
        # Actually, we need to remove the column entirely for true rank reduction
        U_final = []
        for row in U_new:
            U_final.append([row[j] for j in range(rank) if j != term_idx])
        V_final = []
        for row in V_new:
            V_final.append([row[j] for j in range(rank) if j != term_idx])
        W_final = []
        for row in W_new:
            W_final.append([row[j] for j in range(rank) if j != term_idx])
        return True, U_final, V_final, W_final, rank - 1
    
    return False, None, None, None, rank

def _snap_to_half_integer(arr, tol=1e-6):
    """Snap values close to integers or half-integers."""
    result = []
    for row in arr:
        new_row = []
        for val in row:
            if abs(val - round(val)) < tol:
                new_row.append(float(round(val)))
            elif abs(val - (round(val*2)/2.0)) < tol:
                new_row.append(round(val*2)/2.0)
            else:
                new_row.append(float(val))
        result.append(new_row)
    return result

def construct():
    n, m, p = DIMENSIONS
    _start_time = time.time()
    _deadline = _start_time + 130  # 20s safety margin
    
    def _time_left():
        return _deadline - time.time()
    
    best_rank = 60
    best_result = None
    
    # Try different Strassen block configurations
    # m=5, p=6
    # Option 1: j_pairs=[(0,1),(2,3)], k_pairs=[(0,1),(2,3),(4,5)] -> rank 54
    # Option 2: j_pairs=[(0,1),(2,3),(4,?)], but m=5 so only 2 full pairs
    # Option 3: Different k pairings
    # Option 4: Try to use 3 j-pairs by including j=4 in a pair (but need 2 indices)
    
    configurations = [
        # (j_pairs, k_pairs, description)
        ([(0,1),(2,3)], [(0,1),(2,3),(4,5)], "standard"),
        ([(0,1),(2,3)], [(0,2),(1,4),(3,5)], "alt_k1"),
        ([(0,1),(2,3)], [(0,3),(1,4),(2,5)], "alt_k2"),
        ([(0,2),(1,3)], [(0,1),(2,3),(4,5)], "alt_j1"),
        ([(0,3),(1,2)], [(0,1),(2,3),(4,5)], "alt_j2"),
        ([(0,1),(2,3)], [(0,1),(2,4),(3,5)], "alt_k3"),
        ([(0,1),(2,3)], [(0,4),(1,5),(2,3)], "alt_k4"),
    ]
    
    for j_pairs, k_pairs, desc in configurations:
        if _time_left() < 5:
            break
        
        U, V, W = _build_strassen_decomp(n, m, p, j_pairs, k_pairs)
        rank = len(U[0])
        
        if not _verify_decomp(U, V, W, n, m, p, rank):
            continue
        
        if rank < best_rank:
            best_rank = rank
            best_result = {"U": U, "V": V, "W": W}
        
        # Try to eliminate redundant terms
        current_U, current_V, current_W, current_rank = U, V, W, rank
        for attempt in range(current_rank):
            if _time_left() < 3:
                break
            
            # Find the term with smallest L2 norm (most likely redundant)
            norms = []
            for r in range(current_rank):
                norm_sq = sum(current_U[i][r]**2 for i in range(n*m))
                norm_sq += sum(current_V[l][r]**2 for l in range(m*p))
                norm_sq += sum(current_W[k][r]**2 for k in range(p*n))
                norms.append(norm_sq)
            
            min_idx = int(np.argmin(norms))
            
            success, new_U, new_V, new_W, new_rank = _try_eliminate_term(
                current_U, current_V, current_W, n, m, p, current_rank, min_idx
            )
            
            if success:
                current_U, current_V, current_W, current_rank = new_U, new_V, new_W, new_rank
                if current_rank < best_rank:
                    best_rank = current_rank
                    best_result = {"U": current_U, "V": current_V, "W": current_W}
            else:
                # Try next smallest
                norms[min_idx] = float('inf')
                if min_idx == 0:
                    break
    
    # Snap to half-integers for cleaner output
    if best_result is not None:
        best_result["U"] = _snap_to_half_integer(best_result["U"])
        best_result["V"] = _snap_to_half_integer(best_result["V"])
        best_result["W"] = _snap_to_half_integer(best_result["W"])
        
        # Final verification
        if not _verify_decomp(best_result["U"], best_result["V"], best_result["W"], n, m, p, best_rank):
            # Fallback to standard
            U, V, W = _build_strassen_decomp(n, m, p, [(0,1),(2,3)], [(0,1),(2,3),(4,5)])
            best_result = {"U": U, "V": V, "W": W}
            best_rank = len(U[0])
    
    return best_result
# EVOLVE-BLOCK-END
