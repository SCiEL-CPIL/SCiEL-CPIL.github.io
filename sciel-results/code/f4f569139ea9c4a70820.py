"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 148.0  # Leave 2s safety margin for 150s limit

    # Ensure arrays are int64 for exact integer arithmetic
    a_int = a.astype(np.int64)
    b_int = b.astype(np.int64)
    
    a_float = a.astype(np.float64)
    b_float = b.astype(np.float64)

    # 1. Initialization via Maximum Weight Perfect Matching on 2n nodes
    # We formulate the initial assignment as a matching problem where we pair facilities with locations.
    # The "weight" of pairing facility i with location j is derived from the linearized objective:
    # w(i, j) = sum_k a[i, k] * b[j, k]
    # This captures the global affinity between the flow structure of facility i and the distance structure of location j.
    # To solve this as a perfect matching, we create a bipartite graph between facilities and locations.
    # However, the directive suggests a complete graph of size 2n. 
    # A simpler and equivalent approach for bipartite matching is to use the Hungarian algorithm on the cost matrix.
    # Cost matrix C[i, j] = -w(i, j) = -sum_k a[i, k] b[j, k]
    # This is equivalent to C = -a @ b.T (if we consider row i of a and column j of b? No.)
    # Let's check the sum: sum_k a[i, k] b[j, k].
    # This is the dot product of row i of A and row j of B.
    # So C[i, j] = - (a_float[i, :] @ b_float[j, :])
    # This is - (a_float @ b_float.T) [i, j] ? No.
    # a_float[i, :] is a vector. b_float[j, :] is a vector.
    # Dot product is sum_k a[i, k] b[j, k].
    # Matrix M where M[i, j] = sum_k a[i, k] b[j, k].
    # M = a_float @ b_float.T ?
    # (a @ b.T)[i, j] = sum_k a[i, k] b.T[k, j] = sum_k a[i, k] b[j, k]. Yes.
    
    # So we want to minimize sum_{i,j} a[i,j] b[p[i], p[j]].
    # The initialization heuristic minimizes a linear surrogate.
    # Let's compute the surrogate cost matrix.
    
    try:
        # Surrogate cost matrix for initialization
        # We want to align high flow with low distance? Or high flow with high distance?
        # The objective is sum a[i,j] b[p[i], p[j]].
        # If a[i,j] is large, we want b[p[i], p[j]] to be small.
        # The linearized term for facility i at location j is sum_k a[i,k] b[j,k].
        # This sum is large if facility i has high flow to facilities that are "close" to location j in the B-metric?
        # Actually, if b[j,k] is small, the term is small.
        # So we want to minimize this sum.
        
        C_init = a_float @ b_float.T
        # Minimize C_init via Hungarian
        row_ind, col_ind = linear_sum_assignment(C_init)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        
    except Exception:
        p = np.random.permutation(n)

    # 2. Refine with Iterative Linearization (Coordinate Descent on Permutation)
    # This is a fast local improvement method.
    
    def compute_linear_cost_matrix(p):
        # C[i, j] = sum_k a[i, k] * b[j, p[k]]
        # M[k, j] = b[j, p[k]] => M = b[:, p]
        # C = a @ M = a @ b[:, p]
        return a_float @ b_float[:, p]

    current_p = p.copy()
    
    # Perform a few iterations of this cheap refinement
    for _ in range(10):
        if time.time() - start_time > time_limit - 10:
            break
        C = compute_linear_cost_matrix(current_p)
        row_ind, col_ind = linear_sum_assignment(C)
        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind
        
        cost_new = np.sum(a_int * b_int[np.ix_(new_p, new_p)])
        cost_old = np.sum(a_int * b_int[np.ix_(current_p, current_p)])
        
        if cost_new < cost_old:
            current_p = new_p
        else:
            break
            
    p = current_p

    # 3. Local Search: 2-Swap with Efficient Delta Calculation
    # We use the delta formula derived for symmetric matrices, but we must be careful if they are not perfectly symmetric.
    # The QAPLIB instances are usually symmetric. We will assume symmetry for speed but the formula works generally if we don't simplify.
    # General Delta for swapping i and j (u=p[i], v=p[j]):
    # Old terms involving i or j:
    # a[i,i]b[u,u] + a[j,j]b[v,v] + a[i,j]b[u,v] + a[j,i]b[v,u]
    # + sum_{k!=i,j} (a[i,k]b[u,p[k]] + a[k,i]b[p[k],u] + a[j,k]b[v,p[k]] + a[k,j]b[p[k],v])
    #
    # New terms:
    # a[i,i]b[v,v] + a[j,j]b[u,u] + a[i,j]b[v,u] + a[j,i]b[u,v]
    # + sum_{k!=i,j} (a[i,k]b[v,p[k]] + a[k,i]b[p[k],v] + a[j,k]b[u,p[k]] + a[k,j]b[p[k],u])
    #
    # Delta = New - Old
    #
    # If A and B are symmetric:
    # a[i,i]b[v,v] + a[j,j]b[u,u] + 2a[i,j]b[u,v] - (a[i,i]b[u,u] + a[j,j]b[v,v] + 2a[i,j]b[u,v])
    # = (a[i,i] - a[j,j])(b[v,v] - b[u,u])
    #
    # Sum part:
    # sum_{k!=i,j} [ a[i,k](b[v,p[k]] - b[u,p[k]]) + a[k,i](b[p[k],v] - b[p[k],u]) 
    #              + a[j,k](b[u,p[k]] - b[v,p[k]]) + a[k,j](b[p[k],u] - b[p[k],v]) ]
    # Using symmetry a[i,k]=a[k,i], b[v,p[k]]=b[p[k],v]:
    # = sum_{k!=i,j} [ 2a[i,k](b[v,p[k]] - b[u,p[k]]) - 2a[j,k](b[v,p[k]] - b[u,p[k]]) ]
    # = 2 sum_{k!=i,j} (a[i,k] - a[j,k]) (b[v,p[k]] - b[u,p[k]])
    
    def get_delta(i, j, p):
        u = p[i]
        v = p[j]
        
        # Diagonal part
        delta_diag = (a_int[i, i] - a_int[j, j]) * (b_int[v, v] - b_int[u, u])
        
        # Off-diagonal part
        diff_a = a_int[i, :] - a_int[j, :]
        diff_b = b_int[v, :] - b_int[u, :]
        
        # Map diff_b to current positions
        diff_b_perm = diff_b[p]
        
        total_sum = np.dot(diff_a, diff_b_perm)
        
        # Subtract k=i and k=j terms
        # k=i: diff_a[i] * diff_b_perm[i] = diff_a[i] * diff_b[u]
        # k=j: diff_a[j] * diff_b_perm[j] = diff_a[j] * diff_b[v]
        term_i = diff_a[i] * diff_b[u]
        term_j = diff_a[j] * diff_b[v]
        
        sum_k = total_sum - term_i - term_j
        
        delta = delta_diag + 2 * sum_k
        return delta

    # Local Search Loop
    best_p = p.copy()
    best_cost = int(np.sum(a_int * b_int[np.ix_(best_p, best_p)]))
    current_p = p.copy()
    current_cost = best_cost
    
    no_improvement_count = 0
    max_no_improvement = 5 # Stop if 5 full scans yield no improvement
    
    while time.time() - start_time < time_limit:
        if time.time() - start_time > time_limit - 5:
            break
            
        improved = False
        best_delta = 0
        best_i = -1
        best_j = -1
        
        # Full scan of all pairs
        # To speed up, we can vectorize the delta calculation for a fixed i over all j?
        # Delta(i, j) = (a_ii - a_jj)(b_vv - b_uu) + 2 * (a_i - a_j) . (b_v - b_u)[p] - corrections
        # This is hard to vectorize over j because b_v depends on j.
        # So we loop i, and for each i, we can vectorize over j?
        # For fixed i, u is fixed.
        # diff_b_j = b[p[j], :] - b[u, :]
        # This is a matrix of size (n, n) if we do it for all j.
        # b[p[:, :]] is not a standard op.
        # Let's stick to the double loop but optimize the inner call.
        
        for i in range(n):
            if time.time() - start_time > time_limit - 1:
                break
            u = current_p[i]
            # Precompute parts that depend only on i
            a_i = a_int[i, :]
            b_u = b_int[u, :]
            
            for j in range(i + 1, n):
                if time.time() - start_time > time_limit - 1:
                    break
                v = current_p[j]
                
                # Inline delta calculation for speed
                # delta_diag = (a_int[i, i] - a_int[j, j]) * (b_int[v, v] - b_int[u, u])
                # diff_a = a_i - a_int[j, :]
                # diff_b = b_int[v, :] - b_u
                # diff_b_perm = diff_b[current_p]
                # total_sum = np.dot(diff_a, diff_b_perm)
                # term_i = diff_a[i] * diff_b[u]
                # term_j = diff_a[j] * diff_b[v]
                # sum_k = total_sum - term_i - term_j
                # delta = delta_diag + 2 * sum_k
                
                # This is still a lot of Python overhead.
                # Let's try to vectorize over j for a fixed i.
                # We want delta(i, j) for all j > i.
                # Let J be the range of j's.
                # v_j = current_p[J]
                # diff_b_J = b[v_J, :] - b[u, :]  (size |J| x n)
                # diff_b_perm_J = diff_b_J[:, current_p] (size |J| x n)
                # diff_a_J = a[i, :] - a[J, :] (size |J| x n)
                # total_sum_J = sum_k diff_a_J[:, k] * diff_b_perm_J[:, k] (size |J|)
                # term_i_J = diff_a_J[:, i] * diff_b_J[:, u]
                # term_j_J = diff_a_J[:, J] * diff_b_J[:, v_J]  (diagonal elements)
                # sum_k_J = total_sum_J - term_i_J - term_j_J
                # delta_diag_J = (a[i, i] - a[J, J]) * (b[v_J, v_J] - b[u, u])
                # delta_J = delta_diag_J + 2 * sum_k_J
                
                # This vectorization reduces the inner loop to a single vectorized operation.
                # Let's implement this.
                
                # We already are in the loop for i. Let's restructure to vectorize j.
                # We should move the j-loop vectorization outside the j-loop.
                
                # Actually, let's just break and restructure the outer loop to vectorize j.
                pass
        
        # Restructured Loop
        improved = False
        best_delta = 0
        best_i = -1
        best_j = -1
        
        for i in range(n):
            if time.time() - start_time > time_limit - 1:
                break
            
            u = current_p[i]
            a_i = a_int[i, :]
            b_u = b_int[u, :]
            
            # Indices for j > i
            Js = np.arange(i + 1, n)
            if len(Js) == 0:
                continue
                
            v_Js = current_p[Js]
            a_Js = a_int[Js, :]
            b_v_Js = b_int[v_Js, :]
            
            # diff_a_Js: (|J|, n)
            diff_a_Js = a_i[None, :] - a_Js
            
            # diff_b_Js: (|J|, n)
            diff_b_Js = b_v_Js - b_u[None, :]
            
            # diff_b_perm_Js: (|J|, n)
            # We need diff_b_Js[:, current_p]
            # This is a fancy indexing operation.
            diff_b_perm_Js = diff_b_Js[:, current_p]
            
            # total_sum_Js: (|J|,)
            total_sum_Js = np.sum(diff_a_Js * diff_b_perm_Js, axis=1)
            
            # term_i_Js: (|J|,)
            # diff_a_Js[:, i] * diff_b_Js[:, u]
            term_i_Js = diff_a_Js[:, i] * diff_b_Js[:, u]
            
            # term_j_Js: (|J|,)
            # diff_a_Js[:, Js] * diff_b_Js[:, v_Js]
            # We need the element (k, Js[k]) of diff_a_Js and (k, v_Js[k]) of diff_b_Js
            term_j_Js = diff_a_Js[np.arange(len(Js)), Js] * diff_b_Js[np.arange(len(Js)), v_Js]
            
            # sum_k_Js: (|J|,)
            sum_k_Js = total_sum_Js - term_i_Js - term_j_Js
            
            # delta_diag_Js: (|J|,)
            # (a[i, i] - a[J, J]) * (b[v_J, v_J] - b[u, u])
            delta_diag_Js = (a_int[i, i] - a_int[Js, Js]) * (b_int[v_Js, v_Js] - b_int[u, u])
            
            # delta_Js: (|J|,)
            delta_Js = delta_diag_Js + 2 * sum_k_Js
            
            # Find the minimum delta
            min_idx = np.argmin(delta_Js)
            min_delta = delta_Js[min_idx]
            
            if min_delta < best_delta:
                best_delta = min_delta
                best_i = i
                best_j = Js[min_idx]
        
        if best_delta < 0:
            # Apply swap
            i, j = best_i, best_j
            current_p[i], current_p[j] = current_p[j], current_p[i]
            current_cost += int(best_delta)
            improved = True
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= max_no_improvement:
                break

        if current_cost < best_cost:
            best_cost = current_cost
            best_p = current_p.copy()

    return best_p.tolist()

# EVOLVE-BLOCK-END
