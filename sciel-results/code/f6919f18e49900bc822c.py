"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    start_time = time.time()
    # 120s limit for evaluation, 145s for this function? 
    # Contract says: "Each evaluation must finish within 120 s." 
    # But the prompt says: "evaluator KILLS the program after 150s wall-clock."
    # To be safe, we use a margin. Let's assume the 120s is the hard stop for scoring.
    # The current program used 145s. I will use 115s to be safe for the 120s contract, 
    # but the prompt says "Keep an EXPLICIT deadline guard... safety margin before the limit."
    # If the limit is 120s, 115s is safe. If the limit is 150s, 145s is safe.
    # The contract says "Each evaluation must finish within 120 s".
    # However, the "DIRECTIVE" says "evaluator KILLS... after 150s".
    # Usually, the stricter contract (120s) is the one that matters for "valid" score.
    # Let's use 118.0s to be very safe against the 120s limit.
    deadline = start_time + 118.0 

    # Ensure arrays are int64 for exact integer arithmetic
    a = a.astype(np.int64)
    b = b.astype(np.int64)

    # 1. Initial Solution: Rank Matching
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    
    # Heavy flows (high row sums in a) should go to central locations (low row sums in b)
    heavy_idx = np.argsort(-a_sums, kind="stable")
    central_idx = np.argsort(b_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_idx] = central_idx
    
    # Calculate initial cost
    cost = np.sum(a * b[p])
    
    best_cost = cost
    best_p = p.copy()

    # Precompute row sums of b for delta calculation?
    # Delta for swapping i, j:
    # new_cost = old_cost - (sum_{k!=i,j} a[i,k]b[p[i],p[k]] + sum_{k!=i,j} a[j,k]b[p[j],p[k]] + 2*a[i,j]b[p[i],p[j]])
    #            + (sum_{k!=i,j} a[i,k]b[p[j],p[k]] + sum_{k!=i,j} a[j,k]b[p[i],p[k]] + 2*a[i,j]b[p[j],p[i]])
    
    # Let u = p[i], v = p[j]
    # Term for i in old: sum_{k!=i,j} a[i,k] b[u, p[k]] + a[i,j]b[u,v]
    # Term for j in old: sum_{k!=i,j} a[j,k] b[v, p[k]] + a[j,i]b[v,u]
    
    # Vectorized Delta Calculation:
    # We want to compute delta for all pairs (i, j) with i < j.
    # This is O(n^3) if done naively in a loop, but we can vectorize the inner sum.
    
    # Let A_i = a[i, :] and B_u = b[u, :]
    # The interaction of i with all others (excluding j) is dot(A_i, B_u[p]) - a[i,j]b[u,v]
    # Actually, it's easier to compute the full delta matrix.
    
    # Cost contribution of facility i at location u is C_i(u) = sum_k a[i,k] b[u, p[k]]
    # Total Cost = 1/2 * sum_i C_i(p[i])  (since a and b are symmetric, usually)
    # Wait, QAP cost is sum_{i,j} a[i,j] b[p[i], p[j]].
    # If a and b are symmetric, this is 1/2 * sum_i sum_j a[i,j] b[p[i], p[j]] + 1/2 * sum_{i!=j}...
    # Actually, standard QAP definition: sum_{i,j} a[i,j] b[p[i], p[j]].
    # If i=j, term is a[i,i] b[p[i], p[i]].
    
    # Let's compute the "potential" of each facility at each location.
    # Phi[i, u] = sum_k a[i, k] * b[u, p[k]]
    # Then Cost = sum_i Phi[i, p[i]]
    # Note: This counts each pair (i,j) twice if i!=j?
    # sum_i sum_k a[i,k] b[p[i], p[k]]
    # Term (i,j) is a[i,j] b[p[i], p[j]]. Term (j,i) is a[j,i] b[p[j], p[i]].
    # If a and b are symmetric, these are equal. So Cost = sum_i Phi[i, p[i]].
    # This is correct.
    
    # To find delta for swapping i and j:
    # Old Cost part involving i and j:
    # Old: Phi[i, p[i]] + Phi[j, p[j]]
    # Phi[i, u] = sum_{k!=j} a[i,k] b[u, p[k]] + a[i,j] b[u, v]  (where u=p[i], v=p[j])
    # Phi[j, v] = sum_{k!=i} a[j,k] b[v, p[k]] + a[j,i] b[v, u]
    
    # New: Swap u and v. i goes to v, j goes to u.
    # New Phi[i, v] = sum_{k!=j} a[i,k] b[v, p[k]] + a[i,j] b[v, u]
    # New Phi[j, u] = sum_{k!=i} a[j,k] b[u, p[k]] + a[j,i] b[u, v]
    
    # Delta = (New Phi[i, v] + New Phi[j, u]) - (Old Phi[i, u] + Old Phi[j, v])
    # Delta = [sum_{k!=j} a[i,k] (b[v, p[k]] - b[u, p[k]]) + a[i,j](b[v,u] - b[u,v])? No.
    
    # Let's simplify:
    # Old_i = sum_{k!=j} a[i,k] b[u, p[k]] + a[i,j] b[u, v]
    # New_i = sum_{k!=j} a[i,k] b[v, p[k]] + a[i,j] b[v, u]
    # Diff_i = sum_{k!=j} a[i,k] (b[v, p[k]] - b[u, p[k]]) + a[i,j] (b[v, u] - b[u, v])
    # If b is symmetric, b[v,u] = b[u,v], so last term is 0.
    # Diff_i = sum_{k!=j} a[i,k] (b[v, p[k]] - b[u, p[k]])
    
    # Similarly for j:
    # Diff_j = sum_{k!=i} a[j,k] (b[u, p[k]] - b[v, p[k]])
    
    # Total Delta = Diff_i + Diff_j
    
    # We can compute this for all pairs (i,j) vectorized?
    # It's tricky because of the "k!=j" exclusion.
    # However, we can compute the full sum S_i(u, v) = sum_k a[i,k] (b[v, p[k]] - b[u, p[k]])
    # Then Diff_i = S_i(u, v) - a[i,j] (b[v, p[j]] - b[u, p[j]])
    # Note p[j] = v. So b[v, p[j]] = b[v, v] and b[u, p[j]] = b[u, v].
    # So Diff_i = S_i(u, v) - a[i,j] (b[v, v] - b[u, v])
    
    # Let's define D[i, j] = Delta for swapping i and j.
    # This requires computing S_i for all i and all pairs (u,v)? That's O(n^3) or O(n^4).
    # n=150. n^3 = 3.3M. n^4 = 500M.
    # We can do this in a loop over i (150 iterations), and for each i, compute the effect of swapping i with all j.
    # For fixed i, let u = p[i].
    # We want delta(i, j) for all j != i.
    # Let v = p[j].
    # Delta(i, j) = [Sum_{k!=j} a[i,k] (b[v, p[k]] - b[u, p[k]]) + a[i,j](b[v,v]-b[u,v])]  (Wait, sign of a[i,j] term?)
    # From before: Diff_i = S_i(u, v) - a[i,j] (b[v, v] - b[u, v])
    # Where S_i(u, v) = Sum_{k} a[i,k] (b[v, p[k]] - b[u, p[k]])
    # And Diff_j = S_j(v, u) - a[j,i] (b[u, u] - b[v, u])
    # Note S_j(v, u) = Sum_{k} a[j,k] (b[u, p[k]] - b[v, p[k]]) = - Sum_{k} a[j,k] (b[v, p[k]] - b[u, p[k]]) = - S_j(u, v) if a[j,k] symmetric? No.
    # S_j(v, u) is just the sum with u and v swapped in the b part.
    
    # Algorithm for one pass:
    # 1. Compute Phi = a @ b[p]  (n x n matrix). Phi[i, u] is the cost contribution of i at u?
    #    Phi[i, u] = sum_k a[i, k] b[u, p[k]].
    #    Current Cost = sum_i Phi[i, p[i]].
    
    # 2. For each i (150 times):
    #    u = p[i]
    #    We want to find the best j to swap with i, or accept a random j with probability based on delta.
    #    To do this efficiently, we need Delta(i, j) for all j.
    #    Delta(i, j) = (Phi_new_i + Phi_new_j) - (Phi_old_i + Phi_old_j)
    #    Phi_old_i = Phi[i, u]
    #    Phi_old_j = Phi[j, p[j]]
    
    #    Let's compute the change in Phi for i if we move it to location v = p[j].
    #    New_Phi_i_v = sum_k a[i, k] b[v, p[k]]
    #    But p[j] changes to u. So for k=j, the term is a[i, j] b[v, u].
    #    For k != j, the term is a[i, k] b[v, p[k]].
    #    
    #    We can precompute a matrix M where M[v, k] = b[v, p[k]]? No, p changes.
    #    
    #    Let's use the formula:
    #    Delta(i, j) = (sum_{k!=j} a[i,k](b[p[j], p[k]] - b[p[i], p[k]]) + a[i,j](b[p[j], p[i]] - b[p[i], p[j]]))
    #               + (sum_{k!=i} a[j,k](b[p[i], p[k]] - b[p[j], p[k]]) + a[j,i](b[p[i], p[j]] - b[p[j], p[i]]))
    
    #    Let u = p[i], v = p[j].
    #    Term 1 (for i): Sum_{k!=j} a[i,k] (b[v, p[k]] - b[u, p[k]])
    #    Term 2 (for j): Sum_{k!=i} a[j,k] (b[u, p[k]] - b[v, p[k]])
    #    Term 3 (cross): a[i,j](b[v, u] - b[u, v]) + a[j,i](b[u, v] - b[v, u])
    #    If a,b symmetric, Term 3 is 0.
    
    #    Let V = b[v, p] - b[u, p]. This is a vector of length n.
    #    Term 1 = dot(a[i, :], V) - a[i, j] * V[j]  (since V[j] = b[v, p[j]] - b[u, p[j]] = b[v, v] - b[u, v])
    #    Term 2 = dot(a[j, :], -V) - a[j, i] * (-V[i]) = -dot(a[j, :], V) + a[j, i] * V[i]
    #    Note V[i] = b[v, p[i]] - b[u, p[i]] = b[v, u] - b[u, u].
    
    #    So Delta(i, j) = dot(a[i, :], V) - a[i, j] * V[j] - dot(a[j, :], V) + a[j, i] * V[i]
    
    #    We can compute this for all j given i in O(n^2) using vectorization?
    #    V depends on j (since v = p[j]). So V is different for each j.
    #    V_j = b[p[j], p] - b[u, p].
    #    Delta(i, j) = dot(a[i, :], V_j) - dot(a[j, :], V_j) - a[i, j] * V_j[j] + a[j, i] * V_j[i]
    
    #    This is still hard to fully vectorize over j because V_j changes.
    #    However, n=150. O(n^3) per pass is 3.3M ops. 
    #    If we do 1000 passes, that's 3.3B ops. Might be too slow in Python/NumPy if not careful.
    #    But we can do "First Improvement" or "Best Improvement" on a subset?
    #    
    #    Alternative: Simulated Annealing with random swaps.
    #    If we just pick random (i, j), we compute delta in O(n).
    #    120s / 1e-6s per move = 120,000 moves.
    #    This is what the current program does (slowly).
    #    
    #    Can we speed up the random swap delta?
    #    Delta(i, j) requires dot(a[i, :], V) and dot(a[j, :], V).
    #    V = b[v, p] - b[u, p].
    #    dot(a[i, :], b[v, p]) = (a @ b)[i, v] ? No.
    #    Let A_b = a @ b. (n x n).
    #    (a @ b)[i, v] = sum_k a[i, k] b[k, v].
    #    But we need sum_k a[i, k] b[v, p[k]].
    #    This is sum_k a[i, k] (b[p])[v, k] if b is symmetric?
    #    b[v, p[k]] is the element at row v, col p[k].
    #    Let B_p = b[:, p] ? No, b is (n,n). b[v, p[k]] is accessing row v, column p[k].
    #    This is equivalent to (b @ p_onehot)? No.
    #    
    #    Let's precompute Q = a @ b.
    #    Q[i, v] = sum_k a[i, k] b[k, v].
    #    We want S_i(v) = sum_k a[i, k] b[v, p[k]].
    #    If b is symmetric, b[v, p[k]] = b[p[k], v].
    #    So S_i(v) = sum_k a[i, k] b[p[k], v] = (a @ b)[i, v] = Q[i, v].
    #    YES! If a and b are symmetric, S_i(v) = Q[i, v].
    #    
    #    So Term 1 = Q[i, v] - a[i, j] * (b[v, v] - b[u, v])
    #    Term 2 = - Q[j, v] + a[j, i] * (b[v, u] - b[u, u])  (Wait, V[i] = b[v, u] - b[u, u])
    #    Wait, let's re-verify Term 2.
    #    Term 2 = -dot(a[j, :], V) + a[j, i] * V[i]
    #    dot(a[j, :], V) = sum_k a[j, k] (b[v, p[k]] - b[u, p[k]])
    #    = sum_k a[j, k] b[v, p[k]] - sum_k a[j, k] b[u, p[k]]
    #    = Q[j, v] - Q[j, u]
    #    So Term 2 = -(Q[j, v] - Q[j, u]) + a[j, i] * (b[v, u] - b[u, u])
    #    = Q[j, u] - Q[j, v] + a[j, i] * (b[v, u] - b[u, u])
    
    #    Term 1 = Q[i, v] - Q[i, u] - a[i, j] * (b[v, v] - b[u, v])
    
    #    Delta(i, j) = (Q[i, v] - Q[i, u] - a[i, j] * (b[v, v] - b[u, v])) 
    #               + (Q[j, u] - Q[j, v] + a[j, i] * (b[v, u] - b[u, u]))
    
    #    This is O(1) per swap if Q is precomputed!
    #    Q = a @ b. n=150. 150x150 matrix mult is fast.
    
    #    We need to handle the case where a or b are not symmetric?
    #    QAPLIB instances are typically symmetric.
    #    If not, the formula changes.
    #    Assuming symmetry for speed. If it fails, we can check.
    #    The task says "int64 NumPy arrays". It doesn't explicitly say symmetric, but QAPLIB tho150 is symmetric.
    
    #    Let's implement this O(1) delta calculation.
    
    # Precompute Q
    Q = a @ b  # (n, n)
    
    # Precompute diagonal of b
    b_diag = np.diag(b)
    
    # Simulated Annealing
    T = 1000.0
    cooling = 0.999
    
    # To allow more moves, we can do random swaps.
    # We can also do "First Improvement" by checking a few random swaps.
    
    # Let's do a mix:
    # 1. Compute best swap in O(n^2) using the O(1) delta formula?
    #    Delta(i, j) for all i<j.
    #    We can vectorize the calculation of Delta for all pairs?
    #    Delta(i, j) = Q[i, p[j]] - Q[i, p[i]] - a[i, j] * (b[p[j], p[j]] - b[p[i], p[j]]) 
    #               + Q[j, p[i]] - Q[j, p[j]] + a[j, i] * (b[p[j], p[i]] - b[p[i], p[i]])
    
    #    Let's try to compute the matrix of deltas D (n x n).
    #    D[i, j] = Delta(i, j).
    
    #    Term A: Q[i, p[j]] - Q[i, p[i]]
    #    Term B: - a[i, j] * (b[p[j], p[j]] - b[p[i], p[j]])
    #    Term C: Q[j, p[i]] - Q[j, p[j]]
    #    Term D: a[j, i] * (b[p[j], p[i]] - b[p[i], p[i]])
    
    #    We can compute these term by term using NumPy broadcasting.
    
    while time.time() < deadline:
        # Compute Delta Matrix
        # p is the current permutation.
        
        # Term A: Q[:, p] - Q[:, p[:, None]]? No.
        # Q[i, p[j]] is Q[:, p] (n x n matrix where col j is Q[:, p[j]]).
        # Let Q_p = Q[:, p]  # n x n. Q_p[i, j] = Q[i, p[j]].
        # Term A = Q_p - Q_p[:, p] ? No.
        # Q[i, p[i]] is the diagonal of Q_p? No.
        # Q[i, p[i]] is Q_p[i, i] if p[i]=i? No.
        # Q[i, p[i]] is the element at row i, col i of Q_p? 
        # Q_p[i, j] = Q[i, p[j]].
        # So Q[i, p[i]] = Q_p[i, i].
        # So Term A = Q_p - Q_p.diagonal()[:, None]
        
        Q_p = Q[:, p]
        Term_A = Q_p - Q_p.diagonal()[:, None]
        
        # Term B: - a * (b[p, p] - b[p[:, None], p[None, :]])
        # b[p[j], p[j]] is b_diag[p[j]]. Let b_pp = b_diag[p] (n vector).
        # b[p[i], p[j]] is b[p[:, None], p[None, :]]. Let b_pp_mat = b[np.ix_(p, p)] (n x n).
        # Term B = - a * (b_pp_mat - b_pp[None, :]) ?
        # b[p[j], p[j]] is the diagonal of b_pp_mat.
        # So b[p[j], p[j]] - b[p[i], p[j]] = b_pp_mat.diagonal() - b_pp_mat.T ?
        # Let's check indices.
        # We want for pair (i, j): b[p[j], p[j]] - b[p[i], p[j]].
        # In matrix form, row i, col j:
        # b[p[j], p[j]] is the j-th element of b_pp.
        # b[p[i], p[j]] is b_pp_mat[i, j].
        # So Term B = - a * (b_pp[None, :] - b_pp_mat)
        
        b_pp = b_diag[p]
        b_pp_mat = b[np.ix_(p, p)]
        Term_B = -a * (b_pp[None, :] - b_pp_mat)
        
        # Term C: Q[j, p[i]] - Q[j, p[j]]
        # Q[j, p[i]] is Q_p.T? 
        # Q_p[j, i] = Q[j, p[i]].
        # Q[j, p[j]] is Q_p[j, j].
        # So Term C = Q_p.T - Q_p.diagonal()[None, :]
        
        Term_C = Q_p.T - Q_p.diagonal()[None, :]
        
        # Term D: a * (b[p[j], p[i]] - b[p[i], p[i]])
        # b[p[j], p[i]] is b_pp_mat.T[i, j] = b_pp_mat[j, i].
        # b[p[i], p[i]] is b_pp[i].
        # So Term D = a * (b_pp_mat.T - b_pp[:, None])
        
        Term_D = a * (b_pp_mat.T - b_pp[:, None])
        
        Delta = Term_A + Term_B + Term_C + Term_D
        
        # Find the best swap (minimum delta)
        # We only care about i < j.
        # Set diagonal to inf to avoid self-swaps.
        np.fill_diagonal(Delta, np.inf)
        
        # We can pick the global best swap.
        # But doing this every iteration is O(n^2) memory and compute.
        # n=150. 22500 elements. Very fast.
        
        # To diversify, we can pick a random swap with probability, or the best one.
        # Let's try "Best Improvement" first.
        
        # Flatten and find min
        # We need to map flat index back to (i, j).
        # Or just use argmin on the upper triangle.
        
        # Mask lower triangle
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        Delta_masked = np.where(mask, Delta, np.inf)
        
        min_idx = np.unravel_index(np.argmin(Delta_masked), Delta.shape)
        i, j = min_idx
        min_delta = Delta[i, j]
        
        if min_delta < 0:
            # Accept the best improvement
            p[i], p[j] = p[j], p[i]
            cost += min_delta
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
        else:
            # No improvement found.
            # Perturb: pick a random swap
            # Or accept a worse swap with probability exp(-delta/T)
            
            # Pick a few random swaps and check
            # To avoid getting stuck, let's pick a random swap
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            while i == j:
                j = random.randint(0, n - 1)
            
            delta = Delta[i, j]
            
            # Simulated Annealing acceptance
            if delta < 0 or random.random() < np.exp(-delta / T):
                p[i], p[j] = p[j], p[i]
                cost += delta
                if cost < best_cost:
                    best_cost = cost
                    best_p = p.copy()
            
            # Cool down
            T *= cooling
            if T < 1e-6:
                T = 1e-6

    return best_p.tolist()

# EVOLVE-BLOCK-END
