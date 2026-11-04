"""QAP tai256c solver: iterated local search with O(n) swap deltas, Hungarian polish, and adaptive diversification."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment


def solve(a, b):
    n = len(a)
    deadline = time.time() + 115.0  # safety margin under 120s

    # Precompute for fast O(n) swap delta computation
    # We maintain p where p[i] = location of facility i
    # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]

    # For O(n) delta: when swapping p[u] and p[v],
    # delta = sum_{j != u,v} a[u,j]*(b[pv,p[j]] - b[pu,p[j]])
    #       + sum_{j != u,v} a[v,j]*(b[pu,p[j]] - b[pv,p[j]])
    #       + sum_{i != u,v} a[i,u]*(b[p[i],pv] - b[p[i],pu])
    #       + sum_{i != u,v} a[i,v]*(b[p[i],pu] - b[p[i],pv])
    #       + a[u,v]*(b[pv,pu] - b[pu,pv])
    #       + a[v,u]*(b[pu,pv] - b[pv,pu])
    #       + a[u,u]*(b[pv,pv] - b[pu,pu])
    #       + a[v,v]*(b[pu,pu] - b[pv,pv])

    # We can precompute row/col contributions to make this faster.
    # Let's define:
    #   R[i] = sum_j a[i,j] * b[p[i], p[j]]  (row i contribution)
    #   C[i] = sum_j a[j,i] * b[p[j], p[i]]  (col i contribution)
    # Total cost = sum_i R[i] (since each term a[i,j]*b[p[i],p[j]] appears in R[i])
    # Actually total cost = sum_i sum_j a[i,j]*b[p[i],p[j]] = sum_i R[i]

    # When swapping p[u] and p[v]:
    # New p'[u] = pv, p'[v] = pu
    # 
    # For R[u]: was sum_j a[u,j]*b[pu, p[j]], becomes sum_j a[u,j]*b[pv, p'[j]]
    #   where p'[u]=pv, p'[v]=pu, p'[j]=p[j] for j!=u,v
    #   = a[u,u]*b[pv,pv] + a[u,v]*b[pv,pu] + sum_{j!=u,v} a[u,j]*b[pv,p[j]]
    #
    # For R[v]: was sum_j a[v,j]*b[pv, p[j]], becomes sum_j a[v,j]*b[pu, p'[j]]
    #   = a[v,u]*b[pu,pv] + a[v,v]*b[pu,pu] + sum_{j!=u,v} a[v,j]*b[pu,p[j]]
    #
    # For R[k] where k!=u,v: was sum_j a[k,j]*b[p[k],p[j]], becomes sum_j a[k,j]*b[p[k],p'[j]]
    #   = a[k,u]*b[p[k],pv] + a[k,v]*b[p[k],pu] + sum_{j!=u,v} a[k,j]*b[p[k],p[j]]
    #   Difference: a[k,u]*(b[p[k],pv]-b[p[k],pu]) + a[k,v]*(b[p[k],pu]-b[p[k],pv])
    #              = (a[k,u]-a[k,v])*(b[p[k],pv]-b[p[k],pu])

    # So total delta = (new R[u] - old R[u]) + (new R[v] - old R[v]) + sum_{k!=u,v} delta_R[k]
    
    # Let me precompute:
    #   For each location l, define:
    #   row_b[l] = b[l, :]  (row l of b)
    #   col_b[l] = b[:, l]  (col l of b)

    def compute_cost(p):
        return np.sum(a * b[np.ix_(p, p)])

    def two_opt_pass(p, cost, deadline):
        """One pass of 2-opt local search with O(n) delta computation."""
        n = len(p)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Randomize the order to avoid cycling
            order = list(range(n))
            random.shuffle(order)
            for idx_i in order:
                i = order[idx_i]
                if time.time() > deadline:
                    break
                for idx_j in range(idx_i + 1, n):
                    j = order[idx_j]
                    if time.time() > deadline:
                        break
                    pu = p[i]
                    pv = p[j]
                    
                    # Compute delta for swapping p[i] and p[j]
                    # delta_R[i] = a[i,i]*(b[pv,pv]-b[pu,pu]) + a[i,v]*(b[pv,pu]-b[pu,pv]) 
                    #            + sum_{k!=i,j} a[i,k]*(b[pv,p[k]]-b[pu,p[k]])
                    # delta_R[j] = a[j,i]*(b[pu,pv]-b[pv,pu]) + a[j,j]*(b[pu,pu]-b[pv,pv])
                    #            + sum_{k!=i,j} a[j,k]*(b[pu,p[k]]-b[pv,p[k]])
                    # delta_R[k] for k!=i,j = (a[k,i]-a[k,j])*(b[p[k],pv]-b[p[k],pu])
                    
                    # Vectorized computation:
                    # diff_b = b[pv, :] - b[pu, :]  (shape n)
                    # But we need b[pv, p[k]] - b[pu, p[k]] for each k, which is diff_b[p[k]]
                    
                    diff = b[pv, :] - b[pu, :]  # shape (n,)
                    diff_at_p = diff[p]  # shape (n,), diff_at_p[k] = b[pv,p[k]] - b[pu,p[k]]
                    
                    # delta_R[i] = a[i,i]*(b[pv,pv]-b[pu,pu]) + a[i,j]*(b[pv,pu]-b[pu,pv]) + sum_{k!=i,j} a[i,k]*diff_at_p[k]
                    # Note: diff_at_p[i] = b[pv,pu]-b[pu,pu], diff_at_p[j] = b[pv,pv]-b[pu,pv]
                    # So sum_{k} a[i,k]*diff_at_p[k] already includes k=i and k=j
                    sum_i = np.dot(a[i, :], diff_at_p)
                    # But we need to be careful: the formula above for delta_R[i] uses the new p'
                    # Let me re-derive more carefully.
                    
                    # Old R[i] = sum_k a[i,k] * b[pu, p[k]]
                    # New R[i] = sum_k a[i,k] * b[pv, p'[k]] where p'[i]=pv, p'[j]=pu, p'[k]=p[k] for k!=i,j
                    # = a[i,i]*b[pv,pv] + a[i,j]*b[pv,pu] + sum_{k!=i,j} a[i,k]*b[pv,p[k]]
                    
                    # Old R[i] = a[i,i]*b[pu,pu] + a[i,j]*b[pu,pv] + sum_{k!=i,j} a[i,k]*b[pu,p[k]]
                    
                    # delta_R[i] = a[i,i]*(b[pv,pv]-b[pu,pu]) + a[i,j]*(b[pv,pu]-b[pu,pv]) + sum_{k!=i,j} a[i,k]*(b[pv,p[k]]-b[pu,p[k]])
                    
                    # sum_{k} a[i,k]*(b[pv,p[k]]-b[pu,p[k]]) = sum_{k} a[i,k]*diff_at_p[k]
                    # = a[i,i]*diff_at_p[i] + a[i,j]*diff_at_p[j] + sum_{k!=i,j} a[i,k]*diff_at_p[k]
                    # = a[i,i]*(b[pv,pu]-b[pu,pu]) + a[i,j]*(b[pv,pv]-b[pu,pv]) + sum_{k!=i,j} a[i,k]*diff_at_p[k]
                    
                    # So delta_R[i] = a[i,i]*(b[pv,pv]-b[pu,pu]) + a[i,j]*(b[pv,pu]-b[pu,pv]) + sum_{k!=i,j} a[i,k]*diff_at_p[k]
                    # = [a[i,i]*(b[pv,pu]-b[pu,pu]) + a[i,j]*(b[pv,pv]-b[pu,pv]) + sum_{k!=i,j} a[i,k]*diff_at_p[k]] 
                    #   + a[i,i]*(b[pv,pv]-b[pv,pu]) + a[i,j]*(b[pv,pu]-b[pv,pv])
                    # = sum_k a[i,k]*diff_at_p[k] + a[i,i]*(b[pv,pv]-b[pv,pu]) + a[i,j]*(b[pv,pu]-b[pv,pv])
                    # = sum_k a[i,k]*diff_at_p[k] + a[i,i]*(b[pv,pv]-b[pv,pu]) + a[i,j]*(b[pv,pu]-b[pv,pv])
                    
                    # Similarly for delta_R[j]:
                    # Old R[j] = sum_k a[j,k]*b[pv,p[k]]
                    # New R[j] = a[j,i]*b[pu,pv] + a[j,j]*b[pu,pu] + sum_{k!=i,j} a[j,k]*b[pu,p[k]]
                    # delta_R[j] = a[j,i]*(b[pu,pv]-b[pv,pu]) + a[j,j]*(b[pu,pu]-b[pv,pv]) + sum_{k!=i,j} a[j,k]*(b[pu,p[k]]-b[pv,p[k]])
                    # = -sum_k a[j,k]*diff_at_p[k] + a[j,i]*(b[pu,pv]-b[pu,pu]) + a[j,j]*(b[pu,pv]-b[pu,pu])
                    # Wait, let me redo this.
                    # diff_at_p[k] = b[pv,p[k]] - b[pu,p[k]]
                    # So b[pu,p[k]] - b[pv,p[k]] = -diff_at_p[k]
                    # delta_R[j] = a[j,i]*(b[pu,pv]-b[pv,pu]) + a[j,j]*(b[pu,pu]-b[pv,pv]) + sum_{k!=i,j} a[j,k]*(-diff_at_p[k])
                    # = -sum_{k!=i,j} a[j,k]*diff_at_p[k] + a[j,i]*(b[pu,pv]-b[pv,pu]) + a[j,j]*(b[pu,pu]-b[pv,pv])
                    # = -[sum_k a[j,k]*diff_at_p[k] - a[j,i]*diff_at_p[i] - a[j,j]*diff_at_p[j]] + a[j,i]*(b[pu,pv]-b[pv,pu]) + a[j,j]*(b[pu,pu]-b[pv,pv])
                    # = -sum_k a[j,k]*diff_at_p[k] + a[j,i]*diff_at_p[i] + a[j,j]*diff_at_p[j] + a[j,i]*(b[pu,pv]-b[pv,pu]) + a[j,j]*(b[pu,pu]-b[pv,pv])
                    # diff_at_p[i] = b[pv,pu]-b[pu,pu], diff_at_p[j] = b[pv,pv]-b[pu,pv]
                    # = -sum_k a[j,k]*diff_at_p[k] + a[j,i]*(b[pv,pu]-b[pu,pu]) + a[j,j]*(b[pv,pv]-b[pu,pv]) + a[j,i]*(b[pu,pv]-b[pv,pu]) + a[j,j]*(b[pu,pu]-b[pv,pv])
                    # = -sum_k a[j,k]*diff_at_p[k] + a[j,i]*0 + a[j,j]*0
                    # = -sum_k a[j,k]*diff_at_p[k]
                    
                    # Hmm, that's suspiciously clean. Let me verify with a small example.
                    # Actually, I think there's an error. Let me just compute it directly.
                    
                    # For delta_R[k] where k != i,j:
                    # delta_R[k] = a[k,i]*(b[p[k],pv]-b[p[k],pu]) + a[k,j]*(b[p[k],pu]-b[k],pv])
                    # = (a[k,i]-a[k,j])*(b[p[k],pv]-b[p[k],pu])
                    # = (a[k,i]-a[k,j])*diff_col[k] where diff_col[k] = b[p[k],pv]-b[p[k],pu]
                    
                    # Let diff_col = b[:, pv] - b[:, pu]  (shape n)
                    # diff_col_at_p[k] = b[p[k],pv] - b[p[k],pu] = diff_col[p[k]]
                    
                    diff_col = b[:, pv] - b[:, pu]  # shape (n,)
                    diff_col_at_p = diff_col[p]  # shape (n,)
                    
                    # delta_R[k] for k!=i,j = (a[k,i]-a[k,j]) * diff_col_at_p[k]
                    # sum_{k!=i,j} delta_R[k] = sum_{k} (a[k,i]-a[k,j])*diff_col_at_p[k] - (a[i,i]-a[i,j])*diff_col_at_p[i] - (a[j,i]-a[j,j])*diff_col_at_p[j]
                    
                    sum_k_delta = np.dot(a[:, i] - a[:, j], diff_col_at_p)
                    sum_k_delta -= (a[i, i] - a[i, j]) * diff_col_at_p[i]
                    sum_k_delta -= (a[j, i] - a[j, j]) * diff_col_at_p[j]
                    
                    # Now for delta_R[i] and delta_R[j], let me just compute them directly:
                    # delta_R[i] = new_R[i] - old_R[i]
                    # old_R[i] = np.dot(a[i, :], b[pu, p])
                    # new_R[i] = a[i,i]*b[pv,pv] + a[i,j]*b[pv,pu] + np.dot(a[i, [k for k in range(n) if k!=i and k!=j]], b[pv, p[[k for k in range(n) if k!=i and k!=j]]])
                    # This is getting complicated. Let me just use the full O(n) computation.
                    
                    # Actually, let me just compute the delta by evaluating the affected terms directly.
                    # The total cost change is:
                    # delta = sum_{i',j'} a[i',j']*(b[p'[i'],p'[j']] - b[p[i'],p[j']])
                    # Only terms where i' or j' is i or j change.
                    
                    # Let me compute it as:
                    # delta = (new_R[i] - old_R[i]) + (new_R[j] - old_R[j]) + sum_{k!=i,j} (new_R[k] - old_R[k])
                    
                    # old_R[i] = np.dot(a[i, :], b[pu, p])
                    # new_R[i] = a[i,i]*b[pv,pv] + a[i,j]*b[pv,pu] + sum_{k!=i,j} a[i,k]*b[pv,p[k]]
                    #          = np.dot(a[i, :], b[pv, p]) - a[i,i]*b[pv,pu] - a[i,j]*b[pv,pv] + a[i,i]*b[pv,pv] + a[i,j]*b[pv,pu]
                    #          = np.dot(a[i, :], b[pv, p]) + a[i,i]*(b[pv,pv]-b[pv,pu]) + a[i,j]*(b[pv,pu]-b[pv,pv])
                    
                    # old_R[i] = np.dot(a[i, :], b[pu, p])
                    
                    # delta_R[i] = np.dot(a[i, :], b[pv, p]) - np.dot(a[i, :], b[pu, p]) + a[i,i]*(b[pv,pv]-b[pv,pu]) + a[i,j]*(b[pv,pu]-b[pv,pv])
                    #            = np.dot(a[i, :], diff) + a[i,i]*(b[pv,pv]-b[pv,pu]) + a[i,j]*(b[pv,pu]-b[pv,pv])
                    
                    # Similarly:
                    # delta_R[j] = np.dot(a[j, :], b[pu, p]) - np.dot(a[j, :], b[pv, p]) + a[j,i]*(b[pu,pv]-b[pu,pu]) + a[j,j]*(b[pu,pu]-b[pu,pv])
                    #            = -np.dot(a[j, :], diff) + a[j,i]*(b[pu,pv]-b[pu,pu]) + a[j,j]*(b[pu,pu]-b[pu,pv])
                    
                    delta_R_i = np.dot(a[i, :], diff) + a[i, i] * (b[pv, pv] - b[pv, pu]) + a[i, j] * (b[pv, pu] - b[pv, pv])
                    delta_R_j = -np.dot(a[j, :], diff) + a[j, i] * (b[pu, pv] - b[pu, pu]) + a[j, j] * (b[pu, pu] - b[pu, pv])
                    
                    total_delta = delta_R_i + delta_R_j + sum_k_delta
                    
                    if total_delta < 0:
                        p[i], p[j] = p[j], p[i]
                        cost += total_delta
                        improved = True
                        if time.time() > deadline:
                            break
        return p, cost

    def hungarian_reassign(p):
        """Use Hungarian algorithm to find a better assignment."""
        C = np.zeros((n, n), dtype=np.int64)
        for i in range(n):
            C[:, i] += b @ a[:, i]
            C[i, :] += a[i, :] @ b
            C[i, i] -= a[i, i] * b[i, i]
        row_ind, col_ind = linear_sum_assignment(C)
        new_p = np.empty(n, dtype=np.int64)
        new_p[col_ind] = row_ind
        return new_p

    # Initial solution: rank matching
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.astype(np.int64)
    
    best_p = p.copy()
    best_cost = compute_cost(best_p)
    
    # Local search
    p, cost = two_opt_pass(p.copy(), best_cost, deadline)
    if cost < best_cost:
        best_p = p.copy()
        best_cost = cost
    
    # Iterated local search
    no_improve = 0
    max_no_improve = 50
    while time.time() < deadline - 10.0:
        # Perturb
        p = best_p.copy()
        n_swaps = random.randint(2, 5)
        for _ in range(n_swaps):
            i, j = random.sample(range(n), 2)
            p[i], p[j] = p[j], p[i]
        
        # Local search
        p, cost = two_opt_pass(p, compute_cost(p), deadline)
        
        if cost < best_cost:
            best_p = p.copy()
            best_cost = cost
            no_improve = 0
        else:
            no_improve += 1
        
        # Hungarian reassignment periodically
        if no_improve >= max_no_improve or random.random() < 0.1:
            if time.time() < deadline - 5.0:
                new_p = hungarian_reassign(p)
                new_cost = compute_cost(new_p)
                if new_cost < best_cost:
                    best_p = new_p.copy()
                    best_cost = new_cost
                    no_improve = 0
                    # Local search from Hungarian solution
                    p, cost = two_opt_pass(new_p.copy(), best_cost, deadline)
                    if cost < best_cost:
                        best_p = p.copy()
                        best_cost = cost
                no_improve = 0
    
    # Final polish: multiple 2-opt passes
    if time.time() < deadline - 2.0:
        p, cost = two_opt_pass(best_p.copy(), best_cost, deadline)
        if cost < best_cost:
            best_p = p.copy()
            best_cost = cost
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
