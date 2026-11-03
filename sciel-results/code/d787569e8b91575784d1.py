"""Solve QAP tai256c with 2-opt local search and multiple restarts."""
import numpy as np
import time


def solve(a, b):
    n = len(a)
    deadline = time.time() + 110
    
    def cost(p):
        return np.sum(a * b[np.ix_(p, p)])
    
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    best_p = p.copy()
    best_c = cost(p)
    
    def two_opt(p, c):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                if time.time() >= deadline:
                    return p, c, improved
                li = p[i]
                a_i = a[i, :]
                b_li = b[li, :]
                for j in range(i + 1, n):
                    lj = p[j]
                    a_j = a[j, :]
                    b_lj = b[lj, :]
                    
                    # delta = sum_{v not in {i,j}} a[i,v]*(b_lj[v]-b_li[v])
                    #       + a[i,i]*(b_lj[i]-b_li[i]) + a[i,j]*(b_lj[j]-b_li[j])  [but b_lj[j]=b[lj,p[j]]=b[lj,lj], b_li[j]=b[li,p[j]]=b[li,lj]]
                    # Wait, b_lj[v] = b[lj, p[v]], b_li[v] = b[li, p[v]]
                    # For v=i: b_lj[i] = b[lj, p[i]] = b[lj, li], b_li[i] = b[li, p[i]] = b[li, li]
                    # For v=j: b_lj[j] = b[lj, p[j]] = b[lj, lj], b_li[j] = b[li, p[j]] = b[li, lj]
                    
                    # sum_all = a_i @ (b_lj - b_li)  # sum_v a[i,v]*(b_lj[v]-b_li[v])
                    # But this includes v=i and v=j.
                    # The formula I derived has separate terms for v=i and v=j with different b values.
                    # Actually, looking at my derivation again:
                    # The first sum is over v not in {i,j}, using b[p[j],p[v]] - b[p[i],p[v]] = b_lj[v] - b_li[v]
                    # Then separate terms for v=i and v=j.
                    # But b_lj[i] = b[p[j], p[i]] = b[lj, li] and b_li[i] = b[p[i], p[i]] = b[li, li]
                    # So the v=i term in the sum would be a[i,i]*(b[lj,li] - b[li,li])
                    # But in my derivation, the v=i term is a[i,i]*(b[p[j],p[j]] - b[p[i],p[i]]) = a[i,i]*(b[lj,lj] - b[li,li])
                    # These are different! So I can't just use the sum.
                    
                    # Let me just compute it as:
                    # delta = a_i @ (b_lj - b_li)  # this gives sum_v a[i,v]*(b_lj[v]-b_li[v])
                    # But I need to adjust for v=i and v=j.
                    # For v=i: I have a[i,i]*(b_lj[i]-b_li[i]) = a[i,i]*(b[lj,li]-b[li,li]), but I need a[i,i]*(b[lj,lj]-b[li,li])
                    #   Adjustment: a[i,i]*(b[lj,lj]-b[lj,li])
                    # For v=j: I have a[i,j]*(b_lj[j]-b_li[j]) = a[i,j]*(b[lj,lj]-b[li,lj]), but I need a[i,j]*(b[lj,li]-b[li,lj])
                    #   Adjustment: a[i,j]*(b[lj,li]-b[lj,lj])
                    
                    # Similarly for the other terms. This is getting too complex.
                    
                    # Let me just compute the delta directly by summing over all k:
                    # delta = 0
                    # for k in range(n):
                    #   if k == i:
                    #     delta += a[i,i]*(b[lj,lj]-b[li,li]) + a[i,i]*(b[lj,lj]-b[li,li])  # row i, col i
                    #     ... this is too slow in Python.
                    
                    # Given the complexity, let me just compute the cost before and after for the affected pairs.
                    # The affected (u,v) pairs are those with u in {i,j} or v in {i,j}.
                    # Count: 2n + 2n - 4 = 4n - 4 (with double counting of the 4 corner pairs)
                    # Actually, the set is {(u,v): u in {i,j} or v in {i,j}}, which has size 4n - 4.
                    
                    # Let me just compute:
                    # old_cost_affected = sum of a[u,v]*b[p[u],p[v]] for affected pairs
                    # new_cost_affected = sum of a[u,v]*b[p'[u],p'[v]] for affected pairs
                    # delta = new_cost_affected - old_cost_affected
                    
                    # For n=256, this is ~1000 operations per swap. With numpy, I can do it in one shot.
                    p_new = p.copy()
                    p_new[i], p_new[j] = lj, li
                    
                    # Affected pairs: u in {i,j} or v in {i,j}
                    # I'll compute it as:
                    # sum over u in {i,j}, all v + sum over v in {i,j}, u not in {i,j}
                    old_affected = (a[i, :] * b[li, p] + a[j, :] * b[lj, p] +
                                  a[:, i] * b[p, li] + a[:, j] * b[p, lj])
                    new_affected = (a[i, :] * b[lj, p_new] + a[j, :] * b[li, p_new] +
                                  a[:, i] * b[p_new, lj] + a[:, j] * b[p_new, li])
                    # But this double-counts (i,i), (i,j), (j,i), (j,j)
                    # The actual affected set is the union, so I need to subtract the double-counted terms.
                    # Double-counted: (i,i) in row i and col i, (i,j) in row i and col j, (j,i) in row j and col i, (j,j) in row j and col j
                    # So:
                    old_affected -= (a[i,i]*b[li,li] + a[i,j]*b[li,lj] + a[j,i]*b[lj,li] + a[j,j]*b[lj,lj])
                    new_affected -= (a[i,i]*b[lj,lj] + a[i,j]*b[lj,li] + a[j,i]*b[li,lj] + a[j,j]*b[li,li])
                    
                    delta = new_affected.sum() - old_affected.sum()
                    
                    if delta < -1e-9:
                        p[i], p[j] = lj, li
                        c += delta
                        improved = True
                        break
                if improved:
                    break
        return p, c, improved
    
    while time.time() < deadline:
        p, c, _ = two_opt(p, c)
        if c < best_c - 1e-9:
            best_c = c
            best_p = p.copy()
        if time.time() >= deadline:
            break
        for _ in range(5):
            i, j = np.random.choice(n, size=2, replace=False)
            p[i], p[j] = p[j], p[i]
    
    return best_p.tolist()
