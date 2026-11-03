"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree


def _qap_cost(a, b, p):
    """Compute QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))


def _compute_swap_deltas(a, b, p):
    """
    Compute the cost delta for every pair of swaps (i, j) with i > j.
    Returns a matrix D of shape (n, n) where D[i, j] is the delta for swapping p[i] and p[j].
    D[i, i] = 0.
    """
    n = len(p)
    # c[i, k] = sum_l a[i, l] * b[p[k], l]
    # This is (a @ b.T)[i, p[k]]
    # Let M = a @ b.T. Then c[i, k] = M[i, p[k]].
    # c = M[:, p]
    M = a @ b.T
    c = M[:, p]
    
    # delta[i, j] = c[i, j] - c[i, i] - c[j, j] + c[j, i]
    # Let's verify this formula.
    # Cost = sum_{i,j} a[i,j] b[p[i], p[j]]
    # Swap i and j. Let x = p[i], y = p[j].
    # New cost terms involving i or j:
    # Old:
    # a[i,i]b[x,x] + a[i,j]b[x,y] + sum_{k!=i,j} a[i,k]b[x,p[k]]
    # a[j,i]b[y,x] + a[j,j]b[y,y] + sum_{k!=i,j} a[j,k]b[y,p[k]]
    # sum_{k!=i,j} a[k,i]b[p[k],x] + sum_{k!=i,j} a[k,j]b[p[k],y]
    # 
    # New (swap x and y):
    # a[i,i]b[y,y] + a[i,j]b[y,x] + sum_{k!=i,j} a[i,k]b[y,p[k]]
    # a[j,i]b[x,y] + a[j,j]b[x,x] + sum_{k!=i,j} a[j,k]b[x,p[k]]
    # sum_{k!=i,j} a[k,i]b[p[k],y] + sum_{k!=i,j} a[k,j]b[p[k],x]
    #
    # Delta = New - Old
    # = a[i,i](b[y,y]-b[x,x]) + a[i,j](b[y,x]-b[x,y])
    #   + a[j,i](b[x,y]-b[y,x]) + a[j,j](b[x,x]-b[y,y])
    #   + sum_{k!=i,j} a[i,k](b[y,p[k]]-b[x,p[k]])
    #   + sum_{k!=i,j} a[j,k](b[x,p[k]]-b[y,p[k]])
    #   + sum_{k!=i,j} a[k,i](b[p[k],y]-b[p[k],x])
    #   + sum_{k!=i,j} a[k,j](b[p[k],x]-b[p[k],y])
    #
    # Let's look at c[i, j] - c[i, i] - c[j, j] + c[j, i]
    # c[i, j] = sum_l a[i, l] b[p[j], l] = sum_l a[i, l] b[y, l]
    # c[i, i] = sum_l a[i, l] b[x, l]
    # c[j, j] = sum_l a[j, l] b[y, l]
    # c[j, i] = sum_l a[j, l] b[x, l]
    #
    # c[i, j] - c[i, i] = sum_l a[i, l] (b[y, l] - b[x, l])
    #   = a[i,i](b[y,i]-b[x,i]) + a[i,j](b[y,j]-b[x,j]) + sum_{k!=i,j} a[i,k](b[y,p[k]]-b[x,p[k]])
    #   Wait, b[y, l] where l is the index in the sum.
    #   The term l=j corresponds to b[y, p[j]] = b[y,y].
    #   The term l=i corresponds to b[y, p[i]] = b[y,x].
    #   So c[i, j] - c[i, i] = a[i,i](b[y,x]-b[x,x]) + a[i,j](b[y,y]-b[x,y]) + sum_{k!=i,j} a[i,k](b[y,p[k]]-b[x,p[k]])
    #
    # Similarly, c[j, i] - c[j, j] = sum_l a[j, l] (b[x, l] - b[y, l])
    #   = a[j,i](b[x,x]-b[y,x]) + a[j,j](b[x,y]-b[y,y]) + sum_{k!=i,j} a[j,k](b[x,p[k]]-b[y,p[k]])
    #
    # Summing these:
    # (c[i,j]-c[i,i]) + (c[j,i]-c[j,j])
    # = a[i,i](b[y,x]-b[x,x]) + a[i,j](b[y,y]-b[x,y])
    #   + a[j,i](b[x,x]-b[y,x]) + a[j,j](b[x,y]-b[y,y])
    #   + sum_{k!=i,j} a[i,k](b[y,p[k]]-b[x,p[k]])
    #   + sum_{k!=i,j} a[j,k](b[x,p[k]]-b[y,p[k]])
    #
    # This matches the Delta expression derived earlier, EXCEPT for the terms:
    # sum_{k!=i,j} a[k,i](b[p[k],y]-b[p[k],x]) + sum_{k!=i,j} a[k,j](b[p[k],x]-b[p[k],y])
    #
    # Wait, is the formula c[i,j] - c[i,i] - c[j,j] + c[j,i] correct?
    # Let's check the dimensions.
    # c is n x n. c[i, j] is a scalar.
    # The formula gives a scalar.
    # Let's re-verify the delta formula from literature.
    # Standard QAP swap delta:
    # Delta(i, j) = sum_{k} a[i,k](b[p[j],p[k]] - b[p[i],p[k]]) + a[k,i](b[p[k],p[j]] - b[p[k],p[i]]) 
    #              + sum_{k} a[j,k](b[p[i],p[k]] - b[p[j],p[k]]) + a[k,j](b[p[k],p[i]] - b[p[k],p[j]])
    # Note that for k=i and k=j, terms cancel or simplify.
    # Let's split k into i, j, and others.
    # For k=i:
    # a[i,i](b[p[j],p[i]] - b[p[i],p[i]]) + a[i,i](b[p[i],p[j]] - b[p[i],p[i]])
    # = a[i,i](b[y,x] - b[x,x] + b[x,y] - b[x,x]) ... No, p[i]=x, p[j]=y.
    # Term 1 (from i): a[i,i](b[y,x] - b[x,x])
    # Term 2 (from i, second part): a[i,i](b[x,y] - b[x,x]) -- Wait, a[k,i] with k=i is a[i,i]. b[p[i], p[j]] - b[p[i], p[i]] = b[x,y] - b[x,x].
    # Term 3 (from j): a[j,i](b[x,x] - b[y,x])
    # Term 4 (from j, second part): a[i,j](b[x,x] - b[y,y]) -- Wait, a[k,j] with k=i is a[i,j]. b[p[i], p[i]] - b[p[i], p[j]] = b[x,x] - b[x,y].
    # Let's just use the matrix formulation which is cleaner.
    
    # The formula delta[i,j] = c[i,j] - c[i,i] - c[j,j] + c[j,i] is actually:
    # c[i,j] = sum_l a[i,l] b[p[j],l]
    # This represents the interaction of row i of A with row p[j] of B.
    
    # Let's test this hypothesis on a small example.
    # n=2. a=[[a11, a12],[a21, a22]], b=[[b11, b12],[b21, b22]]
    # p = [0, 1]. Cost = a11*b11 + a12*b12 + a21*b21 + a22*b22.
    # Swap 0,1 -> p' = [1, 0]. Cost = a11*b22 + a12*b21 + a21*b12 + a22*b11.
    # Delta = New - Old.
    #
    # c = a @ b.T
    # c[0,0] = a11*b11 + a12*b21
    # c[0,1] = a11*b12 + a12*b22
    # c[1,0] = a21*b11 + a22*b21
    # c[1,1] = a21*b12 + a22*b22
    #
    # Formula: c[0,1] - c[0,0] - c[1,1] + c[1,0]
    # = (a11*b12 + a12*b22) - (a11*b11 + a12*b21) - (a21*b12 + a22*b22) + (a21*b11 + a22*b21)
    # = a11(b12-b11) + a12(b22-b21) - a21(b12-b11) + a22(b21-b22)
    # = a11(b12-b11) + a12(b22-b21) - a21(b12-b11) - a22(b22-b21)
    #
    # Actual Delta:
    # (a11*b22 + a12*b21 + a21*b12 + a22*b11) - (a11*b11 + a12*b12 + a21*b21 + a22*b22)
    # = a11(b22-b11) + a12(b21-b12) + a21(b12-b21) + a22(b11-b22)
    #
    # Are they equal?
    # Formula: a11*b12 - a11*b11 + a12*b22 - a12*b21 - a21*b12 + a21*b11 - a22*b22 + a22*b21
    # Actual:   a11*b22 - a11*b11 + a12*b21 - a12*b12 + a21*b12 - a21*b21 + a22*b11 - a22*b22
    #
    # Let's check term by term.
    # a11*b22 is in Actual, not in Formula.
    # a11*b12 is in Formula, not in Actual.
    # They are NOT equal.
    
    # So the simple matrix formula is incorrect for QAP.
    # The correct delta calculation requires more care.
    
    # Let's derive it properly.
    # Delta(i, j) = sum_{k} [ a[i,k](b[p[j],p[k]] - b[p[i],p[k]]) + a[k,i](b[p[k],p[j]] - b[p[k],p[i]]) 
    #                       + a[j,k](b[p[i],p[k]] - b[p[j],p[k]]) + a[k,j](b[p[k],p[i]] - b[p[k],p[j]]) ]
    
    # Let x = p[i], y = p[j].
    # Delta = sum_k a[i,k](b[y,p[k]] - b[x,p[k]]) + a[k,i](b[p[k],y] - b[p[k],x])
    #         + a[j,k](b[x,p[k]] - b[y,p[k]]) + a[k,j](b[p[k],x] - b[p[k],y])
    
    # Group by k.
    # For k not in {i,j}:
    # Term is a[i,k](b[y,p[k]] - b[x,p[k]]) + a[k,i](b[p[k],y] - b[p[k],x])
    #        + a[j,k](b[x,p[k]] - b[y,p[k]]) + a[k,j](b[p[k],x] - b[p[k],y])
    # = (a[i,k] - a[j,k])(b[y,p[k]] - b[x,p[k]]) + (a[k,i] - a[k,j])(b[p[k],y] - b[p[k],x])
    
    # For k = i:
    # a[i,i](b[y,x] - b[x,x]) + a[i,i](b[x,y] - b[x,x])  -- Wait, p[k]=p[i]=x.
    # First part: a[i,i](b[y,x] - b[x,x])
    # Second part: a[i,i](b[x,y] - b[x,x]) -- No, a[k,i] with k=i is a[i,i]. b[p[k],y] - b[p[k],x] = b[x,y] - b[x,x].
    # Third part: a[j,i](b[x,x] - b[y,x])
    # Fourth part: a[i,j](b[x,x] - b[x,y]) -- No, a[k,j] with k=i is a[i,j]. b[p[k],x] - b[p[k],y] = b[x,x] - b[x,y].
    # Sum for k=i: a[i,i](b[y,x] - b[x,x] + b[x,y] - b[x,x]) + a[j,i](b[x,x] - b[y,x]) + a[i,j](b[x,x] - b[x,y])
    # = a[i,i](b[y,x] + b[x,y] - 2b[x,x]) + a[j,i](b[x,x] - b[y,x]) + a[i,j](b[x,x] - b[x,y])
    
    # For k = j:
    # First part: a[i,j](b[y,y] - b[x,y])
    # Second part: a[j,i](b[y,y] - b[y,x]) -- a[k,i] with k=j is a[j,i]. b[p[j],y] - b[p[j],x] = b[y,y] - b[y,x].
    # Third part: a[j,j](b[x,y] - b[y,y])
    # Fourth part: a[j,j](b[y,x] - b[y,y]) -- a[k,j] with k=j is a[j,j]. b[p[j],x] - b[p[j],y] = b[y,x] - b[y,y].
    # Sum for k=j: a[i,j](b[y,y] - b[x,y]) + a[j,i](b[y,y] - b[y,x]) + a[j,j](b[x,y] - b[y,y] + b[y,x] - b[y,y])
    # = a[i,j](b[y,y] - b[x,y]) + a[j,i](b[y,y] - b[y,x]) + a[j,j](b[x,y] + b[y,x] - 2b[y,y])
    
    # This is getting complicated to vectorize efficiently for all pairs (i,j) simultaneously.
    # However, we can compute the "interaction matrix" C where C[i,k] = sum_l a[i,l] b[p[k],l] = (a @ b.T)[i, p[k]].
    # Let M = a @ b.T. Then C = M[:, p].
    # 
    # Let's look at the term for k not in {i,j}:
    # (a[i,k] - a[j,k])(b[y,p[k]] - b[x,p[k]]) + (a[k,i] - a[k,j])(b[p[k],y] - b[p[k],x])
    # Note that b[y,p[k]] - b[x,p[k]] is not directly in C.
    # C[i,k] = sum_l a[i,l] b[p[k],l].
    # This sums over l. We need specific elements of b.
    
    # Alternative approach:
    # Compute delta for a specific pair (i,j) in O(n).
    # Since n=150, O(n^3) for full search is 3.3M ops, which is fast.
    # But we want to find the best swap quickly.
    # We can compute the delta matrix D in O(n^3) or O(n^2 * n) = O(n^3).
    # 150^3 = 3,375,000. This is very fast in numpy if vectorized properly.
    
    # Let's try to vectorize the O(n) computation for all pairs.
    # Delta(i,j) = sum_{k} [ a[i,k](b[p[j],p[k]] - b[p[i],p[k]]) + a[k,i](b[p[k],p[j]] - b[p[k],p[i]]) 
    #                       + a[j,k](b[p[i],p[k]] - b[p[j],p[k]]) + a[k,j](b[p[k],p[i]] - b[p[k],p[j]]) ]
    
    # Let's define matrices to help.
    # Let B_p = b[p, :]. So B_p[k, l] = b[p[k], l].
    # Then b[p[j], p[k]] = B_p[j, p[k]]? No.
    # b[p[j], p[k]] is element (p[j], p[k]) of b.
    # Let's stick to the indices.
    
    # We can compute the delta for all j for a fixed i in O(n^2) or O(n) with precomputation?
    # No, it depends on p[j] and p[i].
    
    # Let's just use the O(n^3) loop but vectorized over k for each pair (i,j)?
    # Or loop over i, and compute all deltas for j in vectorized fashion?
    # For fixed i:
    # Delta(i,j) = sum_k ...
    # The term depends on p[i] and p[j].
    # Let x = p[i].
    # Delta(i,j) = sum_k [ a[i,k](b[p[j],p[k]] - b[x,p[k]]) + a[k,i](b[p[k],p[j]] - b[p[k],x]) 
    #                    + a[j,k](b[x,p[k]] - b[p[j],p[k]]) + a[k,j](b[p[k],x] - b[p[k],p[j]]) ]
    
    # This is hard to vectorize over j because p[j] varies.
    
    # Let's go back to the O(n^2) swap delta computation using the "C" matrix idea but corrected.
    # The previous formula was wrong.
    # Let's try a different vectorized approach.
    # Compute the cost contribution of each pair (i,j) to the total cost.
    # Cost = sum_{i,j} a[i,j] b[p[i], p[j]]
    # Let S[i,j] = a[i,j] b[p[i], p[j]].
    # Total Cost = sum S[i,j].
    # Swap i and j.
    # New S'[i,j] = a[i,j] b[p[j], p[i]] = a[i,j] b[p[i], p[j]] (since b is symmetric? No, QAP matrices are not necessarily symmetric).
    # Wait, b is a distance matrix, so b[i,j] = b[j,i].
    # If b is symmetric, then b[p[j], p[i]] = b[p[i], p[j]].
    # So S'[i,j] = S[i,j].
    # Similarly S'[j,i] = S[j,i].
    # But S'[i,i] = a[i,i] b[p[j], p[j]] = a[i,i] b[p[j], p[j]].
    # Old S[i,i] = a[i,i] b[p[i], p[i]].
    # So diagonal terms change.
    # And cross terms with k change.
    # S'[i,k] = a[i,k] b[p[j], p[k]]. Old S[i,k] = a[i,k] b[p[i], p[k]].
    # S'[k,i] = a[k,i] b[p[k], p[j]]. Old S[k,i] = a[k,i] b[p[k], p[i]].
    # S'[j,k] = a[j,k] b[p[i], p[k]]. Old S[j,k] = a[j,k] b[p[j], p[k]].
    # S'[k,j] = a[k,j] b[p[k], p[i]]. Old S[k,j] = a[k,j] b[p[k], p[j]].
    
    # So Delta = (S'[i,i] - S[i,i]) + (S'[j,j] - S[j,j])
    #          + (S'[i,j] - S[i,j]) + (S'[j,i] - S[j,i])  <-- These are 0 if b is symmetric.
    #          + sum_{k!=i,j} (S'[i,k] - S[i,k]) + (S'[k,i] - S[k,i]) + (S'[j,k] - S[j,k]) + (S'[k,j] - S[k,j])
    
    # If b is symmetric:
    # Delta = a[i,i](b[p[j],p[j]] - b[p[i],p[i]]) + a[j,j](b[p[i],p[i]] - b[p[j],p[j]])
    #       + sum_{k!=i,j} a[i,k](b[p[j],p[k]] - b[p[i],p[k]]) + a[k,i](b[p[k],p[j]] - b[p[k],p[i]])
    #       + a[j,k](b[p[i],p[k]] - b[p[j],p[k]]) + a[k,j](b[p[k],p[i]] - b[p[k],p[j]])
    
    # This is the same as before.
    
    # Let's compute the delta matrix D in O(n^3) time, but using numpy broadcasting to speed it up.
    # n=150. n^3 = 3.4M.
    # We can create an array of shape (n, n, n) representing the terms for each (i,j,k).
    # This might use too much memory (150^3 * 8 bytes = 27 MB). This is acceptable.
    
    # Let's try to construct D[i,j] directly.
    # D[i,j] = sum_k T[i,j,k]
    # T[i,j,k] = a[i,k](b[p[j],p[k]] - b[p[i],p[k]]) + a[k,i](b[p[k],p[j]] - b[p[k],p[i]]) 
    #           + a[j,k](b[p[i],p[k]] - b[p[j],p[k]]) + a[k,j](b[p[k],p[i]] - b[p[k],p[j]])
    
    # Let's precompute:
    # B_p = b[p, :]  # Shape (n, n). B_p[k, l] = b[p[k], l].
    # Then b[p[i], p[k]] = B_p[i, p[k]].
    # b[p[j], p[k]] = B_p[j, p[k]].
    # b[p[k], p[i]] = B_p[k, p[i]].
    # b[p[k], p[j]] = B_p[k, p[j]].
    
    # So T[i,j,k] = a[i,k](B_p[j, p[k]] - B_p[i, p[k]]) + a[k,i](B_p[k, p[j]] - B_p[k, p[i]])
    #              + a[j,k](B_p[i, p[k]] - B_p[j, p[k]]) + a[k,j](B_p[k, p[i]] - B_p[k, p[j]])
    
    # This still has p[k] and p[i], p[j] inside the indices.
    # B_p[i, p[k]] is a vector over k.
    # Let V_i = B_p[i, p]  # Shape (n,). V_i[k] = B_p[i, p[k]] = b[p[i], p[k]].
    # Then B_p[i, p[k]] = V_i[k].
    # Similarly B_p[j, p[k]] = V_j[k].
    # B_p[k, p[i]] is harder. It's the k-th row of B_p, column p[i].
    # Let W_i = B_p[:, p[i]]  # Shape (n,). W_i[k] = B_p[k, p[i]] = b[p[k], p[i]].
    
    # So T[i,j,k] = a[i,k](V_j[k] - V_i[k]) + a[k,i](W_j[k] - W_i[k])
    #              + a[j,k](V_i[k] - V_j[k]) + a[k,j](W_i[k] - W_j[k])
    
    # Group terms:
    # T[i,j,k] = (a[i,k] - a[j,k])(V_j[k] - V_i[k]) + (a[k,i] - a[k,j])(W_j[k] - W_i[k])
    
    # Let A_diff[i,j,k] = a[i,k] - a[j,k].
    # Let V_diff[i,j,k] = V_j[k] - V_i[k].
    # Let A_T_diff[i,j,k] = a[k,i] - a[k,j].
    # Let W_diff[i,j,k] = W_j[k] - W_i[k].
    
    # D[i,j] = sum_k A_diff[i,j,k] * V_diff[i,j,k] + A_T_diff[i,j,k] * W_diff[i,j,k]
    
    # We can compute this for all i,j in O(n^3) using broadcasting.
    # A_diff: (n, n, n) -> a[i,k] - a[j,k].
    # a is (n, n). a[i,k] is a[i, :]. a[j,k] is a[j, :].
    # A_diff[i,j,k] = a[i,k] - a[j,k].
    # This is a[:, None, :] - a[None, :, :] ? No.
    # a[i,k] depends on i and k.
    # We want a 3D array T1[i,j,k] = a[i,k] - a[j,k].
    # T1 = a[:, None, :] - a[None, :, :]  # Shape (n, n, n).
    # T1[i,j,k] = a[i,k] - a[j,k]. Correct.
    
    # V_diff[i,j,k] = V_j[k] - V_i[k].
    # V is (n, n). V[i,k] = b[p[i], p[k]].
    # V_diff = V[None, :, :] - V[:, None, :]  # Shape (n, n, n).
    # V_diff[i,j,k] = V[j,k] - V[i,k]. Correct.
    
    # A_T_diff[i,j,k] = a[k,i] - a[k,j].
    # A_T = a.T. A_T[k,i] = a[i,k].
    # We want a[k,i] - a[k,j].
    # This is a.T[i, :] - a.T[j, :]? No.
    # a[k,i] is column i of a.
    # A_T_diff[i,j,k] = a[k,i] - a[k,j].
    # Let's use a.T. a.T[i,k] = a[k,i].
    # So A_T_diff[i,j,k] = a.T[i,k] - a.T[j,k].
    # A_T_diff = a.T[:, None, :] - a.T[None, :, :]  # Shape (n, n, n).
    # A_T_diff[i,j,k] = a.T[i,k] - a.T[j,k] = a[k,i] - a[k,j]. Correct.
    
    # W_diff[i,j,k] = W_j[k] - W_i[k].
    # W is (n, n). W[i,k] = b[p[k], p[i]].
    # Note W[i,k] = V[k,i] if b is symmetric?
    # V[k,i] = b[p[k], p[i]]. Yes.
    # So W = V.T.
    # W_diff[i,j,k] = W[j,k] - W[i,k] = V.T[j,k] - V.T[i,k] = V[k,j] - V[k,i].
    # W_diff = V.T[None, :, :] - V.T[:, None, :]  # Shape (n, n, n).
    # W_diff[i,j,k] = V.T[j,k] - V.T[i,k] = V[k,j] - V[k,i]. Correct.
    
    # So D = sum_k (T1 * V_diff + A_T_diff * W_diff)
    # D = np.sum(T1 * V_diff + A_T_diff * W_diff, axis=2)
    
    # This is O(n^3) memory and time.
    # n=150. 150^3 = 3.4M. 3.4M * 8 bytes * 4 arrays = ~100 MB.
    # This is feasible.
    
    # Let's implement this.
    
    B_p = b[p, :]
    # V[i,k] = b[p[i], p[k]] = B_p[i, p[k]]
    V = B_p[:, p]  # Shape (n, n)
    
    # T1[i,j,k] = a[i,k] - a[j,k]
    T1 = a[:, None, :] - a[None, :, :]
    
    # V_diff[i,j,k] = V[j,k] - V[i,k]
    V_diff = V[None, :, :] - V[:, None, :]
    
    # A_T_diff[i,j,k] = a[k,i] - a[k,j]
    A_T_diff = a.T[:, None, :] - a.T[None, :, :]
    
    # W_diff[i,j,k] = V[k,j] - V[k,i]
    # W_diff = V.T[None, :, :] - V.T[:, None, :]
    # Or W_diff = V[None, :, :].T ? No.
    # V[k,j] is element (k,j) of V.
    # We want an array W_diff[i,j,k] = V[k,j] - V[k,i].
    # Let's transpose V to get W. W = V.T. W[k,j] = V[j,k]. No.
    # W[i,k] = b[p[k], p[i]].
    # V[i,k] = b[p[i], p[k]].
    # If b is symmetric, V is symmetric.
    # Let's assume b is symmetric for now (distance matrix).
    # Then W = V.
    # W_diff[i,j,k] = V[k,j] - V[k,i].
    # Let's construct this.
    # V is (n, n).
    # We want W_diff[i,j,k] = V[k,j] - V[k,i].
    # This is V.T[j, :] - V.T[i, :]?
    # V.T[j,k] = V[k,j].
    # So W_diff[i,j,k] = V.T[j,k] - V.T[i,k].
    # W_diff = V.T[None, :, :] - V.T[:, None, :]
    
    W_diff = V.T[None, :, :] - V.T[:, None, :]
    
    D = np.sum(T1 * V_diff + A_T_diff * W_diff, axis=2)
    
    # Set diagonal to 0 (no swap)
    np.fill_diagonal(D, 0)
    
    return D


def _mst_initialization(a, b):
    """
    Generate an initial permutation based on the MST of the flow matrix.
    This is a heuristic to respect global structure.
    """
    n = len(a)
    # Use a as the weight matrix for MST
    # We want to minimize sum a[i,j] b[p[i], p[j]].
    # If we assume that facilities with high flow should be close in the location space,
    # we can build an MST on the flow matrix.
    
    # Create a sparse matrix from a
    # Note: a might have negative values? QAP flows are usually non-negative.
    # If not, MST is not well-defined. Let's assume non-negative.
    a_sparse = csr_matrix(a)
    
    try:
        mst = minimum_spanning_tree(a_sparse)
        mst_adj = mst.toarray()
        
        # Get the adjacency list of the MST
        # We can perform a DFS/BFS to get an ordering
        # This ordering will be our initial permutation p.
        # The idea is that connected nodes in the MST (high flow) should be adjacent in p.
        
        visited = np.zeros(n, dtype=bool)
        order = np.zeros(n, dtype=np.int64)
        idx = 0
        
        def dfs(u):
            nonlocal idx
            visited[u] = True
            order[idx] = u
            idx += 1
            for v in range(n):
                if mst_adj[u, v] > 0 and not visited[v]:
                    dfs(v)
        
        # Start from node 0
        dfs(0)
        
        # If graph is disconnected, visit remaining nodes
        for i in range(n):
            if not visited[i]:
                dfs(i)
        
        # order is a permutation of 0..n-1.
        # We need to map this to locations.
        # We can just use this order as the initial assignment p.
        # p[order[i]] = i ?
        # No, p is the location of facility i.
        # If we want facility order[0] to be at location 0, order[1] at 1, etc.
        # Then p[order[i]] = i.
        
        p = np.empty(n, dtype=np.int64)
        for i in range(n):
            p[order[i]] = i
            
        return p
        
    except Exception:
        # Fallback to rank matching
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p


def _hungarian_initialization(a, b):
    """
    Generate an initial permutation using the Hungarian algorithm on a linearized objective.
    This is the structural change requested: formulate as linear assignment.
    """
    n = len(a)
    # A common linearization is to use the row sums or a combination.
    # Let's try maximizing sum a[i,j] b[p[i], p[j]] is hard.
    # Instead, let's try to match high flow rows to low distance rows.
    # Or use the spectral approach:
    # Compute the top eigenvectors of a and b.
    # Project them and solve LAP.
    
    # Let's try the LAP on the matrix A = a @ b.T
    # This was in the inspiration.
    A = a @ b.T
    # We want to minimize sum -A[i, p[i]]
    row_ind, col_ind = linear_sum_assignment(-A)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p


def solve(a, b):
    n = len(a)
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin before 150s limit

    # 1. Generate multiple initial solutions
    candidates = []
    
    # Candidate 1: Rank Matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    candidates.append(p_rank)
    
    # Candidate 2: Hungarian on A @ B.T
    if time.time() < deadline - 20:
        try:
            p_hung = _hungarian_initialization(a, b)
            candidates.append(p_hung)
        except:
            pass
            
    # Candidate 3: MST-based
    if time.time() < deadline - 20:
        try:
            p_mst = _mst_initialization(a, b)
            candidates.append(p_mst)
        except:
            pass
            
    # Evaluate candidates and pick the best
    best_p = None
    best_cost = float('inf')
    
    for p in candidates:
        cost = _qap_cost(a, b, p)
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
            
    # 2. Local Search with Vectorized Delta
    # We use the O(n^3) vectorized delta calculation.
    # This allows us to find the best swap in O(n^3) time.
    # For n=150, this is ~3.4M ops, which is very fast (<1ms).
    # So we can do many iterations.
    
    p = best_p.copy()
    current_cost = best_cost
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # Compute all swap deltas
        D = _compute_swap_deltas(a, b, p)
        
        # Find the best swap
        min_idx = np.argmin(D)
        min_val = D.flat[min_idx]
        
        if min_val < 0:
            i_idx, j_idx = np.unravel_index(min_idx, D.shape)
            p[i_idx], p[j_idx] = p[j_idx], p[i_idx]
            current_cost += min_val
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            improved = True
        else:
            # No improving swap found. Local minimum.
            # Perform a perturbation (K-best swaps) to escape local minimum.
            # This is Iterated Local Search (ILS).
            if time.time() < deadline - 10:
                # Perturb: do k random swaps
                k = 5
                for _ in range(k):
                    i_rand = np.random.randint(n)
                    j_rand = np.random.randint(n)
                    if i_rand != j_rand:
                        p[i_rand], p[j_rand] = p[j_rand], p[i_rand]
                        
                # Restart local search from perturbed state
                # We don't update best_cost here, we just continue the loop
                # The loop will recompute deltas and find improvements.
                # But we need to ensure we don't get stuck in a bad region.
                # If the perturbed solution is worse, we might want to revert?
                # Standard ILS keeps the best solution found so far.
                # We will continue searching from the perturbed state.
                
                # To avoid infinite loops of bad perturbations, we could track the number
                # of iterations without improvement and increase k.
                # For now, simple ILS.
                
    return best_p.tolist()
# EVOLVE-BLOCK-END
