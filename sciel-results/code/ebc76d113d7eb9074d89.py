# EVOLVE-BLOCK-START
import numpy as np
import time

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    """
    deadline = time.time() + 590  # Safety margin before 600s limit

    def compute_P(support, weights):
        """Compute P[X1+X2+X3 < 2*X4] exactly."""
        n = len(support)
        # Weights are already normalized to sum to 1
        # P = sum_{k,l} w_k w_l * (sum_{i,j: s_i+s_j < 2*s_l - s_k} w_i w_j)
        # Let S_ij = s_i + s_j. Let T_kl = 2*s_l - s_k.
        # We need sum over all i,j,k,l of w_i w_j w_k w_l * [S_ij < T_kl]
        
        # Vectorized approach:
        # Create matrices S (n x n) where S[i,j] = s_i + s_j
        # Create matrix T (n x n) where T[k,l] = 2*s_l - s_k
        # For each pair (k,l), we need the sum of w_i w_j over i,j such that S[i,j] < T[k,l]
        
        # This is O(n^4) if done naively, but we can sort S and use prefix sums of w_i w_j?
        # No, because the condition is S_ij < T_kl, and the weight associated with (i,j) is w_i w_j.
        # So for a fixed threshold t, let F(t) = sum_{i,j: s_i+s_j < t} w_i w_j.
        # Then P = sum_{k,l} w_k w_l F(2*s_l - s_k).
        
        # Steps:
        # 1. Compute all pairwise sums S_ij = s_i + s_j. (n^2 values)
        # 2. Compute all pairwise products W_ij = w_i w_j. (n^2 values)
        # 3. Sort the S_ij values. Let sorted_S be the sorted array, and perm be the indices.
        #    Then sorted_W = W_ij[perm].
        # 4. Compute cumulative sum of sorted_W: CUM[i] = sum_{m=0}^i sorted_W[m].
        # 5. For each k, l, compute t = 2*s_l - s_k.
        #    Find the number of elements in sorted_S strictly less than t.
        #    Let idx = np.searchsorted(sorted_S, t, side='left').
        #    Then F(t) = CUM[idx-1] if idx > 0 else 0.
        # 6. Sum w_k w_l F(t) over all k, l.
        
        S = support[:, None] + support[None, :]
        W = weights[:, None] * weights[None, :]
        
        # Flatten and sort
        flat_S = S.flatten()
        flat_W = W.flatten()
        
        # Sort S, and reorder W accordingly
        order = np.argsort(flat_S, kind='mergesort')
        sorted_S = flat_S[order]
        sorted_W = flat_W[order]
        
        # Cumulative sum
        cum_W = np.cumsum(sorted_W)
        
        # Prepend 0 for easy indexing
        cum_W = np.concatenate(([0.0], cum_W))
        
        # Compute T_kl = 2*s_l - s_k for all k, l
        # T[k, l] = 2*s_l - s_k
        T = 2.0 * support[None, :] - support[:, None]
        
        # For each t in T, find F(t)
        # searchsorted(sorted_S, t, side='left') gives the index of the first element >= t.
        # So the number of elements < t is that index.
        # F(t) = cum_W[index]
        
        flat_T = T.flatten()
        indices = np.searchsorted(sorted_S, flat_T, side='left')
        
        # F_vals = cum_W[indices]
        # Note: cum_W has length n^2 + 1. indices are in [0, n^2].
        F_vals = cum_W[indices]
        
        # Now sum w_k w_l F_vals[k*n + l]
        # W_T = weights[:, None] * weights[None, :]  # This is W[k,l]
        # P = sum_{k,l} W[k,l] * F_vals[k*n+l]
        # F_vals is flattened T, so F_vals.reshape(n,n) is F[k,l]
        F_matrix = F_vals.reshape(n, n)
        P = np.sum(W * F_matrix)
        
        return P

    # Initial support: geometric spacing to cover a wide range efficiently?
    # Or just linear?
    # Let's try a mix.
    # The optimal distribution likely has mass at 0 and some larger values.
    # Let's start with a support that is logarithmically spaced or geometric.
    
    # Strategy:
    # 1. Define a support grid.
    # 2. Optimize weights using gradient ascent (since P is linear in weights? No, P is quadratic in weights? 
    #    Wait, P = sum w_i w_j w_k w_l ... it's 4-linear (multi-linear).
    #    Actually, for fixed support, P is a polynomial of degree 4 in weights.
    #    However, it is linear in each weight vector if we consider the tensor?
    #    No, it's a 4th order tensor contraction.
    #    But we can use coordinate ascent or just simple gradient ascent.
    #    Gradient of P w.r.t w_a:
    #    P = sum_{i,j,k,l} w_i w_j w_k w_l I(s_i+s_j < 2s_l - s_k)
    #    dP/dw_a = sum_{i,j,k,l} [ (delta_{a,i} + delta_{a,j} + delta_{a,k} + delta_{a,l}) ] w_i w_j w_k w_l I(...)
    #    This can be computed efficiently.
    
    # Let's implement gradient ascent.
    
    # Support choice:
    # Let's use a support that includes 0 and some positive numbers.
    # Maybe s_i = i * step?
    # Or s_i = c * i?
    # Let's try n=50 support points.
    
    n = 40
    # Support: 0, 1, 2, ..., n-1?
    # Or maybe spread out more?
    # If we use large values, the condition s_i+s_j < 2s_l - s_k becomes easier to satisfy for large s_l?
    # Let's stick to integers for now, but maybe scale them?
    # The problem is scale invariant?
    # If we scale support by c, s_i -> c s_i.
    # Condition: c s_i + c s_j < 2 c s_l - c s_k => s_i + s_j < 2 s_l - s_k.
    # So the probability P is invariant under scaling of the support!
    # This is a crucial observation. We can fix the support scale arbitrarily.
    # So we can just use integers 0..n-1.
    
    support = np.arange(n, dtype=np.float64)
    
    # Initial weights: uniform
    weights = np.ones(n, dtype=np.float64) / n
    
    # Optimization: Gradient Ascent
    # We want to maximize P.
    # Constraint: sum w = 1, w >= 0.
    # We can project back to simplex after each step.
    
    def gradient_P(support, weights):
        n = len(support)
        S = support[:, None] + support[None, :]
        W = weights[:, None] * weights[None, :]
        
        flat_S = S.flatten()
        flat_W = W.flatten()
        
        order = np.argsort(flat_S, kind='mergesort')
        sorted_S = flat_S[order]
        sorted_W = flat_W[order]
        
        cum_W = np.concatenate(([0.0], np.cumsum(sorted_W)))
        
        T = 2.0 * support[None, :] - support[:, None]
        flat_T = T.flatten()
        indices = np.searchsorted(sorted_S, flat_T, side='left')
        F_vals = cum_W[indices]
        F_matrix = F_vals.reshape(n, n)
        
        # Gradient computation
        # dP/dw_a = sum_{i,j,k,l} (delta_{a,i} + delta_{a,j} + delta_{a,k} + delta_{a,l}) w_i w_j w_k w_l I_{ij,kl}
        # where I_{ij,kl} is 1 if s_i+s_j < 2s_l - s_k.
        
        # Let's compute this efficiently.
        # Term 1 (delta_{a,i}): sum_{i=a, j,k,l} w_i w_j w_k w_l I_{ij,kl}
        # = sum_{j,k,l} w_a w_j w_k w_l I_{aj,kl}
        # = w_a * sum_{j,k,l} w_j w_k w_l I_{aj,kl}
        
        # We can precompute for each i, the sum over j,k,l of w_j w_k w_l I_{ij,kl}.
        # Let G[i] = sum_{j,k,l} w_j w_k w_l I_{ij,kl}
        # Then dP/dw_a = w_a * G[a] + (sum_{j,k,l} w_j w_k w_l I_{a j, kl} * 0? No)
        # Wait, the derivative is:
        # dP/dw_a = sum_{j,k,l} w_a w_j w_k w_l I_{a j, kl}  (from i=a)
        #          + sum_{i,k,l} w_i w_a w_k w_l I_{i a, kl}  (from j=a)
        #          + sum_{i,j,l} w_i w_j w_a w_l I_{i j, a l}  (from k=a)
        #          + sum_{i,j,k} w_i w_j w_k w_a I_{i j, k a}  (from l=a)
        
        # Note that due to symmetry of the condition?
        # I_{ij,kl} is not symmetric in (i,j) and (k,l).
        # But I_{ij,kl} = I_{ji,kl}.
        
        # Let's compute the four terms.
        # Term 1: i=a.
        # Sum_{j,k,l} w_j w_k w_l I_{a j, k l}
        # For fixed a, j: we need sum_{k,l} w_k w_l I_{a j, k l}.
        # This is exactly F(2*s_l - s_k) summed over k,l with weights w_k w_l?
        # No, F(t) was sum_{i',j': s_i'+s_j' < t} w_i' w_j'.
        # Here we have sum_{k,l} w_k w_l [s_a + s_j < 2 s_l - s_k].
        # Let t = s_a + s_j + s_k. Condition: s_a + s_j + s_k < 2 s_l.
        # This is different from the previous F.
        
        # Let's just compute the gradient directly using the sorted arrays logic, but for each a.
        # This might be O(n^3 log n) or similar.
        # With n=40, n^3 = 64000. This is fast.
        
        grad = np.zeros(n)
        
        # We need to compute sums of form:
        # S1[a] = sum_{j,k,l} w_j w_k w_l I(s_a + s_j < 2 s_l - s_k)
        # S2[a] = sum_{i,k,l} w_i w_k w_l I(s_i + s_a < 2 s_l - s_k)  (Same as S1[a] by symmetry of i,j in LHS sum? 
        #        The condition is s_i + s_j < ... . The sum is over i,j. 
        #        In S1, the fixed index is 'a' in the first position of the LHS pair (i,j).
        #        In S2, the fixed index is 'a' in the second position of the LHS pair (i,j).
        #        Since the set of pairs (i,j) is symmetric, and weights are symmetric w_i w_j, 
        #        S1[a] == S2[a].
        
        # So dP/dw_a = 2 * w_a * S1[a] + S3[a] + S4[a]
        
        # S3[a] = sum_{i,j,l} w_i w_j w_l I(s_i + s_j < 2 s_l - s_a)
        # S4[a] = sum_{i,j,k} w_i w_j w_k I(s_i + s_j < 2 s_a - s_k)
        
        # Let's compute S1, S3, S4.
        
        # S1[a]:
        # For each a, sum over j of w_j * (sum_{k,l} w_k w_l I(s_a + s_j < 2 s_l - s_k))
        # Let H[x, y] = sum_{k,l} w_k w_l I(x + y < 2 s_l - s_k) ? No, x is s_a, y is s_j.
        # Condition: s_a + s_j + s_k < 2 s_l.
        
        # Let's precompute for all possible values of s_a + s_j + s_k?
        # This seems complex. Let's just loop over a, j, k, l?
        # n=40. 40^4 = 2.56 million. In Python, this is too slow (seconds).
        # We need vectorization.
        
        # Let's use the sorted sum technique again.
        # For S3[a]: Condition s_i + s_j < 2 s_l - s_a.
        # Let t_a,l = 2 s_l - s_a.
        # We need sum_{l} w_l * (sum_{i,j,k: s_i+s_j < t_a,l} w_i w_j w_k ? No, k is not in the condition for S3?
        # S3[a] = sum_{i,j,l} w_i w_j w_l I(s_i + s_j < 2 s_l - s_a).
        # Here k is not involved! The term was from k=a in the original 4-sum.
        # Original: sum_{i,j,k,l} w_i w_j w_k w_l I(...).
        # Term k=a: sum_{i,j,l} w_i w_j w_a w_l I(s_i + s_j < 2 s_l - s_a).
        # = w_a * sum_{i,j,l} w_i w_j w_l I(s_i + s_j < 2 s_l - s_a).
        
        # So S3[a] = w_a * sum_{l} w_l F(2 s_l - s_a), where F(t) = sum_{i,j} w_i w_j I(s_i+s_j < t).
        # We already have F(t) computed via sorted_S!
        # So S3[a] is easy.
        
        # Term 4: l=a.
        # S4[a] = sum_{i,j,k} w_i w_j w_k I(s_i + s_j < 2 s_a - s_k).
        # = sum_{k} w_k (sum_{i,j} w_i w_j I(s_i + s_j < 2 s_a - s_k)).
        # This is also easy using F(t).
        # S4[a] = sum_{k} w_k F(2 s_a - s_k).
        
        # Now Term 1 & 2 (i=a or j=a):
        # S1[a] = sum_{j,k,l} w_j w_k w_l I(s_a + s_j < 2 s_l - s_k).
        # = sum_{j} w_j (sum_{k,l} w_k w_l I(s_a + s_j < 2 s_l - s_k)).
        # Let G[a, j] = sum_{k,l} w_k w_l I(s_a + s_j + s_k < 2 s_l).
        # This requires summing over k, l.
        # Condition: s_l > (s_a + s_j + s_k) / 2.
        
        # We can compute G[a, j] for all a, j.
        # n=40. 1600 values.
        # For each (a,j), we need sum_{k,l} w_k w_l I(2 s_l > s_a + s_j + s_k).
        # Let's precompute this.
        
        # Precompute G_matrix (n x n)
        G_matrix = np.zeros((n, n))
        for a in range(n):
            for j in range(n):
                # Sum over k, l
                # For each k, find number of l such that 2 s_l > s_a + s_j + s_k
                # Then sum w_l for those l, multiply by w_k.
                # Since support is sorted, we can use searchsorted.
                # 2 s_l > val  => s_l > val / 2.
                # idx = searchsorted(support, val/2, side='right')
                # sum_w_l = sum(weights[idx:])
                # But we need sum_{l} w_l.
                # Precompute suffix sums of weights.
                pass
        
        # Let's do it vectorized.
        # Suffix sums of weights
        w_suffix = np.cumsum(weights[::-1])[::-1]
        # w_suffix[i] = sum(weights[i:])
        
        for a in range(n):
            for j in range(n):
                # vals = s_a + s_j + support  (length n, for k)
                vals = support[a] + support[j] + support
                # thresholds = vals / 2
                thresholds = vals / 2.0
                # indices = searchsorted(support, thresholds, side='right')
                indices = np.searchsorted(support, thresholds, side='right')
                # sum_w_l = w_suffix[indices]
                # Note: if indices[i] == n, sum is 0. w_suffix[n] should be 0.
                # My w_suffix has length n. w_suffix[i] is sum from i to n-1.
                # We need an array of length n+1 for safety, or handle out of bounds.
                # Let's make w_suffix length n+1.
                pass
        
        # Re-do suffix sums properly
        w_suffix = np.zeros(n + 1)
        w_suffix[:n] = np.cumsum(weights[::-1])[::-1]
        # w_suffix[i] = sum(weights[i:n])
        
        G_matrix = np.zeros((n, n))
        for a in range(n):
            for j in range(n):
                vals = support[a] + support[j] + support
                thresholds = vals / 2.0
                indices = np.searchsorted(support, thresholds, side='right')
                # Clamp indices to [0, n]
                # indices are already in [0, n] because support has n elements.
                sum_w_l = w_suffix[indices]
                # G[a, j] = sum_{k} w_k * sum_w_l[k]
                G_matrix[a, j] = np.dot(weights, sum_w_l)
        
        # Now compute S1[a] = sum_{j} w_j G_matrix[a, j]
        S1 = weights @ G_matrix  # (n,)
        
        # Compute S3[a]
        # S3[a] = w_a * sum_{l} w_l F(2 s_l - s_a)
        # F(t) is computed using sorted_S.
        # We need F(2 s_l - s_a) for all a, l.
        # Let's compute F_matrix2[a, l] = F(2 s_l - s_a).
        T2 = 2.0 * support[None, :] - support[:, None]  # T2[a, l] = 2 s_l - s_a
        flat_T2 = T2.flatten()
        indices2 = np.searchsorted(sorted_S, flat_T2, side='left')
        F_vals2 = cum_W[indices2]
        F_matrix2 = F_vals2.reshape(n, n)
        
        # S3_unscaled[a] = sum_{l} w_l F_matrix2[a, l]
        S3_unscaled = F_matrix2 @ weights  # (n,)
        S3 = weights * S3_unscaled
        
        # Compute S4[a]
        # S4[a] = sum_{k} w_k F(2 s_a - s_k)
        # Note F(2 s_a - s_k) is F_matrix2[k, a]?
        # F_matrix2[i, j] = F(2 s_j - s_i).
        # So F(2 s_a - s_k) = F_matrix2[k, a].
        # S4[a] = sum_{k} w_k F_matrix2[k, a]
        # This is (weights @ F_matrix2.T)[a]?
        # F_matrix2.T[a, k] = F_matrix2[k, a].
        # So S4 = F_matrix2.T @ weights.
        S4 = F_matrix2.T @ weights
        
        # Total Gradient
        grad = 2 * weights * S1 + S3 + S4
        
        return grad

    # Optimization Loop
    lr = 0.1
    best_P = compute_P(support, weights)
    best_weights = weights.copy()
    
    for step in range(10000):
        if time.time() > deadline - 30:
            break
        
        grad = gradient_P(support, weights)
        
        # Update
        weights += lr * grad
        
        # Project to simplex
        # Simple projection: normalize and clip
        weights = np.maximum(weights, 0)
        s = weights.sum()
        if s > 1e-12:
            weights /= s
        else:
            weights = np.ones(n) / n
        
        P = compute_P(support, weights)
        if P > best_P:
            best_P = P
            best_weights = weights.copy()
        
        # Adaptive learning rate?
        # If P decreases, reduce lr?
        # For now, fixed lr.
        
        # Early stopping if gradient is small
        if np.linalg.norm(grad) < 1e-8:
            break

    # Final polish: try a few different learning rates or restarts?
    # Given time, just return best found.
    
    support = support.astype(np.float64)
    best_weights = best_weights.astype(np.float64)
    
    return support, best_weights

# EVOLVE-BLOCK-END
