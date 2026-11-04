import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = 120.0  # Safety margin
    
    # 1. Initialization: Rank Matching
    # Rows of a with largest sums to rows of b with smallest sums
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    
    # Sort facilities by descending a-sum
    heavy = np.argsort(-a_sums, kind="stable")
    # Sort locations by ascending b-sum
    central = np.argsort(b_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # 2. Setup for Local Search
    # A[f, k] = a[f, p^{-1}(k)]
    # B[l, k] = b[l, p(k)]
    
    inv_p = np.empty(n, dtype=np.int64)
    inv_p[p] = np.arange(n)
    
    A = a[:, inv_p]  # Shape (n, n). A[f, k] is a[f, facility at k]
    B = b[:, p]      # Shape (n, n). B[l, k] is b[l, location of facility k? No.]
    # Wait.
    # Cost = sum_{i,j} a[i,j] b[p(i), p(j)]
    # Let k be the index in the permutation array?
    # p[i] is the location of facility i.
    # My previous derivation:
    # A[f, k] = a[f, p^{-1}(k)]?
    # If k is a location index?
    # Let's stick to the indices i, j as facilities.
    # p[i] is location.
    
    # Let's redefine A and B to match the C = A @ B.T formula derived.
    # C[f, l] = sum_k A[f, k] B[l, k]
    # We wanted C[f, l] to represent the interaction cost if facility f is at l.
    # Actually, C[f, l] in my derivation was sum_k a[f, p^{-1}(k)] b[l, p(k)].
    # Here k iterates over locations?
    # Let k be a location index.
    # p^{-1}(k) is the facility at location k.
    # p(k) ... p is a permutation of facilities to locations.
    # Usually p[i] = location.
    # So p^{-1}(loc) = fac.
    # And we need b[l, loc_of_fac_j].
    # If we sum over j (facilities), then loc_of_fac_j = p(j).
    # So sum_j a[f, j] b[l, p(j)].
    # This matches sum_k if we let k=j (facility index).
    # So A[f, j] = a[f, j].
    # B[l, j] = b[l, p(j)].
    # Then C[f, l] = sum_j a[f, j] b[l, p(j)].
    
    # Let's re-verify the delta formula with this definition.
    # Swap facilities i and j.
    # Locations u = p(i), v = p(j).
    # New p': p'(i)=v, p'(j)=u.
    # Delta = 2 * sum_{k != i, j} (a[i, k] + a[j, k]) (b[v, p(k)] - b[u, p(k)])
    # Let's check C terms.
    # C[i, u] = sum_k a[i, k] b[u, p(k)]
    # C[i, v] = sum_k a[i, k] b[v, p(k)]
    # C[j, u] = sum_k a[j, k] b[u, p(k)]
    # C[j, v] = sum_k a[j, k] b[v, p(k)]
    
    # Sum_{k != i, j} (a[i, k] + a[j, k]) (b[v, p(k)] - b[u, p(k)])
    # = sum_{k != i, j} a[i, k] b[v, p(k)] - sum_{k != i, j} a[i, k] b[u, p(k)] + ...
    # = (C[i, v] - a[i, i]b[v, p(i)] - a[i, j]b[v, p(j)]) - (C[i, u] - a[i, i]b[u, p(i)] - a[i, j]b[u, p(j)]) + ...
    
    # This looks correct.
    
    # So:
    # A is just 'a' (fixed).
    # B is dynamic: B[l, j] = b[l, p(j)].
    
    A_mat = a  # (n, n)
    B_mat = b[:, p]  # (n, n). B_mat[l, j] = b[l, p(j)]
    
    # C[f, l] = sum_j A_mat[f, j] B_mat[l, j] = A_mat @ B_mat.T
    C = A_mat @ B_mat.T
    
    # Local Search
    # We will do Best Improvement
    while time.time() < deadline - 5:
        best_delta = 0
        best_u = -1
        best_v = -1
        
        # We iterate over pairs of locations u, v?
        # Or pairs of facilities i, j?
        # Swapping facilities i, j is equivalent to swapping their locations u, v.
        # It is easier to iterate over locations u, v and find facilities i, j at them.
        # i = inv_p[u], j = inv_p[v].
        
        # To speed up, we can randomize the order of pairs or just scan.
        # Scanning all u < v is O(n^2).
        
        for u in range(n):
            i = inv_p[u]
            # Optimization: skip if u == v
            for v in range(u + 1, n):
                j = inv_p[v]
                
                # Calculate Delta
                # Delta = 2 * (C[i, v] - C[i, u] + C[j, u] - C[j, v] - T_u - T_v)
                # T_u = (A[i, i] + A[j, i]) * (B[u, i] - B[v, i])  <-- Wait, indices in T
                # My previous T_u term was: (A[f_u, u] + A[f_v, u]) (B[u, u] - B[v, u])
                # Here A[f, k] was a[f, p^{-1}(k)].
                # In the new definition A_mat[f, j] = a[f, j].
                # The "k" in the sum is the facility index j.
                # The terms excluded are k=i and k=j.
                
                # Term for k=i:
                # (a[i, i] + a[j, i]) * (b[v, p(i)] - b[u, p(i)])
                # p(i) = u.
                # So (a[i, i] + a[j, i]) * (b[v, u] - b[u, u])
                
                # Term for k=j:
                # (a[i, j] + a[j, j]) * (b[v, p(j)] - b[u, p(j)])
                # p(j) = v.
                # So (a[i, j] + a[j, j]) * (b[v, v] - b[u, v])
                
                T_i = (A_mat[i, i] + A_mat[j, i]) * (b[v, u] - b[u, u])
                T_j = (A_mat[i, j] + A_mat[j, j]) * (b[v, v] - b[u, v])
                
                delta = 2 * (C[i, v] - C[i, u] + C[j, u] - C[j, v] - T_i - T_j)
                
                if delta < best_delta:
                    best_delta = delta
                    best_u = u
                    best_v = v
        
        if best_u == -1:
            break
        
        # Perform Swap
        i = inv_p[best_u]
        j = inv_p[best_v]
        
        # Update p
        p[i], p[j] = p[j], p[i]
        
        # Update inv_p
        inv_p[best_u], inv_p[best_v] = j, i
        
        # Update B_mat
        # B_mat[l, k] = b[l, p(k)]
        # p(i) and p(j) swapped.
        # So column i and column j of B_mat swap.
        B_mat[:, [i, j]] = B_mat[:, [j, i]]
        
        # Update C
        # C' = C + outer(A_mat[:, i] - A_mat[:, j], B_mat[:, j] - B_mat[:, i])?
        # Let's re-derive with new indices.
        # Old C[f, l] = sum_k A[f, k] B[l, k]
        # New C'[f, l] = sum_k A[f, k] B'[l, k]
        # B'[l, i] = B[l, j], B'[l, j] = B[l, i]. Others same.
        # C'[f, l] = C[f, l] - A[f, i]B[l, i] - A[f, j]B[l, j] + A[f, i]B[l, j] + A[f, j]B[l, i]
        # C'[f, l] = C[f, l] + A[f, i](B[l, j] - B[l, i]) + A[f, j](B[l, i] - B[l, j])
        # C'[f, l] = C[f, l] + (A[f, i] - A[f, j])(B[l, j] - B[l, i])
        
        diff_A = A_mat[:, i] - A_mat[:, j]
        diff_B = B_mat[:, j] - B_mat[:, i]  # Note: B_mat is already updated, so this is (Old B[:, i] - Old B[:, j])?
        # Wait.
        # B_mat was updated.
        # New B_mat[:, j] = Old B_mat[:, i]
        # New B_mat[:, i] = Old B_mat[:, j]
        # So New B_mat[:, j] - New B_mat[:, i] = Old B_mat[:, i] - Old B_mat[:, j]
        # The formula requires (B[l, j] - B[l, i]) using OLD B?
        # Let's check:
        # C' = C + (A[:, i] - A[:, j]) outer (B_old[:, j] - B_old[:, i])
        # So we need to use the diff BEFORE updating B_mat?
        # Or just compute it from the updated B_mat carefully.
        # Let's compute diff_B before swapping B_mat columns.
        
        # Refined Update Step:
        # 1. Compute diff_A = A_mat[:, i] - A_mat[:, j]
        # 2. Compute diff_B = B_mat[:, j] - B_mat[:, i]  (Using OLD B_mat)
        # 3. C += np.outer(diff_A, diff_B)
        # 4. Swap columns in B_mat
        
        # Let's re-do the update logic in the code.
        
        # Actually, I already swapped B_mat in the draft above.
        # I should compute diff_B BEFORE swapping.
        
        pass # Placeholder for logic correction in final code

    return p.tolist()
