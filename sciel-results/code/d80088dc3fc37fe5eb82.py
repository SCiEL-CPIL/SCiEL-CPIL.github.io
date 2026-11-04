"""Rewritten program implementing a rank-1 peeling search for the 2x3x5 matrix multiplication tensor."""
import itertools
import numpy as np

DIMENSIONS = [2, 3, 5]

# Precompute the target tensor T of shape (n*m, m*p, p*n) where T[i*m+j, j*p+k, k*n+i] = 1
# We use the convention: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
# The tensor being decomposed is T[a,b,c] = sum_k U[a,k] V[b,k] W[c,k]
# where a = i*m+j, b = j*p+k, c = k*n+i
# So T has 1s at specific (a,b,c) and 0s elsewhere.

def _build_target_tensor(n, m, p):
    """Build the flattened target tensor of size (n*m, m*p, p*n)."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    return T

def _peel_rank1(T, n, m, p):
    """
    Attempt to find a rank-1 tensor U (length n*m), V (length m*p), W (length p*n)
    such that T - U[i]V[j]W[k] is still a valid tensor (entries 0 or 1) for the
    residual, i.e., the subtraction only affects positions where T==1.
    
    More precisely: find rank-1 summand R = U ⊗ V ⊗ W (outer product) such that:
      - R[a,b,c] is in {0, 1, -1} (but for our purposes, we want R entries to be 0 or 1
        to subtract from T which has 0s and 1s)
      - For all (a,b,c) where T[a,b,c]==1, R[a,b,c] is 0 or 1
      - For all (a,b,c) where T[a,b,c]==0, R[a,b,c] must be 0 (so we don't create negative residuals)
    
    Actually, the standard peeling approach: find a rank-1 tensor R with R[a,b,c] ∈ {0,1}
    such that R ≤ T (element-wise), then residual = T - R is also a {0,1}-valued tensor.
    Then we need the residual to be decomposable.
    
    For exact decomposition, we want R such that T - R is a non-negative integer tensor
    that can itself be decomposed. The simplest sufficient condition is that R is a 
    "valid" rank-1 sub-tensor of T.
    
    A rank-1 tensor R = U ⊗ V ⊗ W with U,V,W ∈ {0,1} vectors is valid if:
      For all (a,b,c): if R[a,b,c] = 1 then T[a,b,c] = 1.
    
    Let's search for such rank-1 tensors.
    """
    T_flat = T.flatten()
    nm, mp, pn = n * m, m * p, p * n
    
    # We need U (nm), V (mp), W (pn) with entries in {0,1}
    # such that for all a,b,c: U[a]*V[b]*W[c] <= T[a,b,c]
    
    # This is equivalent to: the support of U ⊗ V ⊗ W is contained in the support of T.
    # Support of U ⊗ V ⊗ W = { (a,b,c) : U[a]=1 and V[b]=1 and W[c]=1 }
    
    # So we need: if U[a]=1, V[b]=1, W[c]=1, then T[a,b,c]=1.
    # Equivalently: for all (a,b,c) with T[a,b,c]=0, at least one of U[a], V[b], W[c] is 0.
    
    # Let's try to find all maximal rank-1 sub-tensors.
    # Start with U=1, V=1, W=1, then iteratively zero out entries that violate the constraint.
    
    # A better approach: enumerate possible supports.
    # Since nm=6, mp=15, pn=10, the spaces are small enough for some enumeration.
    
    # Let's use a different strategy: for each pair of vectors U and V, determine the 
    # maximum W that works, and check if the resulting R is a valid sub-tensor.
    
    # Actually, let's just do a BFS/DFS over possible U vectors (2^6 = 64 possibilities),
    # for each U, BFS over V vectors (2^15 = 32768), and for each (U,V), check what W can be.
    
    # But that's 64 * 32768 = ~2 million combinations, which is feasible.
    
    # For a given U and V, the constraint on W is:
    # For all a,b,c: if U[a]=1 and V[b]=1, then W[c] must be 0 if T[a,b,c]=0.
    # So W[c] can be 1 only if for ALL (a,b) with U[a]=1 and V[b]=1, we have T[a,b,c]=1.
    
    # Let's implement this.
    
    # Precompute: for each c, the set of (a,b) where T[a,b,c]=0.
    # If U and V have 1s at positions in that set, then W[c] must be 0.
    
    zero_positions = []  # list of (a,b) for each c
    for c in range(pn):
        zeros = []
        for a in range(nm):
            for b in range(mp):
                if T[a, b, c] == 0:
                    zeros.append((a, b))
        zero_positions.append(zeros)
    
    # For each U (2^nm possibilities), for each V (2^mp possibilities), check if a non-zero W exists.
    # To be efficient, we can prune: if U is all zeros or V is all zeros, skip.
    
    nm_mask_max = (1 << nm) - 1
    mp_mask_max = (1 << mp) - 1
    
    best_rank1 = None
    
    for u_mask in range(1, 1 + nm_mask_max):
        # Get support of U
        U_sup = [a for a in range(nm) if (u_mask >> a) & 1]
        if not U_sup:
            continue
        
        for v_mask in range(1, 1 + mp_mask_max):
            V_sup = [b for b in range(mp) if (v_mask >> b) & 1]
            if not V_sup:
                continue
            
            # Determine which c's can have W[c]=1
            W_allowed = [True] * pn
            for c in range(pn):
                # Check: for all (a,b) with a in U_sup and b in V_sup, T[a,b,c] must be 1
                valid = True
                for a in U_sup:
                    for b in V_sup:
                        if T[a, b, c] == 0:
                            valid = False
                            break
                    if not valid:
                        break
                W_allowed[c] = valid
            
            # Find the maximal W: set W[c]=1 for all c where W_allowed[c] is True
            # This gives the largest rank-1 sub-tensor for this (U,V)
            w_mask = 0
            for c in range(pn):
                if W_allowed[c]:
                    w_mask |= (1 << c)
            
            if w_mask == 0:
                continue
            
            # We have a valid rank-1 tensor with U_mask=u_mask, V_mask=v_mask, W_mask=w_mask
            # Store it; we'll use it for peeling
            if best_rank1 is None:
                best_rank1 = (u_mask, v_mask, w_mask)
    
    return best_rank1

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    T = _build_target_tensor(n, m, p)
    T_residual = T.copy()
    
    U_factors = []  # list of U vectors
    V_factors = []
    W_factors = []
    
    # Greedy peeling: repeatedly find a rank-1 sub-tensor and subtract it
    max_rank = nm * mp * pn  # upper bound
    rank = 0
    
    while rank < max_rank:
        # Check if T_residual is zero
        if not T_residual.any():
            break
        
        # Try to find a rank-1 sub-tensor
        rank1 = _peel_rank1(T_residual, n, m, p)
        
        if rank1 is None:
            # Could not peel; fall back to schoolbook for the remaining tensor
            # Count the number of 1s in the residual
            remaining_ones = T_residual.sum()
            rank += remaining_ones
            # Add schoolbook factors for each 1
            for a in range(nm):
                for b in range(mp):
                    for c in range(pn):
                        if T_residual[a, b, c] == 1:
                            u = np.zeros(nm)
                            v = np.zeros(mp)
                            w = np.zeros(pn)
                            u[a] = 1
                            v[b] = 1
                            w[c] = 1
                            U_factors.append(u)
                            V_factors.append(v)
                            W_factors.append(w)
            break
        else:
            u_mask, v_mask, w_mask = rank1
            u = np.zeros(nm)
            v = np.zeros(mp)
            w = np.zeros(pn)
            for a in range(nm):
                if (u_mask >> a) & 1:
                    u[a] = 1
            for b in range(mp):
                if (v_mask >> b) & 1:
                    v[b] = 1
            for c in range(pn):
                if (w_mask >> c) & 1:
                    w[c] = 1
            
            # Subtract the rank-1 tensor from residual
            R = np.einsum('a,b,c->abc', u, v, w)
            T_residual = T_residual - R
            # Verify no negatives
            if (T_residual < 0).any():
                # This shouldn't happen if our constraint was correct, but just in case
                T_residual = np.maximum(T_residual, 0)
            
            U_factors.append(u)
            V_factors.append(v)
            W_factors.append(w)
            rank += 1
    
    # Build the final matrices
    rank_final = len(U_factors)
    if rank_final == 0:
        # Should not happen
        rank_final = 1
        U_factors = [np.zeros(nm)]
        V_factors = [np.zeros(mp)]
        W_factors = [np.zeros(pn)]
    
    U = np.column_stack(U_factors) if len(U_factors) > 1 else np.array([U_factors[0]]).T
    V = np.column_stack(V_factors) if len(V_factors) > 1 else np.array([V_factors[0]]).T
    W = np.column_stack(W_factors) if len(W_factors) > 1 else np.array([W_factors[0]]).T
    
    # Convert to lists of lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
