import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build target tensor support
    support = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                support.append((u, v, w))
    
    target_rank = 12
    
    # Fallback to schoolbook
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(nm)]
    V_sb = [[0] * schoolbook_rank for _ in range(mp)]
    W_sb = [[0] * schoolbook_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    fallback = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    start_time = time.time()
    deadline = 110.0
    
    def compute_tensor(U, V, W):
        """Compute the full tensor from factor matrices."""
        R = U.shape[1]
        C = np.zeros((nm, mp, pn))
        for c in range(R):
            C += np.einsum('u,v,w->uvw', U[:, c], V[:, c], W[:, c])
        return C
    
    def build_target():
        T = np.zeros((nm, mp, pn))
        for (u, v, w) in support:
            T[u, v, w] = 1.0
        return T
    
    T = build_target()
    
    def verify_exact(U, V, W, atol=1e-8):
        """Check if U,V,W exactly decompose T with half-integer entries."""
        for mat in [U, V, W]:
            if not np.allclose(mat * 2, np.round(mat * 2), atol=atol):
                return False
        C = compute_tensor(U, V, W)
        return np.allclose(C, T, atol=atol)
    
    def snap_half(x):
        return np.round(x * 2) / 2.0
    
    def try_pairing(pairs):
        """
        Given a pairing of 24 support entries into 12 pairs,
        try to find half-integer solutions for each pair.
        Returns (U, V, W) if successful, None otherwise.
        """
        U = np.zeros((nm, target_rank))
        V = np.zeros((mp, target_rank))
        W = np.zeros((pn, target_rank))
        
        for c, (e1, e2) in enumerate(pairs):
            u1, v1, w1 = e1
            u2, v2, w2 = e2
            
            # For this component, non-zero U at {u1,u2}, V at {v1,v2}, W at {w1,w2}
            # Constraints:
            #   U[u1]*V[v1]*W[w1] = 1
            #   U[u2]*V[v2]*W[w2] = 1
            # 
            # Let a=U[u1], b=U[u2], c=V[v1], d=V[v2], e=W[w1], f=W[w2]
            #   a*c*e = 1
            #   b*d*f = 1
            #
            # With half-integer constraint: a,b,c,d,e,f in {0, ±0.5, ±1, ±1.5, ...}
            # But they must be non-zero (since product = 1)
            #
            # Simplest: try a=c=e=1, b=d=f=1
            # This works if u1!=u2, v1!=v2, w1!=w2
            # If some indices coincide, we need different values
            
            # Try the trivial solution first
            U[u1, c] = 1.0
            V[v1, c] = 1.0
            W[w1, c] = 1.0
            U[u2, c] = 1.0
            V[v2, c] = 1.0
            W[w2, c] = 1.0
            
            # Check if this works (it always does for the two target entries)
            # But we need to ensure no other entries are non-zero, which is guaranteed
            # by our zero initialization
            
        # Verify the full solution
        if verify_exact(U, V, W):
            return U, V, W
        return None
    
    # Generate all valid pairings and try them
    # Key constraint: each support entry appears exactly once
    # For each pair, the trivial solution (all 1s) works
    
    # But we need to check: does the trivial solution produce the correct tensor?
    # For pair ((u1,v1,w1), (u2,v2,w2)) with all 1s:
    #   Component contributes 1 to T[u1,v1,w1] and T[u2,v2,w2]
    #   But also contributes 1 to T[u1,v2,w1], T[u1,v1,w2], etc. if those are in support!
    #   Wait no - the component is U_c ⊗ V_c ⊗ W_c, so it contributes to ALL
    #   (u,v,w) where U_c[u]!=0, V_c[v]!=0, W_c[w]!=0
    #   So if U_c has non-zeros at {u1,u2}, V_c at {v1,v2}, W_c at {w1,w2},
    #   the component contributes to all 8 combinations (u,v,w) in {u1,u2}x{v1,v2}x{w1,w2}
    #   We need only 2 of these to be in support, and the other 6 must NOT be in support
    #   OR must be cancelled by other components
    
    # This is the key issue! The trivial solution doesn't work in general.
    # We need the "cross terms" to either be zero (not in support) or cancel out.
    
    # Let's check which pairings have the property that the 6 cross terms are not in support
    support_set = set(support)
    
    def is_valid_pairing(pairs):
        """Check if the cross terms of each pair are not in support (or can be handled)."""
        for (e1, e2) in pairs:
            u1, v1, w1 = e1
            u2, v2, w2 = e2
            # The 8 combinations
            for u in [u1, u2]:
                for v in [v1, v2]:
                    for w in [w1, w2]:
                        if (u, v, w) in support_set:
                            # This is either e1 or e2 (ok) or a cross term (problem)
                            if (u, v, w) != e1 and (u, v, w) != e2:
                                return False
        return True
    
    # Try to find a valid pairing
    import random
    np.random.seed(42)
    random.seed(42)
    
    best_solution = None
    
    # First, try to find a pairing where all cross terms are not in support
    for trial in range(200):
        if time.time() > deadline:
            break
        
        indices = list(range(24))
        random.shuffle(indices)
        pairs = []
        for i in range(0, 24, 2):
            pairs.append((support[indices[i]], support[indices[i+1]]))
        
        if is_valid_pairing(pairs):
            sol = try_pairing(pairs)
            if sol:
                U, V, W = sol
                best_solution = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
    
    # If no clean pairing found, try a more sophisticated approach
    if best_solution is None:
        # Use the fact that cross terms can be cancelled
        # Try ALS with proper constraint handling
        
        def try_constrained_als():
            """ALS that respects the sparse structure."""
            for trial in range(50):
                if time.time() > deadline:
                    break
                
                # Random initialization with sparsity
                U = np.zeros((nm, target_rank))
                V = np.zeros((mp, target_rank))
                W = np.zeros((pn, target_rank))
                
                # Initialize each component to cover 2 random support entries
                indices = list(range(24))
                random.shuffle(indices)
                for c in range(target_rank):
                    e1 = support[indices[2*c]]
                    e2 = support[indices[2*c+1]]
                    u1, v1, w1 = e1
                    u2, v2, w2 = e2
                    U[u1, c] = 1.0
                    U[u2, c] = 1.0
                    V[v1, c] = 1.0
                    V[v2, c] = 1.0
                    W[w1, c] = 1.0
                    W[w2, c] = 1.0
                
                # Iterate: for each component, solve the constrained problem
                for it in range(50):
                    if time.time() > deadline:
                        break
                    
                    for c in range(target_rank):
                        # Get the support indices for this component
                        u_indices = np.where(U[:, c] != 0)[0]
                        v_indices = np.where(V[:, c] != 0)[0]
                        w_indices = np.where(W[:, c] != 0)[0]
                        
                        if len(u_indices) == 0 or len(v_indices) == 0 or len(w_indices) == 0:
                            continue
                        
                        # The component contributes to all (u,v,w) in 
                        # u_indices x v_indices x w_indices
                        # We want: sum of contributions = T at those positions
                        # And: contributions = 0 at all other positions
                        
                        # For now, just check if the current solution is good
                        pass
                    
                    # Check convergence
                    C = compute_tensor(U, V, W)
                    residual = np.linalg.norm(C - T)
                    
                    if residual < 1e-8:
                        Us, Vs, Ws = snap_half(U), snap_half(V), snap_half(W)
                        if verify_exact(Us, Vs, Ws):
                            return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}
                    
                    if residual < 1e-6:
                        Us, Vs, Ws = snap_half(U), snap_half(V), snap_half(W)
                        if verify_exact(Us, Vs, Ws, atol=1e-4):
                            return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}
                
                return None
            
            return None
        
        best_solution = try_constrained_als()
    
    if best_solution:
        return best_solution
    else:
        return fallback
