import numpy as np
import time
import itertools
from fractions import Fraction

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
    
    def residual_norm(U, V, W):
        C = compute_tensor(U, V, W)
        return np.linalg.norm(C - T)
    
    def verify_exact(U, V, W, atol=1e-8):
        """Check if U,V,W exactly decompose T with half-integer entries."""
        # Check half-integer constraint
        for mat in [U, V, W]:
            if not np.allclose(mat * 2, np.round(mat * 2), atol=atol):
                return False
        C = compute_tensor(U, V, W)
        return np.allclose(C, T, atol=atol)
    
    def snap_half(x):
        return np.round(x * 2) / 2.0
    
    # Strategy: Try structured initializations based on pairing support entries
    # Each rank-1 component covers some subset of support entries
    # Try all ways to partition 24 entries into 12 groups of 2
    
    best_solution = None
    best_residual = float('inf')
    
    # First, try a gradient descent approach with good initialization
    # Use the fact that we can initialize each rank-1 component to cover 2 support entries
    
    def try_als(U0, V0, W0, max_iters=200, tol=1e-10):
        """Run ALS optimization."""
        U, V, W = U0.copy(), V0.copy(), W0.copy()
        R = U.shape[1]
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update each factor
            for c in range(R):
                # Update U[:, c]
                # C_target - sum_{c'!=c} U_c' ⊗ V_c' ⊗ W_c' = U_c ⊗ V_c ⊗ W_c (approximately)
                # For fixed V_c, W_c: residual_u = T - sum_{c'!=c} U_c'⊗V_c'⊗W_c'
                # We want: U_c ⊗ (V_c ⊗ W_c) ≈ residual
                # This is: for all (v,w): sum_u U_c[u] * (V_c[v]*W_c[w]) = residual[u,v,w]
                # => U_c = argmin ||A @ U_c - b|| where A[v*w, u] = V_c[v]*W_c[w], b[v*w] = residual[:,v,w].ravel()
                
                res = T.copy()
                for c2 in range(R):
                    if c2 != c:
                        res -= np.einsum('u,v,w->uvw', U[:, c2], V[:, c2], W[:, c2])
                
                # Build linear system for U[:,c]
                # A has shape (mp*pn, nm), b has shape (mp*pn,)
                VW = np.outer(V[:, c], W[:, c])  # shape (mp*pn,)
                A = np.tile(VW[:, None], (1, nm))  # shape (mp*pn, nm)
                b = res.reshape(mp * pn)
                
                try:
                    U[:, c] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    break
                
                if time.time() > deadline:
                    break
                
                # Update V[:, c]
                res = T.copy()
                for c2 in range(R):
                    if c2 != c:
                        res -= np.einsum('u,v,w->uvw', U[:, c2], V[:, c2], W[:, c2])
                
                UW = np.outer(U[:, c], W[:, c])  # shape (nm*pn,)
                A = np.tile(UW[:, None], (1, mp))  # shape (nm*pn, mp)
                b = res.reshape(nm * pn)
                
                try:
                    V[:, c] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    break
                
                if time.time() > deadline:
                    break
                
                # Update W[:, c]
                res = T.copy()
                for c2 in range(R):
                    if c2 != c:
                        res -= np.einsum('u,v,w->uvw', U[:, c2], V[:, c2], W[:, c2])
                
                UV = np.outer(U[:, c], V[:, c])  # shape (nm*mp,)
                A = np.tile(UV[:, None], (1, pn))  # shape (nm*mp, pn)
                b = res.reshape(nm * mp)
                
                try:
                    W[:, c] = np.linalg.lstsq(A, b, rcond=None)[0]
                except:
                    break
                
                if time.time() > deadline:
                    break
            
            # Check convergence
            r = residual_norm(U, V, W)
            if r < best_residual:
                best_residual = r
                # Try snapping
                Us, Vs, Ws = snap_half(U), snap_half(V), snap_half(W)
                if verify_exact(Us, Vs, Ws):
                    return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}, r
            
            if r < tol:
                Us, Vs, Ws = snap_half(U), snap_half(V), snap_half(W)
                if verify_exact(Us, Vs, Ws):
                    return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}, r
            
            # Also check if current (unsnapped) is close enough that snapping works
            if r < 1e-6:
                Us, Vs, Ws = snap_half(U), snap_half(V), snap_half(W)
                if verify_exact(Us, Vs, Ws, atol=1e-4):
                    return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}, r
        
        return None, residual_norm(U, V, W)
    
    # Generate structured initializations
    # Each rank-1 component should cover 2 support entries
    # For a pair ((u1,v1,w1), (u2,v2,w2)), we need:
    # U_c[u1]*V_c[v1]*W_c[w1] = 1
    # U_c[u2]*V_c[v2]*W_c[w2] = 1
    # And zero elsewhere (approximately)
    
    def make_init_from_pairs(pairs):
        """Create initial U,V,W from a list of 12 pairs of support entries."""
        U = np.zeros((nm, target_rank))
        V = np.zeros((mp, target_rank))
        W = np.zeros((pn, target_rank))
        
        for c, (e1, e2) in enumerate(pairs):
            u1, v1, w1 = e1
            u2, v2, w2 = e2
            
            # Simple initialization: set the relevant entries
            U[u1, c] = 1.0
            U[u2, c] = 1.0
            V[v1, c] = 1.0
            V[v2, c] = 1.0
            W[w1, c] = 1.0
            W[w2, c] = 1.0
        
        return U, V, W
    
    # Try several random pairings
    np.random.seed(42)
    
    for trial in range(100):
        if time.time() > deadline:
            break
        
        # Random pairing of 24 support entries into 12 pairs
        indices = list(range(len(support)))
        np.random.shuffle(indices)
        pairs = []
        for i in range(0, 24, 2):
            pairs.append((support[indices[i]], support[indices[i+1]]))
        
        U0, V0, W0 = make_init_from_pairs(pairs)
        
        # Add small random perturbation to avoid degeneracy
        U0 += np.random.randn(nm, target_rank) * 0.1
        V0 += np.random.randn(mp, target_rank) * 0.1
        W0 += np.random.randn(pn, target_rank) * 0.1
        
        sol, res = try_als(U0, V0, W0, max_iters=100)
        if sol:
            best_solution = sol
            break
    
    # Also try pure random initializations
    if best_solution is None:
        for trial in range(50):
            if time.time() > deadline:
                break
            
            U0 = np.random.randn(nm, target_rank)
            V0 = np.random.randn(mp, target_rank)
            W0 = np.random.randn(pn, target_rank)
            
            sol, res = try_als(U0, V0, W0, max_iters=100)
            if sol:
                best_solution = sol
                break
    
    if best_solution:
        return best_solution
    
    # Structured enumeration: try to find rank-12 decomposition with half-integer entries
    # For matrix multiplication, rank-1 tensors have a specific structure
    # Each rank-1 component can be parameterized by which (i,j,k) indices it affects
    
    def try_structured_enumeration():
        """Try to find a rank-12 decomposition using structured search."""
        if time.time() > deadline - 5:
            return None
        
        # For a 2x3 x 3x4 matrix multiplication, the tensor has 24 non-zero entries
        # We need to cover all 24 entries with 12 rank-1 components
        # Each rank-1 component can cover at most 2 entries (if we want half-integer values)
        
        # Try to pair up the 24 support entries into 12 pairs
        # For each pair, determine if there exists a rank-1 tensor with half-integer entries
        # that covers both entries with value 1 and is zero elsewhere
        
        # Precompute all possible rank-1 components with half-integer entries
        # that have exactly 2 non-zero entries with value 1
        
        half_ints = [-1.0, -0.5, 0.0, 0.5, 1.0]
        
        # For each pair of support entries, check if we can form a rank-1 component
        pair_solutions = {}
        
        for idx1 in range(len(support)):
            for idx2 in range(idx1 + 1, len(support)):
                u1, v1, w1 = support[idx1]
                u2, v2, w2 = support[idx2]
                
                # For a rank-1 tensor U_c ⊗ V_c ⊗ W_c to have:
                # U_c[u1]*V_c[v1]*W_c[w1] = 1
                # U_c[u2]*V_c[v2]*W_c[w2] = 1
                # And zero at all other positions
                
                # The support of U_c ⊗ V_c ⊗ W_c is the Cartesian product of
                # supports of U_c, V_c, W_c
                
                # For exactly 2 non-zero entries, we need:
                # |supp(U_c)| * |supp(V_c)| * |supp(W_c)| = 2
                # So one factor has support size 2, others have support size 1
                # Or all have support size 1 (giving 1 entry, not 2)
                
                # Case 1: U_c has 2 non-zero entries, V_c and W_c have 1 each
                # Then the 2 entries must share the same v and w
                if v1 == v2 and w1 == w2:
                    # U_c[u1]*V_c[v1]*W_c[w1] = 1
                    # U_c[u2]*V_c[v2]*W_c[w2] = 1
                    # With V_c[v1] = V_c[v2] = a, W_c[w1] = W_c[w2] = b
                    # U_c[u1]*a*b = 1, U_c[u2]*a*b = 1
                    # So U_c[u1] = U_c[u2] = 1/(a*b)
                    # For half-integers: a, b in half_ints, 1/(a*b) in half_ints
                    for a in half_ints:
                        if a == 0: continue
                        for b in half_ints:
                            if b == 0: continue
                            val = 1.0 / (a * b)
                            if val in half_ints:
                                U_c = {u1: val, u2: val}
                                V_c = {v1: a}
                                W_c = {w1: b}
                                pair_solutions[(idx1, idx2)] = (U_c, V_c, W_c)
                                break
                        else:
                            continue
                        break
                
                # Case 2: V_c has 2 non-zero entries, U_c and W_c have 1 each
                if u1 == u2 and w1 == w2:
                    for a in half_ints:
                        if a == 0: continue
                        for b in half_ints:
                            if b == 0: continue
                            val = 1.0 / (a * b)
                            if val in half_ints:
                                U_c = {u1: a}
                                V_c = {v1: val, v2: val}
                                W_c = {w1: b}
                                pair_solutions[(idx1, idx2)] = (U_c, V_c, W_c)
                                break
                        else:
                            continue
                        break
                
                # Case 3: W_c has 2 non-zero entries, U_c and V_c have 1 each
                if u1 == u2 and v1 == v2:
                    for a in half_ints:
                        if a == 0: continue
                        for b in half_ints:
                            if b == 0: continue
                            val = 1.0 / (a * b)
                            if val in half_ints:
                                U_c = {u1: a}
                                V_c = {v1: b}
                                W_c = {w1: val, w2: val}
                                pair_solutions[(idx1, idx2)] = (U_c, V_c, W_c)
                                break
                        else:
                            continue
                        break
        
        # Now try to find a perfect matching of 24 entries into 12 pairs
        # where each pair has a solution
        
        # This is a combinatorial problem - try a greedy approach with backtracking
        used = [False] * len(support)
        assignment = [None] * len(support)
        
        def backtrack(pos):
            if time.time() > deadline - 3:
                return False
            
            if pos == len(support):
                return True
            
            if used[pos]:
                return backtrack(pos + 1)
            
            for j in range(pos + 1, len(support)):
                if used[j]:
                    continue
                
                if (pos, j) in pair_solutions:
                    used[pos] = used[j] = True
                    assignment[pos] = j
                    assignment[j] = pos
                    
                    if backtrack(pos + 1):
                        return True
                    
                    used[pos] = used[j] = False
                    assignment[pos] = assignment[j] = None
            
            return False
        
        if backtrack(0):
            # Build the solution
            U_sol = np.zeros((nm, target_rank))
            V_sol = np.zeros((mp, target_rank))
            W_sol = np.zeros((pn, target_rank))
            
            for c in range(target_rank):
                idx1 = c * 2
                idx2 = assignment[idx1]
                U_c, V_c, W_c = pair_solutions[(idx1, idx2)]
                
                for u, val in U_c.items():
                    U_sol[u, c] = val
                for v, val in V_c.items():
                    V_sol[v, c] = val
                for w, val in W_c.items():
                    W_sol[w, c] = val
            
            if verify_exact(U_sol, V_sol, W_sol):
                return {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}
        
        return None
    
    # Try structured enumeration
    best_solution = try_structured_enumeration()
    
    if best_solution:
        return best_solution
    else:
        return fallback
