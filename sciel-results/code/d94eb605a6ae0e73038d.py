"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

import numpy as np
import time
import itertools
from scipy.optimize import linprog

# EVOLVE-BLOCK-START
def _build_tensor_entries(n, m, p):
    """Build list of (a, b, c) where tensor T[a,b,c] = 1."""
    A_dim = n * m
    B_dim = m * p
    C_dim = p * n
    entries = []
    for a in range(A_dim):
        i = a // m
        j = a % m
        for b in range(B_dim):
            j2 = b // p
            k = b % p
            if j == j2:
                c = k * n + i
                entries.append((a, b, c))
    return entries

def _schoolbook(n, m, p):
    """Return schoolbook decomposition with rank n*m*p."""
    A_dim = n * m
    B_dim = m * p
    C_dim = p * n
    rank = n * m * p
    U = [[0] * rank for _ in range(A_dim)]
    V = [[0] * rank for _ in range(B_dim)]
    W = [[0] * rank for _ in range(C_dim)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a][col] = 1
                V[b][col] = 1
                W[c][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, entries, n, m, p):
    """Verify that the decomposition exactly reproduces the tensor."""
    A_dim = n * m
    B_dim = m * p
    C_dim = p * n
    rank = len(U[0]) if U else 0
    # Build the tensor from the decomposition
    tensor = np.zeros((A_dim, B_dim, C_dim), dtype=float)
    for l in range(rank):
        for a in range(A_dim):
            if U[a][l] != 0:
                for b in range(B_dim):
                    if V[b][l] != 0:
                        for c in range(C_dim):
                            if W[c][l] != 0:
                                tensor[a, b, c] += U[a][l] * V[b][l] * W[c][l]
    # Check against entries
    for (a, b, c) in entries:
        if abs(tensor[a, b, c] - 1.0) > 1e-9:
            return False
    # Check that all other entries are zero
    for a in range(A_dim):
        for b in range(B_dim):
            for c in range(C_dim):
                expected = 1.0 if (a, b, c) in set(entries) else 0.0
                if abs(tensor[a, b, c] - expected) > 1e-9:
                    return False
    return True

def _peel_rank(U, V, W, entries, n, m, p, max_iter=1000, deadline=None):
    """Try to reduce rank by peeling off rank-1 tensors."""
    A_dim = n * m
    B_dim = m * p
    C_dim = p * n
    rank = len(U[0]) if U else 0
    
    # Convert to numpy for efficiency
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build tensor from current decomposition
    tensor = np.zeros((A_dim, B_dim, C_dim), dtype=float)
    for l in range(rank):
        for a in range(A_dim):
            if U_np[a, l] != 0:
                for b in range(B_dim):
                    if V_np[b, l] != 0:
                        for c in range(C_dim):
                            if W_np[c, l] != 0:
                                tensor[a, b, c] += U_np[a, l] * V_np[b, l] * W_np[c, l]
    
    # Try to find rank-1 tensors to subtract
    # A rank-1 tensor is u ⊗ v ⊗ w
    # We want to find u, v, w such that tensor - u⊗v⊗w has a lower "effective rank"
    
    # Simple approach: try to find a rank-1 tensor that matches a large portion of the tensor
    # For each nonzero entry (a,b,c), try u[a]=1, v[b]=1, w[c]=1 and see if we can extend
    
    best_U, best_V, best_W = U_np.copy(), V_np.copy(), W_np.copy()
    best_rank = rank
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try to find a rank-1 tensor to subtract
        # Look for a pattern in the tensor
        found = False
        for a in range(A_dim):
            if time.time() > deadline:
                break
            for b in range(B_dim):
                if time.time() > deadline:
                    break
                for c in range(C_dim):
                    if time.time() > deadline:
                        break
                    if tensor[a, b, c] != 0:
                        # Try u[a]=1, v[b]=1, w[c]=1
                        u = np.zeros(A_dim)
                        v = np.zeros(B_dim)
                        w = np.zeros(C_dim)
                        u[a] = 1
                        v[b] = 1
                        w[c] = 1
                        
                        # Check if this is a valid rank-1 tensor in the support
                        # i.e., for all (a',b',c') where tensor[a',b',c'] != 0,
                        # we need u[a']*v[b']*w[c'] to be consistent
                        valid = True
                        for (a2, b2, c2) in entries:
                            if tensor[a2, b2, c2] != 0:
                                if u[a2] * v[b2] * w[c2] == 0:
                                    valid = False
                                    break
                        if valid:
                            # Subtract this rank-1 tensor
                            for a2 in range(A_dim):
                                for b2 in range(B_dim):
                                    for c2 in range(C_dim):
                                        tensor[a2, b2, c2] -= u[a2] * v[b2] * w[c2]
                            # Remove one column from U, V, W
                            # Find which column corresponds to this pattern
                            # For now, just reduce rank by 1
                            if best_rank > 1:
                                best_U = best_U[:, :-1]
                                best_V = best_V[:, :-1]
                                best_W = best_W[:, :-1]
                                best_rank -= 1
                            found = True
                            break
                if found:
                    break
            if found:
                break
        
        if not found:
            break
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS
    A_dim = n * m  # 8
    B_dim = m * p  # 32
    C_dim = p * n  # 16
    
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for 150s limit
    
    # Get schoolbook decomposition
    sb = _schoolbook(n, m, p)
    entries = _build_tensor_entries(n, m, p)
    
    # Try to reduce rank using peeling
    U_np, V_np, W_np, rank = _peel_rank(
        sb["U"], sb["V"], sb["W"], entries, n, m, p,
        max_iter=5000, deadline=deadline
    )
    
    # Convert back to lists
    U = U_np.tolist()
    V = V_np.tolist()
    W = W_np.tolist()
    
    # Verify the decomposition
    if not _verify_decomposition(U, V, W, entries, n, m, p):
        # Fall back to schoolbook if verification fails
        U = sb["U"]
        V = sb["V"]
        W = sb["W"]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
