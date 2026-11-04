"""Matrix multiplication 4x4 @ 4x5: rank-minimal tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 5]
N, M, P = 4, 4, 5
NM = N * M
MP = M * P
PN = P * N


def _schoolbook():
    n, m, p = N, M, P
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W):
    rank = len(U[0])
    for i in range(N):
        for j in range(M):
            for k in range(P):
                total = 0
                for r in range(rank):
                    total += U[i * M + j][r] * V[j * P + k][r] * W[k * N + i][r]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True


def _rank_of(decomp):
    return len(decomp["U"][0])


def _try_merge_columns(U, V, W, a, b):
    """Try to merge columns a and b if they are proportional in a way that preserves the product."""
    rank = len(U[0])
    if a >= rank or b >= rank or a == b:
        return None
    
    # Check if U[:,a] = s * U[:,b] for some s
    s = None
    for r in range(NM):
        if U[r][a] != 0 and U[r][b] != 0:
            s = U[r][a] / U[r][b]
            break
        elif U[r][a] != 0 or U[r][b] != 0:
            return None
    
    if s is None:
        # Check if both are zero
        if all(U[r][a] == 0 for r in range(NM)) and all(U[r][b] == 0 for r in range(NM)):
            s = 1.0
        else:
            return None
    
    # Verify U proportionality
    for r in range(NM):
        if abs(U[r][a] - s * U[r][b]) > 1e-9:
            return None
    
    # Similarly for V and W
    sv = None
    for r in range(MP):
        if V[r][a] != 0 and V[r][b] != 0:
            sv = V[r][a] / V[r][b]
            break
        elif V[r][a] != 0 or V[r][b] != 0:
            return None
    
    if sv is None:
        if all(V[r][a] == 0 for r in range(MP)) and all(V[r][b] == 0 for r in range(MP)):
            sv = 1.0
        else:
            return None
    
    for r in range(MP):
        if abs(V[r][a] - sv * V[r][b]) > 1e-9:
            return None
    
    sw = None
    for r in range(PN):
        if W[r][a] != 0 and W[r][b] != 0:
            sw = W[r][a] / W[r][b]
            break
        elif W[r][a] != 0 or W[r][b] != 0:
            return None
    
    if sw is None:
        if all(W[r][a] == 0 for r in range(PN)) and all(W[r][b] == 0 for r in range(PN)):
            sw = 1.0
        else:
            return None
    
    for r in range(PN):
        if abs(W[r][a] - sw * W[r][b]) > 1e-9:
            return None
    
    # Product scale: s * sv * sw should be 1 for the terms to be identical
    if abs(s * sv * sw - 1.0) > 1e-9:
        return None
    
    # They are identical in product, so we can drop one
    new_U = [row[:b] + row[b+1:] for row in U]
    new_V = [row[:b] + row[b+1:] for row in V]
    new_W = [row[:b] + row[b+1:] for row in W]
    
    return {"U": new_U, "V": new_V, "W": new_W}


def _build_target_tensor():
    """Build the target tensor T[a,b,c] = 1 if (a,b,c) corresponds to valid (i,j,k)."""
    target = np.zeros((NM, MP, PN))
    for i in range(N):
        for j in range(M):
            for k in range(P):
                a = i * M + j
                b = j * P + k
                c = k * N + i
                target[a, b, c] = 1
    return target


def _gradient_descent(U, V, W, target, lr, max_steps, deadline):
    """Run gradient descent to minimize the residual between the tensor estimate and target."""
    for step in range(max_steps):
        if time.time() > deadline - 5:
            break
        
        # Compute current tensor estimate
        est = np.einsum('ar,br,cr->abc', U, V, W)
        residual = est - target
        
        # Check convergence
        if np.max(np.abs(residual)) < 1e-10:
            break
        
        # Gradients
        dU = np.einsum('abc,br,cr->ar', residual, V, W)
        dV = np.einsum('abc,ar,cr->br', residual, U, W)
        dW = np.einsum('abc,ar,br->cr', residual, U, V)
        
        # Update
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
    
    return U, V, W


def _try_random_search(rank, target, deadline, max_trials=3):
    """Try random initialization and gradient descent to find a valid decomposition."""
    for trial in range(max_trials):
        if time.time() > deadline - 10:
            break
        
        # Random initialization
        U = np.random.randn(NM, rank) * 0.5
        V = np.random.randn(MP, rank) * 0.5
        W = np.random.randn(PN, rank) * 0.5
        
        # Run gradient descent
        U, V, W = _gradient_descent(U, V, W, target, lr=0.01, max_steps=300, deadline=deadline)
        
        # Check if we found a valid decomposition
        est = np.einsum('ar,br,cr->abc', U, V, W)
        if np.max(np.abs(est - target)) < 1e-7:
            # Found a valid decomposition!
            # Convert to lists
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    return None


def construct():
    deadline = time.time() + 110
    best = _schoolbook()
    best_rank = _rank_of(best)
    
    # Try to reduce rank by merging proportional columns
    U, V, W = best["U"], best["V"], best["W"]
    rank = _rank_of(best)
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for a in range(rank):
            if time.time() > deadline - 10:
                break
            for b in range(a + 1, rank):
                if time.time() > deadline - 10:
                    break
                merged = _try_merge_columns(U, V, W, a, b)
                if merged is not None:
                    if _verify(merged["U"], merged["V"], merged["W"]):
                        U, V, W = merged["U"], merged["V"], merged["W"]
                        rank = _rank_of({"U": U, "V": V, "W": W})
                        changed = True
                        best = {"U": U, "V": V, "W": W}
                        best_rank = rank
                        break
            if changed:
                break
    
    # Try random search for lower ranks
    if time.time() < deadline - 20:
        target = _build_target_tensor()
        
        # Try various ranks below the current best
        ranks_to_try = [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 65, 70, 75]
        
        for r in ranks_to_try:
            if time.time() > deadline - 15:
                break
            if r >= best_rank:
                continue
            
            result = _try_random_search(r, target, deadline, max_trials=2)
            if result is not None:
                if _verify(result["U"], result["V"], result["W"]):
                    best = result
                    best_rank = r
                    break
    
    return best
