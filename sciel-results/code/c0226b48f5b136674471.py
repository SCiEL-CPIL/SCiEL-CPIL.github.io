"""Matrix multiplication 2x4x5 rank-minimal tensor decomposition search."""
import time
import random
import math
import numpy as np

DIMENSIONS = [2, 4, 5]
n, m, p = DIMENSIONS
N = n * m  # 8
M = m * p  # 20
P = p * n  # 10

# Deadline guard
T_START = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0

def time_left():
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - T_START)

def tensor_coeff(i, j, k, r):
    """Return U[i,j,r] * V[j,k,r] * W[k,i,r] for a given factorization."""
    # We'll compute this via the matrices
    pass

def verify(U, V, W, rank):
    """Check if the decomposition is exact. Returns True if valid."""
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Must equal 1 for all i,j,k (since it's matrix mult identity tensor)
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape for broadcasting
    # U_arr: (n, m, rank)
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # Compute via einsum
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    # Check if T is all ones
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

def verify_exact(U, V, W, rank):
    """Check with tighter tolerance for integer/half-integer entries."""
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    # For exact integer/half-integer, T should be exactly 1
    # Use a very tight tolerance
    if not np.allclose(T, 1.0, atol=1e-10):
        return False
    return True

def schoolbook():
    """Return the schoolbook decomposition of rank n*m*p = 40."""
    rank = n * m * p
    U = [[0] * rank for _ in range(N)]
    V = [[0] * rank for _ in range(M)]
    W = [[0] * rank for _ in range(P)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def try_rank(rank, max_iters=5000, seed=42):
    """Try to find a decomposition of given rank using random search + refinement."""
    if time_left() < 2:
        return None
    
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    
    # Initialize with random factors
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    # Use a structured initialization based on schoolbook but with rank < n*m*p
    # This is hard, so let's try random initialization with refinement
    
    for attempt in range(20):  # Multiple random restarts
        if time_left() < 5:
            break
        
        # Random initialization
        U = np_rng.normal(size=(N, rank))
        V = np_rng.normal(size=(M, rank))
        W = np_rng.normal(size=(P, rank))
        
        # Scale to get reasonable magnitudes
        U *= 1.0 / math.sqrt(rank)
        V *= 1.0 / math.sqrt(rank)
        W *= 1.0 / math.sqrt(rank)
        
        # Gradient descent on the loss: sum_{i,j,k} (sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] - 1)^2
        lr = 0.1
        for it in range(max_iters):
            if time_left() < 5:
                break
            
            # Compute T[i,j,k]
            U3 = U.reshape(n, m, rank)
            V3 = V.reshape(m, p, rank)
            W3 = W.reshape(p, n, rank)
            
            T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
            err = T - 1.0
            loss = np.sum(err * err)
            
            if loss < best_err:
                best_err = loss
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if loss < 1e-12:
                break
            
            # Gradients
            # dL/dU[i,j,r] = 2 * err[i,j,k] * V[j,k,r] * W[k,i,r] summed over k
            # dL/dV[j,k,r] = 2 * err[i,j,k] * U[i,j,r] * W[k,i,r] summed over i
            # dL/dW[k,i,r] = 2 * err[i,j,k] * U[i,j,r] * V[j,k,r] summed over j
            
            dU = 2 * np.einsum('ijk,jkr,kir->ijr', err, V3, W3)
            dV = 2 * np.einsum('ijk,ijr,kir->jkr', err, U3, W3)
            dW = 2 * np.einsum('ijk,ijr,jkr->kir', err, U3, V3)
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        if best_err < 1e-10:
            break
    
    if best_err < 1e-10 and best_U is not None:
        # Try to snap to integer/half-integer values
        U_snap = np.round(best_U * 2) / 2
        V_snap = np.round(best_V * 2) / 2
        W_snap = np.round(best_W * 2) / 2
        
        # Verify the snapped version
        U3 = U_snap.reshape(n, m, rank)
        V3 = V_snap.reshape(m, p, rank)
        W3 = W_snap.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        
        if np.allclose(T, 1.0, atol=1e-9):
            # Convert to lists
            U_list = U_snap.tolist()
            V_list = V_snap.tolist()
            W_list = W_snap.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    """Main entry point. Returns the best decomposition found."""
    global T_START
    T_START = time.time()
    
    # Start with schoolbook
    result = schoolbook()
    best_rank = n * m * p  # 40
    
    # Try to find lower rank decompositions
    # Known bounds: for 2x4x5, the rank is between some lower bound and 40
    # Let's try ranks from a lower bound up to 39
    
    # Lower bound: max(N*M/P, M*P/N, P*N/M) approximately, but more carefully:
    # The tensor has dimension 8*20*10 = 1600 entries, but structured
    # A simple lower bound is max(n*m, m*p, p*n) = max(8, 20, 10) = 20
    # But that's not tight. Let's just try various ranks.
    
    for rank in range(20, best_rank):
        if time_left() < 10:
            break
        
        res = try_rank(rank, max_iters=2000, seed=rank * 1000 + 42)
        if res is not None:
            # Verify
            if verify(res["U"], res["V"], res["W"], rank):
                best_rank = rank
                result = res
                break  # Found the lowest rank, stop
    
    return result

# For testing
if __name__ == "__main__":
    result = construct()
    print(f"Best rank: {len(result['U'][0])}")
    print(f"U shape: ({len(result['U'])}, {len(result['U'][0])})")
    print(f"V shape: ({len(result['V'])}, {len(result['V'][0])})")
    print(f"W shape: ({len(result['W'])}, {len(result['W'][0])})")
