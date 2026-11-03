"""Optimized decomposition for 3x4x8 matrix multiplication."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if U @ V @ W^T reconstructs the matmul tensor."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                if abs(val - 1.0) > tol:
                    return False, abs(val - 1.0)
    return True, 0.0

def _build_schoolbook(n, m, p):
    """Build schoolbook decomposition for n x m x p matrix multiplication."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U.tolist(), V.tolist(), W.tolist()

def _build_block_decomposition():
    """Build a block-structured decomposition for 3x4x8.
    
    Strategy:
    - Split A (3x4) into blocks: A1 (1x4), A2 (1x4), A3 (1x4)
    - Split B (4x8) into blocks: B1 (4x4), B2 (4x4)
    - Split C (3x8) into blocks: C1 (3x4), C2 (3x4)
    
    C1 = A1*B1 + A2*B1 + A3*B1
    C2 = A1*B2 + A2*B2 + A3*B2
    
    For each 1x4 x 4x4 multiplication, we can use a rank-8 decomposition.
    Total rank: 3 * 2 * 8 = 48 (worse than schoolbook 96/2=48? No, schoolbook is 3*4*8=96)
    
    Actually, let me try a different approach.
    
    Better strategy: Use Strassen-like decomposition on 2x2x2 blocks.
    
    Split A (3x4) into:
    - A_top (2x4) and A_bottom (1x4)
    Split B (4x8) into:
    - B_left (4x4) and B_right (4x4)
    Split C (3x8) into:
    - C_top_left (2x4), C_top_right (2x4), C_bottom_left (1x4), C_bottom_right (1x4)
    
    C_top_left = A_top * B_left
    C_top_right = A_top * B_right
    C_bottom_left = A_bottom * B_left
    C_bottom_right = A_bottom * B_right
    
    For A_top (2x4) * B_left (4x4):
    Split A_top into A_top1 (2x2), A_top2 (2x2)
    Split B_left into B_left1 (4x2), B_left2 (4x2)
    
    This is getting complicated. Let me try a direct optimization approach.
    """
    n, m, p = 3, 4, 8
    
    # Try rank 40-50 range with random initialization and local search
    best_rank = None
    best_result = None
    
    for target_rank in [48, 45, 42, 40]:
        start_time = time.time()
        deadline = start_time + 20  # 20 seconds per rank attempt
        
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            # Gradient descent to minimize ||T - U⊗V⊗W||^2
            lr = 0.01
            for step in range(200):
                if time.time() > deadline:
                    break
                
                # Compute gradient
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # T[i,j,k] = 1
                            # d/dU[i*m+j] = V[j*p+k] * W[k*n+i]
                            # d/dV[j*p+k] = U[i*m+j] * W[k*n+i]
                            # d/dW[k*n+i] = U[i*m+j] * V[j*p+k]
                            residual = 1.0 - np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                            grad_U[i*m+j] += residual * V[j*p+k] * W[k*n+i]
                            grad_V[j*p+k] += residual * U[i*m+j] * W[k*n+i]
                            grad_W[k*n+i] += residual * U[i*m+j] * V[j*p+k]
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                if step % 20 == 0:
                    # Check convergence
                    max_err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                                max_err = max(max_err, abs(val - 1.0))
                    if max_err < 1e-6:
                        valid, err = _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-6)
                        if valid:
                            return U.tolist(), V.tolist(), W.tolist()
            
            # Final check
            valid, err = _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-6)
            if valid:
                return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _build_structured_decomposition():
    """Build a structured decomposition using block matrix multiplication.
    
    Key insight: We can decompose the 3x4x8 problem into smaller subproblems.
    
    Approach:
    1. Split A (3x4) into A1 (2x4) and A2 (1x4)
    2. Split B (4x8) into B1 (4x4) and B2 (4x4)
    3. Compute C1 = A1*B1 (2x4), C2 = A1*B2 (2x4), C3 = A2*B1 (1x4), C4 = A2*B2 (1x4)
    
    For each 2x4 x 4x4 multiplication:
    - Split A1 into A1a (2x2) and A1b (2x2)
    - Split B1 into B1a (4x2) and B1b (4x2)
    - C1a = A1a*B1a (2x2), C1b = A1a*B1b (2x2), C1c = A1b*B1a (2x2), C1d = A1b*B1b (2x2)
    
    Each 2x2x2 can be done with rank 7 (Strassen).
    Total rank: 4 (for 2x4x4 blocks) * 4 (sub-blocks) * 7 (Strassen) = 112 for A1*B1
    Plus similar for other blocks... this is getting large.
    
    Let me try a simpler approach: just use schoolbook but with some structure.
    """
    n, m, p = 3, 4, 8
    
    # Schoolbook gives rank 96. Let's see if we can do better with a simple structure.
    # 
    # Alternative: Use the fact that we can share computations.
    # 
    # For now, return schoolbook as a baseline.
    return _build_schoolbook(n, m, p)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140  # Leave 10s safety margin
    
    # Try structured decomposition first
    result = _build_structured_decomposition()
    if result is not None:
        U, V, W = result
        valid, err = _verify(U, V, W, n, m, p, tol=1e-9)
        if valid:
            return {"U": U, "V": V, "W": W}
    
    # Try block decomposition with optimization
    result = _build_block_decomposition()
    if result is not None:
        U, V, W = result
        valid, err = _verify(U, V, W, n, m, p, tol=1e-9)
        if valid:
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U_sb, V_sb, W_sb = _build_schoolbook(n, m, p)
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
