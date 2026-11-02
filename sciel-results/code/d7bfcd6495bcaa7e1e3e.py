"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140.0

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    best = schoolbook()
    best_rank = n * m * p

    # Build the tensor T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i
    # T[a,b,c] = 1 if there exists (i,j,k) with a=i*m+j, b=j*p+k, c=k*n+i, else 0
    T = np.zeros((n*m, m*p, p*n))
    nz = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                nz.append((a, b, c))

    # Precompute for each a: list of (b, c) where T[a,b,c]=1
    a_to_bc = {a: [] for a in range(n*m)}
    for a, b, c in nz:
        a_to_bc[a].append((b, c))

    # For each b: list of (a, c) where T[a,b,c]=1
    b_to_ac = {b: [] for b in range(m*p)}
    for a, b, c in nz:
        b_to_ac[b].append((a, c))

    # For each c: list of (a, b) where T[a,b,c]=1
    c_to_ab = {c: [] for c in range(p*n)}
    for a, b, c in nz:
        c_to_ab[c].append((a, b))

    # Vectorized ALS: for fixed V,W, solve for U row by row
    # For each a, the equations are:
    #   sum_t U[a,t] * V[b,t] * W[c,t] = 1  for (b,c) in a_to_bc[a]
    #   sum_t U[a,t] * V[b,t] * W[c,t] = 0  for all other (b,c)
    # This is a linear system: M_a @ U[a,:] = rhs_a
    # where M_a has rows indexed by all (b,c) pairs, and M_a[(b,c), t] = V[b,t]*W[c,t]

    # Actually, we only need the nonzero constraints plus enough zero constraints to make it square/determined.
    # But the full system is (m*p)*(p*n) x r which is huge. Let's use a subset.
    # Better approach: use the fact that we only need to satisfy the 60 nonzero entries and 
    # ensure the rest are zero. But that's still a large system.
    
    # Alternative: use gradient descent with momentum, which is more flexible.
    
    def snap_and_verify(U, V, W):
        """Try to snap to half-integers and verify exactness."""
        U2 = np.round(U * 2) / 2
        V2 = np.round(V * 2) / 2
        W2 = np.round(W * 2) / 2
        E = np.einsum('at,bt,ct->abc', U2, V2, W2)
        err = np.sum((T - E) ** 2)
        if err < 1e-9:
            return U2, V2, W2
        return None

    # Try gradient-based optimization with multiple restarts
    # Loss = sum_{a,b,c} (sum_t U[a,t]*V[b,t]*W[c,t] - T[a,b,c])^2
    
    # Precompute for faster gradient computation
    # For gradient w.r.t. U[a,t]: 2 * sum_{b,c} (E[a,b,c] - T[a,b,c]) * V[b,t] * W[c,t]
    # This can be computed as: 2 * (E - T) @ (V[:,t] outer W[:,t]) for each a,t
    
    # More efficiently: 
    # grad_U = 2 * einsum('abc,bt,ct->at', E - T, V, W)
    # grad_V = 2 * einsum('abc,at,ct->bt', E - T, U, W)  
    # grad_W = 2 * einsum('abc,at,bt->ct', E - T, U, V)
    
    def compute_loss(U, V, W):
        E = np.einsum('at,bt,ct->abc', U, V, W)
        return np.sum((E - T) ** 2)

    def compute_grads(U, V, W):
        E = np.einsum('at,bt,ct->abc', U, V, W)
        diff = E - T
        grad_U = 2.0 * np.einsum('abc,bt,ct->at', diff, V, W)
        grad_V = 2.0 * np.einsum('abc,at,ct->bt', diff, U, W)
        grad_W = 2.0 * np.einsum('abc,at,bt->ct', diff, U, V)
        return grad_U, grad_V, grad_W, np.sum(diff * diff)

    np.random.seed(42)

    # Try ranks from lower bound to best_rank-1
    lb = max(n*m, m*p, p*n)  # = max(12, 20, 15) = 20

    for r in range(lb, best_rank):
        if time.time() > deadline - 20:
            break

        # Number of trials depends on rank
        num_trials = 30 if r > 40 else 50

        for trial in range(num_trials):
            if time.time() > deadline - 15:
                break

            # Initialize with small random values
            scale = 1.0 / np.sqrt(r)
            U = scale * np.random.randn(n*m, r)
            V = scale * np.random.randn(m*p, r)
            W = scale * np.random.randn(p*n, r)

            # Adam optimizer
            lr = 0.01
            beta1 = 0.9
            beta2 = 0.999
            eps = 1e-8
            m_U = np.zeros_like(U)
            v_U = np.zeros_like(U)
            m_V = np.zeros_like(V)
            v_V = np.zeros_like(V)
            m_W = np.zeros_like(W)
            v_W = np.zeros_like(W)
            t_step = 0

            best_loss = np.inf
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

            for it in range(3000):
                if time.time() > deadline - 10:
                    break

                grad_U, grad_V, grad_W, loss = compute_grads(U, V, W)

                if loss < best_loss:
                    best_loss = loss
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

                if loss < 1e-10:
                    break

                t_step += 1
                bc1_1 = 1.0 - beta1 ** t_step
                bc1_2 = 1.0 - beta2 ** t_step

                m_U = beta1 * m_U + (1 - beta1) * grad_U
                v_U = beta2 * v_U + (1 - beta2) * grad_U ** 2
                m_U_hat = m_U / bc1_1
                v_U_hat = v_U / bc1_2
                U -= lr * m_U_hat / (np.sqrt(v_U_hat) + eps)

                m_V = beta1 * m_V + (1 - beta1) * grad_V
                v_V = beta2 * v_V + (1 - beta2) * grad_V ** 2
                m_V_hat = m_V / bc1_1
                v_V_hat = v_V / bc1_2
                V -= lr * m_V_hat / (np.sqrt(v_V_hat) + eps)

                m_W = beta1 * m_W + (1 - beta1) * grad_W
                v_W = beta2 * v_W + (1 - beta2) * grad_W ** 2
                m_W_hat = m_W / bc1_1
                v_W_hat = v_W / bc1_2
                W -= lr * m_W_hat / (np.sqrt(v_W_hat) + eps)

            # Check best found in this trial
            if best_loss < 1e-8:
                result = snap_and_verify(best_U, best_V, best_W)
                if result is not None:
                    U2, V2, W2 = result
                    return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}

    return best
# EVOLVE-BLOCK-END
