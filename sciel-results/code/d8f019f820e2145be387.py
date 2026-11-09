"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    target_rank = 22  # Attempt to find a rank 22 decomposition
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Helper to verify a decomposition
    def verify(U, V, W):
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Note: The problem statement says W[k*n+i, r].
        # Let's check the convention:
        # U[i, j, c] -> U[i*m + j, c]
        # V[j, k, c] -> V[j*p + k, c]
        # W[k, i, c] -> W[k*n + i, c]
        
        rank = U.shape[1]
        T = np.zeros((n, m, p), dtype=int)
        for c in range(rank):
            # Extract vectors
            u_vec = U[:, c].reshape(n, m)
            v_vec = V[:, c].reshape(m, p)
            w_vec = W[:, c].reshape(p, n)
            
            # The tensor term is u_vec[i,j] * v_vec[j,k] * w_vec[k,i]
            # We need to sum this over c for each i,j,k
            # This is a 3-way outer product contraction
            # T[i,j,k] += u_vec[i,j] * v_vec[j,k] * w_vec[k,i]
            
            for i in range(n):
                for j in range(m):
                    u_val = u_vec[i, j]
                    if u_val == 0: continue
                    for k in range(p):
                        v_val = v_vec[j, k]
                        if v_val == 0: continue
                        w_val = w_vec[k, i]
                        T[i, j, k] += u_val * v_val * w_val
        
        # Check if T matches the multiplication tensor
        # T[i,j,k] should be 1 if i==j and j==k? No.
        # Matrix multiplication C = A B. C[i,k] = sum_j A[i,j] B[j,k].
        # The tensor T maps (A, B) -> C.
        # T is the identity map on the space of linear maps?
        # The "matrix multiplication tensor" coefficients are 1 if the indices match the multiplication rule, 0 otherwise.
        # Specifically, T_{i,j,k} = 1 if i==j and j==k? No.
        # The tensor T such that T(A,B) = AB.
        # T_{i,j,k} is the coefficient of A_{i,j} B_{j,k} in C_{i,k}.
        # So T_{i,j,k} = 1 if the indices align?
        # C_{i,k} = \sum_j A_{i,j} B_{j,k}.
        # So for a fixed i,k, the term A_{i,j} B_{j,k} has coefficient 1.
        # The tensor is indexed by (i, j, k) corresponding to A_{i,j}, B_{j,k}, C_{i,k}.
        # So T[i,j,k] = 1 if the indices are consistent with the multiplication sum?
        # Actually, the standard definition of the matrix multiplication tensor $T_{n,m,p}$ has entries $T_{i,j,k} = 1$ if $i$ is row of A, $j$ is col of A / row of B, $k$ is col of B.
        # Wait, the output is C of size n x p.
        # The tensor lives in $V_A^* \otimes V_B^* \otimes V_C$.
        # $T = \sum_{i,j,k} e_i \otimes e_j \otimes e_k$ where the sum is over all valid indices?
        # No, $T = \sum_{i=1}^n \sum_{j=1}^m \sum_{k=1}^p e_i \otimes e_j \otimes e_k$?
        # If we contract T with A and B:
        # $\langle T, A \otimes B \rangle = \sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k}$.
        # We want this to equal $\sum_{i,k} C_{i,k}$? No, it should produce the matrix C.
        # The result is a tensor in $V_C$.
        # $C_{i,k} = \sum_{j} A_{i,j} B_{j,k}$.
        # So the tensor should be $T = \sum_{i,j,k} e_i \otimes e_j \otimes e_k$?
        # If $T_{i,j,k} = 1$ for all i,j,k, then $\sum_{i,j,k} A_{i,j} B_{j,k} = (\sum A) (\sum B)$, which is wrong.
        
        # The correct tensor for matrix multiplication is:
        # $T = \sum_{i=1}^n \sum_{j=1}^m \sum_{k=1}^p e_i \otimes e_j \otimes e_k$ is NOT correct.
        # The tensor is $T = \sum_{i=1}^n \sum_{k=1}^p \sum_{j=1}^m e_i \otimes e_j \otimes e_k$?
        # Let's check the contraction:
        # Result $C_{i,k} = \sum_{j} A_{i,j} B_{j,k}$.
        # This can be written as $\sum_{i,j,k} \delta_{i,j,k} A_{i,j} B_{j,k} \delta_{i,k}$? No.
        # The tensor $T$ is defined such that $T(A,B) = AB$.
        # The components are $T_{i,j,k}$?
        # Usually, the matrix multiplication tensor is $\mu_{n,m,p} = \sum_{i,j,k} e_i \otimes e_j \otimes e_k$ where the sum is over all triples? No.
        # It is $\sum_{i=1}^n \sum_{j=1}^m \sum_{k=1}^p e_i \otimes e_j \otimes e_k$?
        # If I take $A=I, B=I$, $C=I$.
        # $\sum_{i,j,k} T_{i,j,k} \delta_{i,j} \delta_{j,k} = \sum_i T_{i,i,i}$.
        # If $T_{i,j,k}=1$ for all, sum is n. But $C=I$ has trace n.
        # But for general A, B, we need $C_{i,k} = \sum_j A_{i,j} B_{j,k}$.
        # This implies the tensor is $\sum_{i,j,k} e_i \otimes e_j \otimes e_k$?
        # No, the indices of the tensor must match the contraction.
        # $T \in \mathbb{R}^{n \times m} \otimes \mathbb{R}^{m \times p} \otimes \mathbb{R}^{n \times p}$.
        # $T = \sum_{i=1}^n \sum_{j=1}^m \sum_{k=1}^p E_{i,j} \otimes E_{j,k} \otimes E_{i,k}$.
        # Here $E_{i,j}$ is the matrix with 1 at (i,j).
        # So the scalar coefficient for the basis element $E_{i,j} \otimes E_{j,k} \otimes E_{i,k}$ is 1.
        # All other coefficients are 0.
        # So $T[i, j, k]$ in a flattened sense?
        # The basis of the first space is indexed by $(i,j)$.
        # The basis of the second space is indexed by $(j,k)$.
        # The basis of the third space is indexed by $(i,k)$.
        # So the tensor is a 3-way array $T_{(i,j), (j,k), (i,k)}$.
        # It is 1 if the indices are consistent (the $j$ in first matches $j$ in second, etc), and 0 otherwise.
        # Specifically, for a fixed $j$, the term $A_{i,j} B_{j,k}$ contributes to $C_{i,k}$.
        # So for each $j$, we have a "slice" of the tensor.
        # $T = \sum_{j=1}^m \sum_{i=1}^n \sum_{k=1}^p E_{i,j} \otimes E_{j,k} \otimes E_{i,k}$.
        
        # So, $T[i, j, k]$ (where i is row of A, j is col of A/row of B, k is col of B)
        # The value is 1 if the indices form a valid multiplication term.
        # In the flattened representation used by the evaluator:
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i (Note: problem says W[k*n+i, r])
        
        # So for each c, we add $U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]$ to $T[i,j,k]$.
        # The target $T[i,j,k]$ is 1 if the indices are valid?
        # Wait, the tensor is fixed.
        # The target tensor is $T_{target}$.
        # $T_{target}[i, j, k] = 1$ if the term $A_{i,j} B_{j,k}$ contributes to $C_{i,k}$.
        # This is ALWAYS true for valid indices $i \in [0,n), j \in [0,m), k \in [0,p)$.
        # So $T_{target}[i, j, k] = 1$ for ALL $i,j,k$?
        # No, that would mean $C_{i,k} = \sum_{j} A_{i,j} B_{j,k}$?
        # If $T_{target}[i,j,k] = 1$ for all i,j,k, then
        # $\sum_{i,j,k} 1 \cdot A_{i,j} B_{j,k} = (\sum A) (\sum B)$.
        # This is NOT matrix multiplication.
        
        # Let's re-read the tensor definition carefully.
        # The tensor is in $V_A^* \otimes V_B^* \otimes V_C$.
        # $V_A$ has basis $e_{i,j}$ ($i \in n, j \in m$).
        # $V_B$ has basis $f_{j,k}$ ($j \in m, k \in p$).
        # $V_C$ has basis $g_{i,k}$ ($i \in n, k \in p$).
        # The multiplication map $M: V_A \otimes V_B \to V_C$ is linear.
        # $M(A, B) = AB$.
        # The associated tensor $T \in V_A^* \otimes V_B^* \otimes V_C$ satisfies
        # $\langle T, A \otimes B \rangle = M(A,B)$? No, $\langle T, A \otimes B \rangle$ is a vector in $V_C$?
        # No, $T$ is a 3-linear form? No, it's a tensor.
        # $T = \sum_{i,j,k} E_{i,j} \otimes E_{j,k} \otimes E_{i,k}$.
        # Here $E_{i,j}$ is the dual basis element?
        # If we contract $T$ with $A$ and $B$:
        # $T(A, B) = \sum_{i,j,k} A_{i,j} B_{j,k} E_{i,k}$.
        # This equals $\sum_{i,k} (\sum_j A_{i,j} B_{j,k}) E_{i,k} = AB$.
        # So the coefficients of the tensor in the basis $E_{i,j} \otimes E_{j,k} \otimes E_{i,k}$ are 1.
        # And 0 for any other combination?
        # The basis of $V_A^*$ is indexed by $(i,j)$.
        # The basis of $V_B^*$ is indexed by $(j,k)$.
        # The basis of $V_C$ is indexed by $(i,k)$.
        # The tensor is a sum of terms $u \otimes v \otimes w$.
        # The only non-zero entries in the "dense" representation of the tensor (if we aligned indices) would be those where the indices match the multiplication structure.
        # However, the "dense" tensor is just the identity map on the middle index summed over?
        # Actually, the tensor is sparse in the sense that for a fixed $(i,j)$ in A and $(j,k)$ in B, the output is fixed at $(i,k)$.
        # So, if we view the tensor as a 3D array $T_{\alpha, \beta, \gamma}$ where $\alpha \in [0, nm), \beta \in [0, mp), \gamma \in [0, np)$,
        # $T_{\alpha, \beta, \gamma} = 1$ if $\alpha, \beta, \gamma$ correspond to a valid multiplication term, else 0.
        # Specifically:
        # $\alpha = i \cdot m + j$
        # $\beta = j \cdot p + k$
        # $\gamma = k \cdot n + i$ (Wait, the problem says $W[k*n+i, r]$)
        # So for each valid $(i,j,k)$, we have $T[i \cdot m + j, j \cdot p + k, k \cdot n + i] = 1$.
        # All other entries are 0.
        
        # So the target tensor is sparse. It has $n \cdot m \cdot p$ ones.
        
        # My verification function above computed T by summing contributions.
        # I need to check if the computed T matches the target T.
        # Target T has 1s at specific indices.
        
        target = np.zeros((n*m, m*p, p*n), dtype=int)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    target[idx_u, idx_v, idx_w] = 1
        
        return np.array_equal(T, target)

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # Attempt to find a lower rank decomposition
    # We will try rank 22 to 35
    for r in range(22, 36):
        if time.time() > deadline:
            break
            
        # Random restarts
        for restart in range(200):
            if time.time() > deadline:
                break
                
            # Initialize U, V with random sparse integers
            # Heuristic: most entries are 0
            density = 0.2
            U = np.random.choice([-1, 0, 1], size=(n*m, r), p=[density/2, 1-density, density/2])
            V = np.random.choice([-1, 0, 1], size=(m*p, r), p=[density/2, 1-density, density/2])
            
            # Solve for W?
            # T_target[u, v, w] = sum_c U[u, c] V[v, c] W[w, c]
            # For a fixed w, we have:
            # sum_c (U[u, c] V[v, c]) W[w, c] = T_target[u, v, w]
            # This is a linear system for W[:, w] (vector of size r).
            # Variables: W[0, w], ..., W[r-1, w].
            # Equations: one for each (u, v). Total n*m*m*p = 3*3*3*4 = 108 equations.
            # If r < 108, the system is overdetermined. We can use least squares.
            # If the solution is integer, we are good.
            
            W_sol = np.zeros((p*n, r))
            valid = True
            
            for w_idx in range(p*n):
                # Extract target vector for this w_idx
                # Target[u, v, w_idx]
                # We need to form the matrix M where M[(u,v), c] = U[u, c] * V[v, c]
                # And vector b[(u,v)] = T_target[u, v, w_idx]
                
                M = np.zeros((n*m*m*p, r))
                b = np.zeros(n*m*m*p)
                
                eq_idx = 0
                for u in range(n*m):
                    for v in range(m*p):
                        # Check if this (u,v) pair contributes to this w_idx in the target
                        # Target is non-zero only if indices match.
                        # We can just use the full target tensor slice.
                        # But constructing M this way is slow.
                        
                        # Optimization: Only include equations where Target[u, v, w_idx] != 0?
                        # No, we need ALL equations to be satisfied.
                        # But if Target[u, v, w_idx] == 0, we still need sum_c ... = 0.
                        
                        # This is 108 equations per w_idx. 12 w_idx. 1296 equations total.
                        # r=22.
                        # We can solve the whole system at once?
                        # Variables: W[w, c] for all w, c. Total p*n*r = 12*22 = 264 variables.
                        # Equations: n*m*m*p = 108.
                        # Under-determined.
                        
                        # Let's solve for all W at once.
                        # System: A * W_flat = b_flat
                        # A is (108, 264).
                        # b is (108,).
                        
                        pass # Placeholder, logic moved below
            
            # Better approach: Build the big linear system for W given U, V.
            # Variables: W[w, c] for w in [0, p*n), c in [0, r).
            # Flatten W to W_flat of size p*n*r.
            # Equation for each (u, v, w):
            # sum_c U[u, c] V[v, c] W[w, c] = Target[u, v, w]
            
            # A matrix: rows are (u, v, w), cols are (w, c).
            # A[(u,v,w), (w,c)] = U[u, c] * V[v, c] if w matches, else 0.
            # Note: The column index in A corresponds to the variable W[w', c'].
            # The term W[w, c] only appears in equations where the output index is w.
            # So for each w, the sub-system is independent?
            # Yes! The equation for (u, v, w) only involves variables W[w, c].
            # So we can solve for each w independently.
            
            W_sol = np.zeros((p*n, r))
            is_integer = True
            
            for w in range(p*n):
                # Build system for this w
                # Variables: W[w, 0], ..., W[w, r-1]
                # Equations: for all u, v
                # sum_c U[u, c] V[v, c] W[w, c] = Target[u, v, w]
                
                # If Target[u, v, w] is 0 for all u,v, then W[w, :] can be anything?
                # No, it must satisfy the homogeneous system.
                # But we also need W to be integer.
                
                M = np.zeros((n*m*m*p, r))
                b = np.zeros(n*m*m*p)
                
                eq_idx = 0
                for u in range(n*m):
                    for v in range(m*p):
                        for c in range(r):
                            M[eq_idx, c] = U[u, c] * V[v, c]
                        # Get target value
                        # Target is sparse.
                        # We can precompute target values.
                        b[eq_idx] = target[u, v, w]
                        eq_idx += 1
                
                # Solve M * x = b
                # Least squares
                try:
                    x, residuals, rank, sv = np.linalg.lstsq(M, b, rcond=None)
                except np.linalg.LinAlgError:
                    continue
                
                # Check if x is close to integers
                x_rounded = np.round(x)
                if np.allclose(x, x_rounded, atol=1e-9):
                    W_sol[w, :] = x_rounded
                else:
                    is_integer = False
                    break
            
            if is_integer:
                # Verify exactness
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W_sol.tolist()
                if verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
