import numpy as np
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    adj = np.array(adjacency, dtype=np.int8)
    # Ensure symmetry and zero diagonal? 
    # Task says "unweighted 100-node graphs", usually simple graphs.
    # We'll assume A is symmetric and 0 diagonal.
    
    # Symmetrize just in case
    adj = (adj + adj.T) // 2
    np.fill_diagonal(adj, 0)
    
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin for 150s limit
    
    # Convert to sparse for faster matrix-vector products
    A_sp = sp.csr_matrix(adj)
    
    # Precompute degrees
    deg = adj.sum(axis=1)
    
    def get_cut_size(x):
        # x is +/- 1
        # Cut size = 0.25 * sum_i (deg_i - x_i * (A x)_i)
        S = A_sp @ x
        return 0.25 * np.sum(deg - x * S)
    
    def local_search(x, T_start=10.0, alpha=0.95, steps=5000):
        """Simulated Annealing followed by Greedy Polish."""
        S = A_sp @ x
        C = 0.25 * np.sum(deg - x * S)
        
        T = T_start
        # Use a fixed local seed or global? Global is better for diversity if not resetting.
        # But we want reproducibility per restart? 
        # Let's use a local RNG for each restart to ensure independence?
        # Actually, using a global RNG stream is fine.
        
        for _ in range(steps):
            if time.time() > deadline - 5:
                break
            i = np.random.randint(n)
            gain = x[i] * S[i]
            
            if gain > 0:
                accept = True
            else:
                if T > 1e-6:
                    prob = np.exp(gain / T)
                else:
                    prob = 0.0
                accept = np.random.random() < prob
                
            if accept:
                v = x[i]
                x[i] = -v
                C += gain
                S -= 2 * v * A_sp[:, i].toarray().ravel()
            
            T *= alpha
            
            if T < 1e-6:
                break
        
        # Greedy Polish
        while True:
            if time.time() > deadline - 5:
                break
            gains = x * S
            i = np.argmax(gains)
            max_gain = gains[i]
            if max_gain <= 0:
                break
            v = x[i]
            x[i] = -v
            C += max_gain
            S -= 2 * v * A_sp[:, i].toarray().ravel()
            
        return C, x
    
    best_C = -1
    best_x = None
    
    # 1. Spectral Initialization
    # Solve (L + lambda I) v = 0? 
    # L = D - A.
    # We want to minimize x^T L x subject to x^T x = n?
    # This is equivalent to finding the Fiedler vector of L.
    # L = D - A.
    # Since we want Max Cut, we want to minimize x^T (D-A) x.
    # The smallest eigenvalue of L is 0 (for connected graph). 
    # The second smallest (Fiedler) gives the best partition.
    # However, for Max Cut, the sign of the Fiedler vector is a good heuristic.
    
    D_sp = sp.diags(deg)
    L_sp = D_sp - A_sp
    
    # Use eigsh to find the smallest algebraic eigenvalue (Fiedler vector)
    # k=2 to get the non-trivial one
    try:
        # sigma=0 to shift invert? Or just 'SM' (Smallest Magnitude)?
        # 'SA' is Smallest Algebraic.
        # For L, smallest is 0. Second smallest is Fiedler.
        # Let's try to get the second smallest.
        # scipy.sparse.linalg.eigsh(L_sp, k=2, which='SA')
        # Sometimes fails if graph is disconnected or numerical issues.
        # Fallback to random if fails.
        
        # Shift to avoid 0 eigenvalue issue?
        # L + I has smallest eigenvalue 1.
        # Let's just use 'SM' (smallest magnitude) or 'SA'.
        # 'SA' is safer for Fiedler.
        
        # If n is small, use dense? n=100, dense is fine.
        # L_dense = L_sp.toarray()
        # eigvals, eigvecs = np.linalg.eigh(L_dense)
        # fiedler = eigvecs[:, 1]
        
        # Dense is faster for n=100 and more robust.
        L_dense = L_sp.toarray()
        eigvals, eigvecs = np.linalg.eigh(L_dense)
        
        # The Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue.
        # eigvals are sorted ascending.
        # eigvecs[:, 0] corresponds to eigvals[0] (usually 0).
        # eigvecs[:, 1] corresponds to eigvals[1] (Fiedler).
        fiedler = eigvecs[:, 1]
        
        # Initialize x based on sign of fiedler
        x_spectral = np.sign(fiedler)
        # Handle zeros (rare)
        x_spectral[x_spectral == 0] = 1
        
        C_spec, x_spec = local_search(x_spectral.copy())
        if C_spec > best_C:
            best_C = C_spec
            best_x = x_spec.copy()
            
    except Exception:
        pass
    
    # 2. Random Restarts
    # Run multiple random restarts with SA + Polish
    
    # Estimate how many restarts we can do
    # Each restart takes ~steps * (time per step)
    # Time per step is dominated by A_sp @ e_i (vector update) and exp.
    # Let's run as many as we can until deadline - 5s.
    
    restart_count = 0
    while time.time() < deadline - 5:
        restart_count += 1
        
        # Random initialization
        x_random = np.random.choice([-1, 1], size=n)
        
        # Maybe use a slightly higher T for random starts?
        # Or vary T.
        T_start = 5.0 + np.random.random() * 5.0
        
        C_rand, x_rand = local_search(x_random.copy(), T_start=T_start)
        
        if C_rand > best_C:
            best_C = C_rand
            best_x = x_rand.copy()
        
        # Optional: If we have time, do a second pass on the best solution found so far?
        # Or just keep generating.
        
        # Early exit if we hit a very high cut?
        # Max possible cut is E.
        # If best_C == 0.5 * sum(adj), we can't do better.
        max_possible = 0.5 * np.sum(adj)
        if best_C >= max_possible - 1e-6:
            break

    if best_x is None:
        # Fallback
        best_x = np.random.choice([-1, 1], size=n)
    
    # Convert -1/1 to 0/1
    # 1 -> 1, -1 -> 0
    labels = (best_x + 1) // 2
    return labels.astype(int)
