import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin before 150s kill
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook rank as fallback
    schoolbook_rank = n * m * p
    U_sch = np.zeros((n*m, schoolbook_rank))
    V_sch = np.zeros((m*p, schoolbook_rank))
    W_sch = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sch[i*m+j, col] = 1
                V_sch[j*p+k, col] = 1
                W_sch[k*n+i, col] = 1
                col += 1
    
    best_result = {"U": U_sch.tolist(), "V": V_sch.tolist(), "W": W_sch.tolist()}
    
    def check_exact(U, V, W):
        """Check if U,V,W exactly decompose T"""
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T, T_approx, atol=1e-10)
    
    def try_rounding(U, V, W):
        """Try to round to integers or half-integers and check exactness"""
        candidates = []
        
        # Integer rounding
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        candidates.append((U_int, V_int, W_int))
        
        # Half-integer rounding
        U_half = np.round(U * 2) / 2
        V_half = np.round(V * 2) / 2
        W_half = np.round(W * 2) / 2
        candidates.append((U_half, V_half, W_half))
        
        # Mixed: try rounding each matrix independently to int or half-int
        for u_mode in ['int', 'half']:
            for v_mode in ['int', 'half']:
                for w_mode in ['int', 'half']:
                    U_r = np.round(U) if u_mode == 'int' else np.round(U * 2) / 2
                    V_r = np.round(V) if v_mode == 'int' else np.round(V * 2) / 2
                    W_r = np.round(W) if w_mode == 'int' else np.round(W * 2) / 2
                    candidates.append((U_r, V_r, W_r))
        
        # Small perturbations + rounding
        for _ in range(30):
            U_try = np.round(U + np.random.randn(*U.shape) * 0.15)
            V_try = np.round(V + np.random.randn(*V.shape) * 0.15)
            W_try = np.round(W + np.random.randn(*W.shape) * 0.15)
            candidates.append((U_try, V_try, W_try))
            
            U_try2 = np.round((U + np.random.randn(*U.shape) * 0.15) * 2) / 2
            V_try2 = np.round((V + np.random.randn(*V.shape) * 0.15) * 2) / 2
            W_try2 = np.round((W + np.random.randn(*W.shape) * 0.15) * 2) / 2
            candidates.append((U_try2, V_try2, W_try2))
        
        for U_c, V_c, W_c in candidates:
            if check_exact(U_c, V_c, W_c):
                return U_c, V_c, W_c
        
        return None
    
    # Key insight: initialize by merging schoolbook terms
    # Each schoolbook term is a rank-1 tensor u⊗v⊗w
    # To get rank r < 45, we need to combine terms
    # Strategy: randomly assign schoolbook terms to r groups, sum within each group
    
    def init_by_merging(target_rank):
        """Initialize U,V,W by merging schoolbook terms into target_rank groups"""
        U = np.zeros((n*m, target_rank))
        V = np.zeros((m*p, target_rank))
        W = np.zeros((p*n, target_rank))
        
        # Randomly assign each schoolbook term to one of target_rank groups
        assignment = np.random.randint(0, target_rank, size=schoolbook_rank)
        for idx in range(schoolbook_rank):
            g = assignment[idx]
            U[:, g] += U_sch[:, idx]
            V[:, g] += V_sch[:, idx]
            W[:, g] += W_sch[:, idx]
        
        return U, V, W
    
    def init_by_partial_merge(target_rank):
        """Initialize by keeping some schoolbook terms and merging others"""
        U = np.zeros((n*m, target_rank))
        V = np.zeros((m*p, target_rank))
        W = np.zeros((p*n, target_rank))
        
        if target_rank >= schoolbook_rank:
            U[:, :target_rank] = U_sch
            V[:, :target_rank] = V_sch
            W[:, :target_rank] = W_sch
            return U, V, W
        
        # Keep first (target_rank - merge_count) terms as-is, merge the rest
        # Try different merge strategies
        merge_count = schoolbook_rank - target_rank
        if merge_count <= 0:
            U[:, :target_rank] = U_sch[:, :target_rank]
            V[:, :target_rank] = V_sch[:, :target_rank]
            W[:, :target_rank] = W_sch[:, :target_rank]
            return U, V, W
        
        # Randomly select which terms to keep separate
        keep_separate = np.random.choice(schoolbook_rank, size=min(target_rank, schoolbook_rank), replace=False)
        keep_set = set(keep_separate.tolist())
        
        for idx in range(schoolbook_rank):
            if idx in keep_set:
                # Find its position in the kept list
                pos = list(keep_set).index(idx)
                if pos < target_rank:
                    U[:, pos] += U_sch[:, idx]
                    V[:, pos] += V_sch[:, idx]
                    W[:, pos] += W_sch[:, idx]
            else:
                # Assign to a random kept term
                if len(keep_set) > 0:
                    pos = np.random.choice(list(keep_set))
                    pos = list(keep_set).index(pos)
                    if pos < target_rank:
                        U[:, pos] += U_sch[:, idx]
                        V[:, pos] += V_sch[:, idx]
                        W[:, pos] += W_sch[:, idx]
        
        return U, V, W
    
    # Lower bound is max(n*m, m*p, p*n) = 15
    lower_bound = max(n*m, m*p, p*n)
    
    for target_rank in range(lower_bound, schoolbook_rank + 1):
        if time.time() > deadline - 15:
            break
        
        max_restarts = 15 if target_rank < 30 else 8
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            # Different initialization strategies
            if restart == 0:
                U, V, W = init_by_merging(target_rank)
            elif restart == 1:
                U, V, W = init_by_partial_merge(target_rank)
            elif restart == 2:
                # Random init with small values
                U = np.random.randn(n*m, target_rank) * 0.5
                V = np.random.randn(m*p, target_rank) * 0.5
                W = np.random.randn(p*n, target_rank) * 0.5
            elif restart == 3:
                # Structured: try to exploit symmetry
                U = np.zeros((n*m, target_rank))
                V = np.zeros((m*p, target_rank))
                W = np.zeros((p*n, target_rank))
                # Fill with random ±1 patterns
                for r in range(target_rank):
                    U[:, r] = np.random.choice([-1, 0, 1], size=n*m)
                    V[:, r] = np.random.choice([-1, 0, 1], size=m*p)
                    W[:, r] = np.random.choice([-1, 0, 1], size=p*n)
            else:
                # More random merges with different seeds
                U, V, W = init_by_merging(target_rank)
                # Add small random noise
                U += np.random.randn(*U.shape) * 0.1
                V += np.random.randn(*V.shape) * 0.1
                W += np.random.randn(*W.shape) * 0.1
            
            # Gradient descent with adaptive learning rate
            lr = 0.02
            momentum = 0.9
            vel_U = np.zeros_like(U)
            vel_V = np.zeros_like(V)
            vel_W = np.zeros_like(W)
            
            max_iters = 1500
            best_loss = np.inf
            patience = 0
            
            for it in range(max_iters):
                if time.time() > deadline - 8:
                    break
                
                # Compute T_approx
                T_approx = np.einsum('ar,br,cr->abc', U, V, W)
                diff = T - T_approx
                loss = np.sum(diff**2)
                
                if loss < best_loss:
                    best_loss = loss
                    patience = 0
                else:
                    patience += 1
                
                if patience > 300 and it > 500:
                    break
                
                # Gradients
                grad_U = -2 * np.einsum('abc,br,cr->ar', diff, V, W)
                grad_V = -2 * np.einsum('abc,ar,cr->br', diff, U, W)
                grad_W = -2 * np.einsum('abc,ar,br->cr', diff, U, V)
                
                # Momentum update
                vel_U = momentum * vel_U + lr * grad_U
                vel_V = momentum * vel_V + lr * grad_V
                vel_W = momentum * vel_W + lr * grad_W
                
                U -= vel_U
                V -= vel_V
                W -= vel_W
                
                # Periodically check for exact solution
                if it % 100 == 0:
                    rounded = try_rounding(U, V, W)
                    if rounded is not None:
                        return {"U": rounded[0].tolist(), "V": rounded[1].tolist(), "W": rounded[2].tolist()}
            
            # Final check
            rounded = try_rounding(U, V, W)
            if rounded is not None:
                return {"U": rounded[0].tolist(), "V": rounded[1].tolist(), "W": rounded[2].tolist()}
    
    return best_result
# EVOLVE-BLOCK-END
