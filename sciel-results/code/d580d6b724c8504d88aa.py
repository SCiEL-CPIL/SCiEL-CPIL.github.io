import numpy as np
import random
import time

def construct():
    S_min = 128
    S_max = 256
    U = 128
    N_sums = S_max - S_min + 1  # 129
    N_vars = U + 1  # 129
    
    # Precompute pair structure for each sum
    # For each sum s, store list of (i,j) pairs with i<=j, i+j=s, 0<=i,j<=U
    pairs_by_sum = {s: [] for s in range(S_min, S_max + 1)}
    for i in range(U + 1):
        for j in range(i, U + 1):
            s = i + j
            if S_min <= s <= S_max:
                pairs_by_sum[s].append((i, j))
    
    # Build sparse representation of the quadratic form
    # counts[s] = sum_{(i,j) in pairs[s]} x[i] * x[j]
    # We'll represent this as: for each pair (i,j), it contributes to sum s=i+j
    
    # ADMM formulation:
    # Minimize: t + rho/2 * ||x - z||^2
    # Subject to: counts(x) <= t, counts(x) >= 1, x in [0,1], z in {0,1}
    # 
    # Split:
    # x-update: minimize t + rho/2*||x-z+lambda||^2 s.t. counts(x) <= t, counts(x)>=1, x in [0,1]
    # z-update: z = projection of (x + lambda) onto {0,1}
    # t-update: t = max(counts(x))
    # lambda-update: lambda += x - z
    
    # Simpler approach: Use ADMM with augmented Lagrangian
    # L(x,z,t,lambda) = t + rho/2*||x-z+lambda||^2 + indicator(x in [0,1]) + indicator(z in {0,1})
    #                  + indicator(counts(x) <= t) + indicator(counts(x) >= 1)
    
    # Practical ADMM:
    # 1. x-update: minimize t + rho/2*||x-z+lambda||^2 s.t. x in [0,1], counts(x) <= t, counts(x)>=1
    # 2. z-update: z = round(x + lambda) clipped to [0,1], then threshold to {0,1}
    # 3. t-update: t = max(1, max(counts(x)))
    # 4. lambda-update: lambda += x - z
    
    # For x-update, use projected gradient or simple iteration
    # Since counts is quadratic, gradient w.r.t x[i] involves all pairs containing i
    
    def compute_counts(x):
        """Compute counts for all sums given binary/continuous x"""
        counts = np.zeros(N_sums, dtype=np.float64)
        for s in range(S_min, S_max + 1):
            c = 0.0
            for (i, j) in pairs_by_sum[s]:
                c += x[i] * x[j]
            counts[s - S_min] = c
        return counts
    
    def compute_gradient(x, counts, rho, z, lam):
        """Compute gradient of ADMM objective w.r.t x"""
        grad = rho * (x - z + lam)
        # Add gradient from t term: dt/dx[i] = sum over pairs containing i of x[j]
        # But t = max(counts), so gradient is 1 for argmax counts, 0 otherwise
        # For simplicity, use subgradient: for each sum s, if counts[s] == t, add contribution
        t_val = np.max(counts)
        for s in range(S_min, S_max + 1):
            if counts[s - S_min] >= t_val - 1e-6:
                for (i, j) in pairs_by_sum[s]:
                    if i == j:
                        grad[i] += 1.0  # d(x[i]^2)/dx[i] = 2*x[i], but we're at x[i] so approx
                    else:
                        grad[i] += x[j]
                        grad[j] += x[i]
        return grad
    
    def admm_solve(max_iters=200, rho=1.0, step_size=0.1):
        """Solve BQP relaxation using ADMM"""
        # Initialize
        x = np.full(N_vars, 0.5, dtype=np.float64)
        z = np.round(x)  # Start with binary
        lam = np.zeros(N_vars, dtype=np.float64)
        t = 1.0
        
        best_x = x.copy()
        best_max_count = float('inf')
        
        for it in range(max_iters):
            # Compute current counts
            counts = compute_counts(x)
            current_max = np.max(counts)
            
            # Track best binary solution
            z_binary = (z > 0.5).astype(np.float64)
            z_counts = compute_counts(z_binary)
            if np.all(z_counts >= 0.99) and np.max(z_counts) < best_max_count:
                best_max_count = np.max(z_counts)
                best_x = z_binary.copy()
            
            # x-update: projected gradient step
            grad = compute_gradient(x, counts, rho, z, lam)
            x_new = x - step_size * grad
            x_new = np.clip(x_new, 0, 1)  # Project to [0,1]
            
            # Simple line search or fixed step
            x = x_new
            
            # t-update
            counts = compute_counts(x)
            t = max(1.0, np.max(counts))
            
            # z-update: projection onto {0,1}
            z = (x + lam > 0.5).astype(np.float64)
            
            # lambda-update
            lam = lam + x - z
            
            # Adaptively adjust rho and step_size
            if it % 50 == 49:
                rho *= 1.5
                step_size *= 0.8
        
        return best_x, best_max_count
    
    # Run ADMM with multiple initializations
    best_global_max = float('inf')
    best_global_A = None
    
    start_time = time.time()
    
    # Try different initializations
    for trial in range(10):
        if time.time() - start_time > 50:
            break
            
        if trial == 0:
            x_init = np.full(N_vars, 0.5)
        else:
            # Random binary initialization with density ~0.4
            density = 0.3 + 0.1 * trial / 10
            x_init = (np.random.random(N_vars) < density).astype(np.float64)
        
        # Run ADMM
        x_sol, max_count = admm_solve(max_iters=100, rho=1.0, step_size=0.05)
        
        # Convert to binary set
        A = set(int(i) for i in range(N_vars) if x_sol[i] > 0.5)
        
        # Verify coverage
        if len(A) == 0:
            continue
            
        counts = compute_counts((np.array([1.0 if i in A else 0.0 for i in range(N_vars)])))
        if np.any(counts < 0.99):
            # Try to fix by adding elements
            missing = [s for s in range(S_min, S_max + 1) if counts[s - S_min] < 0.99]
            for _ in range(20):
                if not missing:
                    break
                best_e = -1
                best_cov = 0
                for e in range(U + 1):
                    if e in A:
                        continue
                    cov = 0
                    for ms in missing:
                        if (ms - e) in A or ms == 2 * e:
                            cov += 1
                    if cov > best_cov:
                        best_cov = cov
                        best_e = e
                if best_e != -1:
                    A.add(best_e)
                    x_arr = np.array([1.0 if i in A else 0.0 for i in range(N_vars)])
                    counts = compute_counts(x_arr)
                    missing = [s for s in range(S_min, S_max + 1) if counts[s - S_min] < 0.99]
                else:
                    break
        
        if np.any(counts < 0.99):
            continue
        
        current_max = int(np.max(counts))
        if current_max < best_global_max:
            best_global_max = current_max
            best_global_A = set(A)
    
    # Local search refinement on best solution
    if best_global_A is not None:
        A = set(best_global_A)
        K = len(A)
        
        counts = np.zeros(N_sums, dtype=np.int32)
        A_list = list(A)
        for i_idx in range(len(A_list)):
            a = A_list[i_idx]
            for j_idx in range(i_idx, len(A_list)):
                b = A_list[j_idx]
                s = a + b
                if S_min <= s <= S_max:
                    counts[s - S_min] += 1
        
        current_max = int(np.max(counts))
        
        # Local search
        for step in range(2000):
            if time.time() - start_time > 55:
                break
                
            a = random.choice(list(A))
            b = random.randint(0, U)
            if b in A:
                continue
            
            # Calculate delta
            valid = True
            new_counts = counts.copy()
            
            # Remove pairs with a
            for x in A:
                s1 = a + x
                if S_min <= s1 <= S_max:
                    new_counts[s1 - S_min] -= 1
                    if new_counts[s1 - S_min] < 0:
                        valid = False
                        break
            if not valid:
                continue
            
            # Add pairs with b
            for x in A:
                if x == a:
                    continue
                s2 = b + x
                if S_min <= s2 <= S_max:
                    new_counts[s2 - S_min] += 1
            s2 = 2 * b
            if S_min <= s2 <= S_max:
                new_counts[s2 - S_min] += 1
            
            if np.any(new_counts < 1):
                continue
            
            new_max = int(np.max(new_counts))
            if new_max < current_max:
                A.remove(a)
                A.add(b)
                counts = new_counts
                current_max = new_max
    
    if best_global_A is None:
        # Fallback: dense set
        return list(range(64, 129))
    
    return list(best_global_A)
