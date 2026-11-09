import time
import random
from fractions import Fraction
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import sys

def _get_primes(limit=1000):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    primes_all = _get_primes(1000)
    
    # Strategy: Try MILP first. If it fails or is too slow, fallback to SA.
    # MILP setup
    # We restrict primes to a reasonable subset to keep variable count manageable.
    # Primes up to 300 (approx 62 primes)
    # N = 2000 (search for prefix up to 2000)
    
    def try_milp():
        if time.time() > deadline - 60:
            return None
            
        primes = [p for p in primes_all if p <= 300]
        N = 2000
        
        # Variables:
        # x_{p,r} for each p in primes, r in 0..p-1
        # y_n for n in 1..N
        # z_n for n in 1..N
        
        num_x = sum(p for p in primes)
        num_y = N
        num_z = N
        
        total_vars = num_x + num_y + num_z
        
        # Objective: Maximize sum(z_n)
        # min -sum(z_n)
        c = np.zeros(total_vars)
        # z_n indices: num_x + num_y to num_x + num_y + N - 1
        z_start = num_x + num_y
        c[z_start:z_start+N] = -1.0
        
        # Bounds
        # x, y, z are binary [0, 1]
        lb = np.zeros(total_vars)
        ub = np.ones(total_vars)
        integrality = np.ones(total_vars)
        
        # Constraints:
        # 1. Budget: sum(1/p * x_{p,r}) <= 1
        #    Indices for x are 0..num_x-1
        #    We need to map (p, r) to index
        #    Let's build the constraint matrix A for budget
        #    Row 0 of A: coefficients for x vars
        
        A_rows = []
        b_lower = []
        b_upper = []
        
        # Budget constraint
        budget_coeff = np.zeros(num_x)
        idx = 0
        prime_indices = {} # (p, r) -> index in x
        for p in primes:
            for r in range(p):
                prime_indices[(p, r)] = idx
                budget_coeff[idx] = 1.0 / p
                idx += 1
        
        A_rows.append(budget_coeff)
        b_lower.append(0.0)
        b_upper.append(1.0)
        
        # Coverage constraints:
        # For each n in 1..N:
        # y_n <= sum_{p,r: n % p == r} x_{p,r}
        # sum_{p,r: n % p == r} x_{p,r} - y_n >= 0
        # And y_n >= x_{p,r} for all such p,r? 
        # Actually, if we maximize sum(z) and z <= y, and y <= sum(x), 
        # and we don't have a lower bound on y relative to x, 
        # y could be 0 even if x=1. 
        # But z <= y, so z=0. 
        # We want z=1 if covered. 
        # So we need y_n to be 1 if ANY x covering n is 1.
        # This is an OR.
        # y_n >= x_{p,r} for all (p,r) covering n.
        # And y_n <= sum(x_{p,r}) covering n.
        
        # This creates many constraints.
        # For each n, and for each (p,r) covering n:
        # y_n - x_{p,r} >= 0
        
        # And for each n:
        # sum(x_{p,r}) - y_n >= 0
        
        # This is O(N * K) constraints, where K is avg number of primes covering n.
        # N=2000, K~10. 20,000 constraints.
        # Variables ~ 5000.
        # This is a sparse matrix.
        
        # Let's use scipy sparse matrices for efficiency.
        from scipy.sparse import lil_matrix
        
        # Estimate rows
        # 1 budget row
        # N rows for sum(x) - y >= 0
        # Sum over n of (number of (p,r) covering n) rows for y - x >= 0
        
        # Instead of building explicitly in a loop which is slow in Python,
        # let's just build the dense matrix if it's small enough, or use lil.
        
        # Total rows approx 1 + N + Sum(counts)
        # Sum(counts) = sum over p of N (since each p covers N/p numbers, and we have p residues? No.)
        # For a fixed p, each residue r covers N/p numbers.
        # Total (p,r) pairs covering a specific n is number of p dividing... no, n % p == r.
        # For each p, there is exactly one r such that n % p == r.
        # So for each n, and each p, there is 1 pair (p, n%p).
        # So for each n, there are len(primes) pairs.
        # So rows for y-x constraints: N * len(primes).
        # N=2000, len(primes)~62. 124,000 rows.
        # This is getting large for dense, but fine for sparse.
        
        A = lil_matrix((1 + N + N * len(primes), total_vars))
        b_l = []
        b_u = []
        
        # Row 0: Budget
        A[0, :num_x] = budget_coeff
        b_l.append(0.0)
        b_u.append(1.0)
        
        row_idx = 1
        
        # Rows for y_n <= sum(x)
        # sum(x) - y_n >= 0
        # -sum(x) + y_n <= 0
        # Let's stick to >= 0 form for LinearConstraint?
        # LinearConstraint(A, lb, ub)
        
        for n in range(1, N + 1):
            y_idx = num_x + (n - 1)
            # Coeff for y_n is -1 (if we write sum(x) - y >= 0)
            # Wait, LinearConstraint A x >= lb.
            # sum(x) - y >= 0
            A[row_idx, y_idx] = -1.0
            for p in primes:
                r = n % p
                x_idx = prime_indices[(p, r)]
                A[row_idx, x_idx] = 1.0
            b_l.append(0.0)
            b_u.append(np.inf)
            row_idx += 1
        
        # Rows for y_n >= x_{p,r}
        # y_n - x_{p,r} >= 0
        for n in range(1, N + 1):
            y_idx = num_x + (n - 1)
            for p in primes:
                r = n % p
                x_idx = prime_indices[(p, r)]
                A[row_idx, y_idx] = 1.0
                A[row_idx, x_idx] = -1.0
                b_l.append(0.0)
                b_u.append(np.inf)
                row_idx += 1
        
        # Prefix constraints: z_n >= z_{n+1}
        # z_n - z_{n+1} >= 0
        for n in range(1, N):
            z_idx_n = z_start + (n - 1)
            z_idx_np1 = z_start + n
            A[row_idx, z_idx_n] = 1.0
            A[row_idx, z_idx_np1] = -1.0
            b_l.append(0.0)
            b_u.append(np.inf)
            row_idx += 1
            
        # Constraint z_n <= y_n
        # z_n - y_n <= 0  =>  -z_n + y_n >= 0
        for n in range(1, N + 1):
            z_idx_n = z_start + (n - 1)
            y_idx = num_x + (n - 1)
            A[row_idx, y_idx] = 1.0
            A[row_idx, z_idx_n] = -1.0
            b_l.append(0.0)
            b_u.append(np.inf)
            row_idx += 1
            
        A = A.tocsr()
        
        constraints = LinearConstraint(A, b_l, b_u)
        bounds = Bounds(lb, ub)
        
        try:
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds, 
                       options={"time_limit": 50, "mip_rel_gap": 0.01})
            
            if res.status == 0 or res.status == 2: # Optimal or Locally Optimal
                x_vals = res.x[:num_x]
                classes = []
                for p in primes:
                    for r in range(p):
                        if x_vals[prime_indices[(p, r)]] > 0.5:
                            classes.append([p, r])
                return classes
            else:
                return None
        except Exception:
            return None

    # 1. Try MILP
    classes = try_milp()
    
    if classes is None:
        # Fallback to SA (similar to original but faster/optimized)
        # Use primes >= 13
        sel_primes = [p for p in primes_all if p >= 13]
        sel_residues = [random.randint(0, p-1) for p in sel_primes]
        
        L_cap = 10000
        counts = [0] * (L_cap + 1)
        
        def rebuild_counts():
            for i in range(L_cap + 1):
                counts[i] = 0
            for p, r in zip(sel_primes, sel_residues):
                start = r if r >= 1 else r + p
                for n in range(start, L_cap + 1, p):
                    counts[n] += 1
                    
        def get_L():
            L = 0
            for i in range(1, L_cap + 1):
                if counts[i] > 0:
                    L = i
                else:
                    break
            return L
            
        rebuild_counts()
        current_L = get_L()
        best_L = current_L
        best_primes = sel_primes[:]
        best_residues = sel_residues[:]
        
        T = 10.0
        cooling = 0.995
        
        while time.time() < deadline:
            idx = random.randint(0, len(sel_primes) - 1)
            p = sel_primes[idx]
            old_r = sel_residues[idx]
            new_r = random.randint(0, p - 1)
            
            # Update counts
            start_old = old_r if old_r >= 1 else old_r + p
            for n in range(start_old, L_cap + 1, p):
                counts[n] -= 1
                
            start_new = new_r if new_r >= 1 else new_r + p
            for n in range(start_new, L_cap + 1, p):
                counts[n] += 1
                
            sel_residues[idx] = new_r
            new_L = get_L()
            
            delta = new_L - current_L
            if delta >= 0:
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_primes = sel_primes[:]
                    best_residues = sel_residues[:]
            else:
                prob = pow(2.71828, delta / T)
                if random.random() < prob:
                    current_L = new_L
                else:
                    # Revert
                    sel_residues[idx] = old_r
                    for n in range(start_new, L_cap + 1, p):
                        counts[n] -= 1
                    for n in range(start_old, L_cap + 1, p):
                        counts[n] += 1
                    current_L = get_L()
                    
            T *= cooling
            if T < 0.1:
                T = 0.1
                
        classes = [[p, r] for p, r in zip(best_primes, best_residues)]
        
    # Final validation
    final_sum = sum(Fraction(1, p) for p, r in classes)
    if final_sum > Fraction(1):
         # Remove classes with largest 1/p until valid
         classes.sort(key=lambda x: 1/x[0], reverse=True)
         # Actually, remove smallest primes (largest 1/p)
         classes.sort(key=lambda x: x[0])
         while final_sum > Fraction(1) and classes:
             p, r = classes.pop(0)
             final_sum -= Fraction(1, p)
             
    return classes

