import numpy as np
import time
import sys
from scipy.optimize import milp, LinearConstraint, Bounds
import scipy.sparse as sp

def construct():
    n = 256
    start_time = time.time()
    # Safety margin: return by 85s to ensure we finish before 90s kill
    deadline = start_time + 85.0
    
    # 1. Precompute all Arithmetic Progressions (APs)
    # An AP is defined by start a (0-based), step s >= 1.
    # Indices: a, a+s, ..., a+(k-1)s < n
    progs = []
    for a in range(n):
        for s in range(1, n):
            # Generate indices for this AP
            # We only care about APs with length >= 2 for "discrepancy" usually, 
            # but the task says "including one". Singletons have sum +/-1, max abs is 1.
            # The bottleneck will be longer APs.
            # However, to be safe and correct, we include all.
            # Optimization: We can skip APs of length 1 as they are trivially bounded by 1.
            # If the final max sum is > 1, singletons don't matter.
            # Let's include length >= 2 to reduce constraint count slightly, 
            # but actually, let's just generate all and filter later if needed.
            # For n=256, total APs is roughly n * n / 2 = 32768. This is manageable.
            
            # Efficient generation
            idx = list(range(a, n, s))
            if len(idx) >= 2: # Skip singletons, they are always <= 1
                progs.append(idx)
            
            # If we include singletons, the max abs sum is at least 1.
            # If we find a solution with max sum 1, we are done.
            # If max sum > 1, singletons are irrelevant.
            # Let's assume the optimal max sum is > 1 for n=256 (Erdos discrepancy suggests growth).
            # So ignoring singletons is safe for finding the *minimizing* configuration 
            # as long as the resulting max sum > 1. If it's 1, we return it.
            
    Np = len(progs)
    
    # 2. Identify "Hard" Constraints for MILP
    # MILP with 32k constraints is too slow.
    # We select a subset of constraints that are likely to be the bottleneck.
    # Heuristic: Longer APs and APs with small steps are often harder.
    # Or, we can just pick a random subset or the longest ones.
    # Let's sort by length descending and take the top K.
    # Also, APs with step 1 (contiguous blocks) are important.
    
    prog_lengths = [len(p) for p in progs]
    # Sort indices by length descending
    sorted_indices = np.argsort(prog_lengths)[::-1]
    
    # How many constraints can MILP handle?
    # Variables: 256 binary vars.
    # Constraints: 2 * K (for -D <= sum <= D).
    # Let's try K = 2000 or so.
    # If it's too slow, we reduce K.
    
    max_milp_constraints = 2000
    selected_indices = sorted_indices[:max_milp_constraints]
    
    # Build sparse matrix for these selected constraints
    # M_sel (K x n)
    rows = []
    cols = []
    vals = []
    
    for idx_in_sel, orig_idx in enumerate(selected_indices):
        for pos in progs[orig_idx]:
            rows.append(idx_in_sel)
            cols.append(pos)
            vals.append(1.0)
            
    M_sel = sp.csr_matrix((vals, (rows, cols)), shape=(len(selected_indices), n))
    
    # 3. MILP Formulation
    # Variables: x_i in {0, 1}
    # Sign s_i = 2*x_i - 1
    # Sum over AP p: sum_{i in p} s_i = sum_{i in p} (2*x_i - 1) = 2 * sum_{i in p} x_i - len(p)
    # Let S_p = sum_{i in p} x_i.
    # We want to minimize D such that |2*S_p - len(p)| <= D for all p.
    # This is equivalent to:
    # -D <= 2*S_p - len(p) <= D
    # (len(p) - D) / 2 <= S_p <= (len(p) + D) / 2
    
    # We can't put D in the constraints directly as a variable easily in standard MILP 
    # unless we make it a continuous variable and minimize it.
    # Objective: min D
    # Variables: x (binary, n), D (continuous, 1)
    # Total vars: n + 1
    
    # Constraints for each selected AP p:
    # 2*S_p - len(p) <= D  =>  2*sum(x_i) - D <= len(p)
    # -2*S_p + len(p) <= D => -2*sum(x_i) - D <= -len(p)
    
    # Let's build the matrix A for scipy.optimize.milp
    # Variables order: [x_0, ..., x_{n-1}, D]
    # A has 2 * K rows.
    
    A_rows = []
    A_cols = []
    A_vals = []
    lb = []
    ub = []
    
    for k in range(len(selected_indices)):
        orig_idx = selected_indices[k]
        p = progs[orig_idx]
        L = len(p)
        
        # Constraint 1: 2*sum(x_i) - D <= L
        # Coeffs for x_i in p: 2
        # Coeff for D: -1
        # RHS: L
        for pos in p:
            A_rows.append(k * 2)
            A_cols.append(pos)
            A_vals.append(2.0)
        A_rows.append(k * 2)
        A_cols.append(n) # Index for D
        A_vals.append(-1.0)
        lb.append(-np.inf)
        ub.append(L)
        
        # Constraint 2: -2*sum(x_i) - D <= -L
        # Coeffs for x_i in p: -2
        # Coeff for D: -1
        # RHS: -L
        for pos in p:
            A_rows.append(k * 2 + 1)
            A_cols.append(pos)
            A_vals.append(-2.0)
        A_rows.append(k * 2 + 1)
        A_cols.append(n)
        A_vals.append(-1.0)
        lb.append(-np.inf)
        ub.append(-L)
        
    A = sp.csr_matrix((A_vals, (A_rows, A_cols)), shape=(2 * len(selected_indices), n + 1))
    
    # Objective: minimize D
    # c = [0, ..., 0, 1]
    c = np.zeros(n + 1)
    c[n] = 1.0
    
    # Bounds:
    # x_i in [0, 1]
    # D in [0, inf] (or [1, inf] since max abs sum >= 1)
    bounds = Bounds(lb=[0] * n + [0], ub=[1] * n + [np.inf])
    
    # Integrality: x_i are integer (binary), D is continuous
    integrality = np.array([1] * n + [0])
    
    # Solve MILP
    # Time limit for MILP: 30 seconds.
    milp_deadline = time.time() + 30.0
    
    try:
        # scipy.optimize.milp parameters
        # time_limit is in seconds
        time_remaining = max(1.0, milp_deadline - time.time())
        
        res = milp(
            c=c,
            constraints=LinearConstraint(A, lb=lb, ub=ub),
            integrality=integrality,
            bounds=bounds,
            options={
                "time_limit": time_remaining,
                "presolve": True
            }
        )
        
        if res.status == 0 or res.status == 1: # Optimal or Time Limit reached with incumbent
            # Extract x
            x = res.x[:n]
            # Convert to signs: s_i = 2*x_i - 1
            signs = (2 * x - 1).astype(np.float32)
            # Ensure they are exactly -1 or 1
            signs = np.where(signs > 0, 1.0, -1.0)
        else:
            # Fallback to random if MILP fails completely
            rng = np.random.RandomState(42)
            signs = rng.choice([-1, 1], size=n).astype(np.float32)
            
    except Exception:
        rng = np.random.RandomState(42)
        signs = rng.choice([-1, 1], size=n).astype(np.float32)

    # 4. Polish with Full Constraint Local Search
    # The MILP solution is optimal for the *subset* of constraints.
    # It might violate the other constraints.
    # We need to minimize the max absolute sum over ALL APs.
    
    # Build full incidence matrix M (n x Np_total)
    # We already have `progs` list.
    # Let's build M as float32 for speed.
    Np_total = len(progs)
    M = np.zeros((n, Np_total), dtype=np.float32)
    for pid, idx in enumerate(progs):
        for i in idx:
            M[i, pid] = 1.0
            
    def compute_sums(s):
        return M.T @ s
        
    def max_abs_sum(sums):
        return np.max(np.abs(sums))
        
    current_signs = signs.copy()
    current_sums = compute_sums(current_signs)
    current_score = max_abs_sum(current_sums)
    
    best_signs = current_signs.copy()
    best_score = current_score
    
    rng = np.random.RandomState(123)
    
    # Local Search Loop
    # We have ~50 seconds left.
    # Each step: flip one sign.
    # Cost: O(Np) to update sums? No, we can update incrementally.
    # delta_sums = -2 * sign_i * M[i, :]
    # new_sums = current_sums + delta_sums
    # This is O(Np) per step. Np ~ 32k.
    # 50s / (32k ops) ... numpy vectorization is fast.
    # 32k float ops is ~ microseconds.
    # So we can do many steps.
    
    # Let's use Simulated Annealing to escape local minima.
    T_start = 1.0
    T_end = 0.01
    n_iters = 200000
    
    for it in range(n_iters):
        if time.time() > deadline - 2.0:
            break
            
        # Decay temperature
        T = T_start * (1.0 - it / float(n_iters)) + T_end
        
        i = rng.randint(n)
        sign_i = current_signs[i]
        
        # Calculate change in sums
        delta = -2.0 * sign_i * M[i, :]
        new_sums = current_sums + delta
        
        # Calculate new score
        # We need max(abs(new_sums))
        # This is O(Np).
        new_score = np.max(np.abs(new_sums))
        
        delta_score = new_score - current_score
        
        # Acceptance
        if delta_score <= 0 or rng.rand() < np.exp(-delta_score / T):
            current_signs[i] = -sign_i
            current_sums = new_sums
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_signs = current_signs.copy()
                
    # Final check
    final_score = max_abs_sum(compute_sums(best_signs))
    if final_score < best_score:
        best_score = final_score
        
    return [int(x) for x in best_signs]

if __name__ == "__main__":
    # Simple self-test
    signs = construct()
    print(f"Length: {len(signs)}")
    print(f"First 10: {signs[:10]}")
    
    # Verify max sum
    n = 256
    progs = []
    for a in range(n):
        for s in range(1, n):
            idx = list(range(a, n, s))
            if len(idx) >= 1:
                progs.append(idx)
                
    max_sum = 0
    for p in progs:
        s = sum(signs[i] for i in p)
        if abs(s) > max_sum:
            max_sum = abs(s)
    print(f"Max Abs Sum: {max_sum}")
