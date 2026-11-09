"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

def construct():
    """
    Find a Sidon set containing 0 with all differences 1..30 present,
    minimizing the largest element.
    
    Strategy: Integer Linear Programming (ILP) using scipy.optimize.milp.
    We iterate on the maximum element M and check if a valid set exists.
    """
    
    REQUIRED = list(range(1, 31))
    
    # We know M must be at least 30. 
    # With k=9 elements, we have 36 differences. 
    # If M=30, we can't fit 36 distinct differences in 1..30.
    # So M >= 36 for k=9.
    # Let's start from 30 and go up.
    
    for M in range(30, 200):
        N = M + 1  # Number of x variables (0 to M)
        
        # Variables:
        # x_0 ... x_M (N variables)
        # z_{i,d} for i in 0..M-d, d in 1..M
        # Let's flatten z. 
        # Index for z_{i,d}: i * M + (d - 1)
        # Total z variables: sum_{d=1}^M (M - d + 1) = M(M+1)/2
        
        num_z = M * (M + 1) // 2
        total_vars = N + num_z
        
        # Objective: Minimize 0 (just find feasible solution)
        # Actually, we can just set c = 0.
        c = np.zeros(total_vars)
        
        # Bounds
        # x_i in [0, 1]
        # z_{i,d} in [0, 1]
        lb = np.zeros(total_vars)
        ub = np.ones(total_vars)
        
        # Constraints
        # We will build A_ub, b_ub for A_ub @ x <= b_ub
        # And A_eq, b_eq for A_eq @ x == b_eq
        
        # Let's use LinearConstraint for inequality and equality if needed, 
        # or just build A_ub and b_ub and use milp directly if it accepts that format?
        # milp(c, constraints=LinearConstraint(A, lb, ub), integrality=integrality, bounds=Bounds(lb, ub))
        
        A_ub = lil_matrix((0, total_vars), dtype=np.float64)
        b_ub = []
        
        # 1. x_0 = 1
        # This is equality. Let's handle equalities separately or as two inequalities.
        # x_0 <= 1 (already in bounds)
        # -x_0 <= -1
        row = np.zeros(total_vars)
        row[0] = -1
        A_ub = A_ub.tocsr() # switch to csr for appending? No, lil is better for building.
        # Actually, let's just collect rows.
        
        rows = []
        
        # Constraint: -x_0 <= -1
        r = np.zeros(total_vars)
        r[0] = -1
        rows.append((r, -1))
        
        # Constraint: sum(x_i) <= 32
        r = np.zeros(total_vars)
        r[:N] = 1
        rows.append((r, 32))
        
        # Constraint: sum_i z_{i,d} <= 1 for all d in 1..M
        # For each d, the variables z_{i,d} are at indices:
        # i ranges from 0 to M-d.
        # Index = N + i * M + (d - 1)
        
        for d in range(1, M + 1):
            r = np.zeros(total_vars)
            for i in range(0, M - d + 1):
                idx = N + i * M + (d - 1)
                r[idx] = 1
            rows.append((r, 1))
            
        # Constraint: sum_i z_{i,d} >= 1 for d in 1..30
        # This is -sum_i z_{i,d} <= -1
        for d in range(1, 31):
            r = np.zeros(total_vars)
            for i in range(0, M - d + 1):
                idx = N + i * M + (d - 1)
                r[idx] = -1
            rows.append((r, -1))
            
        # Linking constraints for z_{i,d}
        # z_{i,d} - x_i <= 0
        # z_{i,d} - x_{i+d} <= 0
        # z_{i,d} - x_i - x_{i+d} <= -1
        
        for d in range(1, M + 1):
            for i in range(0, M - d + 1):
                idx_z = N + i * M + (d - 1)
                idx_xi = i
                idx_xj = i + d
                
                # z_{i,d} - x_i <= 0
                r = np.zeros(total_vars)
                r[idx_z] = 1
                r[idx_xi] = -1
                rows.append((r, 0))
                
                # z_{i,d} - x_{i+d} <= 0
                r = np.zeros(total_vars)
                r[idx_z] = 1
                r[idx_xj] = -1
                rows.append((r, 0))
                
                # z_{i,d} - x_i - x_{i+d} <= -1
                r = np.zeros(total_vars)
                r[idx_z] = 1
                r[idx_xi] = -1
                r[idx_xj] = -1
                rows.append((r, -1))
                
        A_ub = np.vstack([r for r, _ in rows])
        b_ub = np.array([b for _, b in rows])
        
        # Integrality: all variables are binary
        integrality = np.ones(total_vars)
        
        # Solve
        try:
            res = milp(c=c, 
                       constraints=LinearConstraint(A_ub, -np.inf, b_ub), 
                       integrality=integrality, 
                       bounds=Bounds(lb, ub))
            if res.success:
                x_sol = res.x[:N]
                # Extract set
                elements = [i for i in range(N) if x_sol[i] > 0.5]
                # Verify
                if 0 not in elements:
                    continue
                diffs = set()
                valid = True
                for i in range(len(elements)):
                    for j in range(i+1, len(elements)):
                        d = elements[j] - elements[i]
                        if d in diffs:
                            valid = False
                            break
                        diffs.add(d)
                    if not valid:
                        break
                if valid and all(d in diffs for d in range(1, 31)):
                    return elements
        except Exception as e:
            pass
            
    # Fallback if ILP fails or is too slow
    # Use a simple greedy construction as a last resort
    a = [0]
    used = set()
    needed = set(range(1, 31))
    for _ in range(32):
        if not needed:
            break
        best_x = None
        best_score = float('inf')
        for x in range(1, 500):
            if x in a:
                continue
            new_diffs = set()
            ok = True
            for e in a:
                d = abs(x - e)
                if d == 0:
                    continue
                if d in used or d in new_diffs:
                    ok = False
                    break
                new_diffs.add(d)
            if not ok:
                continue
            covers = len(new_diffs & needed)
            if covers == 0:
                continue
            # Heuristic: prefer x that covers many needed and is small
            score = x - covers * 10
            if score < best_score:
                best_score = score
                best_x = x
        if best_x is None:
            break
        a.append(best_x)
        a.sort()
        for e in a[:-1]:
            d = abs(best_x - e)
            used.add(d)
            needed.discard(d)
            
    return a

# EVOLVE-BLOCK-END
