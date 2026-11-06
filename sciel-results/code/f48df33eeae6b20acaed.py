import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin
    
    N = 180
    
    # Precompute conflicts and triples
    # triples contains (a, b, c) such that 1/a = 1/b + 1/c with a < b < c
    triples = []
    
    for a in range(1, N + 1):
        a2 = a * a
        # Find factor pairs d, e of a^2 such that d < e
        for d in range(1, int(a2**0.5) + 1):
            if a2 % d != 0:
                continue
            e = a2 // d
            if d >= e:
                continue
            b = a + d
            c = a + e
            if b <= N and c <= N:
                triples.append((a, b, c))
                
    # Attempt to solve using MILP if available
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        # Objective: maximize sum(x_i) -> minimize -sum(x_i)
        c_obj = -np.ones(N)
        
        # Constraints: for each triple, x_a + x_b + x_c <= 2
        # A_ub is the matrix for A_ub @ x <= b_ub
        A_ub = np.zeros((len(triples), N))
        b_ub = np.full(len(triples), 2)
        
        for idx, (a, b, c) in enumerate(triples):
            A_ub[idx, a-1] = 1
            A_ub[idx, b-1] = 1
            A_ub[idx, c-1] = 1
            
        # Bounds: 0 <= x_i <= 1
        bounds = Bounds(lb=np.zeros(N), ub=np.ones(N))
        
        # Integrality: all variables are integer
        integrality = np.ones(N)
        
        # Solve
        res = milp(
            c=c_obj,
            constraints=LinearConstraint(A_ub, ub=b_ub),
            bounds=bounds,
            integrality=integrality
        )
        
        if res.status == 0:
            S = set()
            for i in range(N):
                if res.x[i] > 0.5:
                    S.add(i+1)
            
            # Verify validity
            valid = True
            for a, b, c in triples:
                if a in S and b in S and c in S:
                    valid = False
                    break
            
            if valid:
                return sorted(S)
                
    except Exception:
        pass
        
    # Fallback: Greedy construction
    S = set()
    # Re-compute conflicts for greedy if needed, or just use triples
    # Greedy from N down to 1
    for a in range(N, 0, -1):
        if time.time() > deadline:
            break
            
        can_add = True
        # Check if adding a creates a conflict
        # Conflict if there exists b, c in S such that (a,b,c) is a triple
        # Since a is the smallest in the triple, we check triples where a is the first element
        # But wait, the triple is defined as a < b < c.
        # If we add a, we need to check if there are b, c in S such that 1/a = 1/b + 1/c.
        # This is exactly checking if (a, b, c) is in triples and b, c in S.
        
        # Optimization: Pre-compute for each a, the list of (b,c)
        # We already have triples. We can filter.
        # Or just iterate through triples? 
        # Number of triples is small.
        
        for b, c in [ (t[1], t[2]) for t in triples if t[0] == a ]:
            if b in S and c in S:
                can_add = False
                break
        
        if can_add:
            S.add(a)
            
    return sorted(S)
