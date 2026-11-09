import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    N = 256
    
    # 1. Filter nodes: 2*v must not be a perfect square
    valid_nodes = []
    for v in range(1, N + 1):
        s = 2 * v
        r = int(math.isqrt(s))
        if r * r != s:
            valid_nodes.append(v)
    
    if not valid_nodes:
        return [1]
        
    M = len(valid_nodes)
    
    # 2. Build adjacency list
    # adj[i] is a set of indices j such that valid_nodes[i] + valid_nodes[j] is a square
    adj = [set() for _ in range(M)]
    
    for i in range(M):
        vi = valid_nodes[i]
        for j in range(i + 1, M):
            vj = valid_nodes[j]
            s = vi + vj
            r = int(math.isqrt(s))
            if r * r == s:
                adj[i].add(j)
                adj[j].add(i)
                
    # 3. Use MILP to find maximum independent set
    # Formulation: maximize sum(x_i) subject to x_i + x_j <= 1 for each edge (i,j)
    # x_i in {0, 1}
    
    milp_solution = None
    
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
        
        # Build constraint matrix for edge constraints: x_i + x_j <= 1
        edges = []
        seen = set()
        for i in range(M):
            for j in adj[i]:
                if i < j and (i, j) not in seen:
                    edges.append((i, j))
                    seen.add((i, j))
        
        E = len(edges)
        
        if E == 0:
            # No conflicts, take all nodes
            return valid_nodes[:]
        
        # Objective: maximize sum(x_i), so c = -1 for each variable
        c = -np.ones(M)
        
        # Constraints: x_i + x_j <= 1 for each edge
        # A_ub * x <= b_ub
        A_ub = np.zeros((E, M))
        for k, (i, j) in enumerate(edges):
            A_ub[k, i] = 1.0
            A_ub[k, j] = 1.0
        
        b_ub = np.ones(E)
        
        # Bounds: 0 <= x_i <= 1
        lb = np.zeros(M)
        ub = np.ones(M)
        
        # Integrality: all variables are binary
        integrality = np.ones(M)
        
        constraint = LinearConstraint(A_ub, ub=b_ub)
        bounds = Bounds(lb, ub)
        
        res = milp(c=c, constraints=[constraint], integrality=integrality, bounds=bounds)
        
        if res.status == 0 and res.x is not None:
            # Extract solution
            solution = res.x
            independent_set_indices = [i for i in range(M) if solution[i] > 0.5]
            
            # Verify validity
            is_valid = True
            for i in independent_set_indices:
                for j in independent_set_indices:
                    if i < j and j in adj[i]:
                        is_valid = False
                        break
                if not is_valid:
                    break
            
            if is_valid and independent_set_indices:
                milp_solution = [valid_nodes[i] for i in independent_set_indices]
        
    except Exception:
        pass
    
    # If MILP failed or returned invalid/empty, use Robust Fallback
    if milp_solution:
        return milp_solution
        
    # Fallback: Multi-Restart Greedy with Local Search
    best_set_indices = []
    
    # Number of restarts depends on remaining time
    # Estimate time per restart: O(M^2) roughly. M=245, so ~60k ops. Very fast.
    # We can do many restarts.
    max_restarts = 100
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5: # Leave 5s margin
            break
            
        remaining = set(range(M))
        current_set = []
        
        while remaining:
            # Find node with minimum degree in the induced subgraph of 'remaining'
            min_deg_node = None
            min_deg = float('inf')
            
            # To add randomness for diversity in restarts, we can break ties randomly
            # Or just pick the first one found with min degree.
            # Let's add a small random perturbation to degrees to break ties differently each restart
            # Or simply shuffle the order of checking? 
            # Simpler: Just pick min degree. If ties, pick lowest index? 
            # To get diversity, let's randomize the tie-breaking.
            
            candidates = []
            for u in remaining:
                deg = len(adj[u] & remaining)
                candidates.append((deg, u))
            
            # Sort by degree, then by random factor for tie-breaking
            # Actually, sorting M items M times is O(M^2 log M). M=245, so ~245*245*8 ~ 480k ops.
            # 100 restarts -> 48M ops. Might be tight in Python?
            # Let's optimize: just find min degree.
            
            min_deg = float('inf')
            min_nodes = []
            for u in remaining:
                deg = len(adj[u] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    min_nodes = [u]
                elif deg == min_deg:
                    min_nodes.append(u)
            
            # Pick one randomly from min_nodes
            chosen = random.choice(min_nodes)
            
            current_set.append(chosen)
            remaining.discard(chosen)
            for v in adj[chosen]:
                remaining.discard(v)
                
        # Local Search Polish: Try to add any vertex not in set that doesn't conflict
        current_set_set = set(current_set)
        improved = True
        while improved:
            improved = False
            for v in range(M):
                if v not in current_set_set:
                    # Check if v conflicts with any in current_set
                    conflict = False
                    for u in current_set:
                        if u in adj[v]:
                            conflict = True
                            break
                    if not conflict:
                        current_set.append(v)
                        current_set_set.add(v)
                        improved = True
        
        if len(current_set) > len(best_set_indices):
            best_set_indices = current_set[:]
            
        # Optional: Swap-based local search
        # For each u in set, check if removing u allows adding 2+ nodes
        # This is expensive. Let's skip for now to ensure speed.
        
    result = [valid_nodes[i] for i in best_set_indices]
    
    if not result:
        # Fallback to single node if something went wrong
        return [valid_nodes[0]]
        
    return result
