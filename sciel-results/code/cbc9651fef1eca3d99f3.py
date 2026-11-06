import random
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, csr_matrix

def popcount(x):
    return bin(x).count('1')

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    def is_valid_addition(adj, u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free."""
        # 1. Common neighbors of u and v must be < 3
        common_uv = adj[u] & adj[v]
        if popcount(common_uv) >= 3:
            return False
        
        # 2. For any w in N(v) \ {u}, v becomes a common neighbor of u and w.
        #    So |N(u) ∩ N(w)| must currently be < 2.
        neighbors_v = adj[v] & ~(1 << u)
        while neighbors_v:
            lsb = neighbors_v & -neighbors_v
            w = (lsb.bit_length() - 1)
            common_uw = adj[u] & adj[w]
            if popcount(common_uw) >= 2:
                return False
            neighbors_v ^= lsb
            
        # 3. Similarly for w in N(u) \ {v}
        neighbors_u = adj[u] & ~(1 << v)
        while neighbors_u:
            lsb = neighbors_u & -neighbors_u
            w = (lsb.bit_length() - 1)
            common_vw = adj[v] & adj[w]
            if popcount(common_vw) >= 2:
                return False
            neighbors_u ^= lsb
            
        return True

    def get_missing_edges(adj):
        """Return list of missing edges."""
        missing = []
        for i in range(n):
            mask = ~adj[i] & ((1 << n) - 1) & ~((1 << i) - 1)
            mask &= ~((1 << (i + 1)) - 1)  # Only j > i
            while mask:
                lsb = mask & -mask
                j = (lsb.bit_length() - 1)
                missing.append((i, j))
                mask ^= lsb
        return missing

    def try_add_all(adj):
        """Try to add all possible missing edges until no more can be added."""
        improved = True
        while improved:
            if time.time() > deadline:
                return False
            improved = False
            missing = get_missing_edges(adj)
            random.shuffle(missing)
            for u, v in missing:
                if time.time() > deadline:
                    return False
                if is_valid_addition(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    improved = True
        return True

    def edge_count(adj):
        return sum(popcount(adj[i]) for i in range(n)) // 2

    def extract_edges(adj):
        current_edges = []
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = (lsb.bit_length() - 1)
                if i < j:
                    current_edges.append([i, j])
                mask ^= lsb
        return current_edges

    def validate_graph(edges):
        """Verify the graph is K2,3-free."""
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u] & adj[v]
                if popcount(common) >= 3:
                    return False
        return True

    best_edges = []
    best_count = 0
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Phase 1: Heuristic Search
    # We run a significant number of restarts to find a good initial solution
    # which will be used as a warm start for the MILP.
    for restart in range(1000):
        if time.time() > deadline - 40:  # Reserve time for MILP
            break
        
        # Initialize adjacency bitmasks
        adj = [0] * n
        
        # Greedy construction with random edge order
        edges = all_pairs[:]
        random.shuffle(edges)
        
        for u, v in edges:
            if time.time() > deadline - 40:
                break
            if is_valid_addition(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Hill climbing: add all possible edges
        try_add_all(adj)
        
        # Swap-based local search: try removing one edge and adding two others
        improved = True
        while improved:
            if time.time() > deadline - 40:
                break
            improved = False
            current_count = edge_count(adj)
            
            # Get current edges
            current_edges = extract_edges(adj)
            random.shuffle(current_edges)
            
            for rem_u, rem_v in current_edges:
                if time.time() > deadline - 40:
                    break
                
                # Temporarily remove edge (rem_u, rem_v)
                adj[rem_u] &= ~(1 << rem_v)
                adj[rem_v] &= ~(1 << rem_u)
                
                # Try to add all possible edges
                added = []
                missing = get_missing_edges(adj)
                random.shuffle(missing)
                for u, v in missing:
                    if time.time() > deadline - 40:
                        break
                    if is_valid_addition(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        added.append((u, v))
                
                new_count = edge_count(adj)
                
                if new_count > current_count:
                    # Keep the change
                    improved = True
                else:
                    # Revert
                    for u, v in added:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    adj[rem_u] |= (1 << rem_v)
                    adj[rem_v] |= (1 << rem_u)
        
        # Final hill climbing
        try_add_all(adj)
        
        # Update best
        count = edge_count(adj)
        if count > best_count:
            best_count = count
            best_edges = extract_edges(adj)
            
    # Phase 2: MILP Optimization
    # If we have time, try to improve the best solution found using MILP.
    # We warm-start the MILP with the current best solution by fixing some variables
    # or simply using it as a bound. Since scipy.milp doesn't support warm starts
    # directly, we can try to solve the MILP from scratch but with a time limit.
    # However, for n=20, the MILP might be too large or slow if not careful.
    # A better approach: Use the heuristic result as a lower bound and see if MILP can beat it.
    
    if time.time() < deadline - 10 and best_count > 0:
        try:
            # Map edges to indices
            edge_idx = {}
            idx = 0
            for i in range(n):
                for j in range(i+1, n):
                    edge_idx[(i, j)] = idx
                    idx += 1
            num_edges = idx
            
            # Map auxiliary variables z_{uvw}
            # z is for each pair (u,v) and w != u,v
            z_idx = {}
            z_count = 0
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(n):
                        if k != i and k != j:
                            z_idx[(i, j, k)] = num_edges + z_count
                            z_count += 1
            total_vars = num_edges + z_count
            
            # Objective: maximize sum x
            c = np.zeros(total_vars)
            for i in range(num_edges):
                c[i] = -1  # Minimize -sum x
            
            # Constraints
            # 1. z_{uvw} <= x_{uw}
            # 2. z_{uvw} <= x_{vw}
            # 3. z_{uvw} >= x_{uw} + x_{vw} - 1  => x_{uw} + x_{vw} - z_{uvw} <= 1
            # 4. sum_{w} z_{uvw} <= 2
            
            # We will build a single A matrix for all constraints.
            # Number of constraints:
            # 3 * z_count + num_pairs (for sum)
            num_pairs = n * (n - 1) // 2
            total_constraints = 3 * z_count + num_pairs
            
            A = lil_matrix((total_constraints, total_vars))
            lb = np.zeros(total_constraints)
            ub = np.zeros(total_constraints)
            
            row = 0
            # Constraints 1 & 2: z_{uvw} - x_{uw} <= 0 and z_{uvw} - x_{vw} <= 0
            # Actually, let's group by z variable for efficiency? No, row by row is easier.
            
            # Let's iterate over all z variables
            for (i, j, k), z_i in z_idx.items():
                # Get x indices for (i,k) and (j,k)
                x_ik_idx = edge_idx.get((min(i,k), max(i,k)))
                x_jk_idx = edge_idx.get((min(j,k), max(j,k)))
                
                # Constraint 1: z_{ijk} <= x_{ik}  => z_{ijk} - x_{ik} <= 0
                A[row, z_i] = 1
                A[row, x_ik_idx] = -1
                ub[row] = 0
                row += 1
                
                # Constraint 2: z_{ijk} <= x_{jk}  => z_{ijk} - x_{jk} <= 0
                A[row, z_i] = 1
                A[row, x_jk_idx] = -1
                ub[row] = 0
                row += 1
                
                # Constraint 3: x_{ik} + x_{jk} - z_{ijk} <= 1
                A[row, x_ik_idx] = 1
                A[row, x_jk_idx] = 1
                A[row, z_i] = -1
                ub[row] = 1
                row += 1
            
            # Constraint 4: sum_{k} z_{ijk} <= 2 for each pair (i,j)
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(n):
                        if k != i and k != j:
                            z_i = z_idx[(i, j, k)]
                            A[row, z_i] = 1
                    ub[row] = 2
                    row += 1
            
            A = csr_matrix(A)
            
            # Bounds: 0 <= x <= 1, 0 <= z <= 1
            integrality = np.ones(total_vars)
            bounds = Bounds(lb=np.zeros(total_vars), ub=np.ones(total_vars))
            
            # Solve MILP
            # Note: scipy.optimize.milp might be slow for this size if not optimized well.
            # We use a time limit implicitly by checking deadline before calling.
            # If it takes too long, we just ignore the result.
            # To be safe, we only attempt this if we have > 20s left.
            if time.time() < deadline - 20:
                res = milp(c=c, constraints=LinearConstraint(A, lb, ub), integrality=integrality, bounds=bounds)
                
                if res.success and res.x is not None:
                    # Extract solution
                    x_sol = res.x[:num_edges]
                    new_edges = []
                    for (i, j), idx in edge_idx.items():
                        if x_sol[idx] > 0.5:
                            new_edges.append([i, j])
                    
                    # Verify the solution
                    if validate_graph(new_edges):
                        new_count = len(new_edges)
                        if new_count > best_count:
                            best_count = new_count
                            best_edges = new_edges

        except Exception:
            pass

    return best_edges
