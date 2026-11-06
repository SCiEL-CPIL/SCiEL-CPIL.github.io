import time
import random
import itertools
import sys

def construct():
    start_time = time.time()
    time_limit = 300  # seconds
    
    # 1. Generate Seed: Paley Graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n_seed = q
    edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    
    # Build adjacency matrix for fast lookup
    adj = [[0] * n_seed for _ in range(n_seed)]
    for i, j in edges:
        adj[i][j] = 1
        adj[j][i] = 1
        
    def is_valid_graph(adj, n):
        """Check if graph has no K5 and no independent set of size 5."""
        # Check for K5
        for combo in itertools.combinations(range(n), 5):
            is_clique = True
            for u, v in itertools.combinations(combo, 2):
                if adj[u][v] == 0:
                    is_clique = False
                    break
            if is_clique:
                return False
        # Check for Independent Set of size 5
        for combo in itertools.combinations(range(n), 5):
            is_independent = True
            for u, v in itertools.combinations(combo, 2):
                if adj[u][v] == 1:
                    is_independent = False
                    break
            if is_independent:
                return False
        return True

    def get_k4_and_independent_k4(adj, n):
        """Return list of K4s and list of independent 4-sets."""
        k4s = []
        indep_k4s = []
        for combo in itertools.combinations(range(n), 4):
            is_clique = True
            is_independent = True
            for u, v in itertools.combinations(combo, 2):
                if adj[u][v] == 1:
                    is_independent = False
                else:
                    is_clique = False
            if is_clique:
                k4s.append(combo)
            if is_independent:
                indep_k4s.append(combo)
        return k4s, indep_k4s

    def solve_new_vertex(k4s, indep_k4s, n_curr, time_budget, num_restarts=50):
        """
        Find adjacency vector x of length n_curr for a new vertex.
        Constraints:
        - For each K4 in k4s: not all x[i] are 1. (Sum x[i] <= 3)
        - For each indep K4 in indep_k4s: not all x[i] are 0. (Sum x[i] >= 1)
        """
        best_solution = None
        best_cost_global = float('inf')
        
        # Pre-compute constraints for faster evaluation
        # Constraint: list of indices, type (1 for clique, 0 for indep)
        # We want to minimize violations.
        
        def get_cost(x):
            cost = 0
            # Check K4 constraints: x[i] must not be all 1
            for k4 in k4s:
                # If all are 1, violation
                # Optimization: check if any is 0
                if all(x[i] == 1 for i in k4):
                    cost += 1
            # Check Independent K4 constraints: x[i] must not be all 0
            for ik4 in indep_k4s:
                # If all are 0, violation
                if all(x[i] == 0 for i in ik4):
                    cost += 1
            return cost
        
        for r in range(num_restarts):
            if time.time() > start_time + time_limit - 10:
                break
            
            # Random initialization
            x = [random.randint(0, 1) for _ in range(n_curr)]
            current_cost = get_cost(x)
            
            if current_cost == 0:
                return x
                
            best_x_local = x[:]
            best_cost_local = current_cost
            
            t_start = time.time()
            temperature = 1.0
            cooling = 0.999
            
            while time.time() - t_start < (time_budget / num_restarts):
                if best_cost_local == 0:
                    return best_x_local
                    
                # Pick random variable to flip
                i = random.randint(0, n_curr - 1)
                old_val = x[i]
                x[i] = 1 - x[i]
                
                new_cost = get_cost(x)
                
                # Accept if better, or with probability exp(-(new_cost - current_cost)/temperature)
                if new_cost <= current_cost:
                    current_cost = new_cost
                    if new_cost < best_cost_local:
                        best_cost_local = new_cost
                        best_x_local = x[:]
                else:
                    # Metropolis acceptance
                    # Use a simple approximation for exp to save time if needed, but math.exp is fine
                    if random.random() < pow(2.71828, -(new_cost - current_cost) / temperature):
                        current_cost = new_cost
                    else:
                        x[i] = old_val
                        
                temperature *= cooling
                if temperature < 0.01:
                    temperature = 0.01
                    
            if best_cost_local == 0:
                return best_x_local
            elif best_cost_local < best_cost_global:
                best_cost_global = best_cost_local
                best_solution = best_x_local
                
        return None

    # Try to grow the graph
    current_adj = [row[:] for row in adj]
    current_n = n_seed
    
    max_target = 43 # Try to reach the known best
    
    # Pre-compute K4s and indep K4s for the initial graph
    # This is expensive but necessary for constraint checking
    # We will re-compute them after each successful addition
    
    for new_v in range(n_seed, max_target):
        if time.time() - start_time > time_limit - 30:
            break
            
        # Get constraints from current graph
        # Note: This step is O(n^4) which can be slow for n~40
        # n=37: C(37,4) ~ 66k. n=42: C(42,4) ~ 116k. This is fine.
        k4s, indep_k4s = get_k4_and_independent_k4(current_adj, current_n)
        
        # Time budget for this step: decrease as n grows
        # We have ~270s left. We want to add up to 6 vertices.
        # 270 / 6 = 45s per vertex.
        # But later steps are harder.
        budget = 30.0
        
        sol = solve_new_vertex(k4s, indep_k4s, current_n, budget, num_restarts=100)
        
        if sol is not None:
            # Add new vertex
            new_n = current_n + 1
            # Extend adjacency matrix
            current_adj.append([0] * new_n)
            for i in range(current_n):
                current_adj[i].append(sol[i])
                current_adj[new_n - 1][i] = sol[i]
            current_adj[new_n - 1][new_n - 1] = 0 # No self loop
            
            current_n = new_n
        else:
            # If we fail, we might try a smaller budget or break
            # Breaking is safer to ensure we return a valid graph
            break
            
    # Verify final graph
    if is_valid_graph(current_adj, current_n):
        # Convert to edge list
        final_edges = []
        for i in range(current_n):
            for j in range(i + 1, current_n):
                if current_adj[i][j]:
                    final_edges.append((i, j))
        return current_n, final_edges
    else:
        # Fallback to seed
        return n_seed, edges

if __name__ == "__main__":
    n, e = construct()
    print(f"Constructed graph with {n} vertices and {len(e)} edges")
