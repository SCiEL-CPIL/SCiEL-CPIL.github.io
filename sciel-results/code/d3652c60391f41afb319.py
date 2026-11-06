import time
import random
import itertools

def construct():
    """
    Attempts to find a (5,5)-free graph on 42 vertices using simulated annealing.
    Falls back to the Paley graph on 37 vertices if no solution is found.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    def is_valid_graph(n, adj):
        """Check if graph has no K5 and no independent set of size 5."""
        # adj is a list of bitmasks
        for combo in itertools.combinations(range(n), 5):
            is_clique = True
            is_indep = True
            for i in range(5):
                for j in range(i+1, 5):
                    u, v = combo[i], combo[j]
                    if (adj[u] >> v) & 1:
                        is_indep = False
                    else:
                        is_clique = False
                    if not is_clique and not is_indep:
                        break
                if not is_clique and not is_indep:
                    break
            if is_clique or is_indep:
                return False
        return True

    def cost(n, adj):
        """Count number of K5 and independent sets of size 5."""
        count = 0
        for combo in itertools.combinations(range(n), 5):
            is_clique = True
            is_indep = True
            for i in range(5):
                for j in range(i+1, 5):
                    u, v = combo[i], combo[j]
                    if (adj[u] >> v) & 1:
                        is_indep = False
                    else:
                        is_clique = False
                    if not is_clique and not is_indep:
                        break
                if not is_clique and not is_indep:
                    break
            if is_clique:
                count += 1
            if is_indep:
                count += 1
        return count

    def get_delta_cost(n, adj, u, v):
        """
        Calculate change in cost when flipping edge (u, v).
        Only checks 5-sets containing both u and v.
        """
        delta = 0
        # Generate all 3-subsets of V \ {u, v}
        # V = range(n)
        others = [x for x in range(n) if x != u and x != v]
        for x, y, z in itertools.combinations(others, 3):
            # Set K = {u, v, x, y, z}
            # Check old status
            # Check new status
            # We can optimize by checking edges
            
            # Edges in K: (u,v), (u,x), (u,y), (u,z), (v,x), (v,y), (v,z), (x,y), (x,z), (y,z)
            
            # Old state
            e_uv = (adj[u] >> v) & 1
            e_ux = (adj[u] >> x) & 1
            e_uy = (adj[u] >> y) & 1
            e_uz = (adj[u] >> z) & 1
            e_vx = (adj[v] >> x) & 1
            e_vy = (adj[v] >> y) & 1
            e_vz = (adj[v] >> z) & 1
            e_xy = (adj[x] >> y) & 1
            e_xz = (adj[x] >> z) & 1
            e_yz = (adj[y] >> z) & 1
            
            old_clique = (e_uv == 1 and e_ux == 1 and e_uy == 1 and e_uz == 1 and 
                          e_vx == 1 and e_vy == 1 and e_vz == 1 and 
                          e_xy == 1 and e_xz == 1 and e_yz == 1)
            
            old_indep = (e_uv == 0 and e_ux == 0 and e_uy == 0 and e_uz == 0 and 
                         e_vx == 0 and e_vy == 0 and e_vz == 0 and 
                         e_xy == 0 and e_xz == 0 and e_yz == 0)
            
            # New state (flip u-v)
            new_e_uv = 1 - e_uv
            
            new_clique = (new_e_uv == 1 and e_ux == 1 and e_uy == 1 and e_uz == 1 and 
                          e_vx == 1 and e_vy == 1 and e_vz == 1 and 
                          e_xy == 1 and e_xz == 1 and e_yz == 1)
            
            new_indep = (new_e_uv == 0 and e_ux == 0 and e_uy == 0 and e_uz == 0 and 
                         e_vx == 0 and e_vy == 0 and e_vz == 0 and 
                         e_xy == 0 and e_xz == 0 and e_yz == 0)
            
            if old_clique and not new_clique:
                delta -= 1
            if new_clique and not old_clique:
                delta += 1
            if old_indep and not new_indep:
                delta -= 1
            if new_indep and not old_indep:
                delta += 1
                
        return delta

    def solve(n):
        # Initialize random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        current_cost = cost(n, adj)
        if current_cost == 0:
            return adj
        
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        
        while T > min_T and time.time() < deadline:
            # Pick random edge
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            delta = get_delta_cost(n, adj, u, v)
            
            if delta < 0 or random.random() < pow(2.71828, -delta / T):
                # Accept
                # Flip edge
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                current_cost += delta
                
                if current_cost == 0:
                    return adj
            
            T *= cooling
        
        return None

    # Try to find solution for n=42
    target_n = 42
    best_adj = None
    
    # Run multiple restarts if time permits
    while time.time() < deadline:
        adj = solve(target_n)
        if adj:
            best_adj = adj
            break
        # If we are running out of time, break
        if time.time() > deadline - 5.0:
            break
            
    if best_adj:
        # Convert to edge list
        edges = []
        for i in range(target_n):
            for j in range(i+1, target_n):
                if (best_adj[i] >> j) & 1:
                    edges.append((i, j))
        return target_n, edges
    else:
        # Fallback: Paley graph on 37 vertices
        q = 37
        residues = {(x * x) % q for x in range(1, q)}
        return q, [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]

# EVOLVE-BLOCK-END
