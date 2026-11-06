"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    def check_valid(adj, n, min_deg):
        """Check if graph is C4-free and has min degree >= min_deg."""
        # Check min degree
        if np.min(adj.sum(axis=1)) < min_deg:
            return False
        # Check C4
        # A C4 exists if any two vertices have > 1 common neighbor
        # Common neighbors count is (adj @ adj.T)[i][j] for i != j
        # But we need to be careful: (adj @ adj.T)[i][j] counts paths of length 2.
        # If count > 1, there is a C4.
        # We can compute this once.
        common = adj @ adj.T
        np.fill_diagonal(common, 0)
        if np.any(common > 1):
            return False
        return True

    def get_edges(adj, n):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j]:
                    edges.append([i, j])
        return edges

    def solve_for_n(n, min_deg, max_iterations=20000):
        """
        Try to find a C4-free graph on n vertices with min degree >= min_deg.
        Uses random initialization and repair.
        """
        # We need min_deg edges per vertex.
        # Total edges approx n * min_deg / 2.
        # Probability p should be around min_deg / n.
        # But we start dense and remove edges.
        
        # Try multiple random seeds/initializations
        for seed in range(100):
            if time.time() > deadline:
                return None
            
            random.seed(seed)
            # Initialize with a random graph with high edge probability to ensure high degree
            # p = min_deg / n * 1.5 gives some slack
            p = min(min_deg / n * 1.5, 0.9)
            adj = np.zeros((n, n), dtype=np.uint8)
            
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < p:
                        adj[i, j] = 1
                        adj[j, i] = 1
            
            # Check initial min degree
            if np.min(adj.sum(axis=1)) < min_deg:
                # Try to add edges to low-degree vertices? 
                # Or just restart with different seed.
                # Adding edges might create C4s, but we have repair loop.
                # Let's try to boost degrees first.
                for _ in range(100):
                    if time.time() > deadline:
                        return None
                    degs = adj.sum(axis=1)
                    low_verts = np.where(degs < min_deg)[0]
                    if len(low_verts) == 0:
                        break
                    u = random.choice(low_verts)
                    # Add a random edge
                    v = random.randint(0, n-1)
                    if v != u and adj[u, v] == 0:
                        adj[u, v] = 1
                        adj[v, u] = 1
            
            if np.min(adj.sum(axis=1)) < min_deg:
                continue
                
            # Repair loop: remove edges to break C4s
            for it in range(max_iterations):
                if time.time() > deadline:
                    return None
                
                # Check for C4
                common = adj @ adj.T
                np.fill_diagonal(common, 0)
                
                # Find a pair with common > 1
                # We want to find any i, j such that common[i,j] > 1
                # Using numpy find
                indices = np.argwhere(common > 1)
                
                if len(indices) == 0:
                    # No C4s
                    if np.min(adj.sum(axis=1)) >= min_deg:
                        return adj
                    else:
                        # C4-free but min degree failed. 
                        # This is a dead end for this initialization usually.
                        # Or we could try to add edges, but that creates C4s.
                        # Break to try next seed.
                        break
                
                # Pick a violating pair
                # To be effective, pick one with highest common count? Or random?
                # Random is fine.
                idx = random.choice(indices)
                u, v = idx[0], idx[1]
                
                # We need to reduce common neighbors of u and v to <= 1.
                # Common neighbors are w such that adj[u,w]=1 and adj[v,w]=1.
                common_neighbors = np.where((adj[u] == 1) & (adj[v] == 1))[0]
                
                # We have len(common_neighbors) > 1.
                # We need to remove edges.
                # Removing (u, w) or (v, w) for w in common_neighbors reduces the count by 1.
                # We should remove enough to bring count to 1.
                # num_to_remove = len(common_neighbors) - 1
                
                # Strategy: Remove edges incident to the vertex with higher degree?
                # Or just pick random w's.
                # Let's pick random w's from common_neighbors.
                # We need to remove at least 1 edge to break this specific C4 instance?
                # Actually, removing one edge (u,w) breaks all C4s involving that specific path u-w-v.
                # But u and v might share multiple w's.
                # If we remove (u, w1), the count of common neighbors drops by 1.
                # If count was 3, now it is 2. Still C4 exists.
                # So we should remove edges until count <= 1.
                
                # Let's remove edges for all but one common neighbor.
                # Shuffle common neighbors
                cn_list = list(common_neighbors)
                random.shuffle(cn_list)
                
                # We need to keep 1, remove the rest?
                # Not necessarily, maybe we can remove edges from other C4s first.
                # But let's just fix this pair.
                # Remove edges (u, w) for all w in cn_list[1:]
                # Or (v, w).
                
                # To preserve degree, remove from the vertex with higher degree?
                # Or alternate.
                # Let's just pick one w to remove for now, and loop again.
                # This is safer for time management.
                w = cn_list[0]
                
                # Decide which edge to remove: (u,w) or (v,w)
                # Remove from higher degree vertex to preserve min degree?
                # deg(u) vs deg(v)
                if adj[u].sum() >= adj[v].sum():
                    adj[u, w] = 0
                    adj[w, u] = 0
                else:
                    adj[v, w] = 0
                    adj[w, v] = 0
            
        return None

    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        min_deg = n - 39
        if min_deg < 1:
            continue
        
        # If min_deg is very small, it's easier.
        # For n=47, min_deg=8.
        # For n=46, min_deg=7.
        
        result_adj = solve_for_n(n, min_deg)
        
        if result_adj is not None:
            # Verify one last time
            if check_valid(result_adj, n, min_deg):
                edges = get_edges(result_adj, n)
                return {'n': n, 'edges': edges}
        
        # If we fail for n, try n-1.
        # But we should use remaining time wisely.
        # If we fail quickly, we can try more iterations for lower n.
        
    # Fallback: empty graph on 39 vertices
    return {'n': 39, 'edges': []}

# EVOLVE-BLOCK-END
