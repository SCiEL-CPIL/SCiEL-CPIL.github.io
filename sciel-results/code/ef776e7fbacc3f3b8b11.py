import time
import random
import sys

def construct():
    n = 15
    start_time = time.time()
    deadline = 50.0  # Safety margin before 60s kill
    
    # All possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(all_edges)
    
    # Map edge index to (u, v)
    edge_to_uv = {idx: (i, j) for idx, (i, j) in enumerate(all_edges)}
    
    # Precompute for each edge index, the list of "conflict pairs" (s, t)
    # A conflict exists if adding this edge creates a path s-u-v-t where deg(s)==deg(t)
    # This is dynamic, so we can't precompute fully, but we can precompute the structure.
    
    # For fast validity checking during backtracking:
    # We maintain:
    # - adj_mask: list of int bitmasks for adjacency
    # - deg: list of degrees
    # - edge_present: boolean array of length m
    
    # Heuristic for ordering edges to try adding:
    # We want to add edges that are "likely" to be valid.
    # A simple heuristic: try edges between vertices with currently different degrees first?
    # Or just random shuffle with backtracking.
    
    # Since n=15 is small (105 edges), we can use a recursive backtracking search.
    # To maximize edge count, we can use a branch-and-bound approach:
    # 1. Sort edges by some heuristic (e.g., potential to connect high-degree nodes).
    # 2. Recursively try to include each edge.
    # 3. Prune if the maximum possible remaining edges + current edges < best found.
    
    # However, the constraint is global (degrees change as we add edges).
    # Let's use a randomized restart backtracking approach.
    
    best_count = 0
    best_edges = None
    
    def is_valid_state(adj_mask, deg):
        # Check all paths s-u-v-t
        # For each edge (u, v) in current graph:
        for u in range(n):
            au = adj_mask[u]
            for v in range(u + 1, n):
                if not (au & (1 << v)):
                    continue
                # Edge (u, v) exists
                au_minus_v = au & ~(1 << v)
                av_minus_u = adj_mask[v] & ~(1 << u)
                
                # Iterate neighbors s of u (excluding v)
                s_mask = au_minus_v
                while s_mask:
                    s_bit = s_mask & -s_mask
                    s = s_bit.bit_length() - 1
                    s_mask ^= s_bit
                    
                    # Iterate neighbors t of v (excluding u)
                    # And t != s
                    t_mask = av_minus_u & ~(1 << s)
                    while t_mask:
                        t_bit = t_mask & -t_mask
                        t = t_bit.bit_length() - 1
                        t_mask ^= t_bit
                        
                        if deg[s] == deg[t]:
                            return False
        return True

    def backtrack(edge_idx, adj_mask, deg, current_edges, current_count):
        nonlocal best_count, best_edges
        
        # Pruning: if we can't beat best_count, stop
        # Max possible edges is m. Remaining edges = m - edge_idx.
        if current_count + (m - edge_idx) <= best_count:
            return
        
        # Base case: all edges considered
        if edge_idx == m:
            if current_count > best_count:
                best_count = current_count
                best_edges = current_edges.copy()
            return
        
        # Check time
        if time.time() > deadline:
            return
            
        u, v = edge_to_uv[edge_idx]
        
        # Option 1: Add edge (u, v)
        # Check if adding it keeps validity
        # We need to check if the NEW graph is valid.
        # Since degrees change, we must check the whole graph or incremental changes.
        # For n=15, full check is ~105 edges * ~14^2 paths ~ 20k ops. Very fast.
        
        # Try adding
        new_adj = adj_mask[:]
        new_deg = deg[:]
        new_adj[u] |= (1 << v)
        new_adj[v] |= (1 << u)
        new_deg[u] += 1
        new_deg[v] += 1
        
        if is_valid_state(new_adj, new_deg):
            current_edges.append(edge_idx)
            backtrack(edge_idx + 1, new_adj, new_deg, current_edges, current_count + 1)
            current_edges.pop()
        
        # Option 2: Skip edge (u, v)
        backtrack(edge_idx + 1, adj_mask, deg, current_edges, current_count)

    # To improve performance, we can use a heuristic ordering of edges.
    # Edges that connect vertices with "compatible" degrees might be better.
    # But degrees change. 
    # Let's try a few different random orderings of edges to break symmetry and find different local maxima.
    
    random.seed(42)
    
    # We will run multiple restarts with different edge orderings.
    # Since backtracking is exponential, we limit the depth or use time cuts.
    # For n=15, 105 edges, full backtracking is impossible.
    # We need a smarter search.
    
    # Alternative: Simulated Annealing or Tabu Search is what the parent did.
    # The parent scored 2.0.
    # Let's try a different local search: **Min-Conflict Heuristic**.
    # Start with a random valid graph (e.g., star).
    # Then try to add edges. If invalid, resolve conflicts by removing edges.
    
    # Actually, let's stick to the "Different Design Family" directive.
    # The parent used Greedy + Tabu.
    # Let's use **Integer Programming** formulation solved by a simple branch-and-bound?
    # Or **Genetic Algorithm**?
    
    # Let's try a **Genetic Algorithm** with a custom crossover that preserves validity.
    # Or better: **Hill Climbing with Random Restarts and Heavy Perturbation (Large Neighborhood Search)**.
    
    # Let's implement a Large Neighborhood Search (LNS):
    # 1. Start with a valid graph (e.g., complete graph minus some edges, or just empty).
    # 2. Destroy a random subset of edges (or all edges in a random subset of vertices).
    # 3. Reconstruct by greedily adding edges back in a random order, checking validity.
    # 4. Keep the best solution.
    
    # This is structurally different from simple tabu/greedy.
    
    def destroy_reconstruct(initial_adj, initial_deg, initial_count):
        # Pick a random subset of vertices to "destroy"
        # Or pick a random percentage of edges to remove.
        # Let's remove edges incident to a random set of k vertices.
        k = random.randint(3, 6)
        destroy_vertices = random.sample(range(n), k)
        
        adj = initial_adj[:]
        deg = initial_deg[:]
        count = initial_count
        
        # Remove all edges incident to destroy_vertices
        for v in destroy_vertices:
            # Get all neighbors
            neighbors = []
            mask = adj[v]
            while mask:
                bit = mask & -mask
                u = bit.bit_length() - 1
                neighbors.append(u)
                mask ^= bit
            
            for u in neighbors:
                adj[v] &= ~(1 << u)
                adj[u] &= ~(1 << v)
                deg[v] -= 1
                deg[u] -= 1
                count -= 1
        
        # Reconstruct: Try to add back all possible edges involving destroy_vertices
        # And also edges between destroy_vertices?
        # Let's try to add back any edge that was removed, plus maybe some new ones?
        # No, just re-add the removed ones in random order.
        
        # Collect removed edges
        # We didn't track them, so let's just try to add ALL edges that are currently missing
        # but incident to destroy_vertices.
        
        candidate_edges = []
        for v in destroy_vertices:
            for u in range(n):
                if u > v and not (adj[v] & (1 << u)):
                    candidate_edges.append((v, u))
        
        random.shuffle(candidate_edges)
        
        for u, v in candidate_edges:
            if adj[u] & (1 << v):
                continue
            # Try to add
            new_adj = adj[:]
            new_deg = deg[:]
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            new_deg[u] += 1
            new_deg[v] += 1
            
            if is_valid_state(new_adj, new_deg):
                adj = new_adj
                deg = new_deg
                count += 1
        
        return adj, deg, count

    # Initial state: Empty graph is valid.
    # Let's start with a few good seeds.
    # Seed 1: Star graph (14 edges)
    # Seed 2: Random valid graph
    
    best_adj = [0] * n
    best_deg = [0] * n
    best_count = 0
    
    # Try to build a good initial solution using the greedy method from parent (it was good)
    # But we will use LNS to improve it.
    
    # Quick greedy init
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.shuffle(all_pairs)
    init_adj = [0] * n
    init_deg = [0] * n
    init_count = 0
    for u, v in all_pairs:
        new_adj = init_adj[:]
        new_deg = init_deg[:]
        new_adj[u] |= (1 << v)
        new_adj[v] |= (1 << u)
        new_deg[u] += 1
        new_deg[v] += 1
        if is_valid_state(new_adj, new_deg):
            init_adj = new_adj
            init_deg = new_deg
            init_count += 1
    
    best_adj = init_adj[:]
    best_deg = init_deg[:]
    best_count = init_count
    
    # LNS Loop
    while time.time() < deadline:
        # Destroy and Reconstruct
        new_adj, new_deg, new_count = destroy_reconstruct(best_adj, best_deg, best_count)
        
        if new_count > best_count:
            best_count = new_count
            best_adj = new_adj
            best_deg = new_deg
            
        # Also try a simpler perturbation: remove 1 random edge, add 1 random edge
        # This helps escape local minima that LNS might miss if k is too large.
        if random.random() < 0.5:
            # Find a random edge to remove
            edges_list = []
            for i in range(n):
                for j in range(i+1, n):
                    if best_adj[i] & (1 << j):
                        edges_list.append((i, j))
            
            if edges_list:
                u1, v1 = random.choice(edges_list)
                # Remove
                test_adj = best_adj[:]
                test_deg = best_deg[:]
                test_adj[u1] &= ~(1 << v1)
                test_adj[v1] &= ~(1 << u1)
                test_deg[u1] -= 1
                test_deg[v1] -= 1
                test_count = best_count - 1
                
                # Try to add a random non-edge
                non_edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if not (test_adj[i] & (1 << j)):
                            non_edges.append((i, j))
                
                if non_edges:
                    random.shuffle(non_edges)
                    for u2, v2 in non_edges:
                        new_adj = test_adj[:]
                        new_deg = test_deg[:]
                        new_adj[u2] |= (1 << v2)
                        new_adj[v2] |= (1 << u2)
                        new_deg[u2] += 1
                        new_deg[v2] += 1
                        
                        if is_valid_state(new_adj, new_deg):
                            # Valid swap
                            if test_count + 1 > best_count:
                                best_count = test_count + 1
                                best_adj = new_adj
                                best_deg = new_deg
                            break

    # Convert best_adj to edges
    final_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i] & (1 << j):
                final_edges.append([int(i), int(j)])
                
    return final_edges

if __name__ == "__main__":
    # Simple self-test
    edges = construct()
    print(f"Found {len(edges)} edges")
