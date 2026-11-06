import random
import time
import numpy as np

def construct():
    deadline = time.time() + 115
    n = 16
    target_edges = 65
    
    def compute_score(adj, deg):
        """Compute the maximum degree sum over all triangles using NumPy."""
        max_sum = 0
        worst_tri = None
        
        # Get all edges (u, v) with u < v
        edges = np.argwhere(adj)
        edges = edges[edges[:, 0] < edges[:, 1]]
        
        if len(edges) == 0:
            return 0, None

        for u, v in edges:
            # Find common neighbors w
            # adj[u] is a boolean vector
            common = np.where(adj[u] & adj[v])[0]
            if len(common) == 0:
                continue
            
            # Calculate sums
            sums = deg[u] + deg[v] + deg[common]
            local_max = np.max(sums)
            
            if local_max > max_sum:
                max_sum = local_max
                w_idx = np.argmax(sums)
                w = common[w_idx]
                worst_tri = (u, v, w)
                
        return max_sum, worst_tri
    
    def get_triangles(adj):
        """Get all triangles in the graph."""
        triangles = []
        for u in range(n):
            for v in range(u + 1, n):
                if not adj[u, v]:
                    continue
                for w in range(v + 1, n):
                    if adj[u, w] and adj[v, w]:
                        triangles.append((u, v, w))
        return triangles
    
    def local_search(adj, deg, score, worst_tri, max_iters, deadline, use_targeted=True):
        """Local search with Best-of-N move selection."""
        current_adj = adj.copy()
        current_deg = deg.copy()
        current_score = score
        current_worst = worst_tri
        
        T = 2.0
        cooling = 0.998
        
        all_possible_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        
        for it in range(max_iters):
            if time.time() > deadline - 3:
                break
            
            moves = []
            
            # Targeted moves
            if use_targeted and current_worst is not None:
                u, v, w = current_worst
                candidate_removals = [(u, v), (u, w), (v, w)]
                random.shuffle(candidate_removals)
                
                for e1 in candidate_removals:
                    if not current_adj[e1[0], e1[1]]:
                        continue
                    # Sample 5 random additions
                    for _ in range(5):
                        e2 = random.choice(all_possible_edges)
                        if current_adj[e2[0], e2[1]]:
                            continue
                        moves.append((e1, e2))
            
            # Random moves
            for _ in range(10):
                e1 = random.choice(all_possible_edges)
                if not current_adj[e1[0], e1[1]]:
                    continue
                e2 = random.choice(all_possible_edges)
                if current_adj[e2[0], e2[1]]:
                    continue
                moves.append((e1, e2))
            
            if not moves:
                continue
                
            best_candidate_score = float('inf')
            best_candidate_adj = None
            best_candidate_deg = None
            best_candidate_worst = None
            best_candidate_move = None
            
            # Evaluate candidates
            for e1, e2 in moves:
                u1, v1 = e1
                u2, v2 = e2
                
                if (u1 == u2 and v1 == v2) or (u1 == v2 and v1 == u2):
                    continue
                    
                new_adj = current_adj.copy()
                new_deg = current_deg.copy()
                
                new_adj[u1, v1] = False
                new_adj[v1, u1] = False
                new_deg[u1] -= 1
                new_deg[v1] -= 1
                
                new_adj[u2, v2] = True
                new_adj[v2, u2] = True
                new_deg[u2] += 1
                new_deg[v2] += 1
                
                new_score, new_worst = compute_score(new_adj, new_deg)
                
                if new_score < best_candidate_score:
                    best_candidate_score = new_score
                    best_candidate_adj = new_adj
                    best_candidate_deg = new_deg
                    best_candidate_worst = new_worst
                    best_candidate_move = (e1, e2)
            
            # Acceptance
            if best_candidate_move is not None:
                delta = best_candidate_score - current_score
                if delta < 0 or random.random() < np.exp(-delta / T):
                    current_adj = best_candidate_adj
                    current_deg = best_candidate_deg
                    current_score = best_candidate_score
                    current_worst = best_candidate_worst
            
            T *= cooling
        
        return current_adj, current_deg, current_score, current_worst
    
    def polish(adj, deg, score, worst_tri, deadline):
        """Final high-precision polish: Best-of-N targeted edge swaps."""
        current_adj = adj.copy()
        current_deg = deg.copy()
        current_score = score
        current_worst = worst_tri
        
        all_possible_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        
        for pass_num in range(20):
            if time.time() > deadline - 2:
                break
            
            improved = False
            if current_worst is not None:
                u, v, w = current_worst
                candidate_removals = [(u, v), (u, w), (v, w)]
                random.shuffle(candidate_removals)
                
                for e1 in candidate_removals:
                    if not current_adj[e1[0], e1[1]]:
                        continue
                    
                    # Try all non-edges
                    non_edges = [(a, b) for a, b in all_possible_edges if not current_adj[a, b]]
                    random.shuffle(non_edges)
                    
                    best_e2 = None
                    best_new_score = current_score
                    best_new_adj = None
                    best_new_deg = None
                    best_new_worst = None
                    
                    for e2 in non_edges:
                        u1, v1 = e1
                        u2, v2 = e2
                        
                        if (u1 == u2 and v1 == v2) or (u1 == v2 and v1 == u2):
                            continue
                            
                        new_adj = current_adj.copy()
                        new_deg = current_deg.copy()
                        
                        new_adj[u1, v1] = False
                        new_adj[v1, u1] = False
                        new_deg[u1] -= 1
                        new_deg[v1] -= 1
                        
                        new_adj[u2, v2] = True
                        new_adj[v2, u2] = True
                        new_deg[u2] += 1
                        new_deg[v2] += 1
                        
                        new_score, new_worst = compute_score(new_adj, new_deg)
                        
                        if new_score < best_new_score:
                            best_new_score = new_score
                            best_new_adj = new_adj
                            best_new_deg = new_deg
                            best_new_worst = new_worst
                            best_e2 = e2
                    
                    if best_e2 is not None:
                        current_adj = best_new_adj
                        current_deg = best_new_deg
                        current_score = best_new_score
                        current_worst = best_new_worst
                        improved = True
                        break # Restart pass with new worst
            
            if not improved:
                break
        
        return current_adj, current_deg, current_score, current_worst
    
    # Try multiple random initializations and local searches
    best_score = float('inf')
    best_adj = None
    best_deg = None
    best_worst_tri = None
    
    for seed in range(120):
        if time.time() > deadline - 15:
            break
        random.seed(seed)
        
        # Start from K_{8,8} + random edge, or random near-bipartite
        adj = np.zeros((n, n), dtype=bool)
        
        # Create a random bipartite-like structure
        # Partition vertices into two sets
        perm = list(range(n))
        random.shuffle(perm)
        split = random.randint(7, 9)  # More balanced partitions
        A = set(perm[:split])
        B = set(perm[split:])
        
        # Add all edges between A and B
        for u in A:
            for v in B:
                adj[u, v] = True
                adj[v, u] = True
        
        edge_count = len(A) * len(B)
        
        # Add remaining edges within A or B to reach target
        remaining = target_edges - edge_count
        if remaining > 0:
            # Add random edges within parts
            internal_edges = []
            for part in [A, B]:
                part_list = list(part)
                for i in range(len(part_list)):
                    for j in range(i + 1, len(part_list)):
                        u, v = part_list[i], part_list[j]
                        if not adj[u, v]:
                            internal_edges.append((u, v))
            random.shuffle(internal_edges)
            for u, v in internal_edges[:remaining]:
                adj[u, v] = True
                adj[v, u] = True
        elif remaining < 0:
            # Remove some cross edges
            cross_edges = [(u, v) for u in A for v in B]
            random.shuffle(cross_edges)
            for u, v in cross_edges[:abs(remaining)]:
                adj[u, v] = False
                adj[v, u] = False
        
        deg = adj.sum(axis=1).astype(int)
        score, worst_tri = compute_score(adj, deg)
        
        if score < best_score:
            best_score = score
            best_adj = adj.copy()
            best_deg = deg.copy()
            best_worst_tri = worst_tri
        
        # Local search from this starting point
        current_adj, current_deg, current_score, current_worst = local_search(
            adj, deg, score, worst_tri, 5000, deadline, use_targeted=True
        )
        
        if current_score < best_score:
            best_score = current_score
            best_adj = current_adj.copy()
            best_deg = current_deg.copy()
            best_worst_tri = current_worst
    
    # Final polish pass
    if best_adj is not None:
        polished_adj, polished_deg, polished_score, polished_worst = polish(
            best_adj, best_deg, best_score, best_worst_tri, deadline
        )
        
        if polished_score < best_score:
            best_score = polished_score
            best_adj = polished_adj.copy()
            best_deg = polished_deg.copy()
            best_worst_tri = polished_worst
    
    # Convert best_adj to edge list
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v]:
                edges.append([u, v])
    
    return {'n': n, 'edges': edges}
