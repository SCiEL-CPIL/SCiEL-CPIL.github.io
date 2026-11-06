import random
import time
import numpy as np

def construct():
    n = 31
    target_edges = 91
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    # Precompute all possible edges as indices
    all_edge_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    all_edge_indices = list(range(len(all_edge_pairs)))
    
    def count_c4_vectorized(A):
        """Count C4s using vectorized numpy operations.
        C4_count = (1/2) * sum_{i<j} C((A^2)[i,j], 2)
        """
        A2 = A @ A
        # Get upper triangle
        iu = np.triu_indices(n, 1)
        vals = A2[iu]
        c4_sum = np.sum(vals * (vals - 1) // 2)
        return c4_sum // 2
    
    def build_pg25_points():
        """Build the 31 points of PG(2,5)."""
        points = []
        seen = set()
        for x in range(5):
            for y in range(5):
                for z in range(5):
                    if x == 0 and y == 0 and z == 0:
                        continue
                    if x != 0:
                        key = (1, y * pow(x, 4, 5), z * pow(x, 4, 5))
                    elif y != 0:
                        key = (0, 1, z * pow(y, 4, 5))
                    else:
                        key = (0, 0, 1)
                    if key not in seen:
                        seen.add(key)
                        points.append(key)
        return points
    
    def build_polarity_graph(points):
        """Build a polarity graph of PG(2,5)."""
        edge_set = set()
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                p = points[i]
                q = points[j]
                b = (p[0]*q[0] + p[1]*q[1] + p[2]*q[2]) % 5
                if b == 0:
                    edge_set.add((i, j))
        return edge_set
    
    def edge_set_to_matrix(edge_set):
        A = np.zeros((n, n), dtype=np.int32)
        for u, v in edge_set:
            A[u, v] = 1
            A[v, u] = 1
        return A
    
    # Build initial graph from polarity
    points = build_pg25_points()
    polarity_edges = build_polarity_graph(points)
    
    current_edges = set(polarity_edges)
    
    # Adjust to exactly 91 edges
    while len(current_edges) > target_edges:
        A = edge_set_to_matrix(current_edges)
        best_remove = None
        best_score = float('inf')
        for e in list(current_edges):
            u, v = e
            common = [w for w in range(n) if w != u and w != v and A[u, w] and A[v, w]]
            c4s = 0
            for idx, a in enumerate(common):
                for b in common[idx+1:]:
                    if A[a, b]:
                        c4s += 1
            if c4s < best_score:
                best_score = c4s
                best_remove = e
        if best_remove:
            current_edges.discard(best_remove)
    
    while len(current_edges) < target_edges:
        A = edge_set_to_matrix(current_edges)
        best_add = None
        best_score = float('inf')
        for i in range(n):
            for j in range(i + 1, n):
                if not A[i, j]:
                    common = sum(1 for w in range(n) if A[i, w] and A[j, w])
                    c4s = common * (common - 1) // 2
                    if c4s < best_score:
                        best_score = c4s
                        best_add = (i, j)
        if best_add:
            current_edges.add(best_add)
    
    A = edge_set_to_matrix(current_edges)
    current_c4 = count_c4_vectorized(A)
    
    best_c4 = current_c4
    best_edges_set = set(current_edges)
    
    # Fast local search with vectorized delta computation
    # Key insight: for each edge removal, compute C4s destroyed quickly
    # For each non-edge addition, compute C4s created quickly
    
    improved = True
    max_local_iters = 5000
    
    while improved and time.time() < deadline:
        improved = False
        A = edge_set_to_matrix(current_edges)
        edge_list = list(current_edges)
        random.shuffle(edge_list)
        
        # Precompute common neighbors for all edges (vectorized)
        # A2[i,j] = number of common neighbors of i and j
        A2 = A @ A
        
        for remove_u, remove_v in edge_list:
            if time.time() > deadline:
                break
            
            # C4s destroyed by removing edge (remove_u, remove_v):
            # = number of edges in induced subgraph on N(u) ∩ N(v)
            # where N(u) ∩ N(v) = {w : A[u,w]=1 and A[v,w]=1, w != u, w != v}
            common_mask = (A[remove_u] & A[remove_v]).copy()
            common_mask[remove_u] = 0
            common_mask[remove_v] = 0
            common_list = np.where(common_mask)[0]
            
            if len(common_list) < 2:
                c4s_destroyed = 0
            else:
                # Count edges among common neighbors
                sub_A = A[np.ix_(common_list, common_list)]
                c4s_destroyed = int(np.triu(sub_A, 1).sum())
            
            # Find best edge to add: minimize new C4s created
            # For non-edge (i,j), new C4s = C(common(i,j), 2) where common counts
            # But we need edges among common neighbors, not just pairs
            # Actually: new C4s from adding (i,j) = number of paths i-a-j-b-i
            # = number of edges in induced subgraph on N(i) ∩ N(j)
            
            best_add = None
            best_delta = float('inf')
            
            # Vectorized: for all non-edges, compute common neighbor count
            # A2[i,j] gives common neighbors
            non_edge_mask = (1 - A)
            non_edge_mask = np.triu(non_edge_mask, 1)
            
            # For each non-edge, new C4s = number of edges among common neighbors
            # This is harder to vectorize directly, so we do it semi-vectorized
            non_edges = np.argwhere(non_edge_mask)
            
            # Sample a subset if too many
            if len(non_edges) > 200:
                idx = np.random.choice(len(non_edges), 200, replace=False)
                non_edges_sample = non_edges[idx]
            else:
                non_edges_sample = non_edges
            
            for i, j in non_edges_sample:
                i, j = int(i), int(j)
                common_ij = (A[i] & A[j])
                common_list_ij = np.where(common_ij)[0]
                if len(common_list_ij) < 2:
                    new_c4s = 0
                else:
                    sub_A2 = A[np.ix_(common_list_ij, common_list_ij)]
                    new_c4s = int(np.triu(sub_A2, 1).sum())
                
                delta = -c4s_destroyed + new_c4s
                if delta < best_delta:
                    best_delta = delta
                    best_add = (i, j)
            
            if best_add is not None and best_delta < 0:
                # Apply swap
                current_edges.discard((remove_u, remove_v))
                A[remove_u, remove_v] = 0
                A[remove_v, remove_u] = 0
                
                add_u, add_v = best_add
                current_edges.add((add_u, add_v))
                A[add_u, add_v] = 1
                A[add_v, add_u] = 1
                
                current_c4 += best_delta
                improved = True
                
                if current_c4 < best_c4:
                    best_c4 = current_c4
                    best_edges_set = set(current_edges)
                
                break  # Restart after a change
    
    # Random restarts with perturbation
    remaining_time = deadline - time.time()
    if remaining_time > 15:
        num_restarts = min(int(remaining_time // 15), 10)
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            current_edges = set(best_edges_set)
            # Perturb: remove k random edges, add k random non-edges
            k = random.randint(3, 8)
            edge_list = list(current_edges)
            random.shuffle(edge_list)
            for _ in range(min(k, len(edge_list))):
                current_edges.discard(edge_list.pop())
            
            non_edges = [(i, j) for i in range(n) for j in range(i+1, n) 
                        if (i, j) not in current_edges]
            random.shuffle(non_edges)
            for _ in range(min(k, len(non_edges))):
                current_edges.add(non_edges.pop())
            
            A = edge_set_to_matrix(current_edges)
            current_c4 = count_c4_vectorized(A)
            
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_edges_set = set(current_edges)
            
            # Quick local search for this restart
            local_improved = True
            max_quick_iters = 200
            quick_iter = 0
            while local_improved and quick_iter < max_quick_iters and time.time() < deadline - 3:
                local_improved = False
                quick_iter += 1
                A = edge_set_to_matrix(current_edges)
                edge_list = list(current_edges)
                random.shuffle(edge_list)
                
                for remove_u, remove_v in edge_list:
                    if time.time() > deadline - 3:
                        break
                    
                    common_mask = (A[remove_u] & A[remove_v]).copy()
                    common_mask[remove_u] = 0
                    common_mask[remove_v] = 0
                    common_list = np.where(common_mask)[0]
                    
                    if len(common_list) < 2:
                        c4s_destroyed = 0
                    else:
                        sub_A = A[np.ix_(common_list, common_list)]
                        c4s_destroyed = int(np.triu(sub_A, 1).sum())
                    
                    best_add = None
                    best_delta = float('inf')
                    
                    non_edge_mask = np.triu(1 - A, 1)
                    non_edges = np.argwhere(non_edge_mask)
                    if len(non_edges) > 150:
                        idx = np.random.choice(len(non_edges), 150, replace=False)
                        non_edges_sample = non_edges[idx]
                    else:
                        non_edges_sample = non_edges
                    
                    for i, j in non_edges_sample:
                        i, j = int(i), int(j)
                        common_ij = (A[i] & A[j])
                        common_list_ij = np.where(common_ij)[0]
                        if len(common_list_ij) < 2:
                            new_c4s = 0
                        else:
                            sub_A2 = A[np.ix_(common_list_ij, common_list_ij)]
                            new_c4s = int(np.triu(sub_A2, 1).sum())
                        
                        delta = -c4s_destroyed + new_c4s
                        if delta < best_delta:
                            best_delta = delta
                            best_add = (i, j)
                    
                    if best_add is not None and best_delta < 0:
                        current_edges.discard((remove_u, remove_v))
                        A[remove_u, remove_v] = 0
                        A[remove_v, remove_u] = 0
                        add_u, add_v = best_add
                        current_edges.add((add_u, add_v))
                        A[add_u, add_v] = 1
                        A[add_v, add_u] = 1
                        current_c4 += best_delta
                        local_improved = True
                        
                        if current_c4 < best_c4:
                            best_c4 = current_c4
                            best_edges_set = set(current_edges)
                        break
    
    # Convert to list of lists
    if best_edges_set is None:
        best_edges = [[u, v] for u, v in all_edge_pairs[:target_edges]]
    else:
        best_edges = [[u, v] for u, v in sorted(best_edges_set)]
    
    return best_edges
