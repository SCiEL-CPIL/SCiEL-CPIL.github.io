import numpy as np
import time

def construct():
    n = 40
    deadline = time.time() + 105.0

    # Initialize adjacency matrix with 8 disjoint K5s
    A = np.zeros((n, n), dtype=np.int64)
    adj_list = [[] for _ in range(n)]

    def add_edge(u, v):
        A[u, v] = 1
        A[v, u] = 1
        adj_list[u].append(v)
        adj_list[v].append(u)

    def remove_edge(u, v):
        A[u, v] = 0
        A[v, u] = 0
        adj_list[u].remove(v)
        adj_list[v].remove(u)

    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u = 5 * k + a
                v = 5 * k + b
                add_edge(u, v)

    m = 80
    best_m = m
    best_A = A.copy()
    best_adj_list = [lst.copy() for lst in adj_list]

    def compute_blocked_matrix(adj_list):
        B = np.zeros((n, n), dtype=bool)
        for u in range(n):
            visited_mask = 1 << u

            def dfs(node, depth, mask):
                if depth == 5:
                    return
                for neighbor in adj_list[node]:
                    if mask & (1 << neighbor):
                        continue
                    new_mask = mask | (1 << neighbor)
                    if depth + 1 == 5:
                        B[u, neighbor] = True
                    else:
                        dfs(neighbor, depth + 1, new_mask)

            dfs(u, 0, visited_mask)
        return B

    B = compute_blocked_matrix(adj_list)

    i_idx, j_idx = np.triu_indices(n, k=1)

    def get_valid_non_edges():
        valid_mask = (A[i_idx, j_idx] == 0) & (~B[i_idx, j_idx])
        valid_indices = np.where(valid_mask)[0]
        return [(int(i_idx[k]), int(j_idx[k])) for k in valid_indices]

    def get_all_edges():
        edge_mask = A[i_idx, j_idx] == 1
        edge_indices = np.where(edge_mask)[0]
        return [(int(i_idx[k]), int(j_idx[k])) for k in edge_indices]

    # Spectral heuristic: try to find a good starting point via random regular graph
    def spectral_initialization():
        """Create a random graph with average degree ~5.25 and check C6-freeness."""
        for attempt in range(50):
            # Random graph with p ~ 5.25/39
            p = 5.25 / (n - 1)
            A_try = np.zeros((n, n), dtype=np.int64)
            adj_try = [[] for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    if np.random.random() < p:
                        A_try[i, j] = 1
                        A_try[j, i] = 1
                        adj_try[i].append(j)
                        adj_try[j].append(i)
            
            m_try = int(np.sum(A_try) // 2)
            if m_try < 80:
                continue
            
            # Check C6-freeness
            B_try = compute_blocked_matrix(adj_try)
            # Check if any edge exists that would create a C6 (shouldn't happen in initial construction)
            # Actually we need to verify the graph has no C6
            # The blocked matrix tells us if there's a path of length 5
            # A C6 exists if there's an edge (u,v) where B[u,v] is True
            has_c6 = False
            for i in range(n):
                for j in range(i + 1, n):
                    if A_try[i, j] == 1 and B_try[i, j]:
                        has_c6 = True
                        break
                if has_c6:
                    break
            
            if not has_c6 and m_try > 85:
                return A_try, adj_try, m_try
        
        return None, None, 0

    # Try spectral initialization
    A_spec, adj_spec, m_spec = spectral_initialization()
    if A_spec is not None and m_spec > m:
        A = A_spec.copy()
        adj_list = [lst.copy() for lst in adj_spec]
        m = m_spec
        best_m = m
        best_A = A.copy()
        best_adj_list = [lst.copy() for lst in adj_list]
        B = compute_blocked_matrix(adj_list)

    steps_since_improvement = 0
    max_steps_without_improvement = 1000
    kick_count = 0

    while time.time() < deadline:
        valid_edges = get_valid_non_edges()

        if not valid_edges:
            # Stuck: No edges can be added
            # Perform a kick: remove an edge
            current_edges = get_all_edges()
            if not current_edges:
                break

            # Try to find an edge whose removal unblocks the most edges
            # Heuristic: for each edge, compute how many non-edges become unblocked
            # This is expensive, so sample a few edges
            sample_size = min(10, len(current_edges))
            sample_indices = np.random.choice(len(current_edges), size=sample_size, replace=False)

            best_edge_to_remove = None
            best_unblock_count = -1

            for idx in sample_indices:
                u, v = current_edges[idx]
                
                # Temporarily remove edge
                A[u, v] = 0
                A[v, u] = 0
                adj_list[u].remove(v)
                adj_list[v].remove(u)
                
                # Compute blocked matrix
                B_temp = compute_blocked_matrix(adj_list)
                
                # Count how many non-edges are now unblocked
                valid_mask_temp = (A[i_idx, j_idx] == 0) & (~B_temp[i_idx, j_idx])
                unblock_count = int(np.sum(valid_mask_temp))
                
                # Restore edge
                A[u, v] = 1
                A[v, u] = 1
                adj_list[u].append(v)
                adj_list[v].append(u)
                
                if unblock_count > best_unblock_count:
                    best_unblock_count = unblock_count
                    best_edge_to_remove = (u, v)

            if best_edge_to_remove is None:
                idx = np.random.randint(len(current_edges))
                best_edge_to_remove = current_edges[idx]

            u, v = best_edge_to_remove
            remove_edge(u, v)
            m -= 1
            B = compute_blocked_matrix(adj_list)
            kick_count += 1

            if m < best_m - 5:
                # Revert
                A = best_A.copy()
                adj_list = [lst.copy() for lst in best_adj_list]
                m = best_m
                B = compute_blocked_matrix(adj_list)
                steps_since_improvement = 0
                kick_count = 0
            else:
                steps_since_improvement += 1
                if steps_since_improvement > max_steps_without_improvement:
                    A = best_A.copy()
                    adj_list = [lst.copy() for lst in best_adj_list]
                    m = best_m
                    B = compute_blocked_matrix(adj_list)
                    steps_since_improvement = 0
                    kick_count = 0
            continue

        # Add a valid edge
        # Use a heuristic: prefer edges that connect vertices with higher degree
        # This tends to create denser local structures that are less likely to form long cycles
        valid_with_scores = []
        for u, v in valid_edges:
            score = len(adj_list[u]) + len(adj_list[v])
            valid_with_scores.append((score, u, v))
        
        valid_with_scores.sort(reverse=True)
        
        # Pick from top candidates with some randomness
        top_k = min(10, len(valid_with_scores))
        if top_k > 0:
            idx = np.random.randint(top_k)
            _, u, v = valid_with_scores[idx]
        else:
            u, v = valid_edges[0]

        add_edge(u, v)
        m += 1
        B = compute_blocked_matrix(adj_list)
        kick_count = 0

        if m > best_m:
            best_m = m
            best_A = A.copy()
            best_adj_list = [lst.copy() for lst in adj_list]
            steps_since_improvement = 0
        else:
            steps_since_improvement += 1
            if steps_since_improvement > max_steps_without_improvement:
                A = best_A.copy()
                adj_list = [lst.copy() for lst in best_adj_list]
                m = best_m
                B = compute_blocked_matrix(adj_list)
                steps_since_improvement = 0
                kick_count = 0

    # Convert best_A to list of edges
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_A[u, v] == 1:
                result.append([u, v])

    return result
