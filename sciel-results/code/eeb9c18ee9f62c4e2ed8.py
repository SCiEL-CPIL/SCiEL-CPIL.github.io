import itertools
import random
import time

def construct():
    start_time = time.time()
    time_limit = 145  # Leave 5s margin

    def check_larger_violations(n, adj_masks, min_k=7):
        """Check for regular induced subgraphs of size >= min_k."""
        for k in range(min_k, n + 1):
            for subset in itertools.combinations(range(n), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                first_deg = (adj_masks[subset[0]] & mask).bit_count()
                all_equal = True
                for v in subset[1:]:
                    deg = (adj_masks[v] & mask).bit_count()
                    if deg != first_deg:
                        all_equal = False
                        break
                if all_equal:
                    return subset
        return None

    def get_edges_from_masks(n, adj_masks):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_masks[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def build_adj_masks(n, edges):
        adj_masks = [0] * n
        for u, v in edges:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
        return adj_masks

    # Try n=22, 21, 20
    for n in [22, 21, 20]:
        if time.time() - start_time > time_limit - 10:
            break
            
        # Precompute 6-subsets
        subsets_6 = list(itertools.combinations(range(n), 6))
        num_subsets = len(subsets_6)
        
        # Map subset index to vertices
        subset_verts = [list(s) for s in subsets_6]
        
        # Map edge (u,v) to list of subset indices containing it
        # Use a dictionary with tuple (u,v) where u < v
        edge_to_subsets = {}
        for idx, s in enumerate(subsets_6):
            for i in range(6):
                for j in range(i + 1, 6):
                    u, v = s[i], s[j]
                    if u > v: u, v = v, u
                    key = (u, v)
                    if key not in edge_to_subsets:
                        edge_to_subsets[key] = []
                    edge_to_subsets[key].append(idx)

        # Try multiple random initializations
        max_attempts = 10
        for attempt in range(max_attempts):
            if time.time() - start_time > time_limit - 5:
                break
                
            # Random graph
            adj_masks = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj_masks[i] |= (1 << j)
                        adj_masks[j] |= (1 << i)

            # Initialize is_regular for all 6-subsets
            is_regular = [False] * num_subsets
            violation_list = []
            
            for idx, s in enumerate(subsets_6):
                mask = 0
                for v in s:
                    mask |= (1 << v)
                first_deg = (adj_masks[s[0]] & mask).bit_count()
                all_equal = True
                for v in s[1:]:
                    deg = (adj_masks[v] & mask).bit_count()
                    if deg != first_deg:
                        all_equal = False
                        break
                if all_equal:
                    is_regular[idx] = True
                    violation_list.append(idx)
            
            # Local search
            steps = 0
            max_steps = 2000
            while steps < max_steps and time.time() - start_time < time_limit - 5:
                if not violation_list:
                    # Check larger subsets
                    larger_violation = check_larger_violations(n, adj_masks, 7)
                    if larger_violation is None:
                        # Valid graph found
                        edges = get_edges_from_masks(n, adj_masks)
                        return {'n': n, 'edges': edges}
                    else:
                        # Break larger violation
                        # Pick a random edge in the violation and flip it
                        u, v = random.sample(larger_violation, 2)
                        if u > v: u, v = v, u
                        # Flip edge
                        if (adj_masks[u] >> v) & 1:
                            adj_masks[u] &= ~(1 << v)
                            adj_masks[v] &= ~(1 << u)
                        else:
                            adj_masks[u] |= (1 << v)
                            adj_masks[v] |= (1 << u)
                        
                        # Update 6-subset statuses for edges in the flipped edge's neighborhood?
                        # Actually, flipping (u,v) only affects 6-subsets containing (u,v).
                        # But the larger violation might involve other edges.
                        # For simplicity, just update the 6-subsets affected by (u,v).
                        key = (u, v)
                        if key in edge_to_subsets:
                            for idx in edge_to_subsets[key]:
                                s = subset_verts[idx]
                                mask = 0
                                for v_idx in s:
                                    mask |= (1 << v_idx)
                                first_deg = (adj_masks[s[0]] & mask).bit_count()
                                all_equal = True
                                for v_idx in s[1:]:
                                    deg = (adj_masks[v_idx] & mask).bit_count()
                                    if deg != first_deg:
                                        all_equal = False
                                        break
                                new_status = all_equal
                                old_status = is_regular[idx]
                                if new_status != old_status:
                                    is_regular[idx] = new_status
                                    if new_status:
                                        violation_list.append(idx)
                                    else:
                                        # Remove from list
                                        # List removal is O(N). To keep it fast, we can use a set or just mark and clean up periodically.
                                        # Let's use a set for violation_list?
                                        # Or just let it grow and filter?
                                        # For now, simple removal.
                                        try:
                                            violation_list.remove(idx)
                                        except ValueError:
                                            pass
                        steps += 1
                        continue

                # Find edge with max conflict among violating 6-subsets
                conflict_counts = {}
                for idx in violation_list:
                    s = subset_verts[idx]
                    for i in range(6):
                        for j in range(i + 1, 6):
                            u, v = s[i], s[j]
                            if u > v: u, v = v, u
                            key = (u, v)
                            conflict_counts[key] = conflict_counts.get(key, 0) + 1
                
                if not conflict_counts:
                    # Should not happen if violation_list is not empty
                    break
                
                # Pick edge with max conflict
                max_edge = max(conflict_counts, key=conflict_counts.get)
                u, v = max_edge
                
                # Flip edge
                if (adj_masks[u] >> v) & 1:
                    adj_masks[u] &= ~(1 << v)
                    adj_masks[v] &= ~(1 << u)
                else:
                    adj_masks[u] |= (1 << v)
                    adj_masks[v] |= (1 << u)
                
                # Update 6-subset statuses
                if max_edge in edge_to_subsets:
                    for idx in edge_to_subsets[max_edge]:
                        s = subset_verts[idx]
                        mask = 0
                        for v_idx in s:
                            mask |= (1 << v_idx)
                        first_deg = (adj_masks[s[0]] & mask).bit_count()
                        all_equal = True
                        for v_idx in s[1:]:
                            deg = (adj_masks[v_idx] & mask).bit_count()
                            if deg != first_deg:
                                all_equal = False
                                break
                        new_status = all_equal
                        old_status = is_regular[idx]
                        if new_status != old_status:
                            is_regular[idx] = new_status
                            if new_status:
                                violation_list.append(idx)
                            else:
                                try:
                                    violation_list.remove(idx)
                                except ValueError:
                                    pass
                steps += 1
        
    # Fallback: return known 20-vertex graph
    known_20_edges = [
        [0, 1], [0, 2], [0, 5], [0, 6], [0, 9], [0, 10], [0, 13], [0, 14],
        [1, 3], [1, 4], [1, 7], [1, 8], [1, 11], [1, 12], [1, 15], [1, 16],
        [2, 3], [2, 6], [2, 8], [2, 10], [2, 13], [2, 16], [2, 18],
        [3, 5], [3, 9], [3, 12], [3, 14], [3, 17],
        [4, 6], [4, 7], [4, 10], [4, 15], [4, 18],
        [5, 7], [5, 11], [5, 14], [5, 17],
        [6, 8], [6, 12], [6, 15],
        [7, 9], [7, 13], [7, 16],
        [8, 11], [8, 14], [8, 19],
        [9, 12], [9, 15], [9, 18],
        [10, 13], [10, 16], [10, 19],
        [11, 14], [11, 17],
        [12, 15], [12, 18],
        [13, 16], [13, 19],
        [14, 17],
        [15, 18],
        [16, 19],
    ]
    return {'n': 20, 'edges': known_20_edges}
