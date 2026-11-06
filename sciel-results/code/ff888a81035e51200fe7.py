import itertools
import random
import time

def construct():
    start_time = time.time()
    time_limit = 140  # Leave 10s margin before 150s kill

    # Hardcoded known valid 20-vertex graph
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

    def build_adj_masks(n, edges):
        adj_masks = [0] * n
        for u, v in edges:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
        return adj_masks

    def get_edges_from_masks(n, adj_masks):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_masks[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def is_regular_subset_fast(subset, adj_masks):
        """Check if the given subset induces a regular subgraph using bitmask operations."""
        mask = 0
        for v in subset:
            mask |= (1 << v)
        first_deg = (adj_masks[subset[0]] & mask).bit_count()
        for v in subset[1:]:
            deg = (adj_masks[v] & mask).bit_count()
            if deg != first_deg:
                return False
        return True

    def count_violations_size_6(n, adj_masks):
        """Count the number of 6-subsets that induce regular subgraphs."""
        count = 0
        for subset in itertools.combinations(range(n), 6):
            if is_regular_subset_fast(subset, adj_masks):
                count += 1
        return count

    def find_violating_6_subsets(n, adj_masks):
        """Find all 6-subsets that induce regular subgraphs."""
        violations = []
        for subset in itertools.combinations(range(n), 6):
            if is_regular_subset_fast(subset, adj_masks):
                violations.append(subset)
        return violations

    def is_valid_full(n, adj_masks):
        """Check if no subset of size >= 6 induces a regular subgraph."""
        # Check size 6 first
        if find_violating_6_subsets(n, adj_masks):
            return False
        
        # Check larger sizes
        for k in range(7, n + 1):
            for subset in itertools.combinations(range(n), k):
                if is_regular_subset_fast(subset, adj_masks):
                    return False
        return True

    # Step 1: Verify 20-vertex graph
    adj_20 = build_adj_masks(20, known_20_edges)
    if not is_valid_full(20, adj_20):
        return {'n': 5, 'edges': []}

    # Step 2: Try to find a 21-vertex valid graph using constraint propagation approach
    n = 21
    deadline = start_time + time_limit - 5
    
    # Precompute all 6-subsets for n=21
    all_6_subsets = list(itertools.combinations(range(n), 6))
    
    # For each 6-subset, precompute which edges (pairs) are in it
    # This allows us to quickly determine which edges affect which subsets
    subset_edges = {}
    for subset in all_6_subsets:
        edges_in_subset = set()
        for i in range(6):
            for j in range(i + 1, 6):
                u, v = subset[i], subset[j]
                if u > v:
                    u, v = v, u
                edges_in_subset.add((u, v))
        subset_edges[subset] = edges_in_subset
    
    # Build reverse mapping: edge -> list of subsets containing it
    edge_to_subsets = {}
    for subset, edges in subset_edges.items():
        for edge in edges:
            if edge not in edge_to_subsets:
                edge_to_subsets[edge] = []
            edge_to_subsets[edge].append(subset)
    
    best_adj = None
    best_cost = float('inf')
    
    for restart in range(60):
        if time.time() > deadline:
            break
            
        # Initialize random graph
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.4:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        current_cost = count_violations_size_6(n, adj)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_adj = list(adj)
        
        if current_cost == 0:
            if is_valid_full(n, adj):
                edges = get_edges_from_masks(n, adj)
                return {'n': 21, 'edges': edges}
        
        # Constraint propagation: iteratively flip edges that maximize resolved conflicts
        improved = True
        max_iterations = 200
        iteration = 0
        
        while improved and time.time() < deadline and iteration < max_iterations:
            iteration += 1
            improved = False
            
            violations = find_violating_6_subsets(n, adj)
            
            if not violations:
                if is_valid_full(n, adj):
                    edges = get_edges_from_masks(n, adj)
                    return {'n': 21, 'edges': edges}
                break
            
            # For each edge, compute the net impact of flipping it
            # Net impact = (violations fixed) - (violations created)
            edge_impact = {}
            
            for edge in edge_to_subsets.keys():
                u, v = edge
                was_present = (adj[u] >> v) & 1
                
                # Count violations that contain this edge and are currently regular
                fixed = 0
                created = 0
                
                # Simulate flip
                if was_present:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                
                for subset in edge_to_subsets[edge]:
                    was_regular = is_regular_subset_fast(subset, adj) if not was_present else None
                    is_regular_now = is_regular_subset_fast(subset, adj)
                    
                    if was_present:
                        # We removed the edge
                        # If it's now regular, we created a violation
                        # If it was regular before, we fixed it
                        # We need to check the state before the flip
                        pass
                    
                    # Actually, let's do this more carefully
                    # We need to know if the subset was regular BEFORE the flip
                    # and if it's regular AFTER the flip
                
                # Restore
                if was_present:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
            
            # Better approach: for each edge, compute impact by checking all affected subsets
            # before and after the flip
            edge_impact = {}
            for edge in edge_to_subsets.keys():
                u, v = edge
                was_present = (adj[u] >> v) & 1
                
                # Save state
                save_u = adj[u]
                save_v = adj[v]
                
                # Apply flip
                if was_present:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                
                fixed = 0
                created = 0
                
                for subset in edge_to_subsets[edge]:
                    # Check if regular after flip
                    is_regular_after = is_regular_subset_fast(subset, adj)
                    
                    # Check if regular before flip (need to temporarily restore)
                    adj[u] = save_u
                    adj[v] = save_v
                    is_regular_before = is_regular_subset_fast(subset, adj)
                    
                    # Restore flipped state
                    if was_present:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                    
                    if is_regular_before and not is_regular_after:
                        fixed += 1
                    elif not is_regular_before and is_regular_after:
                        created += 1
                
                # Restore original state
                adj[u] = save_u
                adj[v] = save_v
                
                edge_impact[edge] = fixed - created
            
            # Find the edge with maximum positive impact
            best_edge = None
            best_impact = 0
            for edge, impact in edge_impact.items():
                if impact > best_impact:
                    best_impact = impact
                    best_edge = edge
            
            if best_edge is not None and best_impact > 0:
                u, v = best_edge
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                
                new_cost = count_violations_size_6(n, adj)
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_adj = list(adj)
                    
                    if new_cost == 0:
                        if is_valid_full(n, adj):
                            edges = get_edges_from_masks(n, adj)
                            return {'n': 21, 'edges': edges}
                        break
                else:
                    # Undo
                    if (adj[u] >> v) & 1:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            else:
                # No single edge flip helps; try random perturbation
                for _ in range(10):
                    if time.time() > deadline:
                        break
                    ru = random.randint(0, n - 1)
                    rv = random.randint(0, n - 1)
                    if ru == rv:
                        continue
                    if (adj[ru] >> rv) & 1:
                        adj[ru] &= ~(1 << rv)
                        adj[rv] &= ~(1 << ru)
                    else:
                        adj[ru] |= (1 << rv)
                        adj[rv] |= (1 << ru)
                    new_cost = count_violations_size_6(n, adj)
                    if new_cost < current_cost:
                        current_cost = new_cost
                        improved = True
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best_adj = list(adj)
                        if new_cost == 0:
                            if is_valid_full(n, adj):
                                edges = get_edges_from_masks(n, adj)
                                return {'n': 21, 'edges': edges}
                            break
                    else:
                        # Undo
                        if (adj[ru] >> rv) & 1:
                            adj[ru] &= ~(1 << rv)
                            adj[rv] &= ~(1 << ru)
                        else:
                            adj[ru] |= (1 << rv)
                            adj[rv] |= (1 << ru)
        
        # If we made progress, continue with improved graph
        if best_cost < float('inf') and best_adj is not None:
            if best_cost == 0:
                if is_valid_full(n, best_adj):
                    edges = get_edges_from_masks(n, best_adj)
                    return {'n': 21, 'edges': edges}
    
    # If we found a graph with 0 size-6 violations, check if it's fully valid
    if best_cost == 0 and best_adj is not None:
        if is_valid_full(n, best_adj):
            edges = get_edges_from_masks(n, best_adj)
            return {'n': 21, 'edges': edges}

    # Step 3: Return the known 20-vertex graph
    return {'n': 20, 'edges': known_20_edges}
