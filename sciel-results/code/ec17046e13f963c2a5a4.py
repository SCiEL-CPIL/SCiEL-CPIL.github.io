import time
import random
import numpy as np
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for evaluation overhead

    # Precompute combinations of size 6 for n up to 21
    # C(21, 6) = 54264, manageable
    # We'll generate these on the fly or cache them.

    def check_validity_fast(n, adj):
        """
        Checks if the graph has no regular induced subgraph of size >= 6.
        Uses bitwise operations for speed.
        """
        # Precompute masks for all 6-subsets
        # Actually, iterating combinations is fine if the check is fast.
        # Key optimization: bit_count is very fast.
        
        for k in range(6, n + 1):
            for comb in itertools.combinations(range(n), k):
                mask = 0
                for v in comb:
                    mask |= (1 << v)
                
                # Check if regular
                # For each v in comb, deg = bit_count(adj[v] & mask)
                # All degrees must be equal.
                first_deg = None
                is_reg = True
                for v in comb:
                    deg = (adj[v] & mask).bit_count()
                    if first_deg is None:
                        first_deg = deg
                    elif deg != first_deg:
                        is_reg = False
                        break
                
                if is_reg:
                    return False
        return True

    def count_bad_6_fast(n, adj):
        """
        Counts the number of 6-vertex subsets that induce a regular subgraph.
        """
        count = 0
        for comb in itertools.combinations(range(n), 6):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            
            # Check if regular
            first_deg = None
            is_reg = True
            for v in comb:
                deg = (adj[v] & mask).bit_count()
                if first_deg is None:
                    first_deg = deg
                elif deg != first_deg:
                    is_reg = False
                    break
            
            if is_reg:
                count += 1
        return count

    def find_violation_fast(n, adj, rng, num_trials=100):
        """
        Randomly samples 6-subsets to find a violation.
        Returns mask of violation or None.
        """
        for _ in range(num_trials):
            comb = rng.sample(range(n), 6)
            mask = 0
            for v in comb:
                mask |= (1 << v)
            
            first_deg = None
            is_reg = True
            for v in comb:
                deg = (adj[v] & mask).bit_count()
                if first_deg is None:
                    first_deg = deg
                elif deg != first_deg:
                    is_reg = False
                    break
            
            if is_reg:
                return mask
        return None

    def flip_edge(adj, u, v):
        if (adj[u] >> v) & 1:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

    def local_search_bitwise(n, max_iters, deadline):
        """
        Local search using bitwise representation.
        Starts with random graph, flips edges to reduce violations.
        """
        rng = random.Random()
        adj = [0] * n
        
        # Initialize random graph
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        current_score = count_bad_6_fast(n, adj)
        if current_score == 0:
            if check_validity_fast(n, adj):
                return adj

        best_adj = adj.copy()
        best_score = current_score
        
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random edge to flip
            u, v = rng.choice(edges)
            flip_edge(adj, u, v)
            
            new_score = count_bad_6_fast(n, adj)
            
            if new_score <= current_score:
                current_score = new_score
                if current_score == 0:
                    if check_validity_fast(n, adj):
                        return adj
                if current_score < best_score:
                    best_score = current_score
                    best_adj = adj.copy()
            else:
                # Revert
                flip_edge(adj, u, v)
                
        return None

    def greedy_add_bitwise(n, deadline):
        """
        Greedy vertex addition using bitwise representation.
        """
        rng = random.Random()
        adj = [0] * n
        
        # Start with 5 vertices
        for i in range(5):
            for j in range(i + 1, 5):
                if rng.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Add vertices 5 to n-1
        for new_v in range(5, n):
            if time.time() > deadline:
                break
            
            best_pattern = 0
            best_violations = float('inf')
            
            # Try random patterns
            # We want to minimize new violations involving new_v
            # A violation is a 6-set containing new_v and 5 others that is regular.
            # For a fixed pattern P (neighbors of new_v among 0..new_v-1),
            # and a set S of 5 vertices from 0..new_v-1,
            # S U {new_v} is regular if:
            # 1. deg_S(u) + (1 if u in P else 0) is constant for all u in S U {new_v}
            # 2. deg_new_v = |P ∩ S| must equal that constant.
            
            # Let's just evaluate a few random patterns and count violations.
            # Since we only add one vertex at a time, we can check violations
            # formed by new_v + 5 old vertices.
            
            n_trials = 20
            for _ in range(n_trials):
                pattern = 0
                for i in range(new_v):
                    if rng.random() < 0.5:
                        pattern |= (1 << i)
                
                # Count violations involving new_v
                violations = 0
                # Iterate 5-subsets of 0..new_v-1
                # This is expensive if new_v is large.
                # But we only do this for a few trials.
                # C(20, 5) = 15504. For 20 trials, ~300k checks.
                # Each check is O(5) bit counts. Fast enough.
                
                for five_set in itertools.combinations(range(new_v), 5):
                    mask = 0
                    for v in five_set:
                        mask |= (1 << v)
                    mask |= (1 << new_v)
                    
                    # Check regularity of S U {new_v}
                    # deg_new_v = popcount(pattern & mask)
                    deg_nv = (pattern & mask).bit_count()
                    
                    is_reg = True
                    for u in five_set:
                        # deg_u_in_S = popcount(adj[u] & mask) ... wait, mask includes new_v?
                        # adj[u] doesn't have bit for new_v yet.
                        # So deg_u_in_S = popcount(adj[u] & mask) is correct because adj[u] has 0 for new_v.
                        # But we need deg in S U {new_v}.
                        # deg_u = popcount(adj[u] & mask) + (1 if u in pattern else 0)
                        deg_u = (adj[u] & mask).bit_count()
                        if (pattern >> u) & 1:
                            deg_u += 1
                        if deg_u != deg_nv:
                            is_reg = False
                            break
                    
                    if is_reg:
                        violations += 1
                
                if violations < best_violations:
                    best_violations = violations
                    best_pattern = pattern
                    if violations == 0:
                        break
            
            # Apply best pattern
            adj[new_v] = best_pattern
            for i in range(new_v):
                if (best_pattern >> i) & 1:
                    adj[i] |= (1 << new_v)
        
        # Final check
        if count_bad_6_fast(n, adj) == 0:
            if check_validity_fast(n, adj):
                return adj
        
        return None

    # Strategy 1: Try n=21 with bitwise local search
    # This is the most promising for improvement.
    t_start = time.time()
    target_n = 21
    budget = 40
    for i in range(5):
        if time.time() > t_start + budget:
            break
        adj = local_search_bitwise(target_n, max_iters=200, deadline=t_start + budget)
        if adj is not None:
            edges = []
            for u in range(target_n):
                for v in range(u + 1, target_n):
                    if (adj[u] >> v) & 1:
                        edges.append([u, v])
            return {'n': target_n, 'edges': edges}

    # Strategy 2: Try n=21 with greedy addition
    t_start = time.time()
    target_n = 21
    budget = 30
    for i in range(3):
        if time.time() > t_start + budget:
            break
        adj = greedy_add_bitwise(target_n, deadline=t_start + budget)
        if adj is not None:
            edges = []
            for u in range(target_n):
                for v in range(u + 1, target_n):
                    if (adj[u] >> v) & 1:
                        edges.append([u, v])
            return {'n': target_n, 'edges': edges}

    # Strategy 3: Try n=20 with bitwise local search (should be fast and reliable)
    t_start = time.time()
    target_n = 20
    budget = 20
    for i in range(10):
        if time.time() > t_start + budget:
            break
        adj = local_search_bitwise(target_n, max_iters=500, deadline=t_start + budget)
        if adj is not None:
            edges = []
            for u in range(target_n):
                for v in range(u + 1, target_n):
                    if (adj[u] >> v) & 1:
                        edges.append([u, v])
            return {'n': target_n, 'edges': edges}

    # Strategy 4: Try n=20 with greedy addition
    t_start = time.time()
    target_n = 20
    budget = 10
    for i in range(5):
        if time.time() > t_start + budget:
            break
        adj = greedy_add_bitwise(target_n, deadline=t_start + budget)
        if adj is not None:
            edges = []
            for u in range(target_n):
                for v in range(u + 1, target_n):
                    if (adj[u] >> v) & 1:
                        edges.append([u, v])
            return {'n': target_n, 'edges': edges}

    # Fallback: n=5
    return {'n': 5, 'edges': []}
