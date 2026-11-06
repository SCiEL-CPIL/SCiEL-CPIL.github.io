import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin for 150s limit
    
    n = 16
    # Generate all edges
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)  # 120
    
    # Map (u, v) to edge index
    edge_index_map = {}
    for idx, (u, v) in enumerate(edges):
        edge_index_map[(u, v)] = idx
        edge_index_map[(v, u)] = idx
        
    # Precompute K4s and their edge indices
    k4s = []
    k4_edge_indices = []
    for quad in itertools.combinations(range(n), 4):
        k4s.append(quad)
        idxs = []
        for u, v in itertools.combinations(quad, 2):
            idxs.append(edge_index_map[(u, v)])
        k4_edge_indices.append(idxs)
        
    num_k4s = len(k4s)  # 1820
    
    # Precompute edge_to_k4s for fast updates
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k_idx, edge_idxs in enumerate(k4_edge_indices):
        for e_idx in edge_idxs:
            edge_to_k4s[e_idx].append(k_idx)

    # Helper to check if a K4 is valid (distinct colors >= 5)
    def is_valid_k4(colors, edge_idxs):
        c = [colors[e] for e in edge_idxs]
        # Valid if distinct colors >= 5
        # Invalid if:
        # - Any color appears >= 3 times (max distinct = 4)
        # - Or two colors appear >= 2 times each (max distinct = 4)
        # Simplest check: len(set(c)) < 5
        return len(set(c)) >= 5

    # Objective: Minimize number of colors.
    # Constraint: All K4s valid.
    
    # Strategy:
    # 1. Try to find a valid coloring with K colors for K in [13, 14, 15].
    # 2. Use random restarts and local search (hill climbing) to minimize violations, then minimize colors.
    #    Actually, better: Fix K, random init, local search to satisfy constraints.
    #    If satisfied, return.
    #    If not, try to improve by reducing colors? No, start with small K.
    #    
    #    Alternative: Start with a valid coloring (e.g. seed), then try to reduce colors.
    #    This is safer because we always have a valid solution.
    
    # Let's stick to the "Start Valid, Reduce Colors" approach as it guarantees a valid output.
    # The current program does this but is slow/weak.
    # Inspiration 2 does "Start Random, Fix Violations" which can find lower K faster if K is small enough.
    
    # Let's try a hybrid:
    # Phase 1: Try to find a solution with K=13, 14, 15 using random restarts + local search to fix violations.
    # Phase 2: If Phase 1 fails or time runs out, fall back to seed (valid, 17 colors) or best found.
    
    # Local Search for fixed K:
    # Objective: Minimize number of violated K4s.
    # Move: Change color of a random edge in a violated K4 to a random color in [0, K).
    
    def count_violations(colors):
        count = 0
        for edge_idxs in k4_edge_indices:
            if not is_valid_k4(colors, edge_idxs):
                count += 1
        return count

    def local_search_fix(colors, K, max_steps):
        # colors is a list or array of length 120
        # K is the number of colors allowed (0..K-1)
        # Try to reduce violations to 0
        
        # Track violations per K4? Too expensive to update fully.
        # Just sample violations.
        
        current_colors = colors[:]
        
        # Initial violation count
        # For speed, we can just iterate until no violations found in a sample or max steps.
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
                
            # Find a violation
            # Sample random K4s
            found_violation = False
            # Check all if violations are low? No, sampling is faster.
            # But if we miss a violation, we might think we're done.
            # Let's do a full check every N steps or at the end?
            # Full check is 1820 * 6 ops ~ 11k ops. ~1ms.
            # 110s / 1ms = 110k full checks.
            
            # Let's just do a full check at the start of each "phase" or periodically.
            # For the inner loop, sample 100 K4s.
            
            sample_size = min(100, num_k4s)
            sample_indices = np.random.choice(num_k4s, sample_size, replace=False)
            
            viol_idx = -1
            for idx in sample_indices:
                if not is_valid_k4(current_colors, k4_edge_indices[idx]):
                    viol_idx = idx
                    break
            
            if viol_idx == -1:
                # No violation in sample. Verify fully.
                if count_violations(current_colors) == 0:
                    return current_colors, True
                # If there is a violation, we missed it. Try again.
                continue
            
            # Change a random edge in the violated K4
            edge_idxs = k4_edge_indices[viol_idx]
            e_idx = random.choice(edge_idxs)
            current_colors[e_idx] = random.randint(0, K - 1)
            
        # Final check
        if count_violations(current_colors) == 0:
            return current_colors, True
        return current_colors, False

    best_solution = None
    best_num_colors = float('inf')
    
    # Try K = 13, 14, 15
    for K in [13, 14, 15]:
        if time.time() > deadline - 20:
            break
            
        # Number of restarts depends on time left
        # Let's do a fixed number, say 20 restarts per K
        max_restarts = 20
        steps_per_restart = 5000
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
                
            # Random init
            colors = np.random.randint(0, K, num_edges).tolist()
            
            # Local search
            result, success = local_search_fix(colors, K, steps_per_restart)
            
            if success:
                # Found a valid solution with K colors
                return result  # Return immediately as we want min colors
        
    # If we didn't find 13-15, try to improve the seed solution?
    # Seed uses 17 colors.
    # Let's try to reduce colors from a valid seed-like solution.
    # Or just return the seed if we haven't found anything better.
    
    # Let's try a quick local search to reduce colors from the seed.
    # Seed: (i + j) % 17
    seed_colors = [(i + j) % 17 for i, j in edges]
    
    # Try to map colors to a smaller palette?
    # This is harder.
    
    # Let's try one more thing: Start with K=16, 17 and try to reduce?
    # No, the "Fix Violations" approach is better for finding small K.
    
    # If all else fails, return seed.
    return seed_colors
