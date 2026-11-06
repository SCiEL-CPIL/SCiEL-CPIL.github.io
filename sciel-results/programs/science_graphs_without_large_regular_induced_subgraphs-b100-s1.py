import random
import itertools
import time

def construct():
    # Target max n is 22
    max_n = 22
    
    def is_valid_graph(adj, n):
        """
        Check if the graph defined by adj (list of int bitmasks) on n vertices
        has no regular induced subgraph on k >= 6 vertices.
        """
        # Precompute popcount for all masks up to 2^n? No, n is small, use bin(x).count('1')
        # Or use int.bit_count() in Python 3.8+
        
        # We need to check all subsets S with size k >= 6
        # It is more efficient to iterate k from 6 to n
        for k in range(6, n + 1):
            # Iterate all subsets of size k
            # Using itertools.combinations is slow for n=20, k=6 (74k combos) * 20 checks?
            # Better to use bitmasks.
            # Generate all masks with popcount k
            # For n=20, total masks is 2^20 = 1M. 
            # We can just iterate all masks and filter by popcount?
            # 1M masks * 20 checks is 20M ops. In Python, this might be slow (seconds).
            # Let's optimize the check.
            
            # Actually, we can just check all masks 0..2^n - 1
            # But we only care about masks with popcount >= 6.
            # Let's iterate k from 6 to n.
            # Generating combinations of size k from n items.
            # For n=20, k=6: 74613
            # k=7: 77520
            # k=8: 125970
            # ...
            # Sum is approx 2^20 / 2 = 500k subsets.
            # For each subset, we check regularity.
            # Regularity check: for each v in S, deg(v in S) must be same.
            
            # To speed up, we can break early if found.
            
            # Let's use a helper to check a specific mask
            def check_mask(mask):
                # Get list of vertices in mask
                # This is expensive if done naively.
                # Alternative: 
                # For a fixed mask, we need degrees of all vertices in mask.
                # deg_v = (adj[v] & mask).bit_count()
                # All deg_v must be equal.
                
                # Optimization: 
                # If the mask is regular, all vertices have same degree d.
                # Sum of degrees = 2 * edges_in_mask.
                # Also, min_degree == max_degree.
                
                # Let's just compute degrees for vertices in the mask.
                # We need to iterate through bits set in mask.
                # 
                # Faster way:
                # We can precompute nothing, just loop.
                # 
                # Actually, for n=20, iterating 500k masks and for each iterating up to 20 vertices is 10M operations.
                # Python might take ~1s. With 120s limit, we can afford several attempts.
                
                verts = []
                m = mask
                while m:
                    lsb = m & -m
                    idx = (lsb.bit_length() - 1)
                    verts.append(idx)
                    m ^= lsb
                
                if not verts:
                    return True
                    
                d0 = (adj[verts[0]] & mask).bit_count()
                for v in verts[1:]:
                    dv = (adj[v] & mask).bit_count()
                    if dv != d0:
                        return False
                return True

            # Iterate combinations
            # Using itertools.combinations is standard and reasonably fast in C
            for combo in itertools.combinations(range(n), k):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                if check_mask(mask):
                    return False
        return True

    # Greedy construction
    # Start with n=1
    best_n = 1
    best_adj = [0]
    
    # Try to extend from 1 to 22
    # We will try multiple random restarts if stuck
    max_attempts_per_step = 200
    
    current_n = 1
    current_adj = [0]
    
    for target_n in range(2, max_n + 1):
        success = False
        for _ in range(max_attempts_per_step):
            # Try to add vertex current_n
            # Randomly connect to previous vertices
            # Probability of edge p=0.5 is standard
            new_adj = current_adj[:] + [0]
            for i in range(current_n):
                if random.random() < 0.5:
                    # Connect i and current_n
                    new_adj[i] |= (1 << current_n)
                    new_adj[current_n] |= (1 << i)
            
            if is_valid_graph(new_adj, target_n):
                current_adj = new_adj
                current_n = target_n
                success = True
                break
        
        if not success:
            # If we fail to extend, maybe we can try a different random seed or restart from scratch?
            # Or just accept current_n as max.
            # Given the time limit, we might try a few full restarts.
            break
            
    return {'n': current_n, 'edges': [[i, j] for i in range(current_n) for j in range(i+1, current_n) if (current_adj[i] >> j) & 1]}

# To improve performance and reliability, we can run multiple independent attempts 
# and keep the best result found within the time limit.
# However, the function must return a single result.
# We can structure construct() to run a loop of attempts, each trying to build a graph, 
# and keep the one with max n.

def construct():
    max_n = 22
    best_result = {'n': 1, 'edges': []}
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    def is_valid_graph(adj, n):
        # Check all subsets of size >= 6
        # Optimization: If a graph is valid, it has no regular induced subgraph of size k for any k>=6.
        # We check k=6, 7, ... n
        for k in range(6, n + 1):
            for combo in itertools.combinations(range(n), k):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                
                # Check if G[mask] is regular
                # Get first vertex
                lsb = mask & -mask
                v0 = lsb.bit_length() - 1
                d0 = (adj[v0] & mask).bit_count()
                
                valid_reg = True
                m = mask
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    if v == v0:
                        m ^= lsb
                        continue
                    dv = (adj[v] & mask).bit_count()
                    if dv != d0:
                        valid_reg = False
                        break
                    m ^= lsb
                
                if valid_reg:
                    return False
        return True

    attempts = 0
    while time.time() - start_time < time_limit:
        attempts += 1
        current_n = 1
        current_adj = [0]
        
        for target_n in range(2, max_n + 1):
            if time.time() - start_time > time_limit:
                break
            
            success = False
            # Try random edges for new vertex
            for _ in range(50): # Fewer tries per step to allow more restarts
                if time.time() - start_time > time_limit:
                    break
                new_adj = current_adj[:] + [0]
                for i in range(current_n):
                    if random.random() < 0.5:
                        new_adj[i] |= (1 << current_n)
                        new_adj[current_n] |= (1 << i)
                
                if is_valid_graph(new_adj, target_n):
                    current_adj = new_adj
                    current_n = target_n
                    success = True
                    break
            
            if not success:
                break
        
        if current_n > best_result['n']:
            best_result = {'n': current_n, 'edges': [[i, j] for i in range(current_n) for j in range(i+1, current_n) if (current_adj[i] >> j) & 1]}
            
        # If we reached max_n, we can stop early if we found a valid one
        if best_result['n'] == max_n:
            break

    return best_result
