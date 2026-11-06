import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    # 1. Start with Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n_seed = q
    adj_seed = [0] * n_seed
    for i in range(n_seed):
        for j in range(i + 1, n_seed):
            if (j - i) % q in residues:
                adj_seed[i] |= (1 << j)
                adj_seed[j] |= (1 << i)
                
    # Helper to check if a specific 5-set is a violation
    def check_violation(adj, n, s):
        # s is a tuple of 5 vertices
        is_clique = True
        is_ind = True
        for i in s:
            mask = 0
            for j in s:
                if i != j:
                    mask |= (1 << j)
            if (adj[i] & mask) != mask:
                is_clique = False
            if (adj[i] & mask) != 0:
                is_ind = False
        if is_clique:
            return 'K5'
        if is_ind:
            return 'Ind5'
        return None

    # Find one violation by random sampling
    def find_one_violation(adj, n, max_samples=1000):
        # First, check a few random samples
        for _ in range(max_samples):
            s = tuple(random.sample(range(n), 5))
            v = check_violation(adj, n, s)
            if v:
                return s, v
        # If not found in random samples, do a systematic check? 
        # For n=43, systematic is 1M checks. 
        # If random fails, it's likely valid or very sparse violations.
        # Let's do a systematic check but break early.
        # Actually, if we can't find it in 1000 random samples, it might be valid.
        # But to be safe, we should verify.
        # Verification is expensive. 
        # Strategy: If random search fails, assume valid for now? No, must be sure.
        # Let's just return None and let the caller decide (maybe return as valid).
        # But if it's invalid, we fail the task.
        # Better: If random search fails, do a full check? 
        # Full check for n=43 is ~1M * 25 ops = 25M ops. In Python, ~1-2s.
        # If we do this every step, it's too slow.
        # So we only do full check if we think we are close to valid.
        
        # For now, return None (meaning "no violation found in sample")
        return None, None

    def is_fully_valid(adj, n):
        # Exhaustive check
        for s in itertools.combinations(range(n), 5):
            v = check_violation(adj, n, s)
            if v:
                return False
        return True

    def try_find_valid_graph(n_target, initial_adj=None):
        if initial_adj is None:
            adj = list(adj_seed)
            if n_target > 37:
                # Add new vertices
                for i in range(37, n_target):
                    adj.append(0)
                    for j in range(37):
                        if random.random() < 0.5:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
                    for j in range(37, i):
                        if random.random() < 0.5:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
            else:
                adj = adj[:n_target]
                for i in range(n_target):
                    adj[i] &= (1 << n_target) - 1
        else:
            adj = list(initial_adj)
            
        # Symmetrize
        for i in range(n_target):
            adj[i] &= ~(1 << i)
            for j in range(i+1, n_target):
                if (adj[i] & (1 << j)) != (adj[j] & (1 << i)):
                    if adj[i] & (1 << j):
                        adj[j] |= (1 << i)
                    else:
                        adj[i] |= (1 << j)
                        adj[j] &= ~(1 << i) # Wait, if i has j and j doesn't have i, we want to make them equal. 
                        # Let's just force symmetry by averaging or random choice.
                        # Simpler: 
                        # if adj[i] has j, ensure adj[j] has i.
                        # if adj[j] has i, ensure adj[i] has j.
                        # This might create conflicts. 
                        # Best: 
                        # if (adj[i] & (1<<j)) and not (adj[j] & (1<<i)):
                        #   adj[j] |= (1<<i)
                        # elif (adj[j] & (1<<i)) and not (adj[i] & (1<<j)):
                        #   adj[i] |= (1<<j)
                        # This ensures if either has it, both have it. 
                        # This is "OR" symmetry.
                        
        # Re-do symmetry correctly
        for i in range(n_target):
            for j in range(i+1, n_target):
                if (adj[i] & (1 << j)) and not (adj[j] & (1 << i)):
                    adj[j] |= (1 << i)
                elif (adj[j] & (1 << i)) and not (adj[i] & (1 << j)):
                    adj[i] |= (1 << j)
                    
        # Check if valid
        if is_fully_valid(adj, n_target):
            return adj

        # Local Search
        max_steps = 5000
        for step in range(max_steps):
            if time.time() > deadline:
                return None
                
            v_set, v_type = find_one_violation(adj, n_target, max_samples=200)
            if v_set is None:
                # No violation found in sample, verify fully
                if is_fully_valid(adj, n_target):
                    return adj
                else:
                    # Found a violation in full check but not in sample? 
                    # This means violations are rare. 
                    # We need to find it. 
                    # For now, just break and try another restart? 
                    # Or do a full search for one violation.
                    for s in itertools.combinations(range(n_target), 5):
                        v = check_violation(adj, n_target, s)
                        if v:
                            v_set = s
                            v_type = v
                            break
                    if v_set is None:
                        return adj # Should be valid
                
            # Flip an edge in v_set
            u, v = random.sample(v_set, 2)
            if (adj[u] & (1 << v)):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
        return None

    # Try sizes from 43 down to 38
    for n in range(43, 37, -1):
        for restart in range(50):
            if time.time() > deadline:
                break
            random.seed(12345 + restart * n + int(time.time()*1000))
            result = try_find_valid_graph(n)
            if result is not None:
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if (result[i] & (1 << j)):
                            edges.append((i, j))
                return n, edges
    
    # Fallback
    edges_seed = []
    for i in range(37):
        for j in range(i + 1, 37):
            if (adj_seed[i] & (1 << j)):
                edges_seed.append((i, j))
    return 37, edges_seed
