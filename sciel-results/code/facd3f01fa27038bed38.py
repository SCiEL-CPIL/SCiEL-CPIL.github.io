import time
import random
import numpy as np

P = {'n': 12, 'm': 40}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (limit is 90s)
    
    n = P['n']
    m = P['m']
    total = 1 << n  # 4096
    
    # Precompute popcount for all masks
    popcount = np.zeros(total, dtype=np.int32)
    for i in range(1, total):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    # Build comparable matrix using numpy
    masks_arr = np.arange(total, dtype=np.uint32)
    comp = np.zeros((total, total), dtype=np.uint8)
    
    # Build in blocks to manage memory
    block_size = 256
    for i_start in range(0, total, block_size):
        i_end = min(i_start + block_size, total)
        block = masks_arr[i_start:i_end]
        a = (block[:, None] & masks_arr[None, :]) == block[:, None]
        b = (block[:, None] & masks_arr[None, :]) == masks_arr[None, :]
        diag = np.eye(i_end - i_start, total, dtype=bool)
        comp[i_start:i_end, :] = ((a | b) & ~diag).astype(np.uint8)
    
    # Precompute neighbor lists for faster updates
    neighbors = []
    for i in range(total):
        neighbors.append(np.where(comp[i] == 1)[0])
    
    def evaluate(x):
        """Compute number of comparable pairs in the selected set."""
        selected = np.where(x == 1)[0]
        if len(selected) < 2:
            return 0
        sub = comp[np.ix_(selected, selected)]
        return int(np.sum(np.triu(sub, k=1)))
    
    def greedy_init():
        """Greedy initialization: add sets one by one, choosing the one with max marginal gain."""
        x = np.zeros(total, dtype=np.uint8)
        degree = np.zeros(total, dtype=np.int32)
        
        for _ in range(m):
            candidates = np.where(x == 0)[0]
            if len(candidates) == 0:
                break
            cand_degrees = degree[candidates]
            best_idx = candidates[np.argmax(cand_degrees)]
            x[best_idx] = 1
            nbrs = neighbors[best_idx]
            degree[nbrs] += 1
        
        return x
    
    def random_init():
        """Random initialization."""
        x = np.zeros(total, dtype=np.uint8)
        chosen = np.random.choice(total, size=m, replace=False)
        x[chosen] = 1
        return x
    
    def small_cardinality_init():
        """Initialize with smallest cardinality sets."""
        order = np.argsort(popcount)
        x = np.zeros(total, dtype=np.uint8)
        x[order[:m]] = 1
        return x
    
    def chain_init():
        """Initialize with a chain structure: nested sets of increasing size."""
        x = np.zeros(total, dtype=np.uint8)
        # Create a chain: each set adds one element
        current = 0
        for i in range(m):
            x[current] = 1
            # Add a random new element
            available = [b for b in range(n) if not (current >> b) & 1]
            if available:
                current |= (1 << random.choice(available))
            else:
                break
        return x
    
    def simulated_annealing(x0, t_max=50000, t_min=0.01, alpha=0.99995):
        """Run simulated annealing starting from x0."""
        x = x0.copy()
        
        # Compute initial degree
        degree = np.zeros(total, dtype=np.int32)
        selected = np.where(x == 1)[0]
        for k in selected:
            nbrs = neighbors[k]
            degree[nbrs] += 1
        
        current_score = evaluate(x)
        best_x_local = x.copy()
        best_score_local = current_score
        
        t = t_max
        no_improve = 0
        
        while t > t_min and no_improve < 30000:
            # Check time
            if time.time() > deadline - 5:
                break
            
            # Pick a random swap: add one, remove one
            selected = np.where(x == 1)[0]
            unselected = np.where(x == 0)[0]
            
            if len(selected) == 0 or len(unselected) == 0:
                break
            
            k_add = unselected[np.random.randint(len(unselected))]
            k_remove = selected[np.random.randint(len(selected))]
            
            if k_add == k_remove:
                continue
            
            # Compute delta
            delta = int(degree[k_add]) - int(degree[k_remove])
            
            # Accept?
            if delta >= 0 or random.random() < np.exp(delta / t):
                # Apply swap
                x[k_remove] = 0
                x[k_add] = 1
                
                # Update degrees
                nbrs_remove = neighbors[k_remove]
                degree[nbrs_remove] -= 1
                nbrs_add = neighbors[k_add]
                degree[nbrs_add] += 1
                
                current_score += delta
                
                if current_score > best_score_local:
                    best_score_local = current_score
                    best_x_local = x.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            
            t *= alpha
        
        return best_x_local, best_score_local
    
    def hill_climbing(x0, max_iter=50000):
        """Pure hill climbing: only accept improvements."""
        x = x0.copy()
        
        # Compute initial degree
        degree = np.zeros(total, dtype=np.int32)
        selected = np.where(x == 1)[0]
        for k in selected:
            nbrs = neighbors[k]
            degree[nbrs] += 1
        
        current_score = evaluate(x)
        best_x_local = x.copy()
        best_score_local = current_score
        
        no_improve = 0
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            selected = np.where(x == 1)[0]
            unselected = np.where(x == 0)[0]
            
            if len(selected) == 0 or len(unselected) == 0:
                break
            
            k_add = unselected[np.random.randint(len(unselected))]
            k_remove = selected[np.random.randint(len(selected))]
            
            if k_add == k_remove:
                continue
            
            delta = int(degree[k_add]) - int(degree[k_remove])
            
            if delta > 0:
                x[k_remove] = 0
                x[k_add] = 1
                
                nbrs_remove = neighbors[k_remove]
                degree[nbrs_remove] -= 1
                nbrs_add = neighbors[k_add]
                degree[nbrs_add] += 1
                
                current_score += delta
                
                if current_score > best_score_local:
                    best_score_local = current_score
                    best_x_local = x.copy()
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
            
            if no_improve > 10000:
                break
        
        return best_x_local, best_score_local
    
    # Try multiple initializations
    inits = [
        ("greedy", greedy_init()),
        ("small_card", small_cardinality_init()),
        ("chain", chain_init()),
        ("random1", random_init()),
        ("random2", random_init()),
    ]
    
    best_x = None
    best_score = -1
    
    # Evaluate initial solutions
    for name, x0 in inits:
        if time.time() > deadline - 15:
            break
        score = evaluate(x0)
        if score > best_score:
            best_score = score
            best_x = x0.copy()
    
    # Run SA from each initialization
    for name, x0 in inits:
        if time.time() > deadline - 20:
            break
        x_best, score = simulated_annealing(x0, t_max=30000, t_min=0.01, alpha=0.99995)
        if score > best_score:
            best_score = score
            best_x = x_best
    
    # Run hill climbing from best solution
    if best_x is not None and time.time() < deadline - 15:
        x_best, score = hill_climbing(best_x, max_iter=30000)
        if score > best_score:
            best_score = score
            best_x = x_best
    
    # Additional restarts from best solution if time permits
    if best_x is not None and time.time() < deadline - 25:
        for restart in range(8):
            if time.time() > deadline - 10:
                break
            # Perturb best solution
            x_pert = best_x.copy()
            selected = np.where(x_pert == 1)[0]
            unselected = np.where(x_pert == 0)[0]
            # Swap 3 random pairs
            for _ in range(3):
                if len(selected) == 0 or len(unselected) == 0:
                    break
                k_r = selected[np.random.randint(len(selected))]
                k_a = unselected[np.random.randint(len(unselected))]
                x_pert[k_r] = 0
                x_pert[k_a] = 1
                selected = np.where(x_pert == 1)[0]
                unselected = np.where(x_pert == 0)[0]
            
            x_best, score = simulated_annealing(x_pert, t_max=20000, t_min=0.01, alpha=0.99995)
            if score > best_score:
                best_score = score
                best_x = x_best
            
            # Hill climb after SA
            if time.time() < deadline - 10:
                x_best2, score2 = hill_climbing(x_best, max_iter=10000)
                if score2 > best_score:
                    best_score = score2
                    best_x = x_best2
    
    # Final hill climbing polish
    if best_x is not None and time.time() < deadline - 5:
        x_best, score = hill_climbing(best_x, max_iter=20000)
        if score > best_score:
            best_score = score
            best_x = x_best
    
    # Fallback
    if best_x is None:
        best_x = greedy_init()
    
    selected = np.where(best_x == 1)[0]
    result = []
    for mask in selected:
        s = [b for b in range(n) if (mask >> b) & 1]
        result.append(s)
    
    # Ensure exactly m sets
    assert len(result) == m, f"Expected {m} sets, got {len(result)}"
    # Ensure distinct
    assert len(set(tuple(s) for s in result)) == m, "Sets are not distinct"
    # Ensure types are correct
    for s in result:
        assert isinstance(s, list), f"Set is not a list: {type(s)}"
        for v in s:
            assert isinstance(v, int) and not isinstance(v, bool), f"Value {v} is not a proper int"
    
    return result
