import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    target_n = 22
    
    # Precompute all subsets of size >= 6
    # This might be memory heavy or slow to generate if done every time, 
    # but for n=22, binom(22,6) is ~75k, binom(22,7) is ~170k. Total ~300k subsets.
    # Storing masks is fine.
    
    subsets_by_size = {}
    for size in range(6, target_n + 1):
        subsets = []
        for comb in itertools.combinations(range(target_n), size):
            mask = 0
            for v in comb:
                mask |= (1 << v)
            subsets.append(mask)
        subsets_by_size[size] = subsets
        
    # Local Search
    best_graph = None
    best_violations = float('inf')
    
    # Try multiple restarts
    rng = random.Random(42)
    
    # We will try to find a valid graph.
    # Start with random graphs.
    
    for restart in range(1000):
        if time.time() > deadline - 10:
            break
            
        # Initialize random graph
        adj = [0] * target_n
        for i in range(target_n):
            for j in range(i + 1, target_n):
                if rng.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
        # Local search
        current_violations = _count_violations(adj, target_n, subsets_by_size)
        
        if current_violations == 0:
            return _graph_to_dict(adj, target_n)
            
        # Simulated Annealing parameters
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        
        steps = 0
        max_steps = 5000 # Per restart
        
        while steps < max_steps and T > min_T:
            if time.time() > deadline - 5:
                break
                
            # Pick a random edge to flip
            u = rng.randint(0, target_n - 1)
            v = rng.randint(0, target_n - 1)
            if u == v:
                continue
                
            # Flip edge
            if (adj[u] >> v) & 1:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
            new_violations = _count_violations(adj, target_n, subsets_by_size)
            delta = new_violations - current_violations
            
            if delta <= 0 or rng.random() < pow(2.71828, -delta / T):
                current_violations = new_violations
                if current_violations == 0:
                    return _graph_to_dict(adj, target_n)
            else:
                # Revert
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
            T *= cooling
            steps += 1
            
        # If we found a better graph, keep it? 
        # Actually, we just return immediately if valid.
        # If not, we continue to next restart.
        
    # If n=22 failed, try n=21
    if time.time() < deadline - 15:
        target_n = 21
        subsets_by_size = {}
        for size in range(6, target_n + 1):
            subsets = []
            for comb in itertools.combinations(range(target_n), size):
                mask = 0
                for v in comb:
                    mask |= (1 << v)
                subsets.append(mask)
            subsets_by_size[size] = subsets
            
        for restart in range(1000):
            if time.time() > deadline - 5:
                break
                
            adj = [0] * target_n
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if rng.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        
            current_violations = _count_violations(adj, target_n, subsets_by_size)
            
            if current_violations == 0:
                return _graph_to_dict(adj, target_n)
                
            T = 1.0
            cooling = 0.999
            min_T = 0.01
            steps = 0
            max_steps = 5000
            
            while steps < max_steps and T > min_T:
                if time.time() > deadline - 2:
                    break
                    
                u = rng.randint(0, target_n - 1)
                v = rng.randint(0, target_n - 1)
                if u == v:
                    continue
                    
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
                new_violations = _count_violations(adj, target_n, subsets_by_size)
                delta = new_violations - current_violations
                
                if delta <= 0 or rng.random() < pow(2.71828, -delta / T):
                    current_violations = new_violations
                    if current_violations == 0:
                        return _graph_to_dict(adj, target_n)
                else:
                    if (adj[u] >> v) & 1:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        
                T *= cooling
                steps += 1
                
    # Fallback to n=20
    if time.time() < deadline - 5:
        target_n = 20
        subsets_by_size = {}
        for size in range(6, target_n + 1):
            subsets = []
            for comb in itertools.combinations(range(target_n), size):
                mask = 0
                for v in comb:
                    mask |= (1 << v)
                subsets.append(mask)
            subsets_by_size[size] = subsets
            
        for restart in range(1000):
            if time.time() > deadline - 2:
                break
                
            adj = [0] * target_n
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if rng.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        
            current_violations = _count_violations(adj, target_n, subsets_by_size)
            
            if current_violations == 0:
                return _graph_to_dict(adj, target_n)
                
            T = 1.0
            cooling = 0.999
            min_T = 0.01
            steps = 0
            max_steps = 5000
            
            while steps < max_steps and T > min_T:
                if time.time() > deadline - 1:
                    break
                    
                u = rng.randint(0, target_n - 1)
                v = rng.randint(0, target_n - 1)
                if u == v:
                    continue
                    
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
                new_violations = _count_violations(adj, target_n, subsets_by_size)
                delta = new_violations - current_violations
                
                if delta <= 0 or rng.random() < pow(2.71828, -delta / T):
                    current_violations = new_violations
                    if current_violations == 0:
                        return _graph_to_dict(adj, target_n)
                else:
                    if (adj[u] >> v) & 1:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        
                T *= cooling
                steps += 1
                
    # Last resort: return a small valid graph
    return {'n': 5, 'edges': []}

def _count_violations(adj, n, subsets_by_size):
    count = 0
    for size in range(6, n + 1):
        for mask in subsets_by_size[size]:
            vertices = [i for i in range(n) if (mask >> i) & 1]
            degrees = []
            for v in vertices:
                deg = bin(adj[v] & mask).count('1')
                degrees.append(deg)
            if all(d == degrees[0] for d in degrees):
                count += 1
    return count

def _graph_to_dict(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                edges.append([i, j])
    return {'n': n, 'edges': edges}

