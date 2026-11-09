import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    def compute_penalty_vectorized(adj, n, max_k=8):
        """
        Compute total number of regular induced subgraphs of size 6..max_k.
        Uses vectorized numpy for k=6,7,8.
        Returns (total_penalty, list_of_violations_sample)
        """
        total = 0
        violations = []
        vertices = np.arange(n)
        
        for k in range(6, min(max_k, n) + 1):
            # Generate all combinations as index arrays
            # For n=21, k=6: C(21,6)=54264 combinations
            # For n=21, k=7: C(21,7)=116280
            # For n=21, k=8: C(21,8)=203490
            # Total ~374k subsets for k=6,7,8
            
            # We'll process in chunks to manage memory
            # Generate combinations using numpy
            from itertools import combinations as comb
            combos = list(comb(range(n), k))
            if not combos:
                continue
            
            # Convert to numpy array of shape (num_combos, k)
            idx = np.array(combos, dtype=np.int32)
            num_combos = len(idx)
            
            # Process in chunks of 5000
            chunk_size = 5000
            for start in range(0, num_combos, chunk_size):
                end = min(start + chunk_size, num_combos)
                idx_chunk = idx[start:end]  # shape (batch, k)
                
                # Extract sub-adjacency: adj[i,j] for all i,j in each combination
                # Shape: (batch, k, k)
                sub_adj = adj[idx_chunk][:, :, :]  # (batch, k, k)
                
                # Compute degrees: row sums of sub_adj
                degrees = sub_adj.sum(axis=2)  # (batch, k)
                
                # Check if all degrees in each row are equal
                min_deg = degrees.min(axis=1)  # (batch,)
                max_deg = degrees.max(axis=1)  # (batch,)
                
                # A combination is regular if min == max
                regular_mask = (min_deg == max_deg)
                count = regular_mask.sum()
                total += count
                
                if count > 0 and len(violations) < 10:
                    # Store some violations for guidance
                    for i in np.where(regular_mask)[0][:3]:
                        violations.append(idx[start + i].tolist())
        
        return total, violations

    def simulated_annealing(n, max_iters, deadline):
        """
        Simulated annealing to minimize the number of regular induced subgraphs of size >= 6.
        """
        # Initialize with random graph
        adj = np.zeros((n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i, j] = 1
                    adj[j, i] = 1
        
        # Initial penalty
        current_penalty, _ = compute_penalty_vectorized(adj, n, max_k=8)
        
        if current_penalty == 0:
            # Check all sizes up to n
            full_penalty, _ = compute_penalty_vectorized(adj, n, max_k=n)
            if full_penalty == 0:
                edges = [[i, j] for i in range(n) for j in range(i+1, n) if adj[i,j] == 1]
                return edges, True
        
        best_adj = adj.copy()
        best_penalty = current_penalty
        
        T = 1.0
        cooling = 0.9995
        min_T = 0.001
        
        iter_count = 0
        while time.time() < deadline and T > min_T:
            # Pick a random edge to flip
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            # Flip
            old_val = adj[u, v]
            adj[u, v] = 1 - old_val
            adj[v, u] = 1 - old_val
            
            # Compute new penalty (only check k=6,7,8 for speed during search)
            new_penalty, _ = compute_penalty_vectorized(adj, n, max_k=8)
            
            delta = new_penalty - current_penalty
            
            if delta <= 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
                # Accept
                current_penalty = new_penalty
                if new_penalty < best_penalty:
                    best_penalty = new_penalty
                    best_adj = adj.copy()
            else:
                # Reject: undo flip
                adj[u, v] = old_val
                adj[v, u] = old_val
            
            T *= cooling
            iter_count += 1
            
            # Periodic full check if penalty is low
            if current_penalty == 0 and iter_count % 50 == 0:
                full_penalty, _ = compute_penalty_vectorized(adj, n, max_k=n)
                if full_penalty == 0:
                    edges = [[i, j] for i in range(n) for j in range(i+1, n) if adj[i,j] == 1]
                    return edges, True
        
        # Final check with best_adj
        if best_penalty == 0:
            full_penalty, _ = compute_penalty_vectorized(best_adj, n, max_k=n)
            if full_penalty == 0:
                edges = [[i, j] for i in range(n) for j in range(i+1, n) if best_adj[i,j] == 1]
                return edges, True
        
        # Return best found (even if not fully valid, for fallback)
        edges = [[i, j] for i in range(n) for j in range(i+1, n) if best_adj[i,j] == 1]
        return edges, False

    def fast_validate(n, edges):
        """
        Fast validation using bitmask approach.
        Returns True if valid (no regular induced subgraph of size >= 6).
        """
        adj_masks = [0] * n
        for u, v in edges:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
        
        for k in range(6, n + 1):
            for subset in combinations(range(n), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                
                d0 = (adj_masks[subset[0]] & mask).bit_count()
                regular = True
                for v in subset[1:]:
                    d = (adj_masks[v] & mask).bit_count()
                    if d != d0:
                        regular = False
                        break
                if regular:
                    return False
        return True

    # Main strategy:
    # 1. Try simulated annealing for n=20 with multiple restarts
    # 2. If found, try n=21
    # 3. Fallback to smaller n
    
    best_result = {'n': 5, 'edges': []}
    
    # Try n=20 first with multiple SA restarts
    found_20 = False
    remaining_for_20 = deadline - time.time() - 15  # Reserve 15s for n=21 attempts
    
    for attempt in range(20):
        if time.time() > start_time + remaining_for_20:
            break
        
        random.seed(attempt * 7919 + 31)
        np.random.seed(attempt * 7919 + 31)
        
        edges, is_valid = simulated_annealing(20, max_iters=5000, deadline=start_time + remaining_for_20)
        
        if is_valid:
            # Double-check with fast_validate
            if fast_validate(20, edges):
                best_result = {'n': 20, 'edges': edges}
                found_20 = True
                break
    
    # Try n=21 if we have time
    if found_20 and time.time() < deadline - 5:
        remaining_for_21 = deadline - time.time() - 3
        for attempt in range(10):
            if time.time() > start_time + remaining_for_21:
                break
            
            random.seed(attempt * 104729 + 97)
            np.random.seed(attempt * 104729 + 97)
            
            edges, is_valid = simulated_annealing(21, max_iters=3000, deadline=start_time + remaining_for_21)
            
            if is_valid:
                if fast_validate(21, edges):
                    best_result = {'n': 21, 'edges': edges}
                    break
    
    # Fallback: try n=19 if we didn't find 20
    if not found_20 and time.time() < deadline - 3:
        for attempt in range(15):
            if time.time() > deadline - 2:
                break
            
            random.seed(attempt * 41 + 7)
            np.random.seed(attempt * 41 + 7)
            
            edges, is_valid = simulated_annealing(19, max_iters=3000, deadline=deadline - 2)
            
            if is_valid:
                if fast_validate(19, edges):
                    best_result = {'n': 19, 'edges': edges}
                    break
    
    return best_result
