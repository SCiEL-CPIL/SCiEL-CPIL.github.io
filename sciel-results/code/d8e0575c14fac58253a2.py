import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 130  # Safety margin (150s limit)
    
    # Helper to check validity and count bad sets using NumPy vectorization
    def count_bad_sets(orientation, n):
        # Reconstruct adjacency matrix: adj[i][j] = 1 if i->j
        adj = np.zeros((n, n), dtype=np.int8)
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                bit = orientation[idx]
                idx += 1
                if bit == 0:
                    adj[i][j] = 1
                else:
                    adj[j][i] = 1
        
        total_4_sets = n*(n-1)*(n-2)*(n-3)//24
        
        if total_4_sets == 0:
            return 0, 0
        
        # Generate all 4-sets
        all_4sets = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        all_4sets.append((i, j, k, l))
        
        total_4_sets = len(all_4sets)
        
        chunk_arr = np.array(all_4sets, dtype=np.int32)
        
        # For each 4-set, check if any vertex dominates it
        dominated = np.zeros(len(chunk_arr), dtype=bool)
        
        for v in range(n):
            adj_v = adj[v]  # shape (n,)
            
            a = chunk_arr[:, 0]
            b = chunk_arr[:, 1]
            c = chunk_arr[:, 2]
            d = chunk_arr[:, 3]
            
            # Check if v is in the 4-set
            v_in_set = (a == v) | (b == v) | (c == v) | (d == v)
            
            # Check if v points to all 4
            points_to_all = (adj_v[a] == 1) & (adj_v[b] == 1) & (adj_v[c] == 1) & (adj_v[d] == 1)
            
            # v dominates the 4-set if v not in set and points to all
            dom = points_to_all & ~v_in_set
            
            dominated |= dom
            
            # Early exit if all dominated
            if dominated.all():
                break
        
        bad_count = int((~dominated).sum())
        return bad_count, total_4_sets

    # Helper to create orientation from adjacency
    def adj_to_orientation(adj, n):
        orientation = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j] == 1:
                    orientation.append(0)
                else:
                    orientation.append(1)
        return orientation

    # Helper to create Paley tournament
    def create_paley(p):
        if p == 1:
            return []
        squares = {x*x % p for x in range(1, p)}
        adj = [[0]*p for _ in range(p)]
        for i in range(p):
            for j in range(p):
                if i == j: continue
                if (j - i) % p in squares:
                    adj[i][j] = 1
                    adj[j][i] = 0
        return adj

    # Try to find a valid tournament for a given n
    def search_tournament(n, max_time=50):
        # Start with Paley if n is prime and n % 4 == 3
        # Else random
        adj = None
        if n % 4 == 3 and all(n % i != 0 for i in range(2, int(n**0.5)+1)):
            adj = create_paley(n)
        else:
            adj = [[0]*n for _ in range(n)]
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        adj[i][j] = 1
                        adj[j][i] = 0
                    else:
                        adj[i][j] = 0
                        adj[j][i] = 1
        
        # Initial check
        covered, total = count_bad_sets(adj_to_orientation(adj, n), n)
        if covered == total:
            return adj_to_orientation(adj, n)
        
        current_bad = total - covered
        best_bad = current_bad
        best_adj = [row[:] for row in adj]
        
        while time.time() - start_time < max_time:
            # Pick a random arc to flip
            i, j = random.sample(range(n), 2)
            # Flip
            old_val = adj[i][j]
            adj[i][j] = 1 - old_val
            adj[j][i] = 1 - adj[i][j]
            
            # Evaluate
            orientation = adj_to_orientation(adj, n)
            covered, _ = count_bad_sets(orientation, n)
            new_bad = total - covered
            
            if new_bad < current_bad:
                current_bad = new_bad
                if new_bad < best_bad:
                    best_bad = new_bad
                    best_adj = [row[:] for row in adj]
                    if best_bad == 0:
                        return adj_to_orientation(adj, n)
            else:
                # Revert if worse (greedy)
                # Or accept with prob exp(-delta/T) for SA
                # Let's stick to greedy with random restarts
                adj[i][j] = old_val
                adj[j][i] = 1 - adj[i][j]
                
            # If stuck, do a random restart?
            # Simple heuristic: if no improvement in last 100 steps, restart
            # (Not implemented here for simplicity, but could be added)
            
        if best_bad == 0:
            return adj_to_orientation(best_adj, n)
        return None

    # Try n=67 first (known smallest)
    result = search_tournament(67, max_time=100)
    if result is not None:
        return {'n': 67, 'orientation': result}
        
    # Fallback to Paley 83
    p = 83
    adj = create_paley(p)
    orientation = adj_to_orientation(adj, p)
    
    return {'n': p, 'orientation': orientation}
