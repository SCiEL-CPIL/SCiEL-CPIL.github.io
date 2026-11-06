import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140  # Safety margin for 150s limit
    
    # Known best: Paley tournament on p=67
    p_known = 67
    squares = set(x * x % p_known for x in range(1, p_known))
    orientation_known = []
    for i in range(p_known):
        for j in range(i + 1, p_known):
            if (j - i) % p_known in squares:
                orientation_known.append(0)
            else:
                orientation_known.append(1)
    
    best_n = p_known
    best_orientation = orientation_known
    
    def verify_orientation(orient, n):
        """Verify that every 4-set has a common dominator. Returns True if valid."""
        m = n * (n - 1) // 2
        # Build adjacency: adj[i][j] = True if i -> j
        adj = np.zeros((n, n), dtype=bool)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if orient[idx] == 0:
                    adj[i][j] = True
                else:
                    adj[j][i] = True
                idx += 1
        
        # For each 4-set, check if any vertex dominates the other 3
        for a in range(n):
            for b in range(a + 1, n):
                for c in range(b + 1, n):
                    for d in range(c + 1, n):
                        # Check if a dominates b, c, d
                        if adj[a][b] and adj[a][c] and adj[a][d]:
                            continue
                        # Check if b dominates a, c, d
                        if adj[b][a] and adj[b][c] and adj[b][d]:
                            continue
                        # Check if c dominates a, b, d
                        if adj[c][a] and adj[c][b] and adj[c][d]:
                            continue
                        # Check if d dominates a, b, c
                        if adj[d][a] and adj[d][b] and adj[d][c]:
                            continue
                        # No dominator found
                        return False
        return True
    
    def count_bad_quads(orient, n):
        """Count the number of 4-sets without a common dominator."""
        m = n * (n - 1) // 2
        adj = np.zeros((n, n), dtype=bool)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if orient[idx] == 0:
                    adj[i][j] = True
                else:
                    adj[j][i] = True
                idx += 1
        
        bad = 0
        for a in range(n):
            for b in range(a + 1, n):
                for c in range(b + 1, n):
                    for d in range(c + 1, n):
                        if not (
                            (adj[a][b] and adj[a][c] and adj[a][d]) or
                            (adj[b][a] and adj[b][c] and adj[b][d]) or
                            (adj[c][a] and adj[c][b] and adj[c][d]) or
                            (adj[d][a] and adj[d][b] and adj[d][c])
                        ):
                            bad += 1
        return bad
    
    def try_find_smaller(n_target, max_time):
        """Try to find a valid tournament on n_target vertices."""
        m = n_target * (n_target - 1) // 2
        
        # Precompute edge endpoints
        edge_ends = [(i, j) for i in range(n_target) for j in range(i + 1, n_target)]
        idx_map = [[-1] * n_target for _ in range(n_target)]
        for idx, (i, j) in enumerate(edge_ends):
            idx_map[i][j] = idx
            idx_map[j][i] = idx
        
        # Initialize from Paley(67) minus one vertex
        # Map new vertex i to old vertex i+1 (skip vertex 0 of Paley)
        orient = [0] * m
        idx = 0
        for i in range(n_target):
            for j in range(i + 1, n_target):
                old_i = i + 1
                old_j = j + 1
                if (old_j - old_i) % p_known in squares:
                    orient[idx] = 0
                else:
                    orient[idx] = 1
                idx += 1
        
        # Build adjacency matrix for fast delta computation
        adj = np.zeros((n_target, n_target), dtype=bool)
        for idx, (i, j) in enumerate(edge_ends):
            if orient[idx] == 0:
                adj[i][j] = True
            else:
                adj[j][i] = True
        
        def is_good_quad(a, b, c, d):
            """Check if quad (a,b,c,d) has a common dominator."""
            return (
                (adj[a][b] and adj[a][c] and adj[a][d]) or
                (adj[b][a] and adj[b][c] and adj[b][d]) or
                (adj[c][a] and adj[c][b] and adj[c][d]) or
                (adj[d][a] and adj[d][b] and adj[d][c])
            )
        
        # Count initial bad quads
        bad_count = 0
        for a in range(n_target):
            for b in range(a + 1, n_target):
                for c in range(b + 1, n_target):
                    for d in range(c + 1, n_target):
                        if not is_good_quad(a, b, c, d):
                            bad_count += 1
        
        if bad_count == 0:
            return orient
        
        best_bad = bad_count
        best_orient = orient[:]
        
        # Simulated annealing
        T = 1.0
        min_T = 0.005
        cooling = 0.99995
        
        while time.time() - start_time < max_time:
            if T < min_T:
                break
            
            # Pick a random edge to flip
            e_idx = random.randint(0, m - 1)
            u, v = edge_ends[e_idx]
            
            # Compute delta_bad: how many quads containing edge (u,v) change status
            # A quad contains edge (u,v) if it has both u and v plus two other vertices
            # The other two vertices are chosen from the remaining n-2 vertices
            delta_bad = 0
            others = [i for i in range(n_target) if i != u and i != v]
            
            # For each pair (x, y) from others, check quad (u, v, x, y)
            for i in range(len(others)):
                x = others[i]
                for j in range(i + 1, len(others)):
                    y = others[j]
                    quad = (u, v, x, y)
                    
                    was_good = is_good_quad(*quad)
                    
                    # Flip the edge
                    old_val = orient[e_idx]
                    orient[e_idx] = 1 - old_val
                    if old_val == 0:
                        adj[u][v] = False
                        adj[v][u] = True
                    else:
                        adj[u][v] = True
                        adj[v][u] = False
                    
                    is_good_now = is_good_quad(*quad)
                    
                    # Restore
                    orient[e_idx] = old_val
                    if old_val == 0:
                        adj[u][v] = True
                        adj[v][u] = False
                    else:
                        adj[u][v] = False
                        adj[v][u] = True
                    
                    if was_good and not is_good_now:
                        delta_bad += 1
                    elif not was_good and is_good_now:
                        delta_bad -= 1
            
            # Accept move
            if delta_bad <= 0 or random.random() < np.exp(-delta_bad / T):
                # Actually flip
                old_val = orient[e_idx]
                orient[e_idx] = 1 - old_val
                if old_val == 0:
                    adj[u][v] = False
                    adj[v][u] = True
                else:
                    adj[u][v] = True
                    adj[v][u] = False
                bad_count += delta_bad
                
                if bad_count < best_bad:
                    best_bad = bad_count
                    best_orient = orient[:]
                    if best_bad == 0:
                        return best_orient
            
            T *= cooling
        
        return None
    
    # Try to find smaller tournaments, starting from 66 down
    for n_try in range(66, 30, -1):
        if time.time() - start_time > deadline - 10:
            break
        
        # Allocate time based on how promising this n is
        time_budget = min(15, deadline - (time.time() - start_time) - 5)
        if time_budget < 3:
            break
        
        sol = try_find_smaller(n_try, time_budget)
        if sol is not None:
            # Verify
            if verify_orientation(sol, n_try):
                return {'n': n_try, 'orientation': sol}
    
    return {'n': best_n, 'orientation': best_orientation}
