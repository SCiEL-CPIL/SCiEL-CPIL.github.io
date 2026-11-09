import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 150s limit
    
    def count_invalid_4sets(adj):
        """
        Count invalid 4-sets (score sequence 2,2,1,1).
        Valid 4-sets have a vertex with out-degree 3 in the induced subgraph.
        Total 4-sets = C(n, 4).
        Valid 4-sets = Sum_{v} C(out_degree(v), 3).
        """
        n = adj.shape[0]
        out_degrees = np.sum(adj, axis=1)
        od = out_degrees.astype(np.int64)
        c3 = np.zeros_like(od)
        mask = od >= 3
        c3[mask] = od[mask] * (od[mask] - 1) * (od[mask] - 2) // 6
        total_valid = np.sum(c3)
        total_4sets = n * (n - 1) * (n - 2) * (n - 3) // 24
        return total_4sets - total_valid

    def compute_delta(adj, i, j):
        """
        Compute the change in the number of invalid 4-sets if edge (i, j) is flipped.
        adj[i, j] = 1 if i -> j.
        Flipping means i stops pointing to j, and j starts pointing to i.
        
        The objective function is F = C(n,4) - Sum_v C(d_v, 3).
        We want to minimize F, which is equivalent to maximizing G = Sum_v C(d_v, 3).
        delta_F = - delta_G.
        
        Only degrees of i and j change.
        d_i decreases by 1.
        d_j increases by 1.
        
        delta_G = C(d_j + 1, 3) - C(d_j, 3) + C(d_i - 1, 3) - C(d_i, 3)
        delta_G = C(d_j, 2) - C(d_i - 1, 2)
        
        Note: C(k, 2) = k(k-1)/2.
        If d_i < 2, C(d_i-1, 2) is 0.
        """
        d_i = adj[i, :].sum()
        d_j = adj[j, :].sum()
        
        # C(d_j, 2)
        term_j = d_j * (d_j - 1) // 2
        
        # C(d_i - 1, 2)
        if d_i >= 2:
            term_i = (d_i - 1) * (d_i - 2) // 2
        else:
            term_i = 0
            
        delta_G = term_j - term_i
        # We want to minimize invalid sets.
        # Invalid = Total - Valid.
        # delta_Invalid = - delta_Valid = - delta_G.
        return -delta_G

    def simulated_annealing(adj, n, max_iter=5000, start_temp=1.0, end_temp=0.01):
        """
        Vectorized-ish Simulated Annealing using exact delta.
        """
        current_adj = adj.copy()
        current_invalid = count_invalid_4sets(current_adj)
        best_adj = current_adj.copy()
        best_invalid = current_invalid
        
        # Precompute list of all edges (i, j) with i < j
        # Actually, we can just pick random i, j and handle i>j or i==j
        # For speed, let's just pick random i, j != i
        
        n_edges = n * (n - 1)
        # We will iterate max_iter times
        
        # To make it faster, we can batch process? No, SA is sequential.
        # But compute_delta is O(1) if we maintain degrees.
        # Let's maintain degrees array to make delta O(1).
        
        degrees = np.sum(current_adj, axis=1).astype(np.int64)
        
        for t in range(max_iter):
            if time.time() > deadline:
                break
            
            # Pick random edge
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            
            # Determine current direction
            # adj[i, j] is 1 if i -> j
            # If we flip, i stops pointing to j.
            
            # Calculate delta based on current degrees
            # If i -> j (adj[i, j]==1):
            #   d_i decreases, d_j increases.
            #   delta_G = C(d_j, 2) - C(d_i - 1, 2)
            #   delta_F = - delta_G
            # If j -> i (adj[i, j]==0):
            #   d_i increases, d_j decreases.
            #   delta_G = C(d_i, 2) - C(d_j - 1, 2)
            #   delta_F = - delta_G
            
            if current_adj[i, j] == 1:
                # i -> j
                # i loses 1, j gains 1
                di = degrees[i]
                dj = degrees[j]
                # term_j = C(dj, 2)
                term_j = dj * (dj - 1) // 2
                # term_i = C(di-1, 2)
                if di >= 2:
                    term_i = (di - 1) * (di - 2) // 2
                else:
                    term_i = 0
                delta_G = term_j - term_i
            else:
                # j -> i
                # i gains 1, j loses 1
                di = degrees[i]
                dj = degrees[j]
                # term_i = C(di, 2)
                term_i = di * (di - 1) // 2
                # term_j = C(dj-1, 2)
                if dj >= 2:
                    term_j = (dj - 1) * (dj - 2) // 2
                else:
                    term_j = 0
                delta_G = term_i - term_j
            
            delta_F = -delta_G
            
            # Acceptance probability
            temp = start_temp * (end_temp / start_temp) ** (t / max_iter)
            if delta_F < 0 or random.random() < np.exp(-delta_F / temp):
                # Accept
                # Flip edge
                current_adj[i, j] = 1 - current_adj[i, j]
                current_adj[j, i] = 1 - current_adj[j, i]
                
                # Update degrees
                if current_adj[i, j] == 1:
                    # Now i -> j
                    # Before flip, if it was 1, it became 0 (i stopped pointing). 
                    # Wait, the logic above assumed we are calculating delta for the flip.
                    # If we accept, we apply the flip.
                    # If old was 1 (i->j), new is 0 (j->i).
                    # d_i decreases by 1, d_j increases by 1.
                    degrees[i] -= 1
                    degrees[j] += 1
                else:
                    # Old was 0 (j->i), new is 1 (i->j).
                    # d_i increases by 1, d_j decreases by 1.
                    degrees[i] += 1
                    degrees[j] -= 1
                    
                current_invalid += delta_F
                
                if current_invalid < best_invalid:
                    best_invalid = current_invalid
                    best_adj = current_adj.copy()
                    if best_invalid == 0:
                        break

        return best_adj, best_invalid

    best_n = None
    best_orient = None

    # Try n from 31 upwards.
    # The gap between 31 and 67 is huge. 
    # Random tournaments for n=31 have ~10^5 invalid sets.
    # SA needs to clear them all.
    # With O(1) delta, we can do ~10^5 steps in 1s.
    # Let's try n=31 to 45 with aggressive SA.
    
    for n in range(31, 46):
        if time.time() > deadline:
            break
            
        # Number of restarts depends on remaining time
        # Let's allocate roughly equal time to each n, or more to smaller n?
        # Smaller n is harder to satisfy if it's impossible? 
        # Actually, if n < f(4), it's impossible. 
        # But we don't know f(4) exactly, only bounds.
        # Let's try a few restarts for each n.
        
        max_restarts = 3
        for r in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Generate random tournament
            adj = np.zeros((n, n), dtype=np.int8)
            # Vectorized random generation
            # Create random matrix, make symmetric
            rand_mat = np.random.rand(n, n)
            # Upper triangle
            iu = np.triu_indices(n, 1)
            # Set adj[i, j] = 1 if rand < 0.5 else 0
            # adj is int8
            vals = (rand_mat[iu] < 0.5).astype(np.int8)
            adj[iu] = vals
            # Lower triangle is inverse
            # adj[j, i] = 1 - adj[i, j]
            adj[(iu[1], iu[0])] = 1 - vals
            
            # Run SA
            # Estimate time per restart
            # Let's give each restart ~5 seconds max? 
            # 150s / 15 n's / 3 restarts = 3.3s per restart.
            # 10,000 steps should be very fast (under 1s).
            # So we can do more iterations or more restarts.
            # Let's do 20,000 iterations per restart.
            
            adj_opt, invalid_count = simulated_annealing(adj, n, max_iter=20000, start_temp=5.0, end_temp=0.1)
            
            if invalid_count == 0:
                best_n = n
                orientation = []
                for i in range(n):
                    for j in range(i + 1, n):
                        orientation.append(0 if adj_opt[i, j] == 1 else 1)
                return {'n': n, 'orientation': orientation}
            
            # If we didn't find 0, keep the best one found so far?
            # The task requires returning a VALID tournament.
            # If we fail to find a valid one for small n, we must fall back to a known valid one.
            # The current program falls back to n=83 Paley.
            # We should ensure we always have a valid fallback.
            
    # Fallback: Use the same logic as before for larger n, but faster.
    # Or just return the best invalid one? No, must be valid.
    # Let's try to find a valid one for n=67 (known smallest) using SA?
    # Or just use the Paley 83 fallback which is guaranteed valid.
    # But maybe we can find n < 83?
    
    # Let's try one more time with n=50 to 60, as 31-45 might be impossible.
    for n in range(46, 61):
        if time.time() > deadline:
            break
        # 1 restart
        adj = np.zeros((n, n), dtype=np.int8)
        rand_mat = np.random.rand(n, n)
        iu = np.triu_indices(n, 1)
        vals = (rand_mat[iu] < 0.5).astype(np.int8)
        adj[iu] = vals
        adj[(iu[1], iu[0])] = 1 - vals
        
        adj_opt, invalid_count = simulated_annealing(adj, n, max_iter=10000, start_temp=5.0, end_temp=0.1)
        if invalid_count == 0:
            orientation = []
            for i in range(n):
                for j in range(i + 1, n):
                    orientation.append(0 if adj_opt[i, j] == 1 else 1)
            return {'n': n, 'orientation': orientation}

    # Absolute fallback: Paley 83
    p = 83
    squares = {x * x % p for x in range(1, p)}
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if diff in squares:
                orientation.append(0) # i -> j
            else:
                orientation.append(1) # j -> i
    return {'n': p, 'orientation': orientation}
