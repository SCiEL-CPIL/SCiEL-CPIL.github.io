import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin for output

    def find_k4_exact(adj, n):
        # Returns a tuple (u, v, w, x) forming a K4, or None
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    w_mask = common
                    while w_mask:
                        lsb = w_mask & -w_mask
                        w = lsb.bit_length() - 1
                        w_mask ^= lsb
                        if w < v: continue # w must be > v? No, just distinct.
                        # We need x > w to avoid duplicates?
                        # Actually, just check if w and x are connected.
                        x_mask = common & adj[w]
                        if x_mask:
                            lsb_x = x_mask & -x_mask
                            x = lsb_x.bit_length() - 1
                            # Ensure distinct
                            if x != u and x != v and x != w:
                                return (u, v, w, x)
        return None

    def find_ind_6_exact(adj, n):
        # Returns a list of 6 vertices that are independent, or None
        def backtrack(current, candidates):
            if len(current) == 6:
                return current[:]
            if candidates.bit_count() < 6 - len(current):
                return None
            # Heuristic: pick vertex with min degree in candidates
            # To speed up, just pick first
            cand_list = []
            mask = candidates
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                cand_list.append(v)
                mask ^= lsb
            
            for v in cand_list:
                new_candidates = candidates & ~(1 << v) & ~adj[v]
                res = backtrack(current + [v], new_candidates)
                if res:
                    return res
            return None
        return backtrack([], (1 << n) - 1)

    def is_valid_exact(adj, n):
        if find_k4_exact(adj, n):
            return False
        if find_ind_6_exact(adj, n):
            return False
        return True

    def estimate_penalty(adj, n, samples=50):
        # Estimate number of K4s and Ind6s
        penalty = 0
        # Sample K4s
        for _ in range(samples):
            u, v, w, x = random.sample(range(n), 4)
            # Check if K4
            if ((adj[u] >> v) & 1) and ((adj[u] >> w) & 1) and ((adj[u] >> x) & 1) and \
               ((adj[v] >> w) & 1) and ((adj[v] >> x) & 1) and ((adj[w] >> x) & 1):
                penalty += 1
        # Sample Ind6s
        for _ in range(samples):
            verts = random.sample(range(n), 6)
            is_ind = True
            for i in range(6):
                for j in range(i+1, 6):
                    if (adj[verts[i]] >> verts[j]) & 1:
                        is_ind = False
                        break
                if not is_ind:
                    break
            if is_ind:
                penalty += 1
        return penalty

    # Fallback: Paley(17)
    squares = {x * x % 17 for x in range(1, 17)}
    best_edges = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
    best_n = 17

    for target_n in [35, 34, 33, 32, 31, 30]:
        if time.time() > deadline - 15.0:
            break
        
        found = False
        attempts = 0
        max_attempts = 20 # Fewer, longer SA runs
        
        while not found and attempts < max_attempts and time.time() < deadline - 15.0:
            attempts += 1
            
            # Initialize random graph with p=0.7
            adj = [0] * target_n
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.7:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Simulated Annealing
            T = 10.0
            current_penalty = estimate_penalty(adj, target_n)
            
            while time.time() < deadline - 15.0:
                # Check exact validity if penalty is low
                if current_penalty == 0:
                    if is_valid_exact(adj, target_n):
                        found = True
                        best_n = target_n
                        edges = []
                        for i in range(target_n):
                            for j in range(i + 1, target_n):
                                if (adj[i] >> j) & 1:
                                    edges.append([i, j])
                        best_edges = edges
                        break
                
                # Pick a random edge to flip
                u = random.randint(0, target_n - 1)
                v = random.randint(0, target_n - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                
                # Flip
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                
                new_penalty = estimate_penalty(adj, target_n)
                
                # Accept?
                if new_penalty < current_penalty:
                    current_penalty = new_penalty
                else:
                    delta = new_penalty - current_penalty
                    if delta == 0:
                        current_penalty = new_penalty
                    else:
                        prob = math.exp(-delta / T)
                        if random.random() < prob:
                            current_penalty = new_penalty
                        else:
                            # Revert
                            if (adj[u] >> v) & 1:
                                adj[u] &= ~(1 << v)
                                adj[v] &= ~(1 << u)
                            else:
                                adj[u] |= (1 << v)
                                adj[v] |= (1 << u)
                
                # Cool down
                T *= 0.99
                if T < 0.1:
                    T = 0.1

            if found:
                break

    return {'n': best_n, 'edges': best_edges}
