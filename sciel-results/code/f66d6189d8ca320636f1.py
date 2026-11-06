import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115

    def _popcount(x):
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')

    def alpha_bound(adj, n, limit=10):
        best = 0

        def dfs(candidates_mask, current_size):
            nonlocal best
            if current_size + _popcount(candidates_mask) <= best:
                return
            if candidates_mask == 0:
                if current_size > best:
                    best = current_size
                return
            if best >= limit:
                return

            # Heuristic: pick vertex with max degree in candidates
            v = 0
            max_deg = -1
            temp = candidates_mask
            while temp:
                lsb = temp & (-temp)
                idx = lsb.bit_length() - 1
                deg = _popcount(adj[idx] & candidates_mask)
                if deg > max_deg:
                    max_deg = deg
                    v = idx
                temp ^= lsb
            
            lsb = 1 << v
            rest = candidates_mask & ~lsb
            
            # Include v
            new_cands = rest & ~adj[v]
            dfs(new_cands, current_size + 1)
            
            if best >= limit:
                return
            
            # Exclude v
            dfs(rest, current_size)

        dfs((1 << n) - 1, 0)
        return best

    def generate_greedy_tf(n, seed):
        random.seed(seed)
        adj = [0] * n
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(edges)
        for u, v in edges:
            if (adj[u] & adj[v]) == 0:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def local_search(adj, n, time_limit):
        best_adj = adj[:]
        best_alpha = alpha_bound(adj, n, 10)
        if best_alpha < 10:
            return best_adj, best_alpha

        alpha = best_alpha
        temperature = 1.0
        cooling = 0.99

        while time.time() < time_limit:
            # Find large independent set
            # Simple greedy: pick vertex with min degree in remaining
            is_set = []
            remaining = (1 << n) - 1
            while remaining and len(is_set) < 12:
                # Pick vertex with min degree in remaining
                min_deg = n
                v = -1
                temp = remaining
                while temp:
                    lsb = temp & (-temp)
                    idx = lsb.bit_length() - 1
                    deg = _popcount(adj[idx] & remaining)
                    if deg < min_deg:
                        min_deg = deg
                        v = idx
                    temp ^= lsb
                if v == -1: break
                is_set.append(v)
                remaining &= ~(1 << v)
                remaining &= ~adj[v]
                
            if len(is_set) < 2:
                break

            u, v = random.sample(is_set, 2)
            if (adj[u] >> v) & 1:
                continue

            old_adj = adj[:]
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

            # Remove edges to break triangles
            conflicts = adj[u] & adj[v]
            while conflicts:
                lsb = conflicts & (-conflicts)
                w = lsb.bit_length() - 1
                if random.random() < 0.5:
                    adj[u] &= ~(1 << w)
                    adj[w] &= ~(1 << u)
                else:
                    adj[v] &= ~(1 << w)
                    adj[w] &= ~(1 << v)
                conflicts ^= lsb

            new_alpha = alpha_bound(adj, n, 10)
            delta = new_alpha - alpha

            if delta < 0:
                alpha = new_alpha
                if new_alpha < best_alpha:
                    best_alpha = new_alpha
                    best_adj = adj[:]
                    if new_alpha < 10:
                        return best_adj, best_alpha
            else:
                if random.random() < (2.718281828 ** (-delta / max(temperature, 0.01))):
                    alpha = new_alpha
                else:
                    adj = old_adj[:]

            temperature *= cooling

        return best_adj, best_alpha

    best_n = 1
    best_edges = []

    for n in [39, 38, 37, 36, 35]:
        if time.time() > deadline - 10:
            break

        found = False
        max_restarts = 500
        for r in range(max_restarts):
            if time.time() > deadline - 5:
                break

            seed = r * 7919 + n * 104729
            adj = generate_greedy_tf(n, seed)
            alpha = alpha_bound(adj, n, 10)

            if alpha < 10:
                found = True
                break

            if alpha <= 11 and time.time() < deadline - 15:
                adj, alpha = local_search(adj, n, deadline - 10)
                if alpha < 10:
                    found = True
                    break

        if found:
            best_n = n
            edges = []
            for i in range(n):
                mask = adj[i]
                while mask:
                    lsb = mask & (-mask)
                    j = lsb.bit_length() - 1
                    if i < j:
                        edges.append([i, j])
                    mask ^= lsb
            best_edges = edges
            break

    if best_n > 1:
        return {'n': best_n, 'edges': best_edges}

    return {'n': 1, 'edges': []}
