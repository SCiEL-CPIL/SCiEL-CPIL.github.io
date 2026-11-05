import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def count_paths_len_5(adj, u, v, n):
        if u == v:
            return 0
        
        count = 0
        # DFS to find simple paths of length 5 from u to v
        # Path: u -> x1 -> x2 -> x3 -> x4 -> v
        
        # Neighbors of u (excluding u, v)
        # We will use a recursive helper to avoid recreating closures if possible, 
        # but for n=20, function call overhead is the main cost.
        
        def dfs(current, depth, visited_mask):
            nonlocal count
            if depth == 4:
                # Check if v is a neighbor of current and not visited
                if (adj[current] >> v) & 1 and not (visited_mask >> v) & 1:
                    count += 1
                return
            
            # Next depth
            # Neighbors of current, excluding visited, u, v
            # u and v are excluded because they are endpoints and cannot be internal nodes
            neigh_mask = adj[current] & ~visited_mask & ~(1 << u) & ~(1 << v)
            
            mask = neigh_mask
            while mask:
                lsb = mask & -mask
                idx = (lsb.bit_length() - 1)
                mask ^= lsb
                dfs(idx, depth + 1, visited_mask | (1 << idx))

        neigh_u = adj[u] & ~(1 << u) & ~(1 << v)
        mask = neigh_u
        while mask:
            lsb = mask & -mask
            x1 = (lsb.bit_length() - 1)
            mask ^= lsb
            dfs(x1, 1, (1 << u) | (1 << x1))
            
        return count

    def solve_n(n, time_budget):
        m = n * (n - 1) // 2
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # Try multiple restarts if time permits
        # But we only have one time_budget for the whole solve_n call.
        
        colors = [random.randint(0, 3) for _ in range(m)]
        
        adj = [[0] * n for _ in range(4)]
        
        def update_adj(idx, old_c, new_c):
            u, v = edges[idx]
            if old_c != -1:
                adj[old_c][u] &= ~(1 << v)
                adj[old_c][v] &= ~(1 << u)
            if new_c != -1:
                adj[new_c][u] |= (1 << v)
                adj[new_c][v] |= (1 << u)
                
        for idx, c in enumerate(colors):
            u, v = edges[idx]
            adj[c][u] |= (1 << v)
            adj[c][v] |= (1 << u)

        def get_cost():
            total_cost = 0
            for c in range(4):
                for u in range(n):
                    for v in range(u + 1, n):
                        idx = u * (n - 1) - u * (u - 1) // 2 + (v - u - 1)
                        if colors[idx] == c:
                            total_cost += count_paths_len_5(adj[c], u, v, n)
            return total_cost

        current_cost = get_cost()
        if current_cost == 0:
            return colors[:]

        T = 1.0
        cooling = 0.999
        best_cost = current_cost
        best_colors = colors[:]
        
        end_time = time.time() + time_budget
        
        while time.time() < end_time:
            idx = random.randint(0, m - 1)
            u, v = edges[idx]
            old_c = colors[idx]
            
            best_delta = float('inf')
            best_new_c = -1
            
            for new_c in range(4):
                if new_c == old_c:
                    continue
                
                broken_c6s = count_paths_len_5(adj[old_c], u, v, n)
                created_c6s = count_paths_len_5(adj[new_c], u, v, n)
                
                delta = created_c6s - broken_c6s
                
                if delta < best_delta:
                    best_delta = delta
                    best_new_c = new_c
            
            if best_new_c == -1:
                continue
            
            if best_delta < 0 or random.random() < pow(2.718, -best_delta / T):
                update_adj(idx, old_c, best_new_c)
                colors[idx] = best_new_c
                current_cost += best_delta
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors[:]
                    
                if best_cost == 0:
                    return best_colors[:]
                
            T *= cooling
            if T < 0.01:
                T = 0.01

        return None

    for n in range(17, 8, -1):
        if time.time() > deadline - 2:
            break
        time_budget = 15.0 if n == 17 else 5.0
        res = solve_n(n, time_budget)
        if res is not None:
            return {'n': n, 'colors': res}
            
    n = 9
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            d = j - i
            c = min(d, n - d) - 1
            colors.append(c)
    return {'n': n, 'colors': colors}

if __name__ == "__main__":
    res = construct()
    print(res)
