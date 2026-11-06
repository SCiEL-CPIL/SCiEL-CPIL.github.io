import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin below 120s evaluator limit

    # --- bitset helpers ---
    def popcount(x):
        return bin(x).count('1')

    def build_adj_bitset(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def count_triangles(adj, n):
        """Count triangles using bitsets."""
        count = 0
        for i in range(n):
            neighbors = adj[i]
            while neighbors:
                lsb = neighbors & (-neighbors)
                j = lsb.bit_length() - 1
                neighbors ^= lsb
                if j > i:
                    common = adj[j] & adj[i] & ~(1 << i) & ~(1 << j)
                    count += popcount(common)
        return count // 3

    def has_independent_set_size(adj, n, k):
        """Returns True if there exists an independent set of size k.
        Uses branch and bound with bitsets. Early exit upon finding size k."""
        best_found = [k - 1]

        def dfs(candidates, current_size):
            if best_found[0] >= k:
                return True  # found
            if current_size >= k:
                best_found[0] = k
                return True
            if not candidates:
                return False
            if current_size + popcount(candidates) <= best_found[0]:
                return False

            # Greedy: pick vertex with highest degree in candidate subgraph
            best_v = -1
            best_deg = -1
            temp = candidates
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                temp ^= lsb
                deg = popcount(adj[v] & candidates)
                if deg > best_deg:
                    best_deg = deg
                    best_v = v

            v = best_v
            bit_v = 1 << v
            rest = candidates & ~bit_v

            # Branch include v
            non_neighbors = rest & ~adj[v]
            if dfs(non_neighbors, current_size + 1):
                return True

            # Branch exclude v
            if dfs(rest, current_size):
                return True

            return False

        dfs((1 << n) - 1, 0)
        return best_found[0] >= k

    def verify_graph(n, edges):
        adj = build_adj_bitset(n, edges)
        if count_triangles(adj, n) > 0:
            return False
        if has_independent_set_size(adj, n, 10):
            return False
        return True

    def andrasfai_graph(k):
        """Andrásfai graph A_k on 3k vertices."""
        n = 3 * k
        edges = []
        for a in range(3):
            for i in range(k):
                for b in range(3):
                    if (b - a) % 3 != 1:
                        continue
                    for j in range(k):
                        if (j - i) % k in (0, 1):
                            u = a * k + i
                            v = b * k + j
                            if u < v:
                                edges.append([u, v])
        return {'n': n, 'edges': edges}

    # Baseline: Andrásfai A_9 has n=27, alpha=9
    best = {'n': 27, 'edges': []}
    ag = andrasfai_graph(9)
    if ag and verify_graph(ag['n'], ag['edges']):
        best = ag

    # Simulated Annealing for target n
    def simulate_annealing(target_n, max_steps=5000, t0=1.0, tf=0.01):
        """
        Attempts to find a valid graph on target_n vertices.
        Returns edges if found, else None.
        """
        # Initialize with random sparse graph
        adj = [0] * target_n
        for i in range(target_n):
            for j in range(i + 1, target_n):
                if random.random() < 0.3:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        current_energy = 0
        best_energy = float('inf')
        best_adj = adj[:]
        
        # Helper to calculate energy (penalty)
        def get_energy(adj_state):
            tri_count = count_triangles(adj_state, target_n)
            # Heuristic for independent set
            # If tri_count > 0, energy is high
            # If alpha >= 10, energy is high
            # We use a fast check for alpha
            # To save time, we only do exact alpha check if tri_count == 0
            # Otherwise, we estimate alpha or just penalize triangles heavily
            if tri_count > 0:
                return 1000 * tri_count
            else:
                # Check if alpha >= 10
                # Use a greedy heuristic first
                # If greedy finds 10, we have a problem
                # If not, we need exact check?
                # For speed, let's just return 0 if we assume it's valid? No.
                # Let's do a quick exact check here.
                if has_independent_set_size(adj_state, target_n, 10):
                    return 100
                return 0

        current_energy = get_energy(adj)
        if current_energy == 0:
            return build_edges_from_adj(adj, target_n)

        for step in range(max_steps):
            if time.time() - start_time > deadline - 5:
                break
            
            # Pick a random edge to flip
            u = random.randint(0, target_n - 1)
            v = random.randint(0, target_n - 1)
            if u == v:
                continue
            
            # Flip edge (u, v)
            if adj[u] & (1 << v):
                # Remove
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
            else:
                # Add
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            new_energy = get_energy(adj)
            delta_e = new_energy - current_energy
            
            # Temperature
            t = t0 * (tf / t0) ** (step / max_steps)
            
            if delta_e < 0 or random.random() < pow(2.71828, -delta_e / t):
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_adj = adj[:]
                    if best_energy == 0:
                        return build_edges_from_adj(best_adj, target_n)
            else:
                # Revert
                if adj[u] & (1 << v):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
        if best_energy == 0:
            return build_edges_from_adj(best_adj, target_n)
        return None

    def build_edges_from_adj(adj, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges

    # Try to find larger graphs
    for target_n in range(40, 27, -1):
        if time.time() - start_time > deadline - 10:
            break
        
        found = False
        for attempt in range(20):
            if time.time() - start_time > deadline - 5:
                break
            random.seed(attempt * 12345 + target_n)
            edges = simulate_annealing(target_n, max_steps=2000)
            if edges:
                # Verify
                if verify_graph(target_n, edges):
                    best = {'n': target_n, 'edges': edges}
                    found = True
                    break
        
        if found:
            break

    return best
