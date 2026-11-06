import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    def get_edges(adj, n):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    def exact_check(n, adj):
        """Exact check: no K4 and no independent 6-set."""
        # Check K4: for each edge (u,v), check common neighbors for triangles
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    # Count edges within common neighborhood
                    m = common
                    while m:
                        lsb = m & -m
                        a = lsb.bit_length() - 1
                        m ^= lsb
                        if (common >> a) & 1 and (adj[a] & common) & (1 << a) == 0:
                            # Check if a has any neighbor in common other than itself
                            rest = common & ~(1 << a)
                            if adj[a] & rest:
                                return False
        # Check independent 6-set using bitset backtracking
        all_mask = (1 << n) - 1
        def find_indep_6(remaining, size):
            if size == 6:
                return True
            if remaining.bit_count() < size:
                return False
            m = remaining
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                m ^= lsb
                # Non-neighbors of u within remaining (excluding u)
                non_n = remaining & ~(adj[u] | (1 << u))
                if non_n.bit_count() >= size - 1:
                    if find_indep_6(non_n, size - 1):
                        return True
            return False
        if find_indep_6(all_mask, 6):
            return False
        return True

    def compute_energy(n, adj):
        """Compute Max-SAT style energy: number of K4s + number of independent 6-sets."""
        k4_count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    common = adj[u] & adj[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    # Count triangles in common
                    m = common
                    while m:
                        lsb = m & -m
                        a = lsb.bit_length() - 1
                        m ^= lsb
                        rest = common & ~(1 << a)
                        k4_count += (adj[a] & rest).bit_count()
        k4_count //= 3

        # Count independent 6-sets
        i6_count = 0
        all_mask = (1 << n) - 1
        # For each 5-tuple that is independent, count how many vertices are non-adjacent to all 5
        # Use backtracking for counting
        def count_indep_sets(remaining, size):
            if size == 0:
                return 1
            if remaining.bit_count() < size:
                return 0
            # Pick a vertex to branch on
            lsb = remaining & -remaining
            u = lsb.bit_length() - 1
            rest = remaining ^ lsb
            # Option 1: don't include u
            count = count_indep_sets(rest, size)
            # Option 2: include u (only if u is non-adjacent to all chosen so far - handled by remaining)
            non_n = rest & ~adj[u]
            count += count_indep_sets(non_n, size - 1)
            return count
        i6_count = count_indep_sets(all_mask, 6)
        return k4_count + i6_count

    def delta_energy_add_edge(n, adj, u, v):
        """Energy change when ADDING edge (u,v). K4s increase, I6s decrease."""
        # K4 increase: triangles in N(u) ∩ N(v) excluding u,v
        common = adj[u] & adj[v]
        common &= ~(1 << u)
        common &= ~(1 << v)
        tri_count = 0
        m = common
        while m:
            lsb = m & -m
            a = lsb.bit_length() - 1
            m ^= lsb
            rest = common & ~(1 << a)
            tri_count += (adj[a] & rest).bit_count()
        tri_count //= 3

        # I6 decrease: independent 5-sets in non-neighborhood of both u and v
        all_mask = (1 << n) - 1
        non_n = all_mask & ~(adj[u] | adj[v] | (1 << u) | (1 << v))
        # Count independent 5-sets in non_n
        def count_indep_sets(remaining, size):
            if size == 0:
                return 1
            if remaining.bit_count() < size:
                return 0
            lsb = remaining & -remaining
            u2 = lsb.bit_length() - 1
            rest = remaining ^ lsb
            count = count_indep_sets(rest, size)
            non_n2 = rest & ~adj[u2]
            count += count_indep_sets(non_n2, size - 1)
            return count
        i5_count = count_indep_sets(non_n, 5)

        return tri_count - i5_count

    def delta_energy_remove_edge(n, adj, u, v):
        """Energy change when REMOVING edge (u,v). K4s decrease, I6s increase."""
        # K4 decrease: triangles in N(u) ∩ N(v) excluding u,v (same as add)
        common = adj[u] & adj[v]
        common &= ~(1 << u)
        common &= ~(1 << v)
        tri_count = 0
        m = common
        while m:
            lsb = m & -m
            a = lsb.bit_length() - 1
            m ^= lsb
            rest = common & ~(1 << a)
            tri_count += (adj[a] & rest).bit_count()
        tri_count //= 3

        # I6 increase: independent 5-sets in non-neighborhood of both u and v
        all_mask = (1 << n) - 1
        non_n = all_mask & ~(adj[u] | adj[v] | (1 << u) | (1 << v))
        def count_indep_sets(remaining, size):
            if size == 0:
                return 1
            if remaining.bit_count() < size:
                return 0
            lsb = remaining & -remaining
            u2 = lsb.bit_length() - 1
            rest = remaining ^ lsb
            count = count_indep_sets(rest, size)
            non_n2 = rest & ~adj[u2]
            count += count_indep_sets(non_n2, size - 1)
            return count
        i5_count = count_indep_sets(non_n, 5)

        return -tri_count + i5_count

    def simulated_annealing(n, adj, T_start, T_end, steps):
        """Run SA on the Max-SAT energy landscape."""
        current_energy = compute_energy(n, adj)
        best_energy = current_energy
        best_adj = adj[:]
        
        for s in range(steps):
            if time.time() > deadline - 10:
                break
            
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            is_edge = bool((adj[u] >> v) & 1)
            
            if is_edge:
                d = delta_energy_remove_edge(n, adj, u, v)
            else:
                d = delta_energy_add_edge(n, adj, u, v)
            
            T = T_start * (T_end / T_start) ** (s / max(steps - 1, 1))
            
            if d < 0 or (T > 0 and random.random() < math.exp(-d / T)):
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                current_energy += d
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_adj = adj[:]
                    
                    if best_energy == 0:
                        return best_adj, 0, True
        
        return best_adj, best_energy, best_energy == 0

    def random_init(n, p=0.5):
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def structured_init(n):
        """Initialize with a quasi-random structure to reduce initial energy."""
        adj = [0] * n
        # Use a cyclic tournament-like structure
        for u in range(n):
            for v in range(u + 1, n):
                # Use modular difference with a hash-like pattern
                d = (v - u) % n
                # Include edge if d is in a "good" set
                if d % 3 != 0 and d % 5 != 0:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    best_graph = None
    best_n = 0

    for target_n in [35, 34, 33, 32, 31, 30]:
        if time.time() > deadline - 20:
            break
        
        found = False
        
        # Phase 1: Structured initializations with SA
        for restart in range(5):
            if time.time() > deadline - 15:
                break
            if found:
                break
            
            adj = structured_init(target_n)
            adj, energy, success = simulated_annealing(
                target_n, adj,
                T_start=8.0, T_end=0.01,
                steps=8000
            )
            if success:
                best_graph = adj
                best_n = target_n
                found = True
                break
        
        # Phase 2: Random initializations with SA
        if not found:
            for restart in range(20):
                if time.time() > deadline - 15:
                    break
                if found:
                    break
                
                p = 0.3 + 0.4 * random.random()
                adj = random_init(target_n, p)
                adj, energy, success = simulated_annealing(
                    target_n, adj,
                    T_start=10.0, T_end=0.01,
                    steps=10000
                )
                if success:
                    best_graph = adj
                    best_n = target_n
                    found = True
                    break
        
        if found:
            break
    
    if best_graph is None:
        # Final fallback: Paley graph of order 17
        squares = {x * x % 17 for x in range(1, 17)}
        adj = [0] * 17
        for i in range(17):
            for j in range(i + 1, 17):
                if (j - i) % 17 in squares:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return {'n': 17, 'edges': get_edges(adj, 17)}
    else:
        return {'n': best_n, 'edges': get_edges(best_graph, best_n)}
