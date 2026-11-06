"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit
    n = 18

    def edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def has_k4(adj):
        """Check if graph contains a K4. Returns True if found."""
        for a in range(n):
            for b in range(a + 1, n):
                if not ((adj[a] >> b) & 1):
                    continue
                common = adj[a] & adj[b]
                for c in range(b + 1, n):
                    if not ((common >> c) & 1):
                        continue
                    remaining = common & adj[c] & ~((1 << (c + 1)) - 1)
                    if remaining:
                        return True
        return False

    def find_k4(adj):
        """Find and return a K4 as (a,b,c,d) or None."""
        for a in range(n):
            for b in range(a + 1, n):
                if not ((adj[a] >> b) & 1):
                    continue
                common = adj[a] & adj[b]
                for c in range(b + 1, n):
                    if not ((common >> c) & 1):
                        continue
                    remaining = common & adj[c] & ~((1 << (c + 1)) - 1)
                    if remaining:
                        d_bit = remaining & (-remaining)
                        d = d_bit.bit_length() - 1
                        return (a, b, c, d)
        return None

    def remove_k4(adj):
        """Remove edges to make graph K4-free. Returns True if successful."""
        for _ in range(500):
            k4 = find_k4(adj)
            if k4 is None:
                return True
            a, b, c, d = k4
            pair = random.choice([(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)])
            x, y = pair
            adj[x] &= ~(1 << y)
            adj[y] &= ~(1 << x)
        return not has_k4(adj)

    def max_triangle_free_size(adj, time_limit=3.0):
        """Find the largest induced triangle-free subgraph using improved branch and bound."""
        best = [0]
        
        # Order vertices by degree descending (higher degree vertices are more constraining)
        deg = [bin(adj[v]).count('1') for v in range(n)]
        order = sorted(range(n), key=lambda v: -deg[v])
        
        # Precompute conflict masks: for each vertex v, which other vertices u
        # form a triangle with some third vertex w that is also in our set.
        # Actually, we need: for a given set S, can we add v?
        # v can be added iff for all pairs (u1, u2) in S, not (adj[u1]&adj[u2] & (1<<v)) and 
        # not (adj[u1]&adj[v]) and not (adj[u2]&adj[v])... 
        # Simpler: v can be added iff the induced subgraph on S ∪ {v} is triangle-free.
        # This means: no two vertices in S are both adjacent to v AND adjacent to each other.
        
        t0 = time.time()
        timeout = [False]

        def backtrack(idx, current_mask, current_size):
            if timeout[0]:
                return
            if time.time() - t0 > time_limit:
                timeout[0] = True
                return
            
            if idx == n:
                if current_size > best[0]:
                    best[0] = current_size
                return
            
            # Pruning: even if we include all remaining, can't beat best
            if current_size + (n - idx) <= best[0]:
                return

            v = order[idx]

            # Check if v can be added to current set
            # v can be added iff there is no pair (u1, u2) in current_mask such that
            # adj[v]&adj[u1] and adj[v]&adj[u2] and adj[u1]&adj[u2]
            # Equivalently: the neighbors of v in current_mask form an independent set in the original graph
            neighbors_in_S = current_mask & adj[v]
            can_add = True
            tmp = neighbors_in_S
            while tmp:
                bit = tmp & (-tmp)
                u = bit.bit_length() - 1
                rest = tmp & ~bit
                if (adj[u] & rest) != 0:
                    can_add = False
                    break
                tmp = rest
            
            if can_add:
                backtrack(idx + 1, current_mask | (1 << v), current_size + 1)
            
            # Try excluding v
            if not timeout[0]:
                backtrack(idx + 1, current_mask, current_size)

        backtrack(0, 0, 0)
        return best[0]

    def max_triangle_free_size_fast(adj, time_limit=1.0):
        """Faster version with tighter time limit for local search."""
        best = [0]
        deg = [bin(adj[v]).count('1') for v in range(n)]
        order = sorted(range(n), key=lambda v: -deg[v])
        t0 = time.time()
        timeout = [False]

        def backtrack(idx, current_mask, current_size):
            if timeout[0]:
                return
            if time.time() - t0 > time_limit:
                timeout[0] = True
                return
            if idx == n:
                if current_size > best[0]:
                    best[0] = current_size
                return
            if current_size + (n - idx) <= best[0]:
                return

            v = order[idx]
            neighbors_in_S = current_mask & adj[v]
            can_add = True
            tmp = neighbors_in_S
            while tmp:
                bit = tmp & (-tmp)
                u = bit.bit_length() - 1
                rest = tmp & ~bit
                if (adj[u] & rest) != 0:
                    can_add = False
                    break
                tmp = rest
            
            if can_add:
                backtrack(idx + 1, current_mask | (1 << v), current_size + 1)
            
            if not timeout[0]:
                backtrack(idx + 1, current_mask, current_size)

        backtrack(0, 0, 0)
        return best[0]

    random.seed(42)
    candidates = []

    # Strategy 1: Complete tripartite K_{6,6,6}
    adj1 = [0] * n
    parts3 = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts3[i]:
                for b in parts3[j]:
                    adj1[a] |= (1 << b)
                    adj1[b] |= (1 << a)
    candidates.append(adj1)

    # Strategy 2: Turan T(18,3) with internal edges
    for trial in range(60):
        if time.time() > deadline - 40:
            break
        parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        adj = [0] * n
        for i in range(3):
            for j in range(i + 1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
        for i in range(3):
            wd = 0.2 + random.random() * 0.6
            for a in parts[i]:
                for b in parts[i]:
                    if a < b and random.random() < wd:
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
        if remove_k4(adj):
            candidates.append(adj)

    # Strategy 3: 4-partition
    for trial in range(40):
        if time.time() > deadline - 40:
            break
        vertices = list(range(n))
        random.shuffle(vertices)
        sizes = [5, 5, 4, 4]
        parts = []
        idx = 0
        for size in sizes:
            parts.append(vertices[idx:idx + size])
            idx += size
        adj = [0] * n
        cross_d = 0.5 + random.random() * 0.5
        within_d = random.random() * 0.4
        for i in range(4):
            for j in range(i + 1, 4):
                for a in parts[i]:
                    for b in parts[j]:
                        if random.random() < cross_d:
                            adj[a] |= (1 << b)
                            adj[b] |= (1 << a)
            for a in parts[i]:
                for b in parts[i]:
                    if a < b and random.random() < within_d:
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
        if remove_k4(adj):
            candidates.append(adj)

    # Strategy 4: 5-partition
    for trial in range(30):
        if time.time() > deadline - 40:
            break
        vertices = list(range(n))
        random.shuffle(vertices)
        sizes = [4, 4, 4, 3, 3]
        parts = []
        idx = 0
        for size in sizes:
            parts.append(vertices[idx:idx + size])
            idx += size
        adj = [0] * n
        cross_d = 0.4 + random.random() * 0.6
        within_d = random.random() * 0.3
        for i in range(5):
            for j in range(i + 1, 5):
                for a in parts[i]:
                    for b in parts[j]:
                        if random.random() < cross_d:
                            adj[a] |= (1 << b)
                            adj[b] |= (1 << a)
            for a in parts[i]:
                for b in parts[i]:
                    if a < b and random.random() < within_d:
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
        if remove_k4(adj):
            candidates.append(adj)

    # Strategy 5: Random dense graphs
    for trial in range(50):
        if time.time() > deadline - 40:
            break
        density = 0.3 + random.random() * 0.5
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < density:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        if remove_k4(adj):
            candidates.append(adj)

    # Strategy 6: Blow-up of C5 (pentagon blow-up)
    for trial in range(20):
        if time.time() > deadline - 40:
            break
        # C5 has 5 vertices, blow up each to size ~3-4
        base = [(0,1),(1,2),(2,3),(3,4),(4,0)]
        vertices = list(range(n))
        random.shuffle(vertices)
        # Partition into 5 groups
        group_sizes = [4, 4, 4, 3, 3]
        groups = []
        idx = 0
        for size in group_sizes:
            groups.append(vertices[idx:idx+size])
            idx += size
        adj = [0] * n
        # Connect groups according to C5
        for i in range(5):
            j = (i + 1) % 5
            for a in groups[i]:
                for b in groups[j]:
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)
        # Add some within-group edges
        for i in range(5):
            wd = random.random() * 0.3
            for a in groups[i]:
                for b in groups[i]:
                    if a < b and random.random() < wd:
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
        if remove_k4(adj):
            candidates.append(adj)

    # Strategy 7: Random sparse graphs (more edges to create more triangles)
    for trial in range(40):
        if time.time() > deadline - 40:
            break
        density = 0.4 + random.random() * 0.4
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < density:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        if remove_k4(adj):
            candidates.append(adj)

    # Strategy 8: Complement of sparse graph (dense graphs)
    for trial in range(30):
        if time.time() > deadline - 40:
            break
        density = 0.1 + random.random() * 0.2
        adj = [0] * n
        # Start with complete graph, remove edges
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() >= density:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        if remove_k4(adj):
            candidates.append(adj)

    # Evaluate candidates
    best_adj = None
    best_score = 18
    for adj in candidates:
        if time.time() > deadline - 20:
            break
        score = max_triangle_free_size(adj, time_limit=2.0)
        if score < best_score:
            best_score = score
            best_adj = adj[:]

    if best_adj is None:
        best_adj = candidates[0]
        best_score = max_triangle_free_size(best_adj, time_limit=2.0)

    # Local search with simulated annealing
    adj = best_adj[:]
    current_score = best_score
    best_adj = adj[:]
    best_score = current_score

    temperature = 1.0
    cooling = 0.999

    for iteration in range(15000):
        if time.time() > deadline - 5:
            break
        if best_score <= 6:
            break

        # Pick a random edge to flip
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i

        old_adj = adj[:]
        if (adj[i] >> j) & 1:
            adj[i] &= ~(1 << j)
            adj[j] &= ~(1 << i)
        else:
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)

        if has_k4(adj):
            adj = old_adj
            continue

        score = max_triangle_free_size_fast(adj, time_limit=0.8)
        delta = score - current_score

        if delta <= 0 or random.random() < min(1.0, math.exp(-delta / max(temperature, 0.01))):
            current_score = score
            if score < best_score:
                best_score = score
                best_adj = adj[:]
        else:
            adj = old_adj

        temperature *= cooling

    # Final polish: try to improve best_adj with more careful search
    if time.time() < deadline - 10:
        adj = best_adj[:]
        # Try removing edges that don't help (i.e., removing them doesn't increase max triangle-free size)
        for _ in range(200):
            if time.time() > deadline - 8:
                break
            if best_score <= 6:
                break
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            if not ((adj[i] >> j) & 1):
                continue
            # Try removing this edge
            adj[i] &= ~(1 << j)
            adj[j] &= ~(1 << i)
            if has_k4(adj):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                continue
            score = max_triangle_free_size_fast(adj, time_limit=0.5)
            if score > best_score:
                # Removing this edge made things worse, restore
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
            else:
                best_score = score
                best_adj = adj[:]

    return edges_from_adj(best_adj)
# EVOLVE-BLOCK-END
