import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    # Q3 edges: 000-001, 000-010, 000-100, 001-011, 001-101, 010-011, 010-110,
    # 011-111, 100-101, 100-110, 101-111, 110-111
    q3_edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
    q3_adj = [0]*8
    for u,v in q3_edges:
        q3_adj[u] |= (1<<v)
        q3_adj[v] |= (1<<u)
    q3_deg = [3]*8

    # All 8! = 40320 permutations of 8 vertices, precomputed as tuples
    all_perms = list(itertools.permutations(range(8)))

    def has_q3(adj, subset):
        """Check if induced subgraph on subset contains Q3. adj is list of 16 ints (bitmasks)."""
        # Quick edge count: need at least 12 edges
        ec = 0
        for i in range(8):
            ec += bin(adj[subset[i]] & ((1<<subset[i+1]) - 1) & ~((1<<subset[i]) - 1)).count('1') if i < 7 else 0
        # Actually let's do it properly
        ec = 0
        for i in range(8):
            for j in range(i+1, 8):
                if adj[subset[i]] >> subset[j] & 1:
                    ec += 1
        if ec < 12:
            return False

        # Degree check
        for i in range(8):
            d = 0
            for j in range(8):
                if i != j and (adj[subset[i]] >> subset[j] & 1):
                    d += 1
            if d < 3:
                return False

        # Try permutations
        for perm in all_perms:
            # perm maps Q3 vertex k -> subset position perm[k]
            # Check all 12 edges
            ok = True
            for u, v in q3_edges:
                a = subset[perm[u]]
                b = subset[perm[v]]
                if not (adj[a] >> b & 1):
                    ok = False
                    break
            if ok:
                return True
        return False

    def find_q3(adj):
        """Find one Q3 in the graph. Returns subset tuple or None."""
        for subset in itertools.combinations(range(16), 8):
            if has_q3(adj, subset):
                return subset
        return None

    def count_q3s(adj, max_check=200):
        """Count Q3s up to max_check subsets for speed."""
        count = 0
        checked = 0
        for subset in itertools.combinations(range(16), 8):
            if has_q3(adj, subset):
                count += 1
                if count > max_check:
                    return count
            checked += 1
        return count

    def edge_count(adj):
        return sum(bin(a).count('1') for a in adj) // 2

    def to_edges(adj):
        edges = []
        for i in range(16):
            for j in range(i+1, 16):
                if adj[i] >> j & 1:
                    edges.append([i, j])
        return edges

    def make_adj(edges):
        adj = [0]*16
        for u, v in edges:
            adj[u] |= (1<<v)
            adj[v] |= (1<<u)
        return adj

    # Phase 1: Multiple random starts with repair
    best_adj = [0]*16
    best_edges = 0

    for attempt in range(10):
        if time.time() > deadline:
            break

        # Random dense start
        adj = [0]*16
        for i in range(16):
            for j in range(i+1, 16):
                if random.random() < 0.55:
                    adj[i] |= (1<<j)
                    adj[j] |= (1<<i)

        # Repair: remove edges from found Q3s
        repair_iters = 0
        while time.time() < deadline:
            subset = find_q3(adj)
            if subset is None:
                break
            # Pick a random edge within the subset to remove
            edges_in_sub = []
            for i in range(8):
                for j in range(i+1, 8):
                    if adj[subset[i]] >> subset[j] & 1:
                        edges_in_sub.append((subset[i], subset[j]))
            if not edges_in_sub:
                break
            u, v = random.choice(edges_in_sub)
            adj[u] &= ~(1<<v)
            adj[v] &= ~(1<<u)
            repair_iters += 1
            if repair_iters > 500:
                break

        # Phase 2: Greedy add edges
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle order for diversity
            pairs = [(i,j) for i in range(16) for j in range(i+1,16)]
            random.shuffle(pairs)
            for i, j in pairs:
                if time.time() > deadline:
                    break
                if adj[i] >> j & 1:
                    continue
                # Try adding
                adj[i] |= (1<<j)
                adj[j] |= (1<<i)
                if find_q3(adj) is not None:
                    adj[i] &= ~(1<<j)
                    adj[j] &= ~(1<<i)
                else:
                    improved = True

        ec = edge_count(adj)
        if ec > best_edges:
            best_edges = ec
            best_adj = adj[:]

    # Phase 3: Simulated annealing on best_adj
    if time.time() < deadline - 20:
        adj = best_adj[:]
        cur_edges = best_edges
        T = 5.0
        T_min = 0.1
        cool_rate = 0.999

        sa_iters = 0
        while T > T_min and time.time() < deadline - 5:
            # Pick a random pair
            i = random.randint(0, 15)
            j = random.randint(0, 15)
            if i == j:
                continue
            if i > j:
                i, j = j, i

            sa_iters += 1
            if sa_iters % 5000 == 0 and time.time() > deadline:
                break

            had_edge = adj[i] >> j & 1

            if had_edge:
                # Remove edge
                adj[i] &= ~(1<<j)
                adj[j] &= ~(1<<i)
                new_edges = cur_edges - 1
                delta = -1
            else:
                # Add edge
                adj[i] |= (1<<j)
                adj[j] |= (1<<i)
                # Check if Q3-free
                if find_q3(adj) is not None:
                    # Revert
                    adj[i] &= ~(1<<j)
                    adj[j] &= ~(1<<i)
                    continue
                new_edges = cur_edges + 1
                delta = 1

            # Accept if improvement or with probability exp(delta/T) for worsening
            if delta >= 0:
                cur_edges = new_edges
            else:
                import math
                if random.random() < math.exp(delta / T):
                    cur_edges = new_edges
                else:
                    # Revert
                    if had_edge:
                        adj[i] |= (1<<j)
                        adj[j] |= (1<<i)
                    else:
                        adj[i] &= ~(1<<j)
                        adj[j] &= ~(1<<i)

            T *= cool_rate

            if cur_edges > best_edges:
                best_edges = cur_edges
                best_adj = adj[:]

    # Final greedy add pass
    if time.time() < deadline - 3:
        adj = best_adj[:]
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            pairs = [(i,j) for i in range(16) for j in range(i+1,16)]
            random.shuffle(pairs)
            for i, j in pairs:
                if adj[i] >> j & 1:
                    continue
                adj[i] |= (1<<j)
                adj[j] |= (1<<i)
                if find_q3(adj) is None:
                    improved = True
                    best_edges = edge_count(adj)
                    best_adj = adj[:]
                else:
                    adj[i] &= ~(1<<j)
                    adj[j] &= ~(1<<i)

    return to_edges(best_adj)
