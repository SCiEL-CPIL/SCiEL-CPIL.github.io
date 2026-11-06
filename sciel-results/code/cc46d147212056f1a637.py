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

    def count_triangles(adj):
        """Count the number of triangles in the graph."""
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    common = adj[i] & adj[j] & ~((1 << (j + 1)) - 1)
                    count += bin(common).count('1')
        return count

    def max_triangle_free_size(adj, time_limit=3.0):
        """Find the largest induced triangle-free subgraph using branch and bound."""
        best = [0]
        
        # Order vertices by degree descending
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
            
            # Pruning: even if we include all remaining, can't beat best
            if current_size + (n - idx) <= best[0]:
                return

            v = order[idx]

            # Check if v can be added to current set
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

    def greedy_edge_insertion(seed_val, num_restarts=1):
        """
        Greedy edge insertion: start from empty graph, repeatedly add the edge
        that creates the most new triangles while keeping the graph K4-free.
        """
        random.seed(seed_val)
        best_adj = None
        best_score = 18
        best_triangles = 0

        for restart in range(num_restarts):
            adj = [0] * n
            # Start with a random sparse base to break symmetry
            if restart > 0:
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < 0.1:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)

            # Greedily add edges
            for _ in range(200):  # Max 200 edges to add
                if time.time() > deadline - 30:
                    break

                # Find all candidate edges that can be added without creating K4
                best_gain = -1
                best_edge = None
                candidates = []

                for i in range(n):
                    for j in range(i + 1, n):
                        if (adj[i] >> j) & 1:
                            continue

                        # Check if adding edge (i,j) creates a K4
                        # A K4 would form if there exist k,l such that:
                        # k is common neighbor of i,j (in current graph) and l is also common neighbor
                        # and k-l is an edge, and i-k, i-l, j-k, j-l are all edges
                        # Actually: adding (i,j) creates K4 iff there exist k,l both adjacent to both i and j
                        # and k-l is an edge
                        common_ij = adj[i] & adj[j]
                        # Check if any two vertices in common_ij are adjacent
                        creates_k4 = False
                        tmp = common_ij
                        while tmp:
                            bit = tmp & (-tmp)
                            k = bit.bit_length() - 1
                            rest = tmp & ~bit
                            if (adj[k] & rest) != 0:
                                creates_k4 = True
                                break
                            tmp = rest

                        if creates_k4:
                            continue

                        # Calculate triangle gain: number of common neighbors of i and j
                        gain = bin(common_ij).count('1')

                        if gain > best_gain:
                            best_gain = gain
                            best_edge = (i, j)
                        elif gain == best_gain and gain > 0:
                            candidates.append((i, j))

                if best_edge is None:
                    break

                # If multiple edges have same gain, pick randomly among them for diversity
                if candidates and random.random() < 0.5:
                    i, j = random.choice(candidates)
                else:
                    i, j = best_edge

                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

            # Evaluate this graph
            if has_k4(adj):
                continue

            tri_count = count_triangles(adj)
            if tri_count > best_triangles:
                best_triangles = tri_count
                best_adj = adj[:]

            # Also evaluate triangle-free size
            score = max_triangle_free_size(adj, time_limit=2.0)
            if score < best_score:
                best_score = score
                best_adj = adj[:]

        return best_adj, best_score

    def local_search_improve(adj, current_score, time_budget=30.0):
        """Local search: try edge flips to reduce max triangle-free size."""
        t0 = time.time()
        best_adj = adj[:]
        best_score = current_score
        temperature = 0.5
        cooling = 0.9995

        for iteration in range(20000):
            if time.time() - t0 > time_budget:
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

        return best_adj, best_score

    random.seed(42)

    # Phase 1: Greedy edge insertion with multiple seeds
    best_adj = None
    best_score = 18

    for seed in range(10):
        if time.time() > deadline - 60:
            break
        adj, score = greedy_edge_insertion(seed, num_restarts=1)
        if adj is not None:
            if score < best_score:
                best_score = score
                best_adj = adj[:]

    if best_adj is None:
        # Fallback: complete tripartite K_{6,6,6}
        best_adj = [0] * n
        parts3 = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        for i in range(3):
            for j in range(i + 1, 3):
                for a in parts3[i]:
                    for b in parts3[j]:
                        best_adj[a] |= (1 << b)
                        best_adj[b] |= (1 << a)
        best_score = max_triangle_free_size(best_adj, time_limit=2.0)

    # Phase 2: Local search improvement
    if time.time() < deadline - 20:
        adj = best_adj[:]
        current_score = best_score
        best_adj, best_score = local_search_improve(adj, current_score, time_budget=40.0)

    # Phase 3: Final polish - try removing edges that don't increase max triangle-free size
    if time.time() < deadline - 10 and best_score > 6:
        adj = best_adj[:]
        for _ in range(300):
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

    # Phase 4: Try adding edges that reduce max triangle-free size
    if time.time() < deadline - 5 and best_score > 6:
        adj = best_adj[:]
        for _ in range(200):
            if time.time() > deadline - 3:
                break
            if best_score <= 6:
                break
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            if (adj[i] >> j) & 1:
                continue
            # Try adding this edge
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            if has_k4(adj):
                adj[i] &= ~(1 << j)
                adj[j] &= ~(1 << i)
                continue
            score = max_triangle_free_size_fast(adj, time_limit=0.5)
            if score < best_score:
                best_score = score
                best_adj = adj[:]
            else:
                adj[i] &= ~(1 << j)
                adj[j] &= ~(1 << i)

    return edges_from_adj(best_adj)
# EVOLVE-BLOCK-END
