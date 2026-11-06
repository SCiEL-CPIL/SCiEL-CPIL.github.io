import time
import random
import numpy as np
from itertools import combinations

def generate_paley(q):
    if q % 4 != 1:
        return None
    squares = set()
    for i in range(1, q):
        squares.add((i * i) % q)
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in squares:
                edges.append([i, j])
    return {'n': q, 'edges': edges}

def generate_srg_seed(n=35, k=10, lam=3, mu=4):
    """Generate a random regular graph and iteratively improve to approach SRG parameters."""
    adj = [set() for _ in range(n)]
    # Start with random k-regular graph
    for _ in range(200):
        adj = [set() for _ in range(n)]
        perm = list(range(n))
        random.shuffle(perm)
        # Try to build k-regular
        for i in range(n):
            u = perm[i]
            candidates = [v for v in range(n) if v != u and v not in adj[u] and len(adj[v]) < k]
            if len(candidates) < k:
                continue
            random.shuffle(candidates)
            for v in candidates:
                if len(adj[u]) >= k:
                    break
                adj[u].add(v)
                adj[v].add(u)
        # Check regularity
        if all(len(adj[i]) == k for i in range(n)):
            break

    # Improve: minimize deviation from SRG parameters
    for iteration in range(500):
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        is_edge = v in adj[u]
        if is_edge:
            adj[u].remove(v)
            adj[v].remove(u)
        else:
            adj[u].add(v)
            adj[v].add(u)

        # Check validity
        if has_k4_fast(adj, n) or has_independent_set_fast(adj, n, 6):
            if is_edge:
                adj[u].add(v)
                adj[v].add(u)
            else:
                adj[u].remove(v)
                adj[v].remove(u)
            continue

        # Check if it's still "reasonable" (not too many triangles or independent sets)
        tri_count = count_triangles(adj, n)
        if tri_count > 200:  # too many triangles
            if is_edge:
                adj[u].add(v)
                adj[v].add(u)
            else:
                adj[u].remove(v)
                adj[v].remove(u)
            continue

    return adj

def build_adjacency(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def get_bitmasks(adj, n):
    masks = [0] * n
    for u in range(n):
        mask = 0
        for v in adj[u]:
            mask |= (1 << v)
        masks[u] = mask
    return masks

def has_k4_fast(adj, n):
    """Fast K4 check using bitmasks."""
    masks = get_bitmasks(adj, n)
    for u in range(n):
        mu = masks[u]
        for v in range(u + 1, n):
            if not (mu & (1 << v)):
                continue
            # u-v edge exists
            common_uv = mu & masks[v]
            # Remove u and v from common
            common_uv &= ~(1 << u)
            common_uv &= ~(1 << v)
            # Need two adjacent vertices in common_uv
            cw = common_uv
            while cw:
                lsb = cw & (-cw)
                w = lsb.bit_length() - 1
                cw ^= lsb
                # Check if w is adjacent to any vertex in common_uv \ {w}
                rest = common_uv & ~(1 << w)
                if masks[w] & rest:
                    return True
    return False

def has_independent_set_fast(adj, n, k):
    """Fast independent set check using bitmasks with aggressive pruning."""
    if n < k:
        return False
    masks = get_bitmasks(adj, n)

    def backtrack(start, forbidden, size):
        if size == k:
            return True
        remaining = k - size
        # Count available vertices
        avail = 0
        for v in range(start, n):
            if not (forbidden & (1 << v)):
                avail += 1
        if avail < remaining:
            return False

        # Greedy bound: use coloring-like bound
        # Simple: try vertices in order
        for v in range(start, n):
            if forbidden & (1 << v):
                continue
            if backtrack(v + 1, forbidden | masks[v], size + 1):
                return True
        return False

    return backtrack(0, 0, 0)

def count_triangles(adj, n):
    """Count triangles in the graph."""
    masks = get_bitmasks(adj, n)
    count = 0
    for u in range(n):
        mu = masks[u]
        for v in range(u + 1, n):
            if not (mu & (1 << v)):
                continue
            common = mu & masks[v]
            common &= ~(1 << u)
            common &= ~(1 << v)
            count += bin(common).count('1')
    return count // 3

def get_independent_sets_size_k(adj, n, k, limit=500):
    """Return list of independent sets of size k."""
    if n < k:
        return []
    masks = get_bitmasks(adj, n)
    sets = []

    def backtrack(start, current_set, neighbor_mask):
        if len(sets) >= limit:
            return
        if len(current_set) == k:
            sets.append(tuple(current_set))
            return
        remaining = k - len(current_set)
        if n - start < remaining:
            return
        for v in range(start, n):
            if not (neighbor_mask & (1 << v)):
                current_set.append(v)
                backtrack(v + 1, current_set, neighbor_mask | masks[v])
                current_set.pop()
                if len(sets) >= limit:
                    return

    backtrack(0, [], 0)
    return sets

def compute_spectral_gap(adj, n):
    """Compute the spectral gap (lambda_1 - lambda_2) of the adjacency matrix."""
    if n <= 1:
        return 0.0
    A = np.zeros((n, n))
    for u in range(n):
        for v in adj[u]:
            A[u, v] = 1.0
    try:
        eigenvalues = np.linalg.eigvalsh(A)
        # Sort in descending order
        eigenvalues = np.sort(eigenvalues)[::-1]
        # Spectral gap: difference between largest and second largest
        return eigenvalues[0] - eigenvalues[1]
    except:
        return 0.0

def try_add_vertex(adj, n, deadline, masks):
    """Find a valid set of neighbors for a new vertex."""
    ind_sets_5 = get_independent_sets_size_k(adj, n, 5)

    if not ind_sets_5:
        N = set()
        remaining = list(range(n))
        random.shuffle(remaining)
        for v in remaining:
            neigh_in_N = [u for u in N if u in adj[v]]
            triangle_formed = False
            for i in range(len(neigh_in_N)):
                for j in range(i + 1, len(neigh_in_N)):
                    if neigh_in_N[j] in adj[neigh_in_N[i]]:
                        triangle_formed = True
                        break
                if triangle_formed:
                    break
            if not triangle_formed:
                N.add(v)
        return N

    active_sets = [set(s) for s in ind_sets_5]

    def backtrack(N, active_sets_copy):
        if not active_sets_copy:
            return N.copy()
        if time.time() > deadline:
            return None

        best_set_idx = -1
        best_candidates = None
        min_candidates = float('inf')

        indices = list(range(len(active_sets_copy)))
        random.shuffle(indices)

        for idx in indices:
            target_set = active_sets_copy[idx]
            candidates = []
            for v in target_set:
                valid = True
                neigh_v_in_N = [u for u in N if u in adj[v]]
                for i in range(len(neigh_v_in_N)):
                    for j in range(i + 1, len(neigh_v_in_N)):
                        if neigh_in_N[j] in adj[neigh_in_N[i]]:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    candidates.append(v)

            if len(candidates) < min_candidates:
                min_candidates = len(candidates)
                best_set_idx = idx
                best_candidates = candidates
            if min_candidates == 0:
                break

        if min_candidates == 0:
            return None

        random.shuffle(best_candidates)

        for v in best_candidates:
            new_N = N | {v}
            new_active_sets = []
            for s in active_sets_copy:
                if v not in s:
                    new_active_sets.append(s)
            result = backtrack(new_N, new_active_sets)
            if result is not None:
                return result
        return None

    for _ in range(30):
        res = backtrack(set(), [set(s) for s in ind_sets_5])
        if res is not None:
            return res
    return None

def spectral_local_search(adj, n, masks, deadline):
    """Local search guided by spectral gap maximization."""
    improved = False
    best_gap = compute_spectral_gap(adj, n)

    for iteration in range(200):
        if time.time() > deadline:
            break

        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue

        is_edge = v in adj[u]

        if is_edge:
            adj[u].remove(v)
            adj[v].remove(u)
            masks[u] &= ~(1 << v)
            masks[v] &= ~(1 << u)
        else:
            adj[u].add(v)
            adj[v].add(u)
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)

        if has_k4_fast(adj, n) or has_independent_set_fast(adj, n, 6):
            # Revert
            if is_edge:
                adj[u].add(v)
                adj[v].add(u)
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
            else:
                adj[u].remove(v)
                adj[v].remove(u)
                masks[u] &= ~(1 << v)
                masks[v] &= ~(1 << u)
            continue

        # Compute new spectral gap
        new_gap = compute_spectral_gap(adj, n)

        # Accept if spectral gap improves or randomly (simulated annealing-like)
        if new_gap >= best_gap or random.random() < 0.1:
            best_gap = new_gap
            # Try to add a vertex
            new_N = try_add_vertex(adj, n, deadline, masks)
            if new_N is not None:
                adj.append(set(new_N))
                for u2 in new_N:
                    adj[u2].add(n)
                new_mask = 0
                for u2 in new_N:
                    new_mask |= (1 << u2)
                masks.append(new_mask)
                for u2 in new_N:
                    masks[u2] |= (1 << n)
                n += 1
                improved = True
            # Keep the edge change even if we couldn't add a vertex
        else:
            # Revert
            if is_edge:
                adj[u].add(v)
                adj[v].add(u)
                masks[u] |= (1 << v)
                masks[v] |= (1 << u)
            else:
                adj[u].remove(v)
                adj[v].remove(u)
                masks[u] &= ~(1 << v)
                masks[v] &= ~(1 << u)

    return improved, n

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    best_adj = None
    best_n = 0
    best_masks = None

    # Try multiple seed constructions
    seeds = []

    # Seed 1: Paley 17
    g17 = generate_paley(17)
    if g17:
        p17_adj = build_adjacency(17, g17['edges'])
        if not has_k4_fast(p17_adj, 17) and not has_independent_set_fast(p17_adj, 17, 6):
            seeds.append((p17_adj, 17))

    # Seed 2: SRG-like 35-vertex graph
    srg_adj = generate_srg_seed(35, 10, 3, 4)
    if not has_k4_fast(srg_adj, 35) and not has_independent_set_fast(srg_adj, 35, 6):
        seeds.append((srg_adj, 35))

    # Seed 3: Another SRG attempt with different parameters
    srg_adj2 = generate_srg_seed(35, 11, 3, 4)
    if not has_k4_fast(srg_adj2, 35) and not has_independent_set_fast(srg_adj2, 35, 6):
        seeds.append((srg_adj2, 35))

    # Pick the best seed (largest n)
    if seeds:
        seeds.sort(key=lambda x: -x[1])
        adj, n = seeds[0]
        masks = get_bitmasks(adj, n)
    else:
        # Fallback: small graph
        return {'n': 1, 'edges': []}

    # Main loop: try to add vertices
    while n < 36 and time.time() < deadline:
        added = False
        if time.time() > deadline - 5:
            break

        new_N = try_add_vertex(adj, n, deadline, masks)

        if new_N is not None:
            adj.append(set(new_N))
            for u in new_N:
                adj[u].add(n)
            new_mask = 0
            for u in new_N:
                new_mask |= (1 << u)
            masks.append(new_mask)
            for u in new_N:
                masks[u] |= (1 << n)
            n += 1
            added = True

        if not added:
            # Spectral local search
            if time.time() < deadline - 10:
                improved, n = spectral_local_search(adj, n, masks, deadline)
                if improved:
                    continue

            # Final attempt: more aggressive perturbations
            success = False
            for attempt in range(15):
                if time.time() > deadline - 5:
                    break
                if n >= 2:
                    u = random.randint(0, n - 1)
                    v = random.randint(0, n - 1)
                    if u != v:
                        is_edge = v in adj[u]
                        if is_edge:
                            adj[u].remove(v)
                            adj[v].remove(u)
                            masks[u] &= ~(1 << v)
                            masks[v] &= ~(1 << u)
                        else:
                            adj[u].add(v)
                            adj[v].add(u)
                            masks[u] |= (1 << v)
                            masks[v] |= (1 << u)

                        if has_k4_fast(adj, n) or has_independent_set_fast(adj, n, 6):
                            if is_edge:
                                adj[u].add(v)
                                adj[v].add(u)
                                masks[u] |= (1 << v)
                                masks[v] |= (1 << u)
                            else:
                                adj[u].remove(v)
                                adj[v].remove(u)
                                masks[u] &= ~(1 << v)
                                masks[v] &= ~(1 << u)
                        else:
                            new_N = try_add_vertex(adj, n, deadline, masks)
                            if new_N is not None:
                                adj.append(set(new_N))
                                for u2 in new_N:
                                    adj[u2].add(n)
                                new_mask = 0
                                for u2 in new_N:
                                    new_mask |= (1 << u2)
                                masks.append(new_mask)
                                for u2 in new_N:
                                    masks[u2] |= (1 << n)
                                n += 1
                                success = True
                                break
                            else:
                                if is_edge:
                                    adj[u].add(v)
                                    adj[v].add(u)
                                    masks[u] |= (1 << v)
                                    masks[v] |= (1 << u)
                                else:
                                    adj[u].remove(v)
                                    adj[v].remove(u)
                                    masks[u] &= ~(1 << v)
                                    masks[v] &= ~(1 << u)
            if not success:
                break

    # Build edge list
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append([u, v])

    return {'n': n, 'edges': edges}
