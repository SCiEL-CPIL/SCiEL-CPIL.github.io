import time
import random
import sys

_START = time.time()
_DEADLINE = 145.0  # Safety margin before 150s kill

def _remaining_time():
    return _DEADLINE - (time.time() - _START)

def _bit_count(x):
    return bin(x).count('1')

def _has_triangle_bitmask(adj, n):
    """Check if adjacency bitmask list has a triangle."""
    for i in range(n):
        ni = adj[i]
        j_mask = ni
        while j_mask:
            lsb = j_mask & -j_mask
            j = lsb.bit_length() - 1
            j_mask ^= lsb
            if j > i:
                if adj[i] & adj[j]:
                    return True
    return False

def _greedy_color_bound(available, adj):
    """
    Upper bound on the size of the maximum independent set in the subgraph induced by 'available'.
    Uses greedy coloring on the subgraph.
    """
    if not available:
        return 0
    unassigned = available
    k = 0
    while unassigned:
        k += 1
        # Pick a vertex from unassigned
        lsb = unassigned & -unassigned
        v = lsb.bit_length() - 1
        unassigned ^= lsb
        # Remove v and its neighbors from unassigned?
        # In independent set context:
        # We are coloring the graph G[available].
        # A color class is an independent set in G.
        # So we pick v, and then we can only pick non-neighbors of v in the next step of the *same* color?
        # No, greedy coloring:
        # Color 1: Pick v. Remove v and its neighbors from 'unassigned' for this color?
        # No, if we pick v for color 1, we cannot pick its neighbors for color 1.
        # So the candidates for color 1 are unassigned \ N(v).
        # Let's implement standard greedy coloring bound.
        candidates = unassigned & ~adj[v]
        # Actually, the standard bound is:
        # While unassigned is not empty:
        #   Pick a vertex v
        #   Let C be a maximal independent set in G[unassigned] containing v? No.
        #   Let C be the set of vertices in unassigned that are non-adjacent to v?
        #   Remove C from unassigned?
        #   Increment k.
        # This gives an upper bound on the chromatic number, which is an upper bound on the independence number?
        # Wait. $\chi(G) \ge \omega(G)$. $\alpha(G) \le n / \chi(G)$? No.
        # The bound is: $\alpha(G) \le$ number of colors in a greedy coloring?
        # Yes, if we color G with k colors, then the largest independent set is at most the size of the largest color class?
        # No, an independent set can take vertices from multiple color classes?
        # No, an independent set in G is a clique in the complement.
        # The standard bound for MIS is:
        # $\alpha(G) \le$ size of the largest color class in a greedy coloring of G?
        # Let's check. If we color G with k colors, each color class is an independent set.
        # So $\alpha(G) \ge \max |C_i|$. This is a lower bound.
        # We need an upper bound.
        # The bound is: $\alpha(G) \le$ number of colors used?
        # No. Consider a star. $\chi=2$, $\alpha=n-1$.
        # The correct bound is: $\alpha(G) \le$ sum of something?
        # Actually, the common bound is:
        # If we greedily color G, and use k colors, then $\alpha(G) \le$ ?
        # Wait, the bound is $\alpha(G) \le$ number of vertices in the largest independent set found? No.
        
        # Let's use the standard "Maximal Independent Set" bound?
        # A simpler bound: $\alpha(G) \le$ size of a maximal clique in the complement? No, that's $\omega(\bar{G}) = \alpha(G)$.
        
        # Let's use the bound: $\alpha(G) \le$ number of colors in a greedy coloring of the *complement*?
        # $\chi(\bar{G}) = \omega(G)$.
        
        # Let's stick to the branch and bound with a simple upper bound:
        # $\alpha(G) \le$ number of vertices in G? (Too weak)
        # $\alpha(G) \le$ size of a greedy maximal independent set? (Lower bound)
        
        # Correct Upper Bound for MIS:
        # Greedy coloring of G. Let the colors be 1..k.
        # It is known that $\alpha(G) \le$ number of colors? NO.
        # Example: $C_5$. $\chi=3$. $\alpha=2$. $2 \le 3$.
        # Example: $K_3$. $\chi=3$. $\alpha=1$. $1 \le 3$.
        # Example: Star $K_{1,3}$. $\chi=2$. $\alpha=3$. $3 \not\le 2$.
        # So $\chi(G)$ is NOT an upper bound for $\alpha(G)$.
        
        # The bound is: $\alpha(G) \le$ size of the largest independent set? (Tautology)
        
        # Let's use the "Greedy Coloring Bound" correctly:
        # If we color G with k colors, then $\alpha(G) \le$ ?
        # Actually, the bound is $\alpha(G) \le$ number of vertices in the graph?
        
        # Let's use a different bound:
        # Pick a vertex v.
        # $\alpha(G) = \max(1 + \alpha(G \setminus (N[v])), \alpha(G \setminus \{v\}))$.
        # This is the recursion.
        # To prune, we need an upper bound on $\alpha(G \setminus \{v\})$.
        # A common upper bound is the size of a maximal independent set in the complement?
        
        # Let's use the "Degree" bound:
        # If the minimum degree of G is $\delta$, then $\alpha(G) \le n \delta / (n-1)$? (Caro-Wei is lower bound).
        
        # Let's use the "Greedy Independent Set" size as a lower bound for the best, and use the number of remaining vertices as the upper bound?
        # That's weak.
        
        # Let's use the bound: $\alpha(G) \le$ size of the largest independent set found by a greedy algorithm? No.
        
        # I will use the bound: $\alpha(G) \le$ number of colors in a greedy coloring of the *complement* graph?
        # $\chi(\bar{G})$ is the clique cover number of G.
        # If G can be covered by k cliques, then any independent set in G can contain at most 1 vertex from each clique.
        # So $\alpha(G) \le k$.
        # Yes! This is the correct bound.
        # So we need to greedily color the *complement* of the available subgraph.
        # Coloring the complement is equivalent to finding a clique cover of the original graph.
        # Greedy clique cover:
        # While unassigned:
        #   Pick a vertex v.
        #   Find a maximal clique in G[unassigned] containing v.
        #   Remove that clique from unassigned.
        #   Increment k.
        # Return k.
        
        # This is expensive.
        # Alternative: Greedy coloring of G (original).
        # If we color G with k colors, does that bound $\alpha(G)$? No.
        
        # Let's use the bound: $\alpha(G) \le$ size of the largest independent set in a greedy maximal independent set? No.
        
        # I will use the simple bound: $\alpha(G) \le$ number of vertices in G.
        # And rely on the recursion to prune.
        # To speed it up, I will use the "Maximal Independent Set" size as a lower bound for `best`.
        
        pass
    return 0

def _max_independent_set_size_fast(adj, n, limit=10):
    """
    Fast exact max independent set using bitset branch and bound.
    Returns min(actual_alpha, limit).
    """
    best = [0]
    full_mask = (1 << n) - 1
    adj_local = adj
    
    def _greedy_clique_cover_bound(available):
        """Upper bound: greedy clique cover of G[available]."""
        unassigned = available
        k = 0
        while unassigned:
            k += 1
            lsb = unassigned & -unassigned
            v = lsb.bit_length() - 1
            unassigned ^= lsb
            # Build a maximal clique containing v in G[available]
            clique = 1 << v
            remaining = unassigned & adj_local[v]
            while remaining:
                lsb_u = remaining & -remaining
                u = lsb_u.bit_length() - 1
                remaining ^= lsb_u
                clique |= lsb_u
                remaining &= adj_local[u]
            unassigned &= ~clique
        return k

    def _bnb(available, current_size):
        if not available:
            if current_size > best[0]:
                best[0] = current_size
            return

        if current_size + _bit_count(available) <= best[0]:
            return

        bound = _greedy_clique_cover_bound(available)
        if current_size + bound <= best[0]:
            return

        # Pick vertex with max degree in G[available]
        v_mask = available
        max_deg = -1
        v = -1
        while v_mask:
            lsb = v_mask & -v_mask
            u = lsb.bit_length() - 1
            v_mask ^= lsb
            deg = _bit_count(adj_local[u] & available)
            if deg > max_deg:
                max_deg = deg
                v = u

        if v == -1:
            return

        # Branch 1: Include v
        _bnb(available & ~(1 << v) & ~adj_local[v], current_size + 1)

        # Branch 2: Exclude v
        _bnb(available & ~(1 << v), current_size)

    _bnb(full_mask, 0)
    return min(best[0], limit)

def _validate_bitmask(adj, n):
    if _has_triangle_bitmask(adj, n):
        return False
    if _max_independent_set_size_fast(adj, n, limit=10) >= 10:
        return False
    return True

def _cyclic_graph(n, shifts):
    adj = [0] * n
    shift_set = set()
    for s in shifts:
        shift_set.add(s % n)
        shift_set.add((-s) % n)
    shift_set.discard(0)
    for i in range(n):
        for s in shift_set:
            j = (i + s) % n
            if j > i:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _shifts_triangle_free(n, shifts):
    dist_set = set()
    for s in shifts:
        dist_set.add(s % n)
        dist_set.add((-s) % n)
    dist_set.discard(0)
    for a in shifts:
        for b in shifts:
            c = (a + b) % n
            d = (a - b) % n
            if c in dist_set and c != 0:
                return True
            if d in dist_set and d != 0:
                return True
    return False

def _try_circulant_systematic(n, max_shifts=None):
    if max_shifts is None:
        max_shifts = n // 2
    half = n // 2
    best_adj = None
    best_alpha = float('inf')

    for seed in range(100):
        if _remaining_time() < 5:
            break
        rng = random.Random(9999 + seed)
        available = list(range(1, half + 1))
        rng.shuffle(available)
        current_shifts = []
        for s in available:
            trial = current_shifts + [s]
            if not _shifts_triangle_free(n, trial):
                current_shifts.append(s)
        if not current_shifts:
            continue
        adj = _cyclic_graph(n, current_shifts)
        if _has_triangle_bitmask(adj, n):
            continue
        alpha = _max_independent_set_size_fast(adj, n, limit=10)
        if alpha < 10:
            return adj
        if alpha < best_alpha:
            best_alpha = alpha
            best_adj = adj

    for attempt in range(500):
        if _remaining_time() < 3:
            break
        rng = random.Random(4242 + attempt)
        num_shifts = rng.randint(2, min(half, 18))
        shifts = rng.sample(range(1, half + 1), num_shifts)
        if _shifts_triangle_free(n, shifts):
            continue
        adj = _cyclic_graph(n, shifts)
        if _has_triangle_bitmask(adj, n):
            continue
        alpha = _max_independent_set_size_fast(adj, n, limit=10)
        if alpha < 10:
            return adj
        if alpha < best_alpha:
            best_alpha = alpha
            best_adj = adj

    return best_adj

def _greedy_triangle_free_bitmask(n, rng):
    adj = [0] * n
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(edges)
    for i, j in edges:
        if _remaining_time() < 2:
            break
        if adj[i] & adj[j]:
            continue
        adj[i] |= (1 << j)
        adj[j] |= (1 << i)
    return adj

def _local_search_bitmask(adj, n, rng, max_iter=2000):
    adj = adj[:]
    alpha = _max_independent_set_size_fast(adj, n, limit=10)
    if alpha < 10:
        return adj, alpha

    def _find_one_mis(adj, n):
        remaining = (1 << n) - 1
        selected = []
        while remaining:
            best_v = -1
            best_deg = n + 1
            for v in range(n):
                if remaining & (1 << v):
                    deg = _bit_count(adj[v] & remaining)
                    if deg < best_deg:
                        best_deg = deg
                        best_v = v
            if best_v == -1:
                break
            selected.append(best_v)
            remaining &= ~(1 << best_v)
            remaining &= ~adj[best_v]
        return selected

    no_improve_count = 0
    for _ in range(max_iter):
        if _remaining_time() < 3:
            break
        target_is = _find_one_mis(adj, n)
        if len(target_is) < 10:
            alpha = _max_independent_set_size_fast(adj, n, limit=10)
            if alpha < 10:
                break
        addable_in_is = []
        for idx_i in range(len(target_is)):
            for idx_j in range(idx_i + 1, len(target_is)):
                i, j = target_is[idx_i], target_is[idx_j]
                if not (adj[i] & (1 << j)):
                    if not (adj[i] & adj[j]):
                        addable_in_is.append((i, j))
        if not addable_in_is:
            no_improve_count += 1
            if no_improve_count > 30:
                break
            continue
        improved = False
        for _ in range(15):
            if _remaining_time() < 2:
                break
            i, j = rng.choice(addable_in_is)
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            new_alpha = _max_independent_set_size_fast(adj, n, limit=10)
            if new_alpha < alpha:
                alpha = new_alpha
                improved = True
                no_improve_count = 0
                if alpha < 10:
                    break
            else:
                adj[i] &= ~(1 << j)
                adj[j] &= ~(1 << i)
        if alpha < 10:
            break
        if not improved:
            no_improve_count += 1
            if no_improve_count > 40:
                break
    return adj, alpha

def _try_construction_bitmask(n, seed, max_local_iters=1000):
    rng = random.Random(seed)
    adj = _greedy_triangle_free_bitmask(n, rng)
    alpha = _max_independent_set_size_fast(adj, n, limit=10)
    if alpha < 10:
        return adj, alpha
    if _remaining_time() > 5:
        adj_ls, alpha_ls = _local_search_bitmask(adj, n, rng, max_local_iters)
        if alpha_ls < 10:
            return adj_ls, alpha_ls
        if alpha_ls < alpha:
            adj, alpha = adj_ls, alpha_ls
    for attempt in range(5):
        if _remaining_time() < 5:
            break
        rng2 = random.Random(seed * 1000 + attempt)
        adj2 = _greedy_triangle_free_bitmask(n, rng2)
        alpha2 = _max_independent_set_size_fast(adj2, n, limit=10)
        if alpha2 < 10:
            return adj2, alpha2
        if alpha2 < alpha and _remaining_time() > 5:
            adj_ls2, alpha_ls2 = _local_search_bitmask(adj2, n, rng2, max_local_iters)
            if alpha_ls2 < alpha:
                adj, alpha = adj_ls2, alpha_ls2
    return None

def construct():
    best_adj = None
    best_n = 0
    best_alpha = float('inf')

    # Prioritize n=40
    for n_target in [40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30]:
        if _remaining_time() < 8:
            break
        if best_n >= n_target:
            continue
        adj = _try_circulant_systematic(n_target)
        if adj is not None:
            alpha = _max_independent_set_size_fast(adj, n_target, limit=10)
            if alpha < 10 and n_target > best_n:
                best_adj = adj
                best_n = n_target
                best_alpha = alpha
                break

    # If circulant didn't work for 40, try local search on 40
    if best_n < 40 and _remaining_time() > 10:
        for seed in range(20):
            if _remaining_time() < 5:
                break
            result = _try_construction_bitmask(40, seed)
            if result is not None:
                adj_res, alpha_res = result
                if alpha_res < 10:
                    best_adj = adj_res
                    best_n = 40
                    best_alpha = alpha_res
                    break
                elif alpha_res < best_alpha:
                    best_adj = adj_res
                    best_n = 40
                    best_alpha = alpha_res

    # Fallback to 39
    if best_n < 39:
        for n_target in [39, 38, 37]:
            if _remaining_time() < 5:
                break
            if best_n >= n_target:
                continue
            for seed in range(20):
                if _remaining_time() < 3:
                    break
                result = _try_construction_bitmask(n_target, seed)
                if result is not None:
                    adj_res, alpha_res = result
                    if alpha_res < 10:
                        best_adj = adj_res
                        best_n = n_target
                        best_alpha = alpha_res
                        break
                    elif alpha_res < best_alpha:
                        best_adj = adj_res
                        best_n = n_target
                        best_alpha = alpha_res
            if best_n >= n_target:
                break

    if best_adj is None:
        n = 26
        rng = random.Random(42)
        adj = _greedy_triangle_free_bitmask(n, rng)
        best_adj = adj
        best_n = n

    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])

    return {'n': best_n, 'edges': edges}
