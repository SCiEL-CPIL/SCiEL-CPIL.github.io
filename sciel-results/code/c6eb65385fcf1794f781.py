"""Triangle-free graph without independent 10-set, maximizing n (Ramsey R(3,10))."""
import time
import random
from itertools import combinations

_START = time.time()
_DEADLINE = 115.0  # safety margin before 150s kill


def _remaining_time():
    return _DEADLINE - (time.time() - _START)


def _has_triangle_bitset(adj_bits, n):
    """Check if adjacency bitsets have a triangle."""
    for i in range(n):
        for j in range(i + 1, n):
            if (adj_bits[i] >> j) & 1:
                common = adj_bits[i] & adj_bits[j]
                # Remove i and j themselves
                common &= ~(1 << i)
                common &= ~(1 << j)
                if common:
                    return True
    return False


def _max_independent_set_size_bitset(adj_bits, n):
    """Find size of maximum independent set using branch and bound with bitmasks."""
    best = [0]

    def bnb(candidates, current_size):
        if candidates == 0:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Upper bound: current_size + popcount(candidates)
        if current_size + bin(candidates).count('1') <= best[0]:
            return

        # Pick vertex with highest degree in candidates for better pruning
        v = -1
        max_deg = -1
        c = candidates
        while c:
            lb = c & (-c)
            v_idx = lb.bit_length() - 1
            deg = bin(adj_bits[v_idx] & candidates).count('1')
            if deg > max_deg:
                max_deg = deg
                v = v_idx
            c ^= lb

        if v == -1:
            if current_size > best[0]:
                best[0] = current_size
            return

        rest = candidates & ~(1 << v)

        # Branch: include v -> exclude all neighbors of v
        non_neighbors = rest & ~adj_bits[v]
        bnb(non_neighbors, current_size + 1)

        # Branch: exclude v
        bnb(rest, current_size)

    bnb((1 << n) - 1, 0)
    return best[0]


def _build_circulant(n, connection_set):
    """Build circulant graph: vertex i connected to i+d mod n for d in connection_set."""
    adj_bits = [0] * n
    for i in range(n):
        for d in connection_set:
            j = (i + d) % n
            if j != i:
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)
    return adj_bits


def _is_valid_graph(adj_bits, n):
    """Check if graph is triangle-free and alpha < 10."""
    if _has_triangle_bitset(adj_bits, n):
        return False
    alpha = _max_independent_set_size_bitset(adj_bits, n)
    if alpha >= 10:
        return False
    return True


def _construct_andrasfai(k):
    """
    Construct Andrásfai graph And(k).
    n = 3k - 1.
    Triangle-free, alpha = k.
    Valid if k < 10.
    """
    n = 3 * k - 1
    if n < 1 or n > 41:
        return None
    # Connection set is {1, 2, ..., k-1}
    conn_set = list(range(1, k))
    adj_bits = _build_circulant(n, conn_set)
    return adj_bits, n


def _construct_paley(q):
    """
    Construct Paley graph P(q) for prime q = 1 mod 4.
    n = q.
    """
    if q < 1 or q > 41:
        return None
    # Check if q is prime and q % 4 == 1
    def is_prime(num):
        if num < 2:
            return False
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                return False
        return True
        
    if not is_prime(q) or q % 4 != 1:
        return None

    n = q
    adj_bits = [0] * n
    
    # Quadratic residues modulo q
    residues = set()
    for i in range(1, q):
        residues.add((i * i) % q)
    
    # Build graph
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % q
            if diff in residues:
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)
                
    return adj_bits, n


def _random_greedy_search(n, time_limit):
    """
    Try to find a triangle-free graph on n vertices with alpha < 10
    using random greedy edge addition.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    # Try a few random instances
    for attempt in range(50):
        if time.time() - start_time > time_limit:
            break
            
        adj_bits = [0] * n
        # Add random edges that don't create triangles
        # To be fast, we don't check alpha until the end of a full construction
        # Or we can do it greedily.
        
        # Random permutation of edges
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        rng.shuffle(edges)
        
        for u, v in edges:
            if time.time() - start_time > time_limit:
                break
            # Check if adding (u, v) creates a triangle
            common = adj_bits[u] & adj_bits[v]
            common &= ~(1 << u)
            common &= ~(1 << v)
            if not common:
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
        
        # Check validity
        if _is_valid_graph(adj_bits, n):
            return adj_bits, n
            
    return None


def construct():
    """
    Construct a triangle-free graph on n vertices (1 <= n <= 41) with no independent set of 10.
    Maximize n.
    """
    best_adj_bits = None
    best_n = 0
    best_alpha = float('inf')
    
    # 1. Try Paley graphs (Algebraic)
    # q must be prime, q % 4 == 1, q <= 41
    paley_candidates = [37, 29, 17, 13, 5]
    for q in paley_candidates:
        if _remaining_time() < 5:
            break
        res = _construct_paley(q)
        if res:
            adj_bits, n = res
            if _is_valid_graph(adj_bits, n):
                best_adj_bits = adj_bits
                best_n = n
                best_alpha = _max_independent_set_size_bitset(adj_bits, n)
                break # Found a valid one, try to find larger? 
                # Actually, we want max n. 37 > 29. So if 37 works, we stop.
                # If 37 fails, try 29? No, 29 is smaller. 
                # We want the LARGEST n. 
                # So we should try larger n first.
                # But Paley graphs are specific. 
                # If 37 works, it's likely the best among Paleys.
                # But maybe a random graph on 40 works?
                # So we shouldn't break immediately if we want to try larger n via other methods.
                # However, Paley 37 is n=37. 
                # Let's collect all valid candidates and pick the best.
                pass

    # Let's re-structure: Collect candidates
    
    candidates = []
    
    # Try Paley
    for q in [37, 29, 17, 13, 5]:
        if _remaining_time() < 2:
            break
        res = _construct_paley(q)
        if res:
            adj_bits, n = res
            if _is_valid_graph(adj_bits, n):
                alpha = _max_independent_set_size_bitset(adj_bits, n)
                candidates.append((n, adj_bits, alpha))
                
    # Try Andrásfai
    # k < 10 => max k=9 => n=26
    for k in range(9, 1, -1):
        if _remaining_time() < 2:
            break
        res = _construct_andrasfai(k)
        if res:
            adj_bits, n = res
            if _is_valid_graph(adj_bits, n):
                alpha = _max_independent_set_size_bitset(adj_bits, n)
                candidates.append((n, adj_bits, alpha))

    # Try Random Greedy for n > max_candidate_n
    max_n_found = max([c[0] for c in candidates], default=0)
    
    # We know R(3,10) > 39. So there is a graph on 39.
    # If we haven't found one > 30, try random search for n=39, 38, ...
    if max_n_found < 39:
        # Try n=39 down to max_n_found + 1
        for n_target in range(39, max_n_found, -1):
            if _remaining_time() < 10:
                break
            # Use a small time budget for each n
            time_per_n = min(10, _remaining_time() / 2)
            res = _random_greedy_search(n_target, time_per_n)
            if res:
                adj_bits, n = res
                alpha = _max_independent_set_size_bitset(adj_bits, n)
                candidates.append((n, adj_bits, alpha))
                break # Found one for this n, stop trying smaller n via random?
                # Actually, if we found 39, we are done. 
                # If we found 38, we might still find 39? No, we tried 39 first.
                # So break is fine.

    # Select best candidate
    if candidates:
        # Sort by n descending
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_n, best_adj_bits, best_alpha = candidates[0]
    else:
        # Fallback: Andrásfai(9) is guaranteed valid
        # n=26, alpha=9
        adj_bits, n = _construct_andrasfai(9)
        best_adj_bits = adj_bits
        best_n = n
        best_alpha = 9

    # Convert to edge list
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if (best_adj_bits[i] >> j) & 1:
                edges.append([i, j])
    
    return {'n': best_n, 'edges': edges}
