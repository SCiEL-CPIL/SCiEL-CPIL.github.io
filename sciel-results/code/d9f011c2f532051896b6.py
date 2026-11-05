"""Triangle-free graph without independent 10-set, maximizing n (Ramsey R(3,10))."""
import time
import random
from itertools import combinations, product

_START = time.time()
_DEADLINE = 115.0  # safety margin before 150s kill


def _remaining_time():
    return _DEADLINE - (time.time() - _START)


def _has_triangle_bitset(adj_bits, n):
    """Check if bitset adjacency has a triangle using bit operations."""
    for i in range(n):
        for j in range(i + 1, n):
            if (adj_bits[i] >> j) & 1:
                # Check if any common neighbor
                common = adj_bits[i] & adj_bits[j]
                if common & ~((1 << (j + 1)) - 1):
                    # There's a common neighbor k > j
                    return True
                # Also check k < i
                if common & ((1 << i) - 1):
                    return True
    return False


def _max_independent_set_size_bitset(adj_bits, n):
    """Find size of maximum independent set using branch and bound with coloring."""
    best = [0]

    def bnb(candidates, current_size):
        if not candidates:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Trivial upper bound
        if current_size + len(candidates) <= best[0]:
            return
        
        v = candidates[0]
        rest = candidates[1:]
        
        neighbors_mask = adj_bits[v]
        non_neighbors = [u for u in rest if not ((neighbors_mask >> u) & 1)]
        bnb(non_neighbors, current_size + 1)
        
        bnb(rest, current_size)

    bnb(list(range(n)), 0)
    return best[0]


def _max_independent_set_fast(adj_bits, n, limit=10):
    """Optimized maximum independent set with early exit at limit."""
    best = [0]
    
    def bnb(candidates, current_size):
        if not candidates:
            if current_size > best[0]:
                best[0] = current_size
                if best[0] >= limit:
                    return True
            return False
        
        if current_size + len(candidates) <= best[0]:
            return False
        
        # Pick vertex with max degree in the induced subgraph
        max_deg = -1
        max_v = candidates[0]
        for v in candidates:
            deg = sum(1 for u in candidates if u != v and ((adj_bits[v] >> u) & 1))
            if deg > max_deg:
                max_deg = deg
                max_v = v
        
        rest = [u for u in candidates if u != max_v]
        
        # Include max_v
        non_neighbors = [u for u in rest if not ((adj_bits[max_v] >> u) & 1)]
        if bnb(non_neighbors, current_size + 1):
            return True
        
        # Exclude max_v
        if bnb(rest, current_size):
            return True
        
        return False

    bnb(list(range(n)), 0)
    return best[0]


def _build_triangle_free_bitset(n, seed=0):
    """
    Build a triangle-free graph using bitset representation.
    Returns list of n integers, each a bitmask of neighbors.
    """
    rng = random.Random(seed)
    adj_bits = [0] * n
    
    edges_order = list(combinations(range(n), 2))
    rng.shuffle(edges_order)
    
    for i, j in edges_order:
        common = adj_bits[i] & adj_bits[j]
        if common:
            continue
        adj_bits[i] |= (1 << j)
        adj_bits[j] |= (1 << i)
    
    return adj_bits


def _is_circulant_triangle_free(n, shifts):
    """Check if circulant graph with given shifts is triangle-free."""
    shift_set = set()
    for s in shifts:
        shift_set.add(s % n)
        shift_set.add((-s) % n)
    # Triangle exists if a+b+c = 0 mod n for a,b,c in shift_set
    # i.e., for any a,b in shift_set, (n-a-b) % n is also in shift_set
    for a in shift_set:
        for b in shift_set:
            c = (-a - b) % n
            if c in shift_set:
                return False
    return True


def _build_circulant(n, shifts):
    """Build circulant graph with given shift set."""
    adj_bits = [0] * n
    for i in range(n):
        for s in shifts:
            j = (i + s) % n
            if j > i:
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)
    return adj_bits


def _enumerate_circulants(n):
    """Enumerate all circulant graphs on n vertices (symmetric shift sets)."""
    # Shifts are from 1 to n//2
    max_shift = n // 2
    all_shifts = list(range(1, max_shift + 1))
    
    for r in range(1, len(all_shifts) + 1):
        for combo in combinations(all_shifts, r):
            shifts = list(combo)
            if _is_circulant_triangle_free(n, shifts):
                yield shifts


def _local_search_bitset(adj_bits, n, seed=0):
    """
    Local search to reduce independence number while maintaining triangle-free.
    """
    rng = random.Random(seed)
    adj_bits = adj_bits[:]
    
    alpha = _max_independent_set_fast(adj_bits, n, limit=10)
    
    if alpha < 10:
        return adj_bits, alpha
    
    all_edges = [(a, b) for a in range(n) for b in range(a + 1, n)]
    
    for _ in range(500):
        if _remaining_time() < 5:
            break
        
        i, j = rng.choice(all_edges)
        if (adj_bits[i] >> j) & 1:
            continue
        
        common = adj_bits[i] & adj_bits[j]
        if common:
            continue
        
        adj_bits[i] |= (1 << j)
        adj_bits[j] |= (1 << i)
        
        new_alpha = _max_independent_set_fast(adj_bits, n, limit=10)
        if new_alpha < alpha:
            alpha = new_alpha
            if alpha < 10:
                break
        else:
            adj_bits[i] &= ~(1 << j)
            adj_bits[j] &= ~(1 << i)
    
    return adj_bits, alpha


def _bitset_to_edges(adj_bits, n):
    """Convert bitset adjacency to edge list."""
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj_bits[i] >> j) & 1:
                edges.append([i, j])
    return edges


def construct():
    """
    Construct a triangle-free graph on n vertices (1 <= n <= 41) with no independent set of 10.
    Maximize n. Uses systematic circulant enumeration.
    """
    best_adj_bits = None
    best_n = 0
    best_alpha = float('inf')
    
    # Phase 1: Systematic circulant enumeration for high n
    for n_target in range(40, 25, -1):
        if _remaining_time() < 10:
            break
        
        found = False
        
        # Try known good constructions first
        # Andrásfai-type: shifts that are not multiples of 3
        if n_target % 3 == 0:
            k = n_target // 3
            shifts = [s for s in range(1, k) if s % 3 != 0]
            if _is_circulant_triangle_free(n_target, shifts):
                adj_bits = _build_circulant(n_target, shifts)
                alpha = _max_independent_set_fast(adj_bits, n_target, limit=10)
                if alpha < 10:
                    best_adj_bits = adj_bits
                    best_n = n_target
                    best_alpha = alpha
                    found = True
        
        # Enumerate circulants
        if not found:
            count = 0
            for shifts in _enumerate_circulants(n_target):
                if _remaining_time() < 5:
                    break
                count += 1
                if count % 10000 == 0 and _remaining_time() < 2:
                    break
                
                adj_bits = _build_circulant(n_target, shifts)
                alpha = _max_independent_set_fast(adj_bits, n_target, limit=10)
                
                if alpha < 10:
                    best_adj_bits = adj_bits
                    best_n = n_target
                    best_alpha = alpha
                    found = True
                    break
                elif alpha < best_alpha:
                    best_alpha = alpha
        
        if found:
            break
    
    # Phase 2: Random + local search as backup
    if best_adj_bits is None or best_n < 30:
        for n_target in range(max(best_n, 25), 41):
            if _remaining_time() < 10:
                break
            
            for seed in range(30):
                if _remaining_time() < 5:
                    break
                
                adj_bits = _build_triangle_free_bitset(n_target, seed)
                alpha = _max_independent_set_fast(adj_bits, n_target, limit=10)
                
                if alpha < 10:
                    if n_target > best_n:
                        best_adj_bits = adj_bits
                        best_n = n_target
                        best_alpha = alpha
                    break
                
                if _remaining_time() > 10:
                    adj_bits_ls, alpha_ls = _local_search_bitset(adj_bits, n_target, seed)
                    if alpha_ls < 10:
                        if n_target > best_n:
                            best_adj_bits = adj_bits_ls
                            best_n = n_target
                            best_alpha = alpha_ls
                        break
    
    # Fallback: Andrásfai graph And(9) - 27 vertices
    if best_adj_bits is None:
        n = 27
        adj_bits = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                d = (j - i) % n
                if 1 <= d <= 8 and d % 3 != 0:
                    adj_bits[i] |= (1 << j)
                    adj_bits[j] |= (1 << i)
        best_adj_bits = adj_bits
        best_n = n
    
    edges = _bitset_to_edges(best_adj_bits, best_n)
    
    return {'n': best_n, 'edges': edges}
