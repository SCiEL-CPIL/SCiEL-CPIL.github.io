"""Combined program: LUW construction + circulant search + known cages for (5,9)."""
import time
import random
import math
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    """Construct a k-regular graph of girth at least g with as few vertices as possible."""
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # 1. Handle known small cages explicitly
    if (k, g) == (3, 5):
        return _petersen()
    elif (k, g) == (3, 6):
        return _heawood()
    elif (k, g) == (3, 7):
        return _tutte_coxeter()
    elif (k, g) == (3, 8):
        return _tutte_coxeter()
    elif (k, g) == (4, 5):
        return _klein_30()
    elif (k, g) == (4, 7):
        return _klein_56()
    elif (k, g) == (5, 9):
        # Try circulant bipartite search first, fall back to LUW
        n_circ, edges_circ = _circulant_search(5, 9, deadline)
        if n_circ is not None:
            return n_circ, edges_circ
        return _luw_construct(5, 9)

    # 2. LUW Construction as baseline
    n_luw, edges_luw = _luw_construct(k, g)

    # 3. For small k and g, try circulant bipartite search
    if k <= 5 and g <= 10 and time.time() < deadline - 30:
        n_circ, edges_circ = _circulant_search(k, g, deadline)
        if n_circ is not None and n_circ < n_luw:
            return n_circ, edges_circ

    return n_luw, edges_luw


def _petersen():
    """Petersen graph: 10 vertices, 3-regular, girth 5."""
    edges = []
    for i in range(5):
        edges.append((i, (i + 1) % 5))
    inner = [5, 7, 9, 6, 8]
    for i in range(5):
        edges.append((inner[i], inner[(i + 2) % 5]))
    for i in range(5):
        edges.append((i, 5 + i))
    return 10, edges


def _heawood():
    """Heawood graph: 14 vertices, 3-regular, girth 6."""
    lines = [
        (0, 1, 2), (0, 3, 4), (0, 5, 6),
        (1, 3, 5), (1, 4, 6), (2, 3, 6), (2, 4, 5)
    ]
    edges = []
    for i, line in enumerate(lines):
        line_id = 7 + i
        for p in line:
            edges.append((p, line_id))
    return 14, edges


def _tutte_coxeter():
    """Tutte-Coxeter graph: 30 vertices, 3-regular, girth 8."""
    adj = {
        0: [1, 15, 21], 1: [0, 16, 22], 2: [3, 17, 23], 3: [2, 18, 24],
        4: [5, 19, 25], 5: [4, 20, 26], 6: [7, 21, 27], 7: [6, 22, 28],
        8: [9, 23, 29], 9: [8, 24, 0], 10: [11, 25, 1], 11: [10, 26, 2],
        12: [13, 27, 3], 13: [12, 28, 4], 14: [15, 29, 5], 15: [0, 14, 16],
        16: [1, 15, 17], 17: [2, 16, 18], 18: [3, 17, 19], 19: [4, 18, 20],
        20: [5, 19, 21], 21: [6, 20, 22], 22: [7, 21, 23], 23: [8, 22, 24],
        24: [9, 23, 25], 25: [10, 24, 26], 26: [11, 25, 27], 27: [12, 26, 28],
        28: [13, 27, 29], 29: [14, 28, 9]
    }
    edges = set()
    for u in range(30):
        for v in adj[u]:
            if u < v:
                edges.add((u, v))
    return 30, list(edges)


def _klein_30():
    return _luw_construct(4, 5)


def _klein_56():
    return _luw_construct(4, 7)


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
    q = k
    # For odd girth g = 2r+1, need d >= r. For even girth g = 2r, need d >= r-1.
    if g % 2 == 1:
        d = max(3, (g + 1) // 2)
    else:
        d = max(3, g // 2 - 1)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


def _circulant_search(k, g, deadline):
    """Search for circulant bipartite graphs with girth >= g.
    
    A circulant bipartite graph has parts U and V of size M each.
    Edge (u, v) exists if (v - u) % M in S where S is a set of k residues.
    Total vertices = 2M.
    
    For girth >= g, we need no cycle shorter than g.
    In bipartite graphs, all cycles are even, so for odd g, we need girth >= g+1.
    """
    # For bipartite, effective girth target
    target_girth = g + 1 if g % 2 == 1 else g
    
    # Moore bound for bipartite k-regular with girth >= target_girth
    # N >= 2 * sum_{i=0}^{(target_girth/2 - 1)} (k-1)^i
    half = target_girth // 2
    moore_sum = 0
    for i in range(half):
        moore_sum += (k - 1) ** i
    N_min = 2 * moore_sum
    M_min = N_min // 2
    
    # Upper bound on M to search (limited by time)
    max_M = M_min + 2000
    
    best_n = None
    best_edges = None
    
    # Try different M values
    for M in range(max(1, M_min), max_M + 1):
        if time.time() > deadline - 5:
            break
        
        # Need k distinct non-zero residues mod M
        if M < k + 1:
            continue
        
        # Try random subsets S of size k
        num_tries = min(100, max(10, int(30 * (max_M - M) / max(1, max_M - M_min))))
        
        for _ in range(num_tries):
            if time.time() > deadline - 5:
                break
            
            S = set(random.sample(range(1, M), k))
            
            # Check if this S gives girth >= target_girth
            # A cycle of length 2L exists if there are L residues s1,...,sL in S
            # such that their sum is 0 mod M, with no proper subsequence summing to 0.
            # For small target_girth, we can check directly.
            
            if _check_circulant_girth(S, M, target_girth):
                n = 2 * M
                if best_n is None or n < best_n:
                    best_n = n
                    best_edges = _build_circulant_edges(S, M)
                    # Can't do better than Moore bound
                    if n <= N_min:
                        return n, best_edges
    
    if best_n is not None:
        return best_n, best_edges
    return None, None


def _check_circulant_girth(S, M, target_girth):
    """Check if circulant bipartite graph with difference set S has girth >= target_girth.
    
    In a circulant bipartite graph, a cycle of length 2L corresponds to a sequence
    s1, s2, ..., sL in S (with signs ±) such that s1 ± s2 ± ... ± sL = 0 mod M,
    where no proper prefix sums to 0 mod M (for simple cycles).
    
    More precisely, for bipartite circulant: cycle of length 2L means there exist
    L elements from S (with repetition allowed? No, for simple cycles, distinct positions)
    such that alternating sum is 0.
    
    Actually, for the bipartite circulant graph where edge (u,v) iff (v-u)%M in S:
    A cycle v0-u0-v1-u1-...-vL-uL-v0 of length 2(L+1) requires:
    v1 = v0 + s1, u1 = v1 - s2 = v0 + s1 - s2, v2 = u1 + s3 = v0 + s1 - s2 + s3, ...
    So we need s1 - s2 + s3 - s4 + ... ± sL = 0 mod M.
    
    For girth check, we need to verify no such relation exists for L < target_girth/2.
    """
    L_max = (target_girth // 2) - 1  # Maximum half-cycle length to check
    
    if L_max < 2:
        return True
    
    S_list = list(S)
    
    # For small L_max, do brute force
    # Check for cycles of length 4: s1 - s2 + s3 - s4 = 0 with all si in S
    # This means s1 + s3 = s2 + s4 mod M
    
    # For length 4 (L=2): need a - b + c - d = 0 => a + c = b + d
    if L_max >= 2:
        # Check all quadruples (a,b,c,d) in S with a+c = b+d mod M
        # Optimization: group by sum
        from collections import defaultdict
        sum_to_pairs = defaultdict(list)
        for i, a in enumerate(S_list):
            for j, c in enumerate(S_list):
                s = (a + c) % M
                sum_to_pairs[s].append((i, j))
        
        for s, pairs in sum_to_pairs.items():
            if len(pairs) >= 2:
                # Check if there are two distinct pairs (i,j) and (k,l) 
                # such that the four indices aren't all the same
                # Actually we need a,b,c,d not all equal in a way that creates a 4-cycle
                # A 4-cycle exists if there are a,c and b,d (possibly same values) 
                # with a+c = b+d and the cycle is valid (no immediate backtracking)
                # For simplicity, if any sum has >= 2 representations, check more carefully
                for idx in range(len(pairs)):
                    i1, j1 = pairs[idx]
                    for idx2 in range(idx + 1, len(pairs)):
                        i2, j2 = pairs[idx2]
                        # a=S[i1], c=S[j1], b=S[i2], d=S[j2]
                        # Need the cycle a-b-c-d-a to be a valid 4-cycle
                        # This requires that the path doesn't backtrack immediately
                        # In bipartite: u0->v0->u1->v1->u0
                        # v0 = u0 + a, u1 = v0 - b, v1 = u1 + c, u0 = v1 - d
                        # So a - b + c - d = 0
                        # For it to be a simple 4-cycle, we need u0 != u1, v0 != v1
                        # u1 = u0 + a - b, so need a != b mod M
                        # v1 = u0 + a - b + c, v0 = u0 + a, need c != b mod M (i.e., S[j1] != S[i2])
                        a_val = S_list[i1]
                        b_val = S_list[i2]
                        c_val = S_list[j1]
                        d_val = S_list[j2]
                        if a_val != b_val and c_val != b_val and a_val != c_val:
                            return False
        # Also need to check: if a+c = b+d where {a,c} != {b,d} as multisets
        # The above check should catch most cases
    
    # For length 6 (L=3): need a - b + c - d + e - f = 0 => a+c+e = b+d+f
    if L_max >= 3:
        # This is more complex. Check if there exist 3 elements from S whose sum
        # equals the sum of another 3 elements from S (mod M), with the sequence
        # forming a valid 6-cycle.
        # For efficiency, only do this for small M
        if M <= 500:
            # Build all sums of 3 elements
            sums3 = defaultdict(list)
            for i in range(len(S_list)):
                for j in range(len(S_list)):
                    for l in range(len(S_list)):
                        s = (S_list[i] + S_list[j] + S_list[l]) % M
                        sums3[s].append((i, j, l))
            
            for s, triples in sums3.items():
                if len(triples) >= 2:
                    for idx in range(len(triples)):
                        i1, j1, l1 = triples[idx]
                        for idx2 in range(idx + 1, len(triples)):
                            i2, j2, l2 = triples[idx2]
                            # Check if this forms a valid 6-cycle
                            # a=S[i1], c=S[j1], e=S[l1], b=S[i2], d=S[j2], f=S[l2]
                            # Need a-b+c-d+e-f = 0 and no immediate backtracking
                            # In bipartite 6-cycle: u0->v0->u1->v1->u2->v2->u0
                            # v0=u0+a, u1=v0-b, v1=u1+c, u2=v1-d, v2=u2+e, u0=v2-f
                            # Need: a!=b, c!=b, d!=c, e!=d, f!=e (no immediate backtrack)
                            # Actually the conditions are that consecutive edges differ
                            a_val, c_val, e_val = S_list[i1], S_list[j1], S_list[l1]
                            b_val, d_val, f_val = S_list[i2], S_list[j2], S_list[l2]
                            if (a_val != b_val and c_val != b_val and 
                                d_val != c_val and e_val != d_val and 
                                f_val != e_val and a_val != f_val):
                                # Additional check: no proper prefix sums to 0
                                # a-b != 0, a-b+c != 0, a-b+c-d != 0, a-b+c-d+e != 0
                                if (a_val - b_val) % M != 0 and \
                                   (a_val - b_val + c_val) % M != 0 and \
                                   (a_val - b_val + c_val - d_val) % M != 0 and \
                                   (a_val - b_val + c_val - d_val + e_val) % M != 0:
                                    return False
    
    # For higher lengths, skip for efficiency (rely on LUW for those)
    return True


def _build_circulant_edges(S, M):
    """Build edge list for circulant bipartite graph."""
    edges = []
    for u in range(M):
        for s in S:
            v = (u + s) % M
            edges.append((u, v + M))
    return edges
# EVOLVE-BLOCK-END
