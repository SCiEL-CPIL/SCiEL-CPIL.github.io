"""
R_4(C6): 4-colouring of K_n with no monochromatic C6.
Different strategy: Simulated annealing with a fast incremental C6-count objective,
using a structured (quadratic-residue) seed and a vectorized delta evaluation.
We minimise the total number of monochromatic 6-cycles. A zero objective gives a valid colouring.
"""
import time
import random
import sys

DEADLINE_MARGIN = 8.0

def _edge_endpoints(n):
    """Return list of (i,j) for all edges in lexicographic order."""
    eps = []
    for i in range(n):
        for j in range(i + 1, n):
            eps.append((i, j))
    return eps

def _build_adj(colors, n, num_colors=4):
    adj = [[0] * n for _ in range(num_colors)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
    return adj

def _count_c6s_using_edge(a, i, j):
    """
    Count the number of 6-cycles in graph 'a' (list of int bitmasks) that use edge (i,j).
    A 6-cycle using edge (i,j) has the form i-j-v2-v3-v4-v5-i.
    We count paths of length 4 from j to i that avoid i and j internally, 
    where the first vertex after j is v2 and the last before i is v5.
    Actually, a cycle is i-v5-v4-v3-v2-j-i. So we need a path of length 4 from j to i
    in the graph with i and j removed from internal vertices.
    Equivalently, count paths j -> v2 -> v3 -> v4 -> v5 -> i of length 5? No, 6-cycle has 6 vertices.
    Vertices: i, j, v2, v3, v4, v5. Edges: (i,j), (j,v2), (v2,v3), (v3,v4), (v4,v5), (v5,i).
    So we need a path of length 4 from j to i in the graph with edge (i,j) removed? No.
    We need a path j -> v2 -> v3 -> v4 -> v5 -> i of length 5 edges, where internal vertices v2..v5 are distinct from i,j.
    This is equivalent to counting walks of length 4 from N(j) to N(i) that are vertex-disjoint from {i,j}.
    Let's use a more direct enumeration for small n.
    """
    cnt = 0
    a_i = a[i]
    a_j = a[j]
    # v5 in N(i) \ {j}
    m_v5 = a_i & ~(1 << j)
    while m_v5:
        v5 = (m_v5 & -m_v5).bit_length() - 1
        m_v5 &= m_v5 - 1
        # v4 in N(v5) \ {i, j, v5}
        m_v4 = a[v5] & ~(1 << i) & ~(1 << j)
        while m_v4:
            v4 = (m_v4 & -m_v4).bit_length() - 1
            m_v4 &= m_v4 - 1
            # v3 in N(v4) \ {i, j, v5, v4}
            m_v3 = a[v4] & ~(1 << i) & ~(1 << j) & ~(1 << v5)
            while m_v3:
                v3 = (m_v3 & -m_v3).bit_length() - 1
                m_v3 &= m_v3 - 1
                # v2 in N(v3) \ {i, j, v5, v4, v3} AND v2 in N(j)
                m_v2 = a[v3] & a_j & ~(1 << i) & ~(1 << j) & ~(1 << v5) & ~(1 << v4)
                cnt += bin(m_v2).count('1')
    return cnt

def _total_c6s(adj, n, eps):
    total = 0
    for c in range(4):
        a = adj[c]
        for i, j in eps:
            total += _count_c6s_using_edge(a, i, j)
    # Each C6 is counted 6 times (once for each edge in the cycle, once per direction? No, once per edge).
    # A 6-cycle has 6 edges. For each edge (i,j) in the cycle, _count_c6s_using_edge counts it once?
    # Let's verify: if the cycle is i-j-v2-v3-v4-v5-i, then for edge (i,j), we find the path j-v2-v3-v4-v5-i.
    # For edge (j,v2), we would find path v2-v3-v4-v5-i-j? No, the edge is (j,v2). The cycle is ...j-v2...
    # The function counts paths from j to i of length 5 avoiding i,j internally.
    # For edge (i,j), it counts paths j->...->i.
    # For edge (j,v2), it would count paths v2->...->j.
    # So each cycle is counted 6 times (once for each of its 6 edges).
    # But wait, the path j->v2->v3->v4->v5->i is one specific orientation.
    # The cycle could also be traversed i->v5->v4->v3->v2->j->i.
    # The function _count_c6s_using_edge(a, i, j) counts paths from j to i.
    # So for edge (i,j), it counts the path j->v2->v3->v4->v5->i.
    # For edge (j,v2), it counts paths from v2 to j. The path is v2->v3->v4->v5->i->j? No, i->j is an edge.
    # The path from v2 to j of length 5 is v2->v3->v4->v5->i->j. This is a valid path in the graph.
    # So yes, each cycle is counted 6 times.
    # However, we just want to minimize the count, so a constant factor doesn't matter.
    return total

def _quadratic_residue_seed(n):
    """
    Generate a seed coloring based on quadratic residues mod n.
    color(i,j) = 0 if (i-j)^2 is a quadratic residue mod n, etc.
    This is a heuristic to get a structured start.
    """
    # For simplicity, use a random seed but with some structure.
    # Let's try: color(i,j) = ((i*j) % n) % 4
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append((i * j) % 4)
    return colors

def _random_seed(n):
    num_edges = n * (n - 1) // 2
    return [random.randint(0, 3) for _ in range(num_edges)]

def _anneal(n, time_limit, seed_val=42):
    random.seed(seed_val)
    deadline = time.time() + time_limit
    eps = _edge_endpoints(n)
    num_edges = len(eps)
    
    # Try a few different initializations
    best_colors = None
    best_score = float('inf')
    
    for init_try in range(10):
        if time.time() > deadline:
            break
        
        if init_try == 0:
            colors = _quadratic_residue_seed(n)
        else:
            colors = _random_seed(n)
        
        adj = _build_adj(colors, n)
        current_score = _total_c6s(adj, n, eps)
        
        if current_score < best_score:
            best_score = current_score
            best_colors = colors[:]
            if best_score == 0:
                return best_colors
        
        if current_score == 0:
            return colors
        
        # Simulated annealing
        T = 50.0
        T_min = 0.1
        cooling = 0.9995
        
        while time.time() < deadline:
            if current_score == 0:
                return colors
            
            # Pick a random edge
            idx = random.randint(0, num_edges - 1)
            i, j = eps[idx]
            old_c = colors[idx]
            new_c = random.randint(0, 3)
            if new_c == old_c:
                continue
            
            # Compute delta
            delta_removed = _count_c6s_using_edge(adj[old_c], i, j)
            delta_added = _count_c6s_using_edge(adj[new_c], i, j)
            delta = delta_added - delta_removed
            new_score = current_score + delta
            
            if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                # Accept
                colors[idx] = new_c
                adj[old_c][i] &= ~(1 << j)
                adj[old_c][j] &= ~(1 << i)
                adj[new_c][i] |= (1 << j)
                adj[new_c][j] |= (1 << i)
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_colors = colors[:]
                    if best_score == 0:
                        return best_colors
            
            T *= cooling
            if T < T_min:
                T = T_min
    
    return best_colors if best_score == 0 else None

def _is_valid(colors, n):
    """Check if the coloring has no monochromatic C6."""
    adj = _build_adj(colors, n)
    eps = _edge_endpoints(n)
    return _total_c6s(adj, n, eps) == 0

def construct():
    start_time = time.time()
    remaining = 150.0 - DEADLINE_MARGIN
    
    # Try n=19, 18, 17, ...
    # We allocate time based on n.
    # n=19: 30s, n=18: 30s, n=17: 30s, n=16: 20s, n=15: 15s, ...
    # But we must be careful to return the best valid coloring found.
    
    best_result = None
    
    # We'll try to find a valid coloring for the largest n possible.
    # Since 17 is known, we try 19, 18, 17.
    # If we find one for 19, return it. If not, try 18. If not, try 17.
    
    # Time budget allocation
    # Let's give 40s to n=19, 40s to n=18, 40s to n=17, and the rest to smaller n if needed.
    # But if we find a valid 17, we should return it quickly to ensure we don't time out.
    
    attempts = [
        (19, 40.0),
        (18, 40.0),
        (17, 40.0),
        (16, 20.0),
        (15, 10.0),
        (14, 5.0),
        (13, 3.0),
        (12, 2.0),
    ]
    
    for n, tl in attempts:
        if time.time() - start_time > remaining - 2:
            break
        actual_tl = min(tl, remaining - 2)
        if actual_tl < 1.0:
            break
        
        # Try multiple seeds for each n
        found = None
        seeds_tried = 0
        max_seeds = max(1, int(actual_tl / 5.0)) # ~5s per seed
        
        for s in range(max_seeds):
            if time.time() - start_time > remaining - 2:
                break
            seed_val = 42 + n * 100 + s
            res = _anneal(n, time_limit=5.0, seed_val=seed_val)
            if res is not None:
                found = res
                break
        
        if found is not None:
            # Verify
            if _is_valid(found, n):
                return {'n': n, 'colors': found}
            else:
                # Should not happen, but if it does, don't return it
                pass
        
        # If we reach here, we didn't find a valid coloring for this n
        # We continue to smaller n
    
    # Fallback: return a trivially valid small coloring
    # n=4, any coloring is valid (no C6 in K4)
    n = 4
    colors = [0] * 6
    return {'n': n, 'colors': colors}

if __name__ == '__main__':
    res = construct()
    print(res['n'])
    print(res['colors'])
