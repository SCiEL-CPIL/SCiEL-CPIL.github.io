"""Construct a C4-free graph with no independent set of size 11, maximizing n.

Uses finite geometry / block design approach: vertices represent points of a
symmetric design (e.g., biplane), and adjacency is defined by co-incidence on
blocks. In a symmetric (v,k,λ) design, any two points lie on exactly λ blocks.
If λ=1 (a 2-(v,k,1) design / pairwise balanced design), any two points share
exactly one block, giving C4-freeness in the point graph where two points are
adjacent if they are on a common block.

For a 2-(v,k,1) design: each point is on r = (v-1)/(k-1) blocks, each block has
k points. The point graph has degree k-1 (adjacent to all other points on each
of its r blocks). Two points share at most one common neighbor because they
share at most one block.

We use known small designs and extend with randomized edge additions while
maintaining C4-freeness, then verify independence number < 11.
"""

import random
import time
from itertools import combinations

def _is_c4_free(adj, n):
    """Check if adjacency list represents a C4-free graph."""
    for u in range(n):
        for v in range(u + 1, n):
            common = adj[u] & adj[v]
            if len(common) > 1:
                return False
    return True

def _can_add_edge(adj, n, u, v):
    """Check if adding edge (u,v) preserves C4-freeness."""
    if u == v:
        return False
    if v in adj[u]:
        return False
    if len(adj[u] & adj[v]) > 1:
        return False
    for w in adj[v]:
        if w == u:
            continue
        common_uw = adj[u] & adj[w]
        if v in common_uw:
            common_uw.discard(v)
        if len(common_uw) > 0:
            return False
    for w in adj[u]:
        if w == v:
            continue
        common_vw = adj[v] & adj[w]
        if u in common_vw:
            common_vw.discard(u)
        if len(common_vw) > 0:
            return False
    return True

def _has_independent_set(adj, n, k):
    """Check if graph has an independent set of size k using backtracking with bitmasks."""
    if n > 45:
        return True
    
    # Use integer bitmasks for efficiency
    # comp_mask[u] = bitmask of vertices NOT adjacent to u (excluding u)
    comp_mask = [0] * n
    for u in range(n):
        mask = (1 << n) - 1
        mask &= ~(1 << u)
        for v in adj[u]:
            mask &= ~(1 << v)
        comp_mask[u] = mask
    
    def backtrack(candidates_mask, size):
        if size >= k:
            return True
        if candidates_mask == 0:
            return False
        if size + bin(candidates_mask).count('1') < k:
            return False
        
        # Pick vertex with minimum degree in candidate subgraph for better pruning
        best_v = -1
        best_deg = n + 1
        temp = candidates_mask
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            temp ^= lsb
            # Degree in candidate subgraph = number of candidates adjacent to v
            deg = bin(candidates_mask & comp_mask[v]).count('1')
            if deg < best_deg:
                best_deg = deg
                best_v = v
        
        # Include best_v
        new_mask = candidates_mask & comp_mask[best_v]
        if backtrack(new_mask, size + 1):
            return True
        # Exclude best_v
        new_mask2 = candidates_mask & ~(1 << best_v)
        if backtrack(new_mask2, size):
            return True
        return False
    
    return backtrack((1 << n) - 1, 0)

def _construct_from_design(blocks, n):
    """Construct point graph from a list of blocks (each block is a set of point indices).
    Two points are adjacent if they appear together in at least one block.
    This gives C4-freeness if any two points appear in at most one block together."""
    adj = [set() for _ in range(n)]
    for block in blocks:
        block_list = list(block)
        for i in range(len(block_list)):
            for j in range(i + 1, len(block_list)):
                u, v = block_list[i], block_list[j]
                adj[u].add(v)
                adj[v].add(u)
    return adj

def _known_designs():
    """Return list of known small 2-(v,k,1) designs (pairwise balanced designs)."""
    designs = []
    
    # Fano plane: 2-(7,3,1), 7 points, 7 lines of size 3
    fano_blocks = [
        {0, 1, 2}, {0, 3, 4}, {0, 5, 6},
        {1, 3, 5}, {1, 4, 6}, {2, 3, 6}, {2, 4, 5}
    ]
    designs.append((7, fano_blocks))
    
    # Projective plane of order 3: 2-(13,4,1), 13 points, 13 lines of size 4
    # Each line has 4 points, each point on 4 lines, any two points on exactly 1 line
    pg3_blocks = [
        {0, 1, 2, 3}, {0, 4, 5, 6}, {0, 7, 8, 9}, {0, 10, 11, 12},
        {1, 4, 7, 10}, {1, 5, 8, 11}, {1, 6, 9, 12},
        {2, 4, 8, 12}, {2, 5, 9, 10}, {2, 6, 7, 11},
        {3, 4, 9, 11}, {3, 5, 7, 12}, {3, 6, 8, 10}
    ]
    designs.append((13, pg3_blocks))
    
    # Projective plane of order 4: 2-(21,5,1), 21 points, 21 lines of size 5
    # Construct using finite field GF(5)
    # Points: (x,y) in GF(5)^2 plus point at infinity for each slope, plus one more
    # Actually: projective plane of order q has q^2+q+1 points
    # For q=4: 21 points, 21 lines, each line has 5 points
    # Let's construct it properly
    q = 4
    # Points indexed 0..20
    # Represent points as (x, y) in GF(q)^2 for affine points (q^2 = 16)
    # Plus q+1 = 5 points at infinity
    # Total: 16 + 5 = 21
    
    # Use GF(5) since 5 is prime
    # Affine points: (x,y) for x,y in {0,1,2,3,4}
    # Infinity points: one for each slope in {0,1,2,3,4} (slope 4 = -1)
    
    def gf5_add(a, b):
        return (a + b) % 5
    
    def gf5_mul(a, b):
        return (a * b) % 5
    
    # Map (x,y) to index: x*5 + y
    # Infinity points: 20 + slope for slope in 0..4
    
    def point_idx(x, y):
        return x * 5 + y
    
    def inf_idx(slope):
        return 20 + slope
    
    # Lines: for each slope m in 0..4 and each intercept b in 0..4: y = m*x + b
    # Plus the vertical line x = c for each c in 0..4 (these are the "infinity" lines with slope infinity)
    # Wait, in projective plane, lines are parameterized differently.
    
    # Standard construction: lines are sets of points (x,y) satisfying y = m*x + b for m,b in GF(q)
    # Plus lines of form x = c for c in GF(q)
    # But this gives q^2 + q lines, each with q points... that's not right for projective plane.
    
    # Let me use a known explicit construction for PG(2,4)
    # Actually, let me just use a known list. The projective plane of order 4 has 21 points and 21 lines.
    
    # I'll use a standard representation. Points are (x,y,z) in GF(5)^3 up to scaling, not all zero.
    # Number of projective points = (5^3 - 1)/(5-1) = 124/4 = 31. That's PG(2,4) which has order 4? No.
    # PG(2,q) has q^2+q+1 points. For q=4: 21. But (4^3-1)/(4-1) = 63/3 = 21. Yes!
    # So we work in GF(5) since 5 is prime, and PG(2,4) uses GF(5)? No, PG(2,q) uses GF(q).
    # q=4 is not prime, but GF(4) exists. Let me use GF(5) for PG(2,4)? No, that would be PG(2,4) only if 4 is a prime power, which it is (2^2), so GF(4) exists.
    
    # This is getting complex. Let me use a different known design.
    
    # Instead, let's use the complement of a known design or construct directly.
    
    # For now, let's skip PG(2,4) and use other approaches.
    
    return designs

def _try_extend_from_design(n_design, blocks, target_n, start_time, time_limit, trial_seed):
    """Try to extend a design-based graph to target_n vertices."""
    if time.time() - start_time > time_limit:
        return None
    
    adj = _construct_from_design(blocks, n_design)
    
    # Add new vertices
    for new_v in range(n_design, target_n):
        if time.time() - start_time > time_limit:
            return None
        # For each new vertex, try to add edges to existing vertices
        # Start with no edges, then greedily add
        adj.append(set())
        # Try adding edges to all existing vertices in random order
        existing = list(range(new_v))
        random.Random(trial_seed + new_v).shuffle(existing)
        for u in existing:
            if time.time() - start_time > time_limit:
                return None
            if _can_add_edge(adj, len(adj), u, new_v):
                adj[u].add(new_v)
                adj[new_v].add(u)
    
    # Try to add more edges between all vertices
    random.Random(trial_seed).shuffle(list(range(target_n)))
    all_pairs = [(u, v) for u in range(target_n) for v in range(u + 1, target_n) if v not in adj[u]]
    random.Random(trial_seed + 1).shuffle(all_pairs)
    
    for u, v in all_pairs:
        if time.time() - start_time > time_limit:
            return None
        if _can_add_edge(adj, target_n, u, v):
            adj[u].add(v)
            adj[v].add(u)
    
    # Verify C4-free
    if not _is_c4_free(adj, target_n):
        return None
    
    # Check independence number
    if _has_independent_set(adj, target_n, 11):
        return None
    
    edges = []
    for u in range(target_n):
        for v in adj[u]:
            if u < v:
                edges.append([u, v])
    return {'n': target_n, 'edges': edges}

def construct():
    """Construct a C4-free graph with no independent set of size 11, maximizing n."""
    
    def seed_construction():
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}
    
    start_time = time.time()
    time_limit = 100  # seconds
    
    best_result = seed_construction()
    
    # Get known designs
    designs = _known_designs()
    
    # Try to extend from each design
    for target_n in range(39, 30, -1):
        if time.time() - start_time > time_limit:
            break
        
        success = False
        for trial in range(100):
            if time.time() - start_time > time_limit:
                break
            
            # Try extending from each known design
            for n_design, blocks in designs:
                if time.time() - start_time > time_limit:
                    break
                if n_design > target_n:
                    continue
                
                result = _try_extend_from_design(n_design, blocks, target_n, start_time, time_limit, trial * 1000 + n_design)
                if result is not None:
                    best_result = result
                    success = True
                    break
            
            if success:
                break
        
        if success:
            break
    
    # Also try pure greedy construction from scratch as fallback
    if not success and time.time() - start_time < time_limit:
        for target_n in range(39, 30, -1):
            if time.time() - start_time > time_limit:
                break
            
            for trial in range(50):
                if time.time() - start_time > time_limit:
                    break
                
                adj = [set() for _ in range(target_n)]
                random.seed(trial * 1000 + target_n + 777)
                
                possible_edges = [(u, v) for u in range(target_n) for v in range(u + 1, target_n)]
                random.shuffle(possible_edges)
                
                for u, v in possible_edges:
                    if time.time() - start_time > time_limit:
                        break
                    if _can_add_edge(adj, target_n, u, v):
                        adj[u].add(v)
                        adj[v].add(u)
                
                if not _is_c4_free(adj, target_n):
                    continue
                
                if _has_independent_set(adj, target_n, 11):
                    continue
                
                edges = []
                for u in range(target_n):
                    for v in adj[u]:
                        if u < v:
                            edges.append([u, v])
                best_result = {'n': target_n, 'edges': edges}
                break
    
    return best_result
