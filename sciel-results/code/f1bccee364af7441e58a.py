"""Rewritten program: construct k-regular graph of girth g using random Cayley graphs."""
# EVOLVE-BLOCK-START
import random
import itertools
from collections import deque
import time

def construct(k, g):
    """
    Construct a k-regular graph with no cycle shorter than g.
    
    Strategy: 
    1. Attempt to find a Cayley graph on S_6 with k involutory generators 
       that yields a girth of at least g.
    2. If that fails, fall back to the Lazebnik-Ustimenko-Woldar (L-U-W) construction.
    
    Returns (n, edges) where n is number of vertices and edges is list of (i,j) pairs.
    """
    
    # Fallback L-U-W construction
    def luw_construct(k, g):
        """Lazebnik-Ustimenko-Woldar graph CD(d, q)"""
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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

    # Cayley graph search on S_6
    def try_cayley_s6(k, g, max_attempts=2000):
        if k != 5:
            return None
        
        # Generate S_6 elements as permutations
        # We represent permutations as tuples of length 6
        # Identity is (0,1,2,3,4,5)
        
        # Precompute all involutions in S_6
        # An involution is a permutation p such that p(p(i)) == i for all i
        # We can generate all permutations and filter, but 720! is too big.
        # Instead, generate involutions by structure:
        # Types: (2), (2,2), (2,2,2)
        
        involutions = []
        
        # Helper to generate permutations from cycle structures
        def generate_perm_from_cycles(cycles, n):
            # cycles is a list of lists, e.g., [[0,1], [2,3]]
            # Returns a tuple permutation
            perm = list(range(n))
            for cycle in cycles:
                if len(cycle) > 1:
                    for i in range(len(cycle)):
                        perm[cycle[i]] = cycle[(i+1) % len(cycle)]
            return tuple(perm)

        # Generate all (2) involutions
        for i in range(6):
            for j in range(i+1, 6):
                cycles = [[i, j]]
                involutions.append(generate_perm_from_cycles(cycles, 6))
                
        # Generate all (2,2) involutions
        for i in range(6):
            for j in range(i+1, 6):
                for k in range(j+1, 6):
                    for l in range(k+1, 6):
                        # Pairs (i,j) and (k,l)
                        cycles = [[i, j], [k, l]]
                        involutions.append(generate_perm_from_cycles(cycles, 6))
                        # Pairs (i,k) and (j,l)
                        cycles = [[i, k], [j, l]]
                        involutions.append(generate_perm_from_cycles(cycles, 6))
                        # Pairs (i,l) and (j,k)
                        cycles = [[i, l], [j, k]]
                        involutions.append(generate_perm_from_cycles(cycles, 6))

        # Generate all (2,2,2) involutions
        for i in range(6):
            for j in range(i+1, 6):
                for k in range(j+1, 6):
                    for l in range(k+1, 6):
                        for m in range(l+1, 6):
                            for n in range(m+1, 6):
                                # 3 ways to pair 6 items
                                cycles = [[i, j], [k, l], [m, n]]
                                involutions.append(generate_perm_from_cycles(cycles, 6))
                                cycles = [[i, k], [j, l], [m, n]]
                                involutions.append(generate_perm_from_cycles(cycles, 6))
                                cycles = [[i, l], [j, k], [m, n]]
                                involutions.append(generate_perm_from_cycles(cycles, 6))
                                cycles = [[i, m], [j, k], [l, n]]
                                involutions.append(generate_perm_from_cycles(cycles, 6))
                                cycles = [[i, n], [j, k], [l, m]]
                                involutions.append(generate_perm_from_cycles(cycles, 6))
                                cycles = [[i, j], [k, m], [l, n]]
                                involutions.append(generate_perm_from_cycles(cycles, 6))
                                # ... This is getting complex and error prone.
                                # Let's just filter from all permutations if we list them.
                                # But listing 720 perms is easy.
        
        # Let's restart involutions generation by filtering all permutations
        # This is safer.
        involutions = []
        all_perms = list(itertools.permutations(range(6)))
        for p in all_perms:
            # Check if p is involution
            is_inv = True
            for i in range(6):
                if p[p[i]] != i:
                    is_inv = False
                    break
            if is_inv and p != tuple(range(6)): # Exclude identity
                involutions.append(p)
                
        # There should be 75 involutions.
        # print(len(involutions)) # 75
        
        # Precompute multiplication table for S_6
        # index_map: perm -> index
        index_map = {p: i for i, p in enumerate(all_perms)}
        mult_table = [[0]*720 for _ in range(720)]
        for i, p1 in enumerate(all_perms):
            for j, p2 in enumerate(all_perms):
                # p1 * p2 means apply p2 then p1?
                # Standard convention: (p1 * p2)(x) = p1(p2(x))
                # Let's stick to one convention.
                # If we treat permutations as functions, composition p1 o p2
                prod = tuple(p1[p2[i]] for i in range(6))
                mult_table[i][j] = index_map[prod]
        
        # BFS to check generation and girth
        def check_generators(gens):
            # gens is a list of 5 permutations
            # Check if they generate S_6
            # And compute girth
            
            # Map gens to indices
            gen_indices = [index_map[g] for g in gens]
            
            # BFS from identity (index 0)
            dist = [-1] * 720
            dist[0] = 0
            q = deque([0])
            min_cycle = float('inf')
            
            while q:
                u = q.popleft()
                du = dist[u]
                
                # If we already found a cycle shorter than g, we can stop?
                # Yes, if min_cycle < g, we can reject.
                if min_cycle < g:
                    return False, min_cycle
                
                for gi in gen_indices:
                    v = mult_table[u][gi]
                    if dist[v] == -1:
                        dist[v] = du + 1
                        q.append(v)
                    else:
                        # Found a cycle
                        # Cycle length is du + dist[v] + 1
                        # Note: This might count the same cycle multiple times or count non-simple paths?
                        # In BFS tree, any edge (u,v) where v is visited creates a cycle of length dist[u] + dist[v] + 1.
                        # Is this the girth?
                        # The girth is the shortest cycle.
                        # Any cycle must contain an edge (u,v) such that one is parent of the other? No.
                        # Any cycle in a BFS tree from a root:
                        # If the cycle does not pass through the root, it will be detected when an edge connects two nodes in the same level or adjacent levels?
                        # Actually, if we run BFS, the first time we see a node, we set its distance.
                        # If we see an edge to an already visited node, we have a cycle.
                        # The length is dist[u] + dist[v] + 1.
                        # This detects all cycles.
                        cycle_len = du + dist[v] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                            if min_cycle < g:
                                return False, min_cycle
            
            # If we finished BFS and min_cycle >= g, we need to check if graph is connected
            # If dist[i] == -1 for any i, it's not connected.
            # But Cayley graph on a group with generators is connected iff they generate the group.
            # We can check if all dist are >= 0.
            if any(d == -1 for d in dist):
                return False, min_cycle # Not connected, treat as invalid
            
            return min_cycle >= g, min_cycle

        # Random search
        random.seed(42)
        for _ in range(max_attempts):
            # Pick 5 distinct random involutions
            if len(involutions) < 5:
                break
            gens = random.sample(involutions, 5)
            valid, min_c = check_generators(gens)
            if valid:
                # Construct edges
                # We need to map group elements to 0..n-1
                # n = 720
                # Edges are (u, v) where v = u * g for g in gens
                # We only add edge once, e.g., if u < v? Or just list all directed?
                # The task asks for a list of edges (i, j). Usually undirected.
                # "simple (no loops or repeated edges)"
                # So we should add (min(u,v), max(u,v)) to a set, then convert to list.
                
                edges_set = set()
                for u in range(720):
                    for g in gens:
                        v = mult_table[u][index_map[g]]
                        if u < v:
                            edges_set.add((u, v))
                        else:
                            edges_set.add((v, u))
                
                edges = list(edges_set)
                return 720, edges
        
        return None

    # Try Cayley graph
    result = try_cayley_s6(k, g)
    if result:
        return result
    
    # Fallback
    return luw_construct(k, g)
# EVOLVE-BLOCK-END
