"""Rewritten program: construct k-regular graph of girth g using random Cayley graphs on GL(2,p)."""
# EVOLVE-BLOCK-START
import random
import time
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with no cycle shorter than g.
    
    Strategy: Try to build a Cayley graph on GL(2,p) using involutory generators.
    If that fails within time, fall back to the Lazebnik-Ustimenko-Woldar construction.
    
    Returns (n, edges) where n is number of vertices and edges is list of (i,j) pairs.
    """
    start_time = time.time()
    time_limit = 110.0  # Leave 10s margin
    
    # Moore bound calculation
    def moore_bound(k, g):
        if g % 2 == 0:
            return 2 * sum((k-1)**i for i in range(g//2))
        else:
            return 1 + k * sum((k-1)**i for i in range((g-1)//2))
    
    mb = moore_bound(k, g)
    
    # L-U-W construction as reliable fallback
    def luw_construct(k, g):
        """Lazebnik-Ustimenko-Woldar graph CD(d, q)"""
        q = k
        # For girth g, we need d such that girth >= d + q
        # Actually the girth of CD(d,q) is at least d + q for q >= 5
        # We want d + q >= g, so d >= g - q
        # But d must be >= 3
        d = max(3, g - q)
        if d < 3:
            d = 3
        
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
        
        # Rule for CD(d,q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - r
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
    
    # GL(2,p) group construction
    def gl2p_group(p):
        """
        Construct GL(2,p) group elements and multiplication table.
        Returns (n_group, elements, mul_table, involutions)
        where elements is list of (a,b,c,d) representing [[a,b],[c,d]]
        mul_table[i][j] = index of elements[i] * elements[j]
        involutions is list of indices of elements that are involutions (order 2)
        """
        # Generate all matrices in GL(2,p)
        elements = []
        elem_to_idx = {}
        
        for a in range(p):
            for b in range(p):
                for c in range(p):
                    for d in range(p):
                        det = (a * d - b * c) % p
                        if det != 0:
                            elem = (a, b, c, d)
                            elem_to_idx[elem] = len(elements)
                            elements.append(elem)
        
        n_group = len(elements)
        
        # Build multiplication table
        mul_table = [[0] * n_group for _ in range(n_group)]
        for i, e1 in enumerate(elements):
            for j, e2 in enumerate(elements):
                # Matrix multiplication
                a1, b1, c1, d1 = e1
                a2, b2, c2, d2 = e2
                prod = ((a1*a2 + b1*c2) % p,
                        (a1*b2 + b1*d2) % p,
                        (c1*a2 + d1*c2) % p,
                        (c1*b2 + d1*d2) % p)
                mul_table[i][j] = elem_to_idx[prod]
        
        # Find involutions (elements of order 2, excluding identity)
        identity = (1, 0, 0, 1)
        id_idx = elem_to_idx[identity]
        involutions = []
        for i, elem in enumerate(elements):
            if i == id_idx:
                continue
            # Check if elem^2 = identity
            if mul_table[i][i] == id_idx:
                involutions.append(i)
        
        return n_group, elements, mul_table, involutions
    
    # Check girth using BFS from identity
    def check_girth(mul_table, gens, g, n_group):
        """
        Check if the Cayley graph has girth >= g.
        Uses BFS from identity, tracking non-backtracking walks.
        State: (node, last_gen_idx) where last_gen_idx is -1 for start
        """
        # BFS: state is (node, prev_gen)
        # prev_gen = -1 means we're at the start (identity) with no previous generator
        # We want to find if there's a cycle of length < g
        # A cycle of length L means we return to a previously visited state
        # with the same prev_gen (or we can just track visited states)
        
        # Actually, for Cayley graphs, the girth is the length of the shortest cycle.
        # We can do BFS from identity, and if we reach a node that was already visited
        # at a shallower depth with a different path, we have a cycle.
        
        # Better approach: BFS tracking (node, last_generator)
        # Start: (identity, -1) at depth 0
        # From (node, last_gen), we can go to (node * gen, gen) for gen != last_gen_inv
        # But since all gens are involutions, gen_inv = gen, so we just avoid last_gen
        
        id_idx = 0  # We need to find identity index
        # Find identity
        for i in range(n_group):
            e = mul_table[i]
            if all(mul_table[i][j] == j for j in range(n_group)):
                id_idx = i
                break
        
        # BFS
        # visited[node][gen] = depth at which we first reached (node, gen)
        # gen = -1 means no previous generator (only for start)
        
        # Use a dictionary for visited states
        visited = {}
        queue = deque()
        
        # Start state
        visited[(id_idx, -1)] = 0
        queue.append((id_idx, -1, 0))
        
        while queue:
            node, last_gen, depth = queue.popleft()
            
            if depth >= g - 1:
                # We've explored enough; if we haven't found a shorter cycle, girth >= g
                # Actually, we need to check if we can return to identity with depth < g
                # But we're doing BFS from identity, so if we reach identity again at depth d < g,
                # that's a cycle of length d
                continue
            
            for gi, gen in enumerate(gens):
                # Don't backtrack (since gens are involutions, avoid the same generator)
                if last_gen == gi:
                    continue
                
                new_node = mul_table[node][gen]
                state = (new_node, gi)
                
                if state not in visited:
                    visited[state] = depth + 1
                    queue.append((new_node, gi, depth + 1))
                else:
                    # We've seen this state before
                    # If we're back at identity with depth + 1 < g, we have a cycle
                    if new_node == id_idx and depth + 1 < g:
                        return False
        
        # Also check if we can return to identity
        # Actually, the above BFS should catch cycles. Let me reconsider.
        
        # A cycle of length L means there exist g1, g2, ..., gL in Gens such that
        # g1 * g2 * ... * gL = identity, and no proper prefix equals identity,
        # and gi != gi+1 (non-backtracking).
        
        # The BFS above tracks (node, last_gen). If we reach (identity, gi) at depth d,
        # that means we have a walk of length d from identity back to identity,
        # with the last step being gi. This is a cycle of length d if d >= 2.
        
        # Let me redo this more carefully.
        
        return True
    
    # Improved girth check
    def check_girth_v2(mul_table, gens, g, n_group):
        """
        Check if the Cayley graph has girth >= g.
        BFS from identity tracking (node, last_gen).
        """
        # Find identity index
        id_idx = 0
        for i in range(n_group):
            is_id = True
            for j in range(n_group):
                if mul_table[i][j] != j:
                    is_id = False
                    break
            if is_id:
                id_idx = i
                break
        
        # BFS
        # visited[(node, last_gen)] = depth
        visited = {}
        queue = deque()
        
        # Start: at identity, no last generator
        visited[(id_idx, -1)] = 0
        queue.append((id_idx, -1, 0))
        
        while queue:
            node, last_gen, depth = queue.popleft()
            
            if depth >= g - 1:
                continue
            
            for gi, gen in enumerate(gens):
                if last_gen == gi:
                    continue
                
                new_node = mul_table[node][gen]
                state = (new_node, gi)
                
                if state not in visited:
                    visited[state] = depth + 1
                    # If we've returned to identity, we have a cycle of length depth + 1
                    if new_node == id_idx and depth + 1 >= 2:
                        if depth + 1 < g:
                            return False
                    queue.append((new_node, gi, depth + 1))
        
        return True
    
    # Try GL(2,p) for various p
    best_result = None
    
    random.seed(42)
    
    for p in [7, 11, 13, 17, 19, 23]:
        if time.time() - start_time > time_limit:
            break
        
        n_group, elements, mul_table, involutions = gl2p_group(p)
        
        if n_group < mb:
            continue
        
        if len(involutions) < k:
            continue
        
        # Randomly select k involutions and check girth
        max_attempts = 500
        for attempt in range(max_attempts):
            if time.time() - start_time > time_limit:
                break
            
            # Select k random distinct involutions
            gens = random.sample(involutions, k)
            
            if check_girth_v2(mul_table, gens, g, n_group):
                # Found a valid Cayley graph!
                # Build edge list
                edges = []
                for i in range(n_group):
                    for gen in gens:
                        j = mul_table[i][gen]
                        if i < j:
                            edges.append((i, j))
                
                result = (n_group, edges)
                if best_result is None or n_group < best_result[0]:
                    best_result = result
                break
    
    if best_result is not None:
        return best_result
    
    # Fallback to L-U-W
    return luw_construct(k, g)
# EVOLVE-BLOCK-END
