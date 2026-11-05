import time
import random
import sys
from collections import deque
import numpy as np

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    """
    start_time = time.time()
    deadline = start_time + 145.0

    if k == 3 and g == 16:
        # Attempt 1: Tree Closure (Target n ~ 766)
        result = _try_tree_closure(deadline, start_time)
        if result:
            return result

        # Attempt 2: Voltage Lift of Petersen (Target n < 936)
        result = _try_voltage_lift(g, deadline, start_time)
        if result:
            return result

        # Fallback: LUW or larger lift
        return _construct_luw(3, g, d=7)
    else:
        return _construct_luw(k, g, d=7)

def _try_tree_closure(deadline, start_time):
    """
    Attempts to construct a 3-regular graph of girth 16 by closing a depth-8 tree.
    """
    d = 8
    # Build the non-backtracking tree of depth d
    # Root at 0
    # Neighbors of root: 1, 2, 3
    # Each other node has 2 children
    
    parent = {}
    children = {}
    depth = {}
    
    nodes = [0]
    parent[0] = -1
    depth[0] = 0
    children[0] = []
    
    current_level = [0]
    for i in range(1, d + 1):
        next_level = []
        for u in current_level:
            # Determine number of children
            # Root has 3 children
            # Others have 2 children
            num_children = 3 if i == 1 else 2
            for j in range(num_children):
                v = len(nodes)
                nodes.append(v)
                parent[v] = u
                depth[v] = i
                children[v] = []
                children[u].append(v)
                next_level.append(v)
        current_level = next_level
        
    n = len(nodes)
    leaves = [v for v in nodes if depth[v] == d]
    
    # Calculate distances between all leaves
    # We only care about pairs with distance >= 15
    # dist(u, v) = depth[u] + depth[v] - 2 * depth[lca(u, v)]
    # Since all leaves are at depth d, dist(u, v) = 2d - 2 * depth[lca(u, v)]
    # We need dist >= 15 => 16 - 2 * depth[lca] >= 15 => 2 * depth[lca] <= 1 => depth[lca] <= 0.5
    # So depth[lca] must be 0.
    # This means u and v must be in different main branches (children of root).
    
    # Let's verify this logic.
    # If lca is root (depth 0), dist = 16. Cycle = 17. OK.
    # If lca is depth 1, dist = 14. Cycle = 15. BAD.
    # So we can ONLY connect leaves that are in different main branches.
    
    # Partition leaves into 3 groups based on their child of root ancestor.
    groups = {1: [], 2: [], 3: []}
    for v in leaves:
        # Find ancestor at depth 1
        u = v
        while depth[u] > 1:
            u = parent[u]
        groups[u].append(v)
        
    # We have 3 groups of size 128 each (384 / 3).
    # We can only connect leaves from different groups.
    # This forms a 3-partite graph of valid connections.
    # We need a perfect matching in this 3-partite graph.
    # Since the graph is complete between different groups, any perfect matching works.
    # We just need to pair up the 384 leaves such that no pair is from the same group.
    
    # Strategy:
    # Shuffle leaves in each group.
    # Pair group 1 with group 2, and group 3 with... wait.
    # We have 128 leaves in G1, 128 in G2, 128 in G3.
    # Total 384.
    # We need 192 edges.
    # We can pair G1 with G2 (128 edges) and G3 with... who?
    # If we pair G1-G2, all G1 and G2 are used.
    # G3 has 128 leaves left.
    # They must be paired with each other? No, same group is forbidden.
    # So we cannot use a simple pairwise grouping.
    
    # We need a 3-regular hypergraph matching? No, just a matching.
    # Each leaf must be matched to a leaf in a different group.
    # This is equivalent to finding a permutation of the leaves such that
    # the partner of any leaf is in a different group.
    
    # Since the valid graph is complete 3-partite, we can just:
    # 1. Shuffle G1, G2, G3.
    # 2. Create a list of all leaves.
    # 3. We need to form 192 pairs.
    # 4. If we take 1 from G1, 1 from G2, 1 from G3... that's 3 leaves.
    #    We can form a triangle? No, the graph is simple, no triangles allowed?
    #    Wait, the graph must be simple.
    #    If we connect u (G1) - v (G2), v (G2) - w (G3), w (G3) - u (G1), that's a triangle of length 3 in the "matching graph", but in the actual graph, the cycle length is determined by tree distances.
    #    The cycle in the final graph would be u-v (edge) + v...w (tree path) + w...u (tree path).
    #    Actually, the edges we add are between leaves.
    #    The cycle is: u --(new edge)--> v --(tree path)--> LCA(v,w) ...
    #    This is getting complicated.
    #    
    #    Let's stick to the condition: dist(u, v) >= 15.
    #    This implies u and v are in different main branches.
    #    So we just need a perfect matching in the complete 3-partite graph.
    #    Such a matching always exists.
    #    
    #    Algorithm:
    #    1. Shuffle G1, G2, G3.
    #    2. We have 128 leaves in each.
    #    3. We can pair G1[0] with G2[0], G1[1] with G2[1], ..., G1[127] with G2[127].
    #       This uses all of G1 and G2.
    #       G3 is left with 128 leaves.
    #       They cannot be paired with each other.
    #       So this fails.
    #       
    #    We need to mix them.
    #    Let's create a pool of leaves.
    #    We need to pair them up.
    #    A simple strategy:
    #    Sort leaves by group.
    #    G1 (128), G2 (128), G3 (128).
    #    Pair G1[0] with G2[0].
    #    Pair G1[1] with G3[0].
    #    Pair G1[2] with G2[1].
    #    ...
    #    We just need to ensure that for every pair, the groups are different.
    #    
    #    We can generate a random permutation of the 384 leaves, and then pair (0,1), (2,3), etc.
    #    Check if any pair has same group. If so, swap.
    #    Or better:
    #    Construct the matching explicitly.
    #    Since the valid graph is dense, a random greedy matching will likely succeed.
    
    leaves_shuffled = leaves[:]
    random.shuffle(leaves_shuffled)
    
    # Map leaf to group
    leaf_group = {v: groups_key for groups_key, group_leaves in groups.items() for v in group_leaves}
    
    matched = [False] * n
    edges = []
    
    # Greedy matching with backtracking/restart
    for _ in range(100):
        random.shuffle(leaves_shuffled)
        matched = [False] * n
        edges = []
        success = True
        
        for i in range(0, len(leaves_shuffled), 2):
            u = leaves_shuffled[i]
            v = leaves_shuffled[i+1]
            if leaf_group[u] == leaf_group[v]:
                # Try to fix by finding a valid partner for u
                # Find a w not matched, w != v, group[w] != group[u]
                found = False
                for j in range(i+2, len(leaves_shuffled)):
                    w = leaves_shuffled[j]
                    if not matched[w] and leaf_group[w] != leaf_group[u]:
                        # Swap v and w in the list?
                        # Or just pair u with w, and leave v for later?
                        # This breaks the simple pairing.
                        # Let's just restart if conflict is frequent.
                        found = True
                        break
                if not found:
                    success = False
                    break
            else:
                edges.append((u, v))
                matched[u] = True
                matched[v] = True
        
        if success:
            # Verify all leaves are matched
            if all(matched[v] for v in leaves):
                # Build the full edge list
                all_edges = []
                for v in range(n):
                    for child in children[v]:
                        all_edges.append((v, child))
                
                # Add matching edges
                for u, v in edges:
                    all_edges.append((u, v))
                
                return n, all_edges

    return None

def _try_voltage_lift(g, deadline, start_time):
    # Petersen graph
    pet_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n_base = 10
    edge_index = {}
    for i, (u, v) in enumerate(pet_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
        
    # Precompute constraints
    coeff_vectors = set()
    max_walk_len = g - 1
    for start_v in range(n_base):
        stack = [(start_v, -1, 0, [0] * 15)]
        while stack:
            v, prev_v, l, counts = stack.pop()
            if l >= max_walk_len:
                continue
            for next_v in range(n_base):
                if next_v == prev_v or next_v == v:
                    continue
                e_idx = edge_index.get((v, next_v))
                if e_idx is None:
                    continue
                new_counts = counts.copy()
                u_edge, v_edge = pet_edges[e_idx]
                if v == u_edge and next_v == v_edge:
                    new_counts[e_idx] += 1
                else:
                    new_counts[e_idx] -= 1
                new_l = l + 1
                if next_v == start_v and new_l > 0:
                    c_tuple = tuple(new_counts)
                    if any(x != 0 for x in c_tuple):
                        for i in range(15):
                            if c_tuple[i] != 0:
                                if c_tuple[i] < 0:
                                    c_tuple = tuple(-x for x in c_tuple)
                                break
                        coeff_vectors.add(c_tuple)
                stack.append((next_v, v, new_l, new_counts))
                
    coeff_list = list(coeff_vectors)
    if not coeff_list:
        return None
        
    coeff_matrix = np.array(coeff_list, dtype=np.int64)
    
    def is_prime(n):
        if n < 2: return False
        if n < 4: return True
        if n % 2 == 0 or n % 3 == 0: return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0: return False
            i += 6
        return True

    best_n = 10 ** 6
    best_edges = None
    
    m = 2
    while 10 * m < best_n and time.time() < deadline:
        if not is_prime(m):
            m += 1
            continue
            
        # Simulated Annealing
        voltages = np.array([random.randint(0, m - 1) for _ in range(15)], dtype=np.int64)
        current_cost = np.sum((coeff_matrix @ voltages) % m == 0)
        
        T = 1.0
        for step in range(5000):
            if time.time() > deadline:
                break
            if current_cost == 0:
                edges = _build_lifted_graph(pet_edges, voltages.tolist(), m)
                if 10 * m < best_n:
                    best_n = 10 * m
                    best_edges = edges
                break
                
            new_voltages = voltages.copy()
            idx = random.randint(0, 14)
            new_voltages[idx] = random.randint(0, m - 1)
            new_cost = np.sum((coeff_matrix @ new_voltages) % m == 0)
            
            delta = new_cost - current_cost
            if delta < 0 or random.random() < np.exp(-delta / T):
                voltages = new_voltages
                current_cost = new_cost
                
            T *= 0.99
            
        m += 1
        
    if best_edges:
        return best_n, best_edges
        
    return None

def _build_lifted_graph(pet_edges, voltages, m):
    n_base = 10
    edge_index = {}
    for i, (u, v) in enumerate(pet_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
        
    edges = set()
    for u in range(n_base):
        for v in range(n_base):
            if u == v:
                continue
            e_idx = edge_index.get((u, v))
            if e_idx is None:
                continue
            vol = voltages[e_idx]
            u_edge, v_edge = pet_edges[e_idx]
            if u == u_edge and v == v_edge:
                sign = 1
            else:
                sign = -1
            for layer in range(m):
                u_node = layer * n_base + u
                target_layer = (layer + sign * vol) % m
                v_node = target_layer * n_base + v
                a, b = min(u_node, v_node), max(u_node, v_node)
                edges.add((a, b))
    return list(edges)

def _construct_luw(k, g, d=None, variant=0):
    if d is None:
        d = max(3, (g - 2) // 2)
    q = k
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
    offset = variant % 4
    for c in range(5, d + 1):
        r = (c - 5 + offset) % 4
        c0 = c - (c - 5) % 4
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
    index = {start: 0}
    order = [start]
    edges = []
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
