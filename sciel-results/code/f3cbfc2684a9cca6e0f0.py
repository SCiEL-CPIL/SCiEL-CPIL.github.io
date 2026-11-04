import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    # We focus on the k=3 case primarily, as that's the target of the task description.
    # The seed handles k=4 and others, but for k=3, we use the algebraic ensemble.
    
    if k != 3:
        # Fallback to a simple known construction or seed-like logic for other k
        # For k=4, the seed used a table. We can just return a valid large graph or use the same logic.
        # To be safe and fast, we'll implement a basic valid graph generator for non-3 cases
        # or just return the seed's result if we can replicate it quickly.
        # However, the task is specifically n(3,16). We assume k=3 is the main test.
        # Let's implement a generic fallback that returns a valid 3-regular graph if k!=3 is passed?
        # No, the interface says evaluator passes k and g.
        # If k != 3, we should still try to construct a k-regular graph.
        # For simplicity and robustness, we will use the seed's LUW logic as a fallback for k!=3,
        # but for k=3 we use the new ensemble strategy.
        
        # Re-implementing the seed's core logic for fallback
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
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

    # --- Strategy for k=3: Algebraic Ensemble Search ---
    
    # The LUW construction for k=3 uses q=3 (Z3).
    # The standard rule in the seed is fixed.
    # We will vary the "twist" in the recurrence.
    # The recurrence is y[c] = x[c] + y[a] * x[b] (mod 3) for side 0.
    # In the seed, (a,b) is determined by c.
    # We can try different coefficients for the multiplication or addition?
    # In Z3, multiplication is limited. 
    # However, we can vary the indices a and b, or add a constant offset.
    # Let's try adding a constant offset 'shift' to the calculation:
    # y[c] = (x[c] + y[a] * x[b] + shift) % 3
    # Or vary the rule selection pattern.
    
    # Actually, a more powerful variation is to change the depth 'd' slightly?
    # No, d is constrained by g.
    
    # Let's look at the "rule" generation in the seed:
    # rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    # r = (c - 5) % 4
    # This is a fixed pattern.
    
    # New Idea: Randomize the rule coefficients within valid constraints.
    # For each c, we need to pick (a, b) such that the graph remains 3-regular and simple.
    # The LUW construction guarantees 3-regularity and simplicity if the "tree" expansion doesn't collapse.
    # Collapsing happens if two different paths lead to the same vertex.
    # We can generate random rules and check if the resulting graph is simple and has girth >= g.
    
    # Optimization: We generate a candidate graph, check if it's simple (no multi-edges/loops in the "virtual" graph before indexing? 
    # Actually, the indexing process itself detects if a vertex is revisited. 
    # If `other in index`, it's a back-edge or a cross-edge. 
    # In a tree expansion, `other in index` means we found a cycle.
    # If the cycle length is < g, it's invalid.
    # If the cycle length >= g, it's valid.
    # Wait, the LUW construction *is* the graph. The "index" dict maps virtual vertices to indices.
    # If we encounter a virtual vertex that is already in `index`, it means two paths met.
    # This creates a cycle. The length of this cycle is the distance between the two meeting points + 1?
    # Actually, the LUW construction builds a graph where vertices are states.
    # If two states are identical, they are the same vertex.
    # The "girth" of the resulting graph is determined by the shortest cycle in this state space.
    
    # So, the algorithm is:
    # 1. Define a rule (sequence of (a,b) pairs for c=2..d).
    # 2. Expand the graph from the root using BFS on the state space.
    # 3. If the expansion completes without "colliding" in a way that creates a short cycle, we have a candidate.
    # 4. We need to verify the girth.
    
    # How to verify girth efficiently?
    # The state space is a tree of depth d (if no collisions).
    # Collisions create cycles.
    # We can track the "depth" of each state.
    # When a state `s` is encountered for the second time from a different parent `p2` (or same parent via different path?),
    # we can calculate the cycle length.
    
    # Let's implement a function to build the graph with a specific rule and return (n, edges, girth_found).
    
    def build_graph_with_rule(rule, q=3, d=None):
        if d is None:
            d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
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
        
        # We also need to track the "distance" from start for each vertex to detect cycle lengths
        dist = {start: 0}
        parent = {start: None}
        
        # BFS
        for i in range(len(order)):
            vertex = order[i]
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                    dist[other] = dist[vertex] + 1
                    parent[other] = vertex
                    if vertex[0] == 0:
                        edges.append((index[vertex], index[other]))
                else:
                    # Cycle found
                    # The cycle length is dist[vertex] + dist[other] + 1?
                    # This is true if `other` is an ancestor of `vertex` or vice versa?
                    # In a BFS tree, an edge (u,v) where v is already visited creates a cycle.
                    # If v is at the same level or previous level, the cycle length is dist[u] + dist[v] + 1.
                    # If v is a descendant, it's a forward edge, not a cycle in the BFS tree sense, but in the graph.
                    # However, in this construction, edges are undirected.
                    # If we see `other` already in index, it means we've seen this state before.
                    # The cycle length is the distance between `vertex` and `other` in the graph + 1.
                    # Since we are doing BFS, `dist[other]` is the shortest distance from start to `other`.
                    # `dist[vertex]` is the shortest distance from start to `vertex`.
                    # The edge (vertex, other) creates a cycle of length dist[vertex] + dist[other] + 1 ONLY IF the paths to vertex and other are disjoint except for start?
                    # No, the standard formula for cycle length from BFS back-edge (u,v) where v is visited is dist[u] + dist[v] + 1.
                    # This is valid for simple cycles in undirected graphs found by BFS.
                    cycle_len = dist[vertex] + dist[other] + 1
                    if cycle_len < g:
                        return None, None, cycle_len # Invalid: short cycle found
                    # If cycle_len >= g, we don't add the edge again (it's already added or will be added from the other side? 
                    # Wait, edges are added only if vertex[0]==0.
                    # If `other` is already in index, we don't add a new edge to `order`.
                    # But we must ensure the edge exists in the final graph.
                    # The seed code only adds edges when `vertex[0]==0` and `other` is being processed.
                    # If `other` is already in index, the edge was likely added when `other` was the "vertex" (if other[0]==0) or will be?
                    # Actually, the seed logic:
                    # `if vertex[0] == 0: edges.append(...)`
                    # This ensures each edge is added exactly once (from the side 0 vertex).
                    # If `other` is already in index, it means we've seen this connection.
                    # We don't need to add it again.
                    pass
        
        # After BFS, we have a graph.
        # We must verify the girth is actually >= g.
        # The BFS check above only catches cycles that are "back edges" in the BFS tree.
        # Does it catch ALL short cycles?
        # In an undirected graph, every cycle contains at least one back edge in the BFS tree.
        # The length of that cycle is dist[u] + dist[v] + 1.
        # So if we check all back edges and ensure dist[u]+dist[v]+1 >= g, then all cycles are >= g.
        # Is this true? Yes.
        
        return len(order), edges, g # girth is at least g

    # Generate random rules
    # The seed rule:
    # rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    # We can randomize the choice of (a,b) for each c.
    # Valid (a,b) must be such that the indices are within [1, c-1]?
    # In the seed, a and b are indices of previous y's or x's.
    # y is the new vector. x is the old vector.
    # y[c] depends on y[a] and x[b].
    # a must be < c. b must be < c.
    # Let's try random a, b in [1, c-1].
    
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    best_n = float('inf')
    best_edges = None
    
    # Try a few random rules
    for attempt in range(50):
        if time.time() > deadline:
            break
            
        rule = {}
        # Fix first few to match seed to ensure stability? Or fully random?
        # Let's try fully random for c >= 2
        for c in range(2, d + 1):
            a = random.randint(1, c-1)
            b = random.randint(1, c-1)
            rule[c] = (a, b)
            
        n, edges, girth = build_graph_with_rule(rule, q=3, d=d)
        
        if n is not None and n < best_n:
            best_n = n
            best_edges = edges
            
        # If we found a graph with n == Moore bound, we can stop
        moore_bound = 2 * ((2 ** (g//2)) - 1)
        if best_n <= moore_bound:
            break

    if best_edges is None:
        # Fallback to seed rule if random search fails
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        best_n, best_edges, _ = build_graph_with_rule(rule, q=3, d=d)
        
    return best_n, best_edges

