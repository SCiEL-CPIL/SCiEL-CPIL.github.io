import time
import itertools
import random
from math import gcd
from collections import defaultdict

def _lcm(a, b):
    return a * b // gcd(a, b)

def _is_bad_triple(a, b, c):
    lab = _lcm(a, b)
    if lab != _lcm(a, c):
        return False
    return lab == _lcm(b, c)

def construct():
    start_time = time.time()
    deadline = start_time + 85
    n = 128
    elements = list(range(1, n + 1))
    
    # Precompute all bad triples
    bad_triples = []
    for a, b, c in itertools.combinations(elements, 3):
        if _is_bad_triple(a, b, c):
            bad_triples.append((a - 1, b - 1, c - 1))
    
    if not bad_triples:
        return elements
    
    # Build hypergraph adjacency: for each node, which pairs of other nodes would form a bad triple with it
    pairs_by_node = defaultdict(set)
    # Also build a "conflict" graph: two nodes conflict if they appear together in any bad triple
    conflict = [set() for _ in range(n)]
    for a, b, c in bad_triples:
        pairs_by_node[a].add(frozenset((b, c)))
        pairs_by_node[b].add(frozenset((a, c)))
        pairs_by_node[c].add(frozenset((a, b)))
        conflict[a].add(b)
        conflict[a].add(c)
        conflict[b].add(a)
        conflict[b].add(c)
        conflict[c].add(a)
        conflict[c].add(b)
    
    # Convert conflict to bitmasks for fast operations
    conflict_mask = [0] * n
    for i in range(n):
        for j in conflict[i]:
            conflict_mask[i] |= (1 << j)
    
    # Degree in conflict graph
    degree = [bin(conflict_mask[i]).count('1') for i in range(n)]
    
    # Greedy coloring upper bound
    def greedy_color_bound(available_mask):
        """Upper bound on max independent set in conflict graph induced by available_mask."""
        nodes = [i for i in range(n) if (available_mask >> i) & 1]
        if not nodes:
            return 0
        
        color = {}
        num_colors = 0
        for node in nodes:
            used = set()
            for neighbor in conflict[node]:
                if (available_mask >> neighbor) & 1 and neighbor in color:
                    used.add(color[neighbor])
            c = 0
            while c in used:
                c += 1
            color[node] = c
            num_colors = max(num_colors, c + 1)
        
        return num_colors
    
    # Check if a node can be added to a selected set without completing a bad triple
    def can_add(selected_mask, node):
        for pair in pairs_by_node[node]:
            x, y = tuple(pair)
            if (selected_mask >> x) & 1 and (selected_mask >> y) & 1:
                return False
        return True
    
    # Branch and bound
    best_size = 0
    best_mask = 0
    time_exceeded = False
    
    def branch_and_bound(available_mask, selected_mask, current_size):
        nonlocal best_size, best_mask, time_exceeded
        
        if time.time() > deadline - 5:
            time_exceeded = True
            return
        
        if available_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = selected_mask
            return
        
        # Upper bound
        ub = current_size + greedy_color_bound(available_mask)
        if ub <= best_size:
            return
        
        # Pick node with highest degree in induced subgraph
        temp = available_mask
        best_node = -1
        best_deg = -1
        while temp:
            lsb = temp & (-temp)
            node = lsb.bit_length() - 1
            d = bin(conflict_mask[node] & available_mask).count('1')
            if d > best_deg:
                best_deg = d
                best_node = node
            temp ^= lsb
        
        if best_node == -1:
            return
        
        # Branch include
        if can_add(selected_mask, best_node):
            new_selected = selected_mask | (1 << best_node)
            new_available = available_mask & ~(1 << best_node) & ~conflict_mask[best_node]
            branch_and_bound(new_available, new_selected, current_size + 1)
        
        if time_exceeded:
            return
        
        # Branch exclude
        branch_and_bound(available_mask & ~(1 << best_node), selected_mask, current_size)
    
    # Initial greedy solutions
    orderings = [
        sorted(range(n), key=lambda i: degree[i]),
        sorted(range(n), key=lambda i: degree[i], reverse=True),
        list(range(n)),
        list(range(n - 1, -1, -1)),
        sorted(range(n), key=lambda i: -degree[i]),
        sorted(range(n), key=lambda i: i),
        sorted(range(n), key=lambda i: -i),
    ]
    
    for order in orderings:
        if time.time() > deadline - 30:
            break
        sel = 0
        for node in order:
            if can_add(sel, node):
                sel |= (1 << node)
        sz = bin(sel).count('1')
        if sz > best_size:
            best_size = sz
            best_mask = sel
    
    # Run branch and bound
    if time.time() < deadline - 15:
        all_mask = (1 << n) - 1
        branch_and_bound(all_mask, 0, 0)
    
    # Local search: try adding any node that can be added
    if not time_exceeded and time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for node in range(n):
                if time.time() > deadline - 5:
                    break
                if not (best_mask >> node) & 1:
                    if can_add(best_mask, node):
                        best_mask |= (1 << node)
                        improved = True
    
    # WalkSAT-style randomized local search to escape local optima
    if time.time() < deadline - 10:
        rng = random.Random(42)
        current_mask = best_mask
        current_size = best_size
        
        max_iterations = 50000
        for iteration in range(max_iterations):
            if time.time() > deadline - 10:
                break
            
            # Count violated triples (where all 3 elements are in current_mask)
            violated = []
            for a, b, c in bad_triples:
                if (current_mask >> a) & 1 and (current_mask >> b) & 1 and (current_mask >> c) & 1:
                    violated.append((a, b, c))
            
            if not violated:
                # Valid solution, try to add more elements
                added = True
                while added:
                    added = False
                    for node in range(n):
                        if not (current_mask >> node) & 1:
                            if can_add(current_mask, node):
                                current_mask |= (1 << node)
                                added = True
                new_size = bin(current_mask).count('1')
                if new_size > best_size:
                    best_size = new_size
                    best_mask = current_mask
                break
            
            # Pick a random violated triple
            a, b, c = rng.choice(violated)
            
            # For each node in the triple, compute the "cost" of removing it
            # (number of other violated triples it participates in)
            node_costs = {}
            for node in [a, b, c]:
                cost = 0
                for va, vb, vc in violated:
                    if node in (va, vb, vc):
                        cost += 1
                node_costs[node] = cost
            
            # With probability 0.7, remove the node with highest cost
            # With probability 0.3, remove a random node
            if rng.random() < 0.7:
                max_cost = max(node_costs.values())
                candidates = [node for node in [a, b, c] if node_costs[node] == max_cost]
                node_to_remove = rng.choice(candidates)
            else:
                node_to_remove = rng.choice([a, b, c])
            
            current_mask &= ~(1 << node_to_remove)
            new_size = bin(current_mask).count('1')
            
            if new_size > best_size:
                best_size = new_size
                best_mask = current_mask
    
    return [i + 1 for i in range(n) if (best_mask >> i) & 1]
