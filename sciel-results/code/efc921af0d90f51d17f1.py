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
    deadline = start_time + 88
    n = 128
    elements = list(range(1, n + 1))
    
    # Precompute all bad triples and build efficient data structures
    bad_triples = []
    for a, b, c in itertools.combinations(elements, 3):
        if _is_bad_triple(a, b, c):
            bad_triples.append((a - 1, b - 1, c - 1))
    
    if not bad_triples:
        return elements
    
    # For each node, store bitmask of "partner pairs" that would form a bad triple
    # forbidden_pairs[i] is a list of (j, k) such that {i,j,k} is a bad triple
    forbidden_pairs = [[] for _ in range(n)]
    # conflict_mask[i]: bitmask of nodes that appear with i in some bad triple
    conflict_mask = [0] * n
    
    for a, b, c in bad_triples:
        forbidden_pairs[a].append((b, c))
        forbidden_pairs[b].append((a, c))
        forbidden_pairs[c].append((a, b))
        conflict_mask[a] |= (1 << b) | (1 << c)
        conflict_mask[b] |= (1 << a) | (1 << c)
        conflict_mask[c] |= (1 << a) | (1 << b)
    
    degree = [bin(conflict_mask[i]).count('1') for i in range(n)]
    
    # Fast can_add: check if adding node to selected_mask creates a bad triple
    def can_add(selected_mask, node):
        for j, k in forbidden_pairs[node]:
            if (selected_mask >> j) & 1 and (selected_mask >> k) & 1:
                return False
        return True
    
    # Partition by 2-adic valuation for structured bounds
    # Numbers with same 2-adic valuation have special LCM structure
    def v2(x):
        """2-adic valuation (exponent of 2 in prime factorization)"""
        if x == 0:
            return 0
        count = 0
        while x % 2 == 0:
            count += 1
            x //= 2
        return count
    
    # Group elements by their 2-adic valuation
    groups_by_v2 = defaultdict(list)
    for i in range(n):
        groups_by_v2[v2(i + 1)].append(i)
    
    # Stronger upper bound using partition structure
    def partition_bound(available_mask):
        """Upper bound based on 2-adic valuation partitioning."""
        total = 0
        for v, group in groups_by_v2.items():
            # Within each group, count available nodes
            avail_in_group = sum(1 for i in group if (available_mask >> i) & 1)
            total += avail_in_group
        return total
    
    # Greedy coloring upper bound (faster version)
    def greedy_color_bound_fast(available_mask):
        """Fast upper bound: use simple degree-based bound."""
        # Upper bound by counting nodes with degree 0 in induced subgraph
        # plus a greedy estimate
        count = 0
        temp = available_mask
        while temp:
            lsb = temp & (-temp)
            node = lsb.bit_length() - 1
            # If node has no neighbors in available set, it can always be included
            if (conflict_mask[node] & available_mask) == 0:
                count += 1
                available_mask &= ~lsb
            temp ^= lsb
        
        # For remaining nodes, use a simple bound: ceil(remaining / (1 + avg_degree))
        remaining = bin(available_mask).count('1')
        if remaining == 0:
            return count
        # Simple bound: at most remaining/2 + 1 for a graph (very loose)
        # Better: use greedy coloring on a subset
        nodes = [i for i in range(n) if (available_mask >> i) & 1]
        if len(nodes) <= 30:
            color = {}
            num_colors = 0
            for node in nodes:
                used = set()
                for neighbor in range(n):
                    if (conflict_mask[node] >> neighbor) & 1 and (available_mask >> neighbor) & 1:
                        if neighbor in color:
                            used.add(color[neighbor])
                c = 0
                while c in used:
                    c += 1
                color[node] = c
                num_colors = max(num_colors, c + 1)
            return count + num_colors
        else:
            # Loose bound for large sets
            return count + (remaining + 1) // 2
    
    best_size = 0
    best_mask = 0
    
    # Multiple greedy initializations
    orderings = [
        sorted(range(n), key=lambda i: degree[i]),
        sorted(range(n), key=lambda i: degree[i], reverse=True),
        list(range(n)),
        list(range(n - 1, -1, -1)),
    ]
    
    for order in orderings:
        if time.time() > deadline - 40:
            break
        sel = 0
        for node in order:
            if can_add(sel, node):
                sel |= (1 << node)
        sz = bin(sel).count('1')
        if sz > best_size:
            best_size = sz
            best_mask = sel
    
    # Branch and bound with iterative deepening
    time_exceeded = False
    
    def branch_and_bound(available_mask, selected_mask, current_size, depth):
        nonlocal best_size, best_mask, time_exceeded
        
        if time.time() > deadline - 3:
            time_exceeded = True
            return
        
        if available_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = selected_mask
            return
        
        # Upper bound
        ub = current_size + greedy_color_bound_fast(available_mask)
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
            branch_and_bound(new_available, new_selected, current_size + 1, depth + 1)
        
        if time_exceeded:
            return
        
        # Branch exclude
        branch_and_bound(available_mask & ~(1 << best_node), selected_mask, current_size, depth + 1)
    
    # Run B&B if we have time
    if time.time() < deadline - 20:
        all_mask = (1 << n) - 1
        branch_and_bound(all_mask, 0, 0, 0)
    
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
    
    # WalkSAT-style randomized local search with faster violation counting
    if time.time() < deadline - 8:
        rng = random.Random(42)
        current_mask = best_mask
        current_size = best_size
        
        # Precompute for each node, which bad triples it participates in
        triples_by_node = defaultdict(list)
        for idx, (a, b, c) in enumerate(bad_triples):
            triples_by_node[a].append(idx)
            triples_by_node[b].append(idx)
            triples_by_node[c].append(idx)
        
        max_iterations = 100000
        for iteration in range(max_iterations):
            if time.time() > deadline - 8:
                break
            
            # Count violated triples
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
            
            # Compute cost for each node in the triple
            node_costs = {}
            for node in [a, b, c]:
                cost = 0
                for va, vb, vc in violated:
                    if node in (va, vb, vc):
                        cost += 1
                node_costs[node] = cost
            
            # Remove node with highest cost (with some randomness)
            max_cost = max(node_costs.values())
            candidates = [node for node in [a, b, c] if node_costs[node] == max_cost]
            if rng.random() < 0.7:
                node_to_remove = rng.choice(candidates)
            else:
                node_to_remove = rng.choice([a, b, c])
            
            current_mask &= ~(1 << node_to_remove)
            new_size = bin(current_mask).count('1')
            
            if new_size > best_size:
                best_size = new_size
                best_mask = current_mask
    
    # Final polish: try to add nodes one more time
    if time.time() < deadline - 2:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for node in range(n):
                if not (best_mask >> node) & 1:
                    if can_add(best_mask, node):
                        best_mask |= (1 << node)
                        improved = True
    
    return [i + 1 for i in range(n) if (best_mask >> i) & 1]
