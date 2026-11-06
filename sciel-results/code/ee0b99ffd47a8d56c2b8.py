"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
'''Greedy Additive Parsimony with NNI optimization.
   Constructs a tree by adding taxa one by one, choosing the insertion
   point that minimizes the total parsimony score. Followed by local
   search using Nearest Neighbor Interchange (NNI) swaps.'''
import math
import numpy as np
from collections import defaultdict

def _calc_parsimony_score(edges, sequences, labels):
    """
    Calculate total parsimony score for the tree defined by edges.
    """
    n_taxa = len(labels)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Root at 0
    parent = {0: -1}
    visited = {0}
    stack = [0]
    order = []
    while stack:
        node = stack.pop()
        order.append(node)
        for neighbor in adj[node]:
            if neighbor not in visited:
                visited[neighbor] = True
                parent[neighbor] = node
                stack.append(neighbor)
                
    if len(visited) != len(edges) + 1:
        return float('inf')
        
    total_score = 0
    num_sites = len(sequences[labels[0]])
    
    for site in range(num_sites):
        states = {}
        costs = {}
        
        # Leaves
        for label in labels:
            idx = label_to_idx[label]
            states[idx] = {sequences[label][site]}
            costs[idx] = 0
            
        # Internal nodes (post-order)
        for node in reversed(order):
            if node in states:
                continue
                
            child_states = []
            child_cost_sum = 0
            for neighbor in adj[node]:
                if neighbor == parent.get(node, -1):
                    continue
                child_states.append(states[neighbor])
                child_cost_sum += costs[neighbor]
                
            if not child_states:
                continue
                
            intersection = set(child_states[0])
            for cs in child_states[1:]:
                intersection &= cs
                
            if intersection:
                states[node] = intersection
                costs[node] = child_cost_sum
            else:
                union = set()
                for cs in child_states:
                    union |= cs
                states[node] = union
                costs[node] = child_cost_sum + 1
                
        total_score += costs[0]
        
    return total_score

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n < 2:
        return labels[0] + ";" if labels else ";"
    if n == 2:
        return f"({labels[0]}, {labels[1]});"
    if n == 3:
        return f"({labels[0]}, ({labels[1]}, {labels[2]}));"
        
    # Greedy Additive Construction
    # Start with first 3 taxa
    # Tree: (0, (1, 2))
    # Nodes: 0, 1, 2, 3(internal)
    # Edges: (0,3), (3,1), (3,2)
    
    edges = set()
    next_node_id = n
    
    # Initialize with 3 taxa
    # Internal node 3
    edges.add((0, 3))
    edges.add((3, 1))
    edges.add((3, 2))
    
    # Add remaining taxa
    for k in range(3, n):
        best_score = float('inf')
        best_edges = None
        
        # Try inserting taxon k into each existing edge
        for (u, v) in list(edges):
            # Create new node w
            w = next_node_id
            next_node_id += 1
            
            new_edges = set(edges)
            new_edges.discard((u, v))
            new_edges.discard((v, u))
            new_edges.add((u, w))
            new_edges.add((w, v))
            new_edges.add((w, k))
            
            score = _calc_parsimony_score(new_edges, sequences, labels[:k+1])
            if score < best_score:
                best_score = score
                best_edges = new_edges
                best_w = w
                
        if best_edges is not None:
            edges = best_edges
        else:
            # Fallback: attach to node 0
            w = next_node_id
            next_node_id += 1
            edges.add((0, w))
            edges.add((w, k))
            
    # NNI Optimization
    # Perform a few passes of NNI
    for _ in range(5):
        improved = False
        # Generate NNI moves
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        for u, v in list(edges):
            if len(adj[u]) != 3 or len(adj[v]) != 3:
                continue
                
            u_neighbors = [n for n in adj[u] if n != v]
            v_neighbors = [n for n in adj[v] if n != u]
            
            if len(u_neighbors) < 1 or len(v_neighbors) < 2:
                continue
                
            a = u_neighbors[0]
            b = v_neighbors[0]
            d = v_neighbors[1]
            
            # Move 1: (u,b) and (v,a)
            new_edges = set(edges)
            new_edges.discard((u, v))
            new_edges.discard((v, u))
            new_edges.add((u, b))
            new_edges.add((v, a))
            
            score = _calc_parsimony_score(new_edges, sequences, labels)
            if score < _calc_parsimony_score(edges, sequences, labels):
                edges = new_edges
                improved = True
                break # Restart NNI after a change
                
            # Move 2: (u,d) and (v,a)
            new_edges = set(edges)
            new_edges.discard((u, v))
            new_edges.discard((v, u))
            new_edges.add((u, d))
            new_edges.add((v, a))
            
            score = _calc_parsimony_score(new_edges, sequences, labels)
            if score < _calc_parsimony_score(edges, sequences, labels):
                edges = new_edges
                improved = True
                break
                
        if not improved:
            break
            
    # Convert edges to Newick
    # Root at 0
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    def to_newick(node, parent):
        children = [n for n in adj[node] if n != parent]
        if not children:
            return labels[node]
        else:
            child_strs = [to_newick(c, node) for c in children]
            return "(" + ",".join(child_strs) + ")"
            
    newick = to_newick(0, -1) + ";"
    return newick
# EVOLVE-BLOCK-END
