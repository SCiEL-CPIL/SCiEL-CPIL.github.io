"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import numpy as np
import random
import sys
from collections import defaultdict

class Tree:
    def __init__(self):
        self.children = {}
        self.parent = {}
        self.branch_length = {}
        self.root = None
        self.labels = {}
        self.node_count = 0

    def add_leaf(self, node_id, label):
        self.children[node_id] = []
        self.labels[node_id] = label
        self.node_count = max(self.node_count, node_id + 1)

    def add_internal(self, node_id):
        self.children[node_id] = []
        self.node_count = max(self.node_count, node_id + 1)

    def set_parent(self, node_id, parent_id, length):
        self.parent[node_id] = parent_id
        self.branch_length[node_id] = length
        if parent_id in self.children:
            self.children[parent_id].append(node_id)

    def get_newick(self):
        def build(node_id):
            if node_id in self.labels:
                return self.labels[node_id]
            else:
                children = self.children[node_id]
                if not children:
                    return "?"
                child_strings = []
                for c in children:
                    child_str = build(c)
                    bl = self.branch_length.get(c, 0.0)
                    child_strings.append(f"{child_str}:{bl:.4f}")
                return f"({','.join(child_strings)})"
        
        root_str = build(self.root)
        return f"{root_str};"

def _jc_distance(seq1, seq2):
    n = len(seq1)
    if n == 0:
        return 0.0
    mismatches = sum(1 for a, b in zip(seq1, seq2) if a != b)
    p = mismatches / n
    if p >= 0.75:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)

def _compute_distance_matrix(sequences, labels):
    n = len(labels)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _jc_distance(sequences[labels[i]], sequences[labels[j]])
            dist[i, j] = d
            dist[j, i] = d
    return dist

def _neighbor_joining(dist, labels):
    n = len(labels)
    if n < 2:
        return labels[0] if n == 1 else ""
    
    d = dist.copy()
    nodes = list(range(n))
    node_trees = {i: labels[i] for i in range(n)}
    
    dist_dict = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_dict[(i, j)] = d[i, j]
    
    while len(nodes) > 3:
        m = len(nodes)
        
        r = {}
        for i in nodes:
            r[i] = sum(dist_dict[(i, j)] for j in nodes if j != i)
        
        best_q = float('inf')
        best_pair = None
        for idx_i in range(m):
            i = nodes[idx_i]
            for idx_j in range(idx_i + 1, m):
                j = nodes[idx_j]
                q = (m - 2) * dist_dict[(i, j)] - r[i] - r[j]
                if q < best_q:
                    best_q = q
                    best_pair = (i, j)
        
        i, j = best_pair
        
        d_ij = dist_dict[(i, j)]
        delta = (r[i] - r[j]) / (2 * (m - 2)) if m > 2 else 0.0
        li = 0.5 * d_ij + delta
        lj = 0.5 * d_ij - delta
        
        li = max(0.0, li)
        lj = max(0.0, lj)
        
        k = max(nodes) + 1
        new_tree = f"({node_trees[i]}:{li:.4f},{node_trees[j]}:{lj:.4f})"
        node_trees[k] = new_tree
        
        for x in nodes:
            if x == i or x == j:
                continue
            dx = 0.5 * (dist_dict[(i, x)] + dist_dict[(j, x)] - d_ij)
            dx = max(0.0, dx)
            dist_dict[(k, x)] = dx
            dist_dict[(x, k)] = dx
        
        nodes = [x for x in nodes if x != i and x != j]
        nodes.append(k)
    
    if len(nodes) == 3:
        a, b, c = nodes
        d_ab = dist_dict[(a, b)]
        d_bc = dist_dict[(b, c)]
        d_ca = dist_dict[(c, a)]
        
        l_a = max(0.0, (d_ab + d_ca - d_bc) / 2.0)
        l_b = max(0.0, (d_ab + d_bc - d_ca) / 2.0)
        l_c = max(0.0, (d_bc + d_ca - d_ab) / 2.0)
        
        return f"({node_trees[a]}:{l_a:.4f},{node_trees[b]}:{l_b:.4f},{node_trees[c]}:{l_c:.4f});"
    elif len(nodes) == 2:
        a, b = nodes
        d_ab = dist_dict[(a, b)]
        return f"({node_trees[a]}:{d_ab/2:.4f},{node_trees[b]}:{d_ab/2:.4f});"
    else:
        return ";".join(node_trees[x] for x in nodes) + ";"

def _parse_newick_to_tree(newick_string, labels):
    s = newick_string.strip()
    if s.endswith(';'):
        s = s[:-1]
    
    tree = Tree()
    label_to_id = {label: i for i, label in enumerate(labels)}
    
    # Stack-based parser
    # Each element: {'id': int, 'children': [], 'bl': float}
    stack = []
    current = None
    i = 0
    n = len(s)
    next_node_id = len(labels)
    
    while i < n:
        c = s[i]
        if c == '(':
            # Start new internal node
            current = {'id': next_node_id, 'children': [], 'bl': 0.0}
            next_node_id += 1
            if stack:
                stack[-1]['children'].append(current)
            else:
                # This is the root
                pass
            stack.append(current)
            i += 1
        elif c == ')':
            # End current node
            current = stack.pop()
            i += 1
            # Check for branch length
            if i < n and s[i] == ':':
                i += 1
                start = i
                while i < n and s[i].isdigit() or (i < n and s[i] == '.'):
                    i += 1
                bl = float(s[start:i])
                current['bl'] = bl
            # Add to parent
            if stack:
                stack[-1]['children'].append(current)
            else:
                # This is the root
                tree.root = current['id']
                tree.children[current['id']] = []
                tree.branch_length[current['id']] = current['bl']
                # Recursively add children
                def add_children(node_info, parent_id):
                    for child in node_info['children']:
                        child_id = child['id']
                        tree.add_internal(child_id)
                        tree.set_parent(child_id, parent_id, child['bl'])
                        add_children(child, child_id)
                add_children(current, current['id'])
        elif c == ',':
            i += 1
        elif c == ':':
            i += 1
            start = i
            while i < n and (s[i].isdigit() or s[i] == '.'):
                i += 1
            bl = float(s[start:i])
            if stack:
                stack[-1]['bl'] = bl
        else:
            # Label
            start = i
            while i < n and s[i] not in '(),:':
                i += 1
            label = s[start:i]
            node_id = label_to_id[label]
            # Check for branch length
            bl = 0.0
            if i < n and s[i] == ':':
                i += 1
                start = i
                while i < n and (s[i].isdigit() or s[i] == '.'):
                    i += 1
                bl = float(s[start:i])
            
            # Add leaf
            tree.add_leaf(node_id, label)
            # Add to parent
            if stack:
                stack[-1]['children'].append({'id': node_id, 'children': [], 'bl': bl})
    
    # If the tree was just a single leaf or simple case, handle it
    if tree.root is None and stack:
        # The last popped item was the root
        pass
        
    return tree

def _compute_likelihood(tree, data):
    # Felsenstein's algorithm
    # Root the tree
    # Compute partial likelihoods
    pass

def _optimize_ml(tree, sequences, labels):
    # ML Optimization
    pass

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    if len(labels) < 2:
        return labels[0] + ";" if labels else ";"
    
    dist = _compute_distance_matrix(sequences, labels)
    newick = _neighbor_joining(dist, labels)
    
    # If parsing is too hard, just return NJ.
    # But I want ML.
    
    # Let's just return NJ for now to ensure it works.
    return newick
# EVOLVE-BLOCK-END
