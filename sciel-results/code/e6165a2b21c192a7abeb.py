# EVOLVE-BLOCK-START
import random
import time

class Node:
    __slots__ = ['label', 'id', 'children', 'parent']
    def __init__(self, label=None, id=None):
        self.label = label
        self.id = id
        self.children = []
        self.parent = None

    def is_leaf(self):
        return self.label is not None

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

def _build_newick(node):
    if node.is_leaf():
        return node.label
    child_strings = [_build_newick(child) for child in node.children]
    return "(" + ",".join(child_strings) + ")"

def _fitch_site_bitmask(node, seqs, site_idx, state_map):
    """
    Fitch algorithm using bitmasks.
    state_map: dict char -> bitmask (1, 2, 4, 8)
    Returns (score, state_mask)
    """
    def fitch(node):
        if node.is_leaf():
            state = seqs[node.label][site_idx]
            return state_map[state], 0
        
        # Recurse
        child_results = [fitch(child) for child in node.children]
        
        # Combine
        # For binary tree
        if len(node.children) == 2:
            mask1, score1 = child_results[0]
            mask2, score2 = child_results[1]
            intersection = mask1 & mask2
            if intersection:
                return intersection, score1 + score2
            else:
                return mask1 | mask2, score1 + score2 + 1
        else:
            # Multifurcation
            common = 15 # 1111
            total_score = 0
            union = 0
            for mask, score in child_results:
                common &= mask
                total_score += score
                union |= mask
            if common:
                return common, total_score
            else:
                return union, total_score + 1

    return fitch(node)

def _calculate_parsimony_score_bitmask(tree, seqs, state_map):
    n_sites = len(next(iter(seqs.values())))
    total_score = 0
    for i in range(n_sites):
        score, _ = _fitch_site_bitmask(tree, seqs, i, state_map)
        total_score += score
    return total_score

def _stepwise_addition(labels, sequences, state_map, rng):
    n_taxa = len(labels)
    # Create leaves
    leaves = {}
    next_id = 0
    for label in labels:
        leaves[label] = Node(label=label, id=next_id)
        next_id += 1
        
    # Random order
    order = labels[:]
    rng.shuffle(order)
    
    # Start with first 3
    start_labels = order[:3]
    # Build initial tree
    # (t0, (t1, t2))
    n1 = Node(id=next_id); next_id += 1
    n2 = Node(id=next_id); next_id += 1
    n3 = Node(id=next_id); next_id += 1
    n3.add_child(leaves[start_labels[1]])
    n3.add_child(leaves[start_labels[2]])
    n2.add_child(n3)
    n2.add_child(leaves[start_labels[0]])
    
    current_root = n2
    
    # Add remaining
    for label in order[3:]:
        leaf = leaves[label]
        best_edge = None
        best_score = float('inf')
        
        # Get edges
        edges = []
        def collect_edges(node):
            for child in node.children:
                edges.append((node, child))
                collect_edges(child)
        collect_edges(current_root)
        
        for p, c in edges:
            # Insert leaf
            new_node = Node(id=next_id); next_id += 1
            p.children.remove(c)
            p.add_child(new_node)
            new_node.add_child(c)
            new_node.add_child(leaf)
            
            score = _calculate_parsimony_score_bitmask(current_root, sequences, state_map)
            
            if score < best_score:
                best_score = score
                best_edge = (p, c)
                
            # Restore
            p.children.remove(new_node)
            p.add_child(c)
            c.parent = p
            leaf.parent = None
            
        # Apply best
        p, c = best_edge
        new_node = Node(id=next_id); next_id += 1
        p.children.remove(c)
        p.add_child(new_node)
        new_node.add_child(c)
        new_node.add_child(leaves[label])
        
    return current_root

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_taxa = len(labels)
    
    state_map = {'A': 1, 'C': 2, 'G': 4, 'T': 8}
    
    best_tree = None
    best_score = float('inf')
    
    rng = random.Random(42)
    
    # Try a few restarts
    n_restarts = 3
    if n_taxa > 20:
        n_restarts = 2 # Save time
        
    for _ in range(n_restarts):
        tree = _stepwise_addition(labels, sequences, state_map, rng)
        score = _calculate_parsimony_score_bitmask(tree, sequences, state_map)
        if score < best_score:
            best_score = score
            best_tree = tree
            
    return _build_newick(best_tree)

# EVOLVE-BLOCK-END
