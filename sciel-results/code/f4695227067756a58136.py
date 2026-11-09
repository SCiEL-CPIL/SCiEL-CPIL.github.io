"""Reconstruct unrooted tree topology using quartet-based split support
with Jukes-Cantor distances. For each quartet of taxa, determine the most
supported topology and accumulate split support. Then recursively assemble
the tree from the best-supported splits."""

import math
import numpy as np
from itertools import combinations

def _jc_distance(seq1, seq2):
    """Compute Jukes-Cantor corrected distance between two sequences."""
    n = len(seq1)
    if n == 0:
        return 0.0
    diff = sum(1 for a, b in zip(seq1, seq2) if a != b)
    p = diff / n
    if p >= 0.7499:
        return 3.0
    if p <= 0:
        return 0.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)

def reconstruct(sequences):
    """Reconstruct unrooted tree topology using quartet-based split support.
    
    Args:
        sequences: dict mapping taxon labels to DNA strings
        
    Returns:
        Newick string representing the unrooted tree
    """
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"
    
    # Build distance matrix
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _jc_distance(sequences[labels[i]], sequences[labels[j]])
            dist[i, j] = d
            dist[j, i] = d
    
    # Quartet-based split support
    split_support = {}
    
    for i, j, k, l in combinations(range(n), 4):
        # Three possible topologies for quartet (i,j,k,l)
        # ab|cd: score = d(a,c) + d(b,d) - d(a,b) - d(c,d)
        s_ab_cd = dist[i, k] + dist[j, l] - dist[i, j] - dist[k, l]
        s_ac_bd = dist[i, j] + dist[k, l] - dist[i, k] - dist[j, l]
        s_ad_bc = dist[i, l] + dist[j, k] - dist[i, j] - dist[k, l]
        
        scores = [s_ab_cd, s_ac_bd, s_ad_bc]
        best_idx = int(np.argmax(scores))
        
        if best_idx == 0:
            split = frozenset({i, j})
        elif best_idx == 1:
            split = frozenset({i, k})
        else:
            split = frozenset({i, l})
        
        # Normalize: use smaller side for canonical representation
        if len(split) > n // 2:
            split = frozenset(set(range(n)) - set(split))
        
        split_support[split] = split_support.get(split, 0) + 1
    
    # Build tree using recursive split selection
    all_taxa = frozenset(range(n))
    
    def build_tree(taxa_set):
        """Recursively build tree for a set of taxa."""
        if len(taxa_set) <= 3:
            return list(taxa_set)
        
        best_split = None
        best_score = 0
        
        for split, support in split_support.items():
            if support <= 0:
                continue
            # Check if split is a valid non-trivial split of taxa_set
            side1 = split & taxa_set
            side2 = taxa_set - side1
            if len(side1) >= 1 and len(side2) >= 1:
                if support > best_score:
                    best_score = support
                    best_split = side1
        
        if best_split is None:
            return list(taxa_set)
        
        other = taxa_set - best_split
        left = build_tree(best_split)
        right = build_tree(other)
        
        return (left, right)
    
    tree = build_tree(all_taxa)
    
    def to_newick(node):
        """Convert tree structure to Newick string."""
        if isinstance(node, list):
            if len(node) == 1:
                return labels[node[0]]
            return ",".join(labels[i] for i in sorted(node))
        if isinstance(node, tuple):
            left_nw = to_newick(node[0])
            right_nw = to_newick(node[1])
            return f"({left_nw},{right_nw})"
        return str(node)
    
    result = to_newick(tree)
    return result
