import numpy as np
import time
import math
from itertools import combinations

def _jc_distance_matrix(sequences):
    """Compute Jukes-Cantor distance matrix from sequences."""
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    char_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seqs = np.array([[char_to_int[c] for c in sequences[label]] for label in labels], dtype=np.uint8)
    L = seqs.shape[1]
    
    mismatches = (seqs[:, None, :] != seqs[None, :, :]).sum(axis=2).astype(np.float64)
    p = mismatches / L
    
    d = np.zeros((n, n), dtype=np.float64)
    mask = p < 0.75
    d[mask] = -0.75 * np.log(1.0 - 4.0 * p[mask] / 3.0)
    d[~mask] = 10.0
    np.fill_diagonal(d, 0.0)
    
    return d, labels, seqs

def _nj_with_expansion(dist, labels):
    """NJ with pre-allocated expanded distance matrix."""
    n = len(labels)
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},({labels[1]},{labels[2]}));"
    
    max_nodes = 2 * n
    D = np.zeros((max_nodes, max_nodes), dtype=np.float64)
    D[:n, :n] = dist
    
    active = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    next_id = n
    
    while len(active) > 3:
        m = len(active)
        
        r = np.zeros(m, dtype=np.float64)
        for idx, i in enumerate(active):
            for jdx, j in enumerate(active):
                if jdx != idx:
                    r[idx] += D[i, j]
        
        best_q = float('inf')
        best_i_idx = -1
        best_j_idx = -1
        
        for idx_i in range(m):
            for idx_j in range(idx_i + 1, m):
                i = active[idx_i]
                j = active[idx_j]
                q = (m - 2) * D[i, j] - r[idx_i] - r[idx_j]
                if q < best_q:
                    best_q = q
                    best_i_idx = idx_i
                    best_j_idx = idx_j
        
        i = active[best_i_idx]
        j = active[best_j_idx]
        
        new_id = next_id
        next_id += 1
        
        trees[new_id] = f"({trees[i]},{trees[j]})"
        
        for x in active:
            if x != i and x != j:
                d_new_x = 0.5 * (D[i, x] + D[j, x] - D[i, j])
                D[new_id, x] = d_new_x
                D[x, new_id] = d_new_x
        
        active = [x for x in active if x not in (i, j)]
        active.append(new_id)
    
    a, b, c = active
    return f"({trees[a]},({trees[b]},{trees[c]}));"

def _parsimony_quartet_scores(seqs, labels, n):
    """
    Compute parsimony scores for all quartets.
    Returns dict: frozenset of 4 labels -> (score_ab_cd, score_ac_bd, score_ad_bc)
    """
    quartet_scores = {}
    quartets = list(combinations(range(n), 4))
    
    for a, b, c, d in quartets:
        sa, sb, sc, sd = seqs[a], seqs[b], seqs[c], seqs[d]
        
        all_same = (sa == sb) & (sb == sc) & (sc == sd)
        
        compatible_ab_cd = (sa == sb) & (sc == sd) & (sa != sc)
        cost_ab_cd = np.where(all_same, 0, np.where(compatible_ab_cd, 1, 2)).astype(np.float64)
        
        compatible_ac_bd = (sa == sc) & (sb == sd) & (sa != sb)
        cost_ac_bd = np.where(all_same, 0, np.where(compatible_ac_bd, 1, 2)).astype(np.float64)
        
        compatible_ad_bc = (sa == sd) & (sb == sc) & (sa != sb)
        cost_ad_bc = np.where(all_same, 0, np.where(compatible_ad_bc, 1, 2)).astype(np.float64)
        
        score_ab_cd = cost_ab_cd.sum()
        score_ac_bd = cost_ac_bd.sum()
        score_ad_bc = cost_ad_bc.sum()
        
        key = frozenset((labels[a], labels[b], labels[c], labels[d]))
        quartet_scores[key] = (score_ab_cd, score_ac_bd, score_ad_bc)
    
    return quartet_scores

def _build_tree_from_splits(splits, labels, n):
    """
    Build a tree from a set of compatible splits using recursive division.
    """
    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},({labels[1]},{labels[2]}))"
    
    if not splits:
        mid = n // 2
        left = labels[:mid]
        right = labels[mid:]
        return f"({_build_tree_from_splits([], left, len(left))},({_build_tree_from_splits([], right, len(right))}))"
    
    all_taxa = frozenset(labels)
    
    best_split = None
    best_support = -1
    
    for split, support in splits:
        if split <= all_taxa and 1 < len(split) < n:
            if support > best_support:
                best_support = support
                best_split = split
    
    if best_split is None:
        mid = n // 2
        left = labels[:mid]
        right = labels[mid:]
        return f"({_build_tree_from_splits([], left, len(left))},({_build_tree_from_splits([], right, len(right))}))"
    
    left_taxa = list(best_split)
    right_taxa = list(all_taxa - best_split)
    
    left_splits = [(s & best_split, sup) for s, sup in splits if (s & best_split) and 2 <= len(s & best_split) < len(left_taxa)]
    right_splits = [(s & (all_taxa - best_split), sup) for s, sup in splits if (s & (all_taxa - best_split)) and 2 <= len(s & (all_taxa - best_split)) < len(right_taxa)]
    
    left_tree = _build_tree_from_splits(left_splits, left_taxa, len(left_taxa))
    right_tree = _build_tree_from_splits(right_splits, right_taxa, len(right_taxa))
    
    return f"({left_tree},{right_tree})"

def reconstruct(sequences):
    start_time = time.time()
    deadline = 265.0
    
    d, labels, seqs = _jc_distance_matrix(sequences)
    n = len(labels)
    
    tree_newick = _nj_with_expansion(d, labels)
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return tree_newick
    
    if n >= 4 and elapsed < deadline - 30:
        try:
            quartet_scores = _parsimony_quartet_scores(seqs, labels, n)
            
            splits = []
            for key, (s1, s2, s3) in quartet_scores.items():
                taxa = sorted(key)
                a, b, c, d = taxa[0], taxa[1], taxa[2], taxa[3]
                
                scores = [s1, s2, s3]
                best_idx = int(np.argmin(scores))
                second_idx = int(np.argsort(scores)[1])
                
                net_support = scores[second_idx] - scores[best_idx]
                
                if best_idx == 0:
                    side1 = frozenset({a, b})
                    side2 = frozenset({c, d})
                elif best_idx == 1:
                    side1 = frozenset({a, c})
                    side2 = frozenset({b, d})
                else:
                    side1 = frozenset({a, d})
                    side2 = frozenset({b, c})
                
                if len(side1) <= len(side2):
                    split = side1
                else:
                    split = side2
                
                if 2 <= len(split) <= n - 2 and net_support > 0:
                    splits.append((split, net_support))
            
            if splits:
                tree_from_splits = _build_tree_from_splits(splits, labels, n)
                if tree_from_splits:
                    taxa_in_tree = set()
                    for label in labels:
                        if label in tree_from_splits:
                            taxa_in_tree.add(label)
                    if taxa_in_tree == set(labels):
                        tree_newick = tree_from_splits + ";"
        except Exception:
            pass
    
    return tree_newick
