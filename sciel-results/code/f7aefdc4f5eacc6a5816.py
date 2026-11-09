"""Phylogenetic topology recovery via Split-Based Quartet Consensus."""
import time
import math
import random
import numpy as np

_DEADLINE_FRACTION = 0.92
_start_time = None


def _jc_distance_matrix(sequences, labels):
    """Compute JC-corrected distance matrix using numpy vectorization."""
    n = len(labels)
    enc = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seqs = np.array([[enc[c] for c in s] for s in (sequences[lab] for lab in labels)], dtype=np.uint8)
    n_sites = seqs.shape[1]
    
    # Pairwise differences
    diff = (seqs[:, np.newaxis, :] != seqs[np.newaxis, :, :])
    pmat = diff.sum(axis=2).astype(np.float64) / n_sites
    
    pmat = np.clip(pmat, 0, 0.7499)
    dmat = -0.75 * np.log(1.0 - 4.0 * pmat / 3.0)
    np.fill_diagonal(dmat, 0.0)
    return dmat


def _score_quartet(dmat, a, b, c, d):
    """
    Score a quartet (a,b,c,d) using the Minimum Evolution (MinSum) criterion.
    Returns the split (set1, set2) that represents the best topology.
    There are 3 topologies:
    1. (a,b)|(c,d) -> Splits: {a,b}|{c,d}
    2. (a,c)|(b,d) -> Splits: {a,c}|{b,d}
    3. (a,d)|(b,c) -> Splits: {a,d}|{b,c}
    
    For a given topology, the sum of branch lengths can be estimated.
    Under JC/NJ, the length of the internal branch and external branches can be derived.
    A simpler proxy often used is the sum of the three "quartet distances":
    Q1 = d_ab + d_cd
    Q2 = d_ac + d_bd
    Q3 = d_ad + d_bc
    
    The topology with the smallest Q is usually the correct one (for long branches, this can fail, but it's a standard heuristic).
    Actually, the MinEvolve score for a quartet is slightly more complex, but Q-values are a good proxy.
    Let's use the Q-value minimum.
    """
    d_ab = dmat[a, b]
    d_cd = dmat[c, d]
    d_ac = dmat[a, c]
    d_bd = dmat[b, d]
    d_ad = dmat[a, d]
    d_bc = dmat[b, c]
    
    q1 = d_ab + d_cd
    q2 = d_ac + d_bd
    q3 = d_ad + d_bc
    
    min_q = min(q1, q2, q3)
    
    if min_q == q1:
        # Topology (a,b)|(c,d)
        # Split: {a,b} and {c,d}
        return (frozenset([a, b]), frozenset([c, d]))
    elif min_q == q2:
        # Topology (a,c)|(b,d)
        # Split: {a,c} and {b,d}
        return (frozenset([a, c]), frozenset([b, d]))
    else:
        # Topology (a,d)|(b,c)
        # Split: {a,d} and {b,c}
        return (frozenset([a, d]), frozenset([b, c]))


def _splits_compatible(split1, split2, taxa_set):
    """
    Check if two splits are compatible.
    A split is a pair of frozensets (A, B) such that A U B = taxa_set (or subset in recursion).
    For compatibility check in the global set, we check if they cross.
    Splits S1 = (A1, B1) and S2 = (A2, B2) are incompatible if:
    A1 ∩ A2 != empty AND A1 ∩ B2 != empty AND B1 ∩ A2 != empty AND B1 ∩ B2 != empty.
    """
    A1, B1 = split1
    A2, B2 = split2
    
    # If they don't cover the same set of taxa, we need to be careful.
    # In our greedy selection, we assume all splits are on the full set of taxa.
    # If A1 ∩ A2 is empty, they are compatible.
    # If A1 ∩ B2 is empty, they are compatible.
    # If B1 ∩ A2 is empty, they are compatible.
    # If B1 ∩ B2 is empty, they are compatible.
    
    if len(A1 & A2) == 0: return True
    if len(A1 & B2) == 0: return True
    if len(B1 & A2) == 0: return True
    if len(B1 & B2) == 0: return True
    
    return False


def _build_tree_from_splits(taxa, splits):
    """
    Recursively build a Newick string from a set of compatible splits.
    taxa: set of taxon indices
    splits: list of splits (frozenset, frozenset) that are valid for this subset of taxa
    """
    n = len(taxa)
    if n <= 1:
        return str(taxa.pop()) if taxa else ""
    if n == 2:
        a, b = list(taxa)
        return f"({a},{b})"
    if n == 3:
        a, b, c = list(taxa)
        return f"({a},{b},{c})"
    
    # Find a split that partitions 'taxa' into two non-empty sets
    # We filter the global splits to find one that is "active" on this subset
    # A split (A, B) is active on T if A ∩ T is not empty and B ∩ T is not empty.
    # And A ∪ B should cover T? Not necessarily, but if it's from the tree, it should.
    # Actually, if we have the full set of splits from the tree, any split will partition the leaves.
    
    valid_split = None
    for s in splits:
        A, B = s
        A_in_T = A & taxa
        B_in_T = B & taxa
        if len(A_in_T) > 0 and len(B_in_T) > 0:
            # Ensure the split covers all taxa in the current subset?
            # If the split is from the true tree, A ∪ B = All Taxa.
            # So A_in_T ∪ B_in_T = T.
            # We can use this split.
            valid_split = (A_in_T, B_in_T)
            break
            
    if valid_split is None:
        # Fallback: Star tree
        return "(" + ",".join(map(str, taxa)) + ")"
        
    A, B = valid_split
    tree_A = _build_tree_from_splits(A, splits)
    tree_B = _build_tree_from_splits(B, splits)
    
    return f"({tree_A},{tree_B})"


def _star_tree(labels):
    """Fallback: star tree."""
    if len(labels) <= 1:
        return labels[0] if labels else ""
    if len(labels) == 2:
        return f"({labels[0]},{labels[1]})"
    return "(" + ",".join(labels) + ")"


def reconstruct(sequences):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    
    if n <= 1:
        return labels[0] if labels else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"
    
    # Time check
    elapsed = time.time() - _start_time
    remaining = _DEADLINE_FRACTION * 270.0 - elapsed
    if remaining < 10.0:
        return _star_tree(labels)
    
    # 1. Compute Distance Matrix
    dmat = _jc_distance_matrix(sequences, labels)
    
    # 2. Sample Quartets and Score Splits
    # Number of samples depends on n and time
    # For n=32, we need ~29 splits. 5000 samples should be enough.
    num_samples = 5000
    if n > 20:
        num_samples = 8000
    if n > 28:
        num_samples = 10000
        
    split_counts = {}
    indices = list(range(n))
    
    # Pre-compute all pairs? No, just sample.
    for _ in range(num_samples):
        # Pick 4 distinct indices
        a, b, c, d = random.sample(indices, 4)
        
        # Determine best split
        split = _score_quartet(dmat, a, b, c, d)
        
        # Normalize split: Ensure the smaller set is first, or sort by first element
        # For consistent hashing
        s1, s2 = split
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        elif len(s1) == len(s2):
            if min(s1) > min(s2):
                s1, s2 = s2, s1
                
        key = (s1, s2)
        split_counts[key] = split_counts.get(key, 0) + 1
        
        # Time check inside loop
        if _ % 1000 == 0:
            elapsed = time.time() - _start_time
            if elapsed > _DEADLINE_FRACTION * 270.0 - 5.0:
                break

    # 3. Greedy Selection of Compatible Splits
    # Sort splits by frequency (descending)
    sorted_splits = sorted(split_counts.items(), key=lambda x: x[1], reverse=True)
    
    selected_splits = []
    for split, count in sorted_splits:
        # Check compatibility with all selected splits
        compatible = True
        for sel_split in selected_splits:
            if not _splits_compatible(split, sel_split, set(indices)):
                compatible = False
                break
        if compatible:
            selected_splits.append(split)
            # We only need n-3 splits for a binary tree
            if len(selected_splits) >= n - 3:
                break
                
    # 4. Reconstruct Tree
    if not selected_splits:
        return _star_tree(labels)
        
    # The splits use indices. The builder expects indices.
    # We need to map back to labels for the Newick string.
    # My _build_tree_from_splits returns strings of indices.
    # I will modify the builder to return labels directly or map at the end.
    # It's easier to map at the end if the structure is preserved.
    # But the recursive builder joins strings.
    # Let's make the builder work with labels directly by passing a map.
    
    # Actually, let's just use the indices in the Newick and replace them?
    # No, the labels are t0, t1, etc.
    # Let's modify _build_tree_from_splits to accept a label_map.
    
    def _build_tree_indices(taxa, splits, label_map):
        n = len(taxa)
        if n <= 1:
            return label_map[list(taxa)[0]] if taxa else ""
        if n == 2:
            a, b = list(taxa)
            return f"({label_map[a]},{label_map[b]})"
        if n == 3:
            a, b, c = list(taxa)
            return f"({label_map[a]},{label_map[b]},{label_map[c]})"
        
        valid_split = None
        for s in splits:
            A, B = s
            A_in_T = A & taxa
            B_in_T = B & taxa
            if len(A_in_T) > 0 and len(B_in_T) > 0:
                valid_split = (A_in_T, B_in_T)
                break
                
        if valid_split is None:
            return "(" + ",".join(label_map[i] for i in taxa) + ")"
            
        A, B = valid_split
        tree_A = _build_tree_indices(A, splits, label_map)
        tree_B = _build_tree_indices(B, splits, label_map)
        
        return f"({tree_A},{tree_B})"

    label_map = {i: labels[i] for i in range(n)}
    tree = _build_tree_indices(set(indices), selected_splits, label_map)
    
    return tree
