import numpy as np
import math
import time
from itertools import combinations

def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 165.0

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_taxa = len(labels)

    if n_taxa <= 2:
        if n_taxa == 1:
            return f"({labels[0]});"
        else:
            return f"({labels[0]},{labels[1]});"

    # Map sequences to integer arrays
    seq_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    L = len(next(iter(sequences.values())))
    
    seq_array = np.empty((n_taxa, L), dtype=np.int8)
    for i, label in enumerate(labels):
        s = sequences[label]
        seq_array[i] = [seq_map[c] for c in s]

    # Compute pairwise Hamming distances
    diff = (seq_array[:, None, :] != seq_array[None, :, :])
    hamming = diff.sum(axis=2)
    
    p = hamming / L
    
    d_matrix = np.zeros((n_taxa, n_taxa), dtype=np.float64)
    valid_mask = (p > 0) & (p < 0.75)
    
    if np.any(valid_mask):
        d_valid = -0.75 * np.log(1 - (4.0 / 3.0) * p[valid_mask])
        d_matrix[valid_mask] = d_valid
        large_val = np.max(d_valid) * 1.5
    else:
        large_val = 1.0
        
    invalid_mask = p >= 0.75
    d_matrix[invalid_mask] = large_val
    np.fill_diagonal(d_matrix, 0.0)

    # Step 1: Compute Quartet Counts
    # A split is represented by a frozenset of the smaller side (or canonical side)
    # For a split {A, B}, if len(A) < len(B), key is frozenset(A). If len(A) > len(B), key is frozenset(B).
    # If equal, use sorted tuple or frozenset of the one with smaller min element?
    # Standard canonical form: frozenset of the side that doesn't contain the first taxon 't0'?
    # Or simply: frozenset(A) if min(A) < min(B) else frozenset(B).
    
    split_counts = {}
    
    # Iterate over all quartets
    # N=32, C(32,4) = 35960. This is fast.
    for quartet in combinations(range(n_taxa), 4):
        a, b, c, d = quartet
        # Calculate distance sums for the 3 topologies
        # 1: (a,b)|(c,d) -> D[a,b] + D[c,d]
        # 2: (a,c)|(b,d) -> D[a,c] + D[b,d]
        # 3: (a,d)|(b,c) -> D[a,d] + D[b,c]
        
        s1 = d_matrix[a, b] + d_matrix[c, d]
        s2 = d_matrix[a, c] + d_matrix[b, d]
        s3 = d_matrix[a, d] + d_matrix[b, c]
        
        min_sum = min(s1, s2, s3)
        
        if min_sum == s1:
            # Split: {a,b} and {c,d}
            side1 = frozenset([a, b])
            side2 = frozenset([c, d])
            # Canonical key: use the side with the smaller minimum element
            if min(side1) < min(side2):
                key = side1
            else:
                key = side2
        elif min_sum == s2:
            side1 = frozenset([a, c])
            side2 = frozenset([b, d])
            if min(side1) < min(side2):
                key = side1
            else:
                key = side2
        else:
            side1 = frozenset([a, d])
            side2 = frozenset([b, c])
            if min(side1) < min(side2):
                key = side1
            else:
                key = side2
            
        if key in split_counts:
            split_counts[key] += 1
        else:
            split_counts[key] = 1

    # Step 2: Greedy Selection of Compatible Splits
    # Sort splits by count (descending)
    # Note: We must ensure the split is valid for the full set of taxa.
    # A split key is a frozenset of indices.
    # We need to check if it's a "trivial" split (size 1 or N-1)?
    # Quartets generate splits of size 2.
    # As we combine, we might want to infer larger splits?
    # Greedy selection usually works on the set of "candidate" splits.
    # Here, candidates are the splits observed in quartets.
    # This might miss large-scale structure if no quartet supports it directly?
    # But quartets cover all pairs.
    
    # Sort by weight
    sorted_splits = sorted(split_counts.items(), key=lambda x: x[1], reverse=True)
    
    selected_splits = []
    
    def is_compatible(split1, split2):
        # split1, split2 are frozensets
        # They are compatible if one of the four intersections is empty
        # A1 = split1, A2 = all_taxa - split1
        # B1 = split2, B2 = all_taxa - split2
        # Compatible if: A1&B1 == empty OR A1&B2 == empty OR A2&B1 == empty OR A2&B2 == empty
        # Equivalent to: (split1 & split2) is empty OR (split1 & (all - split2)) is empty ...
        # Simpler: 
        # Let S1 = split1, S2 = split2
        # Intersections:
        # i1 = S1 & S2
        # i2 = S1 & (all - S2)
        # i3 = (all - S1) & S2
        # i4 = (all - S1) & (all - S2)
        # If any is empty, compatible.
        
        # Optimization: 
        # If S1 is subset of S2, compatible? No, that's a specific case.
        # General check:
        # len(S1 & S2) == 0 -> Compatible (Disjoint)
        # len(S1 - S2) == 0 -> S1 subset S2 -> Compatible? 
        #   If S1 subset S2, then (all-S1) & (all-S2) is empty? No.
        #   Let's just compute the 4 sets.
        
        # For efficiency, precompute complements?
        # N=32, set ops are fast.
        
        # We need the set of all taxa to define complements.
        # We can pass all_taxa_set.
        
        # Actually, two splits (A|A') and (B|B') are compatible iff
        # A∩B = ∅ or A∩B' = ∅ or A'∩B = ∅ or A'∩B' = ∅.
        
        # Let's use bitmasks for speed if needed, but sets are fine for N=32.
        
        # We'll implement this inside the loop with the global all_taxa_set.
        pass

    all_taxa_set = frozenset(range(n_taxa))
    
    for split, count in sorted_splits:
        compatible = True
        for existing in selected_splits:
            # Check compatibility
            # S1 = split, S2 = existing
            # Intersections:
            i1 = split & existing
            i2 = split & (all_taxa_set - existing)
            i3 = (all_taxa_set - split) & existing
            i4 = (all_taxa_set - split) & (all_taxa_set - existing)
            
            if i1 or i2 or i3 or i4:
                # If none are empty, they are incompatible
                compatible = False
                break
            else:
                # At least one is empty, so compatible
                pass
        
        if compatible:
            selected_splits.append(split)
            
        if time.time() > deadline - 5.0:
            break

    # Step 3: Build Tree from Selected Splits
    # Recursive construction
    # If no splits selected, return star tree.
    
    def build_tree(taxa_list, splits):
        taxa_set = frozenset(taxa_list)
        if len(taxa_list) <= 3:
            # Star tree for 3 or fewer
            if len(taxa_list) == 1:
                return labels[taxa_list[0]]
            elif len(taxa_list) == 2:
                return f"({labels[taxa_list[0]]},{labels[taxa_list[1]]})"
            else:
                return f"({labels[taxa_list[0]]},({labels[taxa_list[1]]},{labels[taxa_list[2]]}))"
        
        # Find a split that partitions taxa_list
        # The split must be a subset of taxa_list or its complement must be a subset?
        # A split in the global tree restricts to this subtree.
        # If a global split S is such that S & taxa_set is not empty and not all of taxa_set,
        # then it partitions this subtree.
        
        found_split = None
        for split in splits:
            intersection = split & taxa_set
            if intersection and intersection != taxa_set:
                # This split partitions the current set
                # We need to ensure it's "valid" for this subset
                # i.e., it doesn't just cut off 1 taxon from the whole tree but is internal here?
                # Any non-trivial partition is valid for recursion.
                found_split = intersection
                break
        
        if found_split is None:
            # No split found, return star
            return "(" + ",".join([labels[t] for t in taxa_list]) + ")"
            # Note: Newick for star with >3 leaves is technically (a,(b,(c,d))) etc.
            # But a "star" in Newick is often represented as ((a,b),(c,d)) etc.
            # To be safe, construct a balanced tree or simple chain.
            # Simple chain:
            if len(taxa_list) == 4:
                return f"({labels[taxa_list[0]]},({labels[taxa_list[1]]},({labels[taxa_list[2]]},{labels[taxa_list[3]]})))"
            # General fallback:
            current = labels[taxa_list[0]]
            for t in taxa_list[1:]:
                current = f"({current},{labels[t]})"
            return current

        # Recurse
        left_taxa = list(found_split)
        right_taxa = list(taxa_set - found_split)
        
        left_tree = build_tree(left_taxa, splits)
        right_tree = build_tree(right_taxa, splits)
        
        return f"({left_tree},{right_tree})"

    final_tree = build_tree(list(range(n_taxa)), selected_splits)
    return final_tree + ";"
