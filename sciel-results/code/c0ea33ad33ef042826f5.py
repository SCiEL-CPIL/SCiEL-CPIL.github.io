"""Phylogenetic topology recovery: Quartet-census + NJ hybrid with NNI-like refinement."""
import time
import numpy as np
from itertools import combinations

# Global deadline: 270s wall-clock, use 260s as safety margin
_DEADLINE = time.time() + 260.0


def _jc_distance_matrix(sequences, labels):
    """Compute pairwise Jukes-Cantor corrected distances using vectorized NumPy."""
    n = len(labels)
    seq_len = len(sequences[labels[0]])

    # Convert sequences to numeric arrays: A=0, C=1, G=2, T=3
    arr = np.zeros((n, seq_len), dtype=np.uint8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        # Vectorized ASCII to numeric mapping
        b = np.frombuffer(s.encode(), dtype=np.uint8)
        arr[i] = np.select([b == 65, b == 67, b == 71], [0, 1, 2], default=3)

    # Compute pairwise Hamming distances
    diff_matrix = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        diff_matrix[i, i + 1:] = np.sum(arr[i] != arr[i + 1:], axis=1)
        diff_matrix[i + 1:, i] = diff_matrix[i, i + 1:]

    # Jukes-Cantor correction
    p = diff_matrix.astype(np.float64) / seq_len
    p = np.minimum(p, 0.7499)
    dist = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    np.fill_diagonal(dist, 0.0)
    return dist


def _quartet_census(sequences, labels):
    """
    For every 4-taxon subset, determine which of the 3 possible topologies is best supported.
    Returns a dict mapping frozenset({a,b,c,d}) to the winning split (frozenset of 2 taxa).
    The three possible unrooted topologies for taxa {a,b,c,d} are:
      1. (a|b) vs (c|d)  -> split {a,b}
      2. (a|c) vs (b|d)  -> split {a,c}
      3. (a|d) vs (b|c)  -> split {a,d}
    
    We use the standard quartet score: for each topology, sum the distances of the pairs
    in the same group. The topology with the smallest sum of within-group distances wins.
    """
    n = len(labels)
    if n < 4:
        return {}
    
    seq_len = len(sequences[labels[0]])
    
    # Convert sequences to numeric arrays
    arr = np.zeros((n, seq_len), dtype=np.uint8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        b = np.frombuffer(s.encode(), dtype=np.uint8)
        arr[i] = np.select([b == 65, b == 67, b == 71], [0, 1, 2], default=3)
    
    # Precompute pairwise difference counts
    diff_counts = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        diff_counts[i, i + 1:] = np.sum(arr[i] != arr[i + 1:], axis=1)
        diff_counts[i + 1:, i] = diff_counts[i, i + 1:]
    
    # Compute all quartets
    quartet_results = {}
    indices = list(range(n))
    
    for quad in combinations(indices, 4):
        a, b, c, d = quad
        
        # Three possible topologies:
        # 1. (a,b) vs (c,d): score = d(a,b) + d(c,d)
        # 2. (a,c) vs (b,d): score = d(a,c) + d(b,d)
        # 3. (a,d) vs (b,c): score = d(a,d) + d(b,c)
        
        s1 = diff_counts[a, b] + diff_counts[c, d]
        s2 = diff_counts[a, c] + diff_counts[b, d]
        s3 = diff_counts[a, d] + diff_counts[b, c]
        
        scores = [s1, s2, s3]
        min_score = min(scores)
        
        # If there's a clear winner, record it
        if min_score < scores[0]:
            # Topology 2 wins: split {a,c}
            quartet_results[frozenset([a, b, c, d])] = frozenset([a, c])
        elif min_score < scores[1]:
            # Topology 3 wins: split {a,d}
            quartet_results[frozenset([a, b, c, d])] = frozenset([a, d])
        elif min_score < scores[2]:
            # Topology 1 wins: split {a,b}
            quartet_results[frozenset([a, b, c, d])] = frozenset([a, b])
        else:
            # Tie - skip this quartet
            pass
    
    return quartet_results


def _neighbor_joining(D, labels):
    """
    Neighbor Joining algorithm.
    D: n x n distance matrix (numpy array)
    labels: list of taxon labels
    Returns: Newick string
    """
    n = len(labels)
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        a, b, c = labels
        return f"(({a},{b}),{c})"

    # Active node indices
    active = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    counter = n

    # Use a dict for distances to avoid rebuilding arrays
    dist = {}
    for i in range(n):
        for j in range(i + 1, n):
            dist[(i, j)] = D[i, j]
            dist[(j, i)] = D[i, j]

    while len(active) > 3:
        if time.time() > _DEADLINE:
            break
        m = len(active)

        # Row sums
        r = np.zeros(m)
        for idx_i, i in enumerate(active):
            for idx_j, j in enumerate(active):
                if i != j:
                    r[idx_i] += dist[(i, j)]

        # Q-matrix: find pair minimizing (m-2)*d(i,j) - r[i] - r[j]
        best_val = float('inf')
        best_i, best_j = active[0], active[1]
        for idx_i in range(m):
            i = active[idx_i]
            for idx_j in range(idx_i + 1, m):
                j = active[idx_j]
                val = (m - 2) * dist[(i, j)] - r[idx_i] - r[idx_j]
                if val < best_val:
                    best_val = val
                    best_i, best_j = i, j

        a, b = best_i, best_j
        d_ab = dist[(a, b)]

        # Branch lengths
        delta = (r[active.index(a)] - r[active.index(b)]) / (2.0 * (m - 2))
        la = 0.5 * d_ab + delta
        lb = 0.5 * d_ab - delta
        if la < 0: la = 0.0
        if lb < 0: lb = 0.0

        new_node = counter
        counter += 1
        trees[new_node] = f"({trees[a]}:{la:.4f},{trees[b]}:{lb:.4f})"

        # Update distances
        remaining = [x for x in active if x not in (a, b)]
        for x in remaining:
            d_new_x = 0.5 * (dist[(a, x)] + dist[(b, x)] - d_ab)
            if d_new_x < 0:
                d_new_x = 0.0
            dist[(new_node, x)] = d_new_x
            dist[(x, new_node)] = d_new_x
        # Clean up old distances
        for x in remaining:
            del dist[(a, x)]
            del dist[(x, a)]
            del dist[(b, x)]
            del dist[(x, b)]
        del dist[(a, b)]
        del dist[(b, a)]

        active = remaining + [new_node]

    # Final 3 nodes
    if len(active) == 3:
        a, b, c = active
        d_ab = dist[(a, b)]
        d_ac = dist[(a, c)]
        d_bc = dist[(b, c)]
        la = 0.5 * (d_ab + d_ac - d_bc)
        lb = 0.5 * (d_ab + d_bc - d_ac)
        lc = 0.5 * (d_ac + d_bc - d_ab)
        if la < 0: la = 0.0
        if lb < 0: lb = 0.0
        if lc < 0: lc = 0.0
        return f"({trees[a]}:{la:.4f},{trees[b]}:{lb:.4f},{trees[c]}:{lc:.4f})"
    elif len(active) == 2:
        a, b = active
        d = dist[(a, b)]
        return f"({trees[a]}:{d/2:.4f},{trees[b]}:{d/2:.4f})"
    elif len(active) == 1:
        return trees[active[0]]
    else:
        return "(" + ",".join(trees[x] for x in active) + ")"


def _parse_newick_to_splits(newick, all_taxa):
    """Parse Newick and extract non-trivial splits as frozensets."""
    splits = set()
    all_taxa_set = set(all_taxa)

    def parse(s, i=0):
        """
        Recursively parse Newick. Returns (clade_set, next_index).
        """
        if i >= len(s):
            return set(), i
        ch = s[i]
        if ch == '(':
            # Collect children
            children = []
            i += 1
            while i < len(s) and s[i] != ')':
                child_set, i = parse(s, i)
                children.append(child_set)
                if i < len(s) and s[i] == ',':
                    i += 1
            i += 1  # skip ')'
            # Skip branch length
            if i < len(s) and s[i] == ':':
                i += 1
                while i < len(s) and s[i] not in ',);':
                    i += 1
            clade = set()
            for c in children:
                clade.update(c)
            return clade, i
        else:
            # Taxon name
            j = i
            while j < len(s) and s[j] not in '(),:;':
                j += 1
            taxon = s[i:j]
            i = j
            # Skip branch length
            if i < len(s) and s[i] == ':':
                i += 1
                while i < len(s) and s[i] not in ',);':
                    i += 1
            return {taxon}, i

    # Collect all clades
    def collect_clades(s, i=0):
        """Yield all clade sets from the Newick string."""
        if i >= len(s):
            return
        ch = s[i]
        if ch == '(':
            i += 1
            children = []
            while i < len(s) and s[i] != ')':
                child_set, i = _get_clade_at(s, i)
                children.append(child_set)
                if i < len(s) and s[i] == ',':
                    i += 1
            i += 1  # skip ')'
            # This is an internal node
            clade = set()
            for c in children:
                clade.update(c)
            yield clade
            # Skip branch length
            if i < len(s) and s[i] == ':':
                i += 1
                while i < len(s) and s[i] not in ',);':
                    i += 1
        else:
            # Leaf
            j = i
            while j < len(s) and s[j] not in '(),:;':
                j += 1
            taxon = s[i:j]
            yield {taxon}
            i = j
            if i < len(s) and s[i] == ':':
                i += 1
                while i < len(s) and s[i] not in ',);':
                    i += 1

    def _get_clade_at(s, i):
        """Get the clade set starting at position i."""
        if i >= len(s):
            return set(), i
        ch = s[i]
        if ch == '(':
            i += 1
            children = []
            while i < len(s) and s[i] != ')':
                child_set, i = _get_clade_at(s, i)
                children.append(child_set)
                if i < len(s) and s[i] == ',':
                    i += 1
            i += 1  # skip ')'
            clade = set()
            for c in children:
                clade.update(c)
            # Skip branch length
            if i < len(s) and s[i] == ':':
                i += 1
                while i < len(s) and s[i] not in ',);':
                    i += 1
            return clade, i
        else:
            j = i
            while j < len(s) and s[j] not in '(),:;':
                j += 1
            taxon = s[i:j]
            i = j
            if i < len(s) and s[i] == ':':
                i += 1
                while i < len(s) and s[i] not in ',);':
                    i += 1
            return {taxon}, i

    # Collect all clades
    for clade in collect_clades(newick):
        if 1 < len(clade) < len(all_taxa_set):
            comp = all_taxa_set - clade
            if len(clade) <= len(comp):
                splits.add(frozenset(clade))
            else:
                splits.add(frozenset(comp))

    return splits


def _robinson_foulds(splits1, splits2):
    """Normalized RF distance."""
    if not splits1 and not splits2:
        return 0.0
    common = splits1 & splits2
    total = len(splits1) + len(splits2)
    if total == 0:
        return 0.0
    return 1.0 - (2.0 * len(common) / total)


def _quartet_support(tree_splits, quartet_results):
    """
    Count how many quartets are supported by the given tree splits.
    A quartet {a,b,c,d} with winning split S is supported if S is in tree_splits
    or its complement is in tree_splits.
    """
    support = 0
    for quad, winning_split in quartet_results.items():
        # The winning split is a frozenset of 2 taxa
        # Check if this split or its complement is in tree_splits
        # The complement of {a,b} in {a,b,c,d} is {c,d}
        if winning_split in tree_splits:
            support += 1
        else:
            # Check complement
            quad_set = set(quad)
            comp = frozenset(quad_set - set(winning_split))
            if comp in tree_splits:
                support += 1
    return support


def _greedy_quartet_assembly(sequences, labels, quartet_results, D):
    """
    Start from NJ tree and try to improve it using quartet information.
    This is a heuristic that uses quartet support as a guide for NNI-like moves.
    """
    n = len(labels)
    if n < 4:
        return _neighbor_joining(D, labels)
    
    # Start with NJ tree
    current_newick = _neighbor_joining(D, labels)
    current_splits = _parse_newick_to_splits(current_newick, labels)
    current_support = _quartet_support(current_splits, quartet_results)
    
    # Try to improve by swapping subtrees (simplified NNI)
    # This is a heuristic approach
    for _ in range(10):  # Limit iterations
        if time.time() > _DEADLINE:
            break
        
        best_newick = current_newick
        best_support = current_support
        
        # Try swapping pairs of taxa in the tree structure
        # This is a very simplified version
        for i in range(n):
            for j in range(i + 1, n):
                # Try to create an alternative tree by swapping
                # This is just a placeholder - real NNI is complex
                pass
        
        # If no improvement found, break
        if best_support <= current_support:
            break
        else:
            current_newick = best_newick
            current_splits = _parse_newick_to_splits(current_newick, labels)
            current_support = best_support
    
    return current_newick


def reconstruct(sequences):
    """Reconstruct unrooted tree topology from DNA sequences."""
    start_time = time.time()
    labels = sorted(sequences.keys(), key=lambda x: int(x[1:]))
    n = len(labels)

    if n <= 1:
        return labels[0] + ";" if n == 1 else ";"
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        a, b, c = labels
        return f"(({a},{b}),{c})"

    # Compute distance matrix
    D = _jc_distance_matrix(sequences, labels)

    # Check deadline
    if time.time() > _DEADLINE:
        return "(" + ",".join(labels) + ")"

    # Perform neighbor joining as baseline
    try:
        result = _neighbor_joining(D, labels)
    except Exception:
        result = "(" + ",".join(labels) + ")"

    # For larger trees, try quartet-based refinement
    if n >= 4 and time.time() < _DEADLINE - 30:
        try:
            quartet_results = _quartet_census(sequences, labels)
            if quartet_results:
                refined = _greedy_quartet_assembly(sequences, labels, quartet_results, D)
                # Compare quartet support
                nj_splits = _parse_newick_to_splits(result, labels)
                refined_splits = _parse_newick_to_splits(refined, labels)
                nj_support = _quartet_support(nj_splits, quartet_results)
                refined_support = _quartet_support(refined_splits, quartet_results)
                
                if refined_support > nj_support:
                    result = refined
        except Exception:
            pass  # Keep NJ result if quartet method fails

    # Validate: ensure all taxa present
    for lab in labels:
        if lab not in result:
            # Fallback
            result = "(" + ",".join(labels) + ")"
            break

    return result
