# EVOLVE-BLOCK-START
import time
import math
import numpy as np
import sys
from itertools import combinations

_START = None
_DEADLINE_FRACTION = 0.85 # Stop at 85% of 270s to be safe

def _now():
    return time.time()

def _init_time():
    global _START
    if _START is None:
        _START = time.time()

def _time_left():
    _init_time()
    return max(0.0, 270.0 * _DEADLINE_FRACTION - (time.time() - _START))

def _seqs_to_array(sequences):
    labels = sorted(sequences, key=lambda x: int(x[1:]))
    seqs = [sequences[l] for l in labels]
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    # Using uint8 for compactness
    arr = np.array([[mapping[c] for c in s] for s in seqs], dtype=np.uint8)
    return labels, arr

def _jc_matrix(arr):
    n, L = arr.shape
    # Calculate pairwise differences efficiently
    diff = (arr[:, None, :] != arr[None, :, :])
    counts = diff.sum(axis=2)
    p = counts / L
    p_safe = np.clip(p, 0, 0.7499)
    d = -0.75 * np.log(1 - (4.0/3.0) * p_safe)
    np.fill_diagonal(d, 0)
    return d

def _nj(dist, labels):
    n = len(labels)
    if n == 1:
        return labels[0], set()
    if n == 2:
        return f"({labels[0]},{labels[1]})", set()
    
    all_leaves = set(labels)
    nodes = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    leaf_sets = {i: {labels[i]} for i in range(n)}
    
    # Copy dist to dict for easy access/modification
    d = {}
    for i in range(n):
        d[i] = {}
        for j in range(n):
            d[i][j] = dist[i][j]
    
    splits = set()
    
    while len(nodes) > 3:
        n_curr = len(nodes)
        total = {a: sum(d[a][b] for b in nodes if b != a) for a in nodes}
        
        best_criterion = float('inf')
        best_a, best_b = -1, -1
        
        for i, a in enumerate(nodes):
            for b in nodes[i+1:]:
                criterion = (n_curr - 2) * d[a][b] - total[a] - total[b]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_a, best_b = a, b
        
        a, b = best_a, best_b
        
        new_id = -1 - len(nodes)
        trees[new_id] = f"({trees[a]},{trees[b]})"
        leaf_sets[new_id] = leaf_sets[a] | leaf_sets[b]
        
        # Record split
        side1 = frozenset(leaf_sets[new_id])
        side2 = frozenset(all_leaves - leaf_sets[new_id])
        # Canonical form: lexicographically smaller frozenset
        if tuple(sorted(side1)) < tuple(sorted(side2)):
            split = side1
        else:
            split = side2
        if 1 < len(split) < n - 1:
            splits.add(split)
            
        d[new_id] = {}
        for x in nodes:
            if x == a or x == b:
                continue
            d[new_id][x] = 0.5 * (d[a][x] + d[b][x] - d[a][b])
            d[x][new_id] = d[new_id][x]
            
        nodes.remove(a)
        nodes.remove(b)
        nodes.append(new_id)
        
    final_tree = "(" + ",".join(trees[x] for x in nodes) + ")"
    return final_tree, splits

def _quartet_likelihood(arr, i, j, k, l):
    """
    Compute ML likelihoods for the three quartet topologies on taxa i,j,k,l.
    Returns (lik_12|34, lik_13|24, lik_14|23) for topologies ((i,j),(k,l)), ((i,k),(j,l)), ((i,l),(j,k))
    Under JC model, we can use a simplified approach: count matches/mismatches.
    """
    # For JC model, the likelihood for a quartet can be computed from site patterns
    # Each site has 4^4 = 256 possible patterns for 4 taxa
    # We can categorize by which taxa match
    
    a = arr[i]
    b = arr[j]
    c = arr[k]
    d = arr[l]
    
    # Count concordant vs discordant patterns for each topology
    # Topology 1: ((a,b),(c,d)) - a,b should be similar; c,d should be similar
    # Topology 2: ((a,c),(b,d)) - a,c should be similar; b,d should be similar
    # Topology 3: ((a,d),(b,c)) - a,d should be similar; b,c should be similar
    
    L = len(a)
    
    # For each site, determine which topology is best supported
    # Under JC, the most likely topology is the one where the two pairs have fewer differences
    
    # Count pairwise differences
    ab_diff = np.sum(a != b)
    cd_diff = np.sum(c != d)
    ac_diff = np.sum(a != c)
    bd_diff = np.sum(b != d)
    ad_diff = np.sum(a != d)
    bc_diff = np.sum(b != c)
    
    # For topology ((a,b),(c,d)): internal branch connects (ab) to (cd)
    # Likelihood proportional to exp(-0.75 * (ab_diff + cd_diff + 2*cross_diff))
    # Simplified: use sum of within-pair distances as proxy
    
    # More accurate: compute actual ML for JC quartet
    # For JC, we can use the fact that P(pattern | tree, branch lengths) can be computed
    # But for speed, use a heuristic based on distance additivity
    
    # Distance-based quartet test:
    # Topology ((a,b),(c,d)) is supported if d_ab + d_cd < d_ac + d_bd and d_ab + d_cd < d_ad + d_bc
    d_ab = -0.75 * math.log(1 - (4.0/3.0) * min(ab_diff / L, 0.7499))
    d_cd = -0.75 * math.log(1 - (4.0/3.0) * min(cd_diff / L, 0.7499))
    d_ac = -0.75 * math.log(1 - (4.0/3.0) * min(ac_diff / L, 0.7499))
    d_bd = -0.75 * math.log(1 - (4.0/3.0) * min(bd_diff / L, 0.7499))
    d_ad = -0.75 * math.log(1 - (4.0/3.0) * min(ad_diff / L, 0.7499))
    d_bc = -0.75 * math.log(1 - (4.0/3.0) * min(bc_diff / L, 0.7499))
    
    # Quartet scores: lower is better for the "wrong" topology
    # For ((a,b),(c,d)): score = d_ab + d_cd - d_ac - d_bd (should be negative if supported)
    score1 = d_ab + d_cd - d_ac - d_bd
    score2 = d_ac + d_bd - d_ab - d_cd
    score3 = d_ad + d_bc - d_ab - d_cd
    
    # Convert to weights: larger weight for more supported topology
    # Use softmax-like weighting
    eps = 1e-10
    w1 = max(0, -score1) + eps
    w2 = max(0, -score2) + eps
    w3 = max(0, -score3) + eps
    
    # Normalize
    total = w1 + w2 + w3
    if total > 0:
        w1 /= total
        w2 /= total
        w3 /= total
    
    return w1, w2, w3

def _quartet_voting_tree(labels, arr):
    """
    Build tree using quartet voting.
    1. For every quartet of taxa, compute ML weights for the 3 topologies
    2. Vote for splits
    3. Build tree from the most supported splits
    """
    n = len(labels)
    if n <= 3:
        if n == 1:
            return labels[0]
        elif n == 2:
            return f"({labels[0]},{labels[1]})"
        else:
            return f"({labels[0]},{labels[1]},{labels[2]})"
    
    # Index mapping
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    # For each quartet, vote on the three possible splits
    # Split representation: frozenset of taxa on one side (canonical: smaller lex)
    split_votes = {}
    
    # Iterate over all quartets
    num_quartets = 0
    max_quartets = 50000  # Safety limit
    
    for i, j, k, l in combinations(range(n), 4):
        if num_quartets >= max_quartets:
            break
        if _time_left() < 5.0:
            break
            
        w1, w2, w3 = _quartet_likelihood(arr, i, j, k, l)
        
        # Topology 1: ((i,j),(k,l)) -> split {i,j} vs {k,l}
        # Topology 2: ((i,k),(j,l)) -> split {i,k} vs {j,l}
        # Topology 3: ((i,l),(j,k)) -> split {i,l} vs {j,k}
        
        # Canonical split: use frozenset of the smaller side
        def canon_split(side1, side2):
            if len(side1) <= len(side2):
                return frozenset(side1)
            else:
                return frozenset(side2)
        
        s1 = canon_split({i, j}, {k, l})
        s2 = canon_split({i, k}, {j, l})
        s3 = canon_split({i, l}, {j, k})
        
        split_votes[s1] = split_votes.get(s1, 0.0) + w1
        split_votes[s2] = split_votes.get(s2, 0.0) + w2
        split_votes[s3] = split_votes.get(s3, 0.0) + w3
        
        num_quartets += 1
    
    if not split_votes:
        # Fallback to NJ
        d = _jc_matrix(arr)
        tree, _ = _nj(d, labels)
        return tree
    
    # Get top splits by vote count
    sorted_splits = sorted(split_votes.items(), key=lambda x: -x[1])
    
    # Greedily select compatible splits
    # Two splits are compatible if neither side of one is a proper subset of a side of the other
    selected_splits = []
    all_leaves = set(range(n))
    
    for split, weight in sorted_splits:
        side1 = split
        side2 = all_leaves - split
        
        # Check compatibility with already selected splits
        compatible = True
        for existing in selected_splits:
            ex_side1 = existing
            ex_side2 = all_leaves - existing
            
            # Check if side1 and ex_side1 are incompatible
            # Incompatible if one is a proper subset of the other
            if side1 < ex_side1 or side1 > ex_side1:
                compatible = False
                break
            if side1 < ex_side2 or side1 > ex_side2:
                compatible = False
                break
            if side2 < ex_side1 or side2 > ex_side1:
                compatible = False
                break
            if side2 < ex_side2 or side2 > ex_side2:
                compatible = False
                break
        
        if compatible:
            selected_splits.append(split)
    
    # Build tree from selected splits using a recursive approach
    # Or use the splits to guide an NJ-like algorithm
    
    # For now, use the split information to weight the distance matrix
    # and run NJ with modified distances
    
    # Alternative: build tree from splits directly
    # Use a simple approach: start with all taxa, merge based on splits
    
    # Let's use the split votes to create a weighted distance matrix
    # and run NJ on it
    
    # Compute base distance matrix
    d = _jc_matrix(arr)
    
    # Modify distances to reflect quartet support
    # For each pair, if they appear together in a highly supported split, reduce their distance
    
    # For simplicity, just use NJ on the original distance matrix
    # but use the quartet votes to select among candidate trees
    
    # Generate multiple candidate trees using different subsets of top splits
    candidates = []
    
    # Candidate 1: NJ on full distance matrix
    tree_nj, splits_nj = _nj(d, labels)
    candidates.append((tree_nj, splits_nj, "NJ"))
    
    # Candidate 2: NJ with distances adjusted by quartet votes
    # For pairs in same side of a high-vote split, reduce distance
    d_adjusted = d.copy()
    for split, weight in sorted_splits[:20]:  # Use top 20 splits
        side1 = split
        side2 = all_leaves - split
        # Reduce distances within side1 and within side2
        for i in side1:
            for j in side1:
                if i != j:
                    d_adjusted[i][j] *= 0.95
        for i in side2:
            for j in side2:
                if i != j:
                    d_adjusted[i][j] *= 0.95
    
    tree_adj, splits_adj = _nj(d_adjusted, labels)
    candidates.append((tree_adj, splits_adj, "Adjusted"))
    
    # Score each candidate by how many of its splits are in the top quartet-voted splits
    top_split_set = set(s for s, w in sorted_splits[:50])
    
    best_candidate = None
    best_score = -1
    
    for tree, splits, name in candidates:
        # Count how many splits in this tree are in the top quartet-voted splits
        common = len(splits & top_split_set)
        if common > best_score:
            best_score = common
            best_candidate = tree
    
    if best_candidate is None:
        best_candidate = tree_nj
    
    return best_candidate

def reconstruct(sequences):
    global _START
    _START = time.time()
    
    labels, arr = _seqs_to_array(sequences)
    n = len(labels)
    
    # For small n, use quartet voting
    # For large n, quartet voting is O(n^4) which may be too slow
    # Use NJ + bootstrap for large n, quartet voting for small n
    
    if n <= 12:
        # Use quartet voting
        tree = _quartet_voting_tree(labels, arr)
        return tree
    else:
        # Use NJ + bootstrap consensus
        d_full = _jc_matrix(arr)
        tree_full, splits_full = _nj(d_full, labels)
        
        candidates = [(tree_full, splits_full)]
        split_counts = {}
        
        # Count splits in the first tree (NJ)
        for s in splits_full:
            split_counts[s] = split_counts.get(s, 0) + 1
            
        L = arr.shape[1]
        num_bootstraps = 0
        max_bootstraps = 2000
        
        while num_bootstraps < max_bootstraps and _time_left() > 5.0:
            indices = np.random.randint(0, L, L)
            arr_boot = arr[:, indices]
            d_boot = _jc_matrix(arr_boot)
            tree_boot, splits_boot = _nj(d_boot, labels)
            candidates.append((tree_boot, splits_boot))
            
            # Update split counts
            for s in splits_boot:
                split_counts[s] = split_counts.get(s, 0) + 1
                
            num_bootstraps += 1
        
        if len(candidates) > 1:
            total_bootstraps = len(candidates)
            threshold = 0.5 * total_bootstraps
            
            # Identify strong splits
            strong_splits = set()
            for s, c in split_counts.items():
                if c > threshold:
                    strong_splits.add(s)
                    
            best_idx = 0
            best_score = -1
            second_best_score = float('inf')
            
            for i in range(len(candidates)):
                splits_i = candidates[i][1]
                common = len(splits_i & strong_splits)
                
                if common > best_score:
                    best_score = common
                    best_idx = i
                    second_best_score = len(splits_i.symmetric_difference(candidates[0][1]))
                elif common == best_score:
                    rf_to_nj = len(splits_i.symmetric_difference(candidates[0][1]))
                    if rf_to_nj < second_best_score:
                        best_idx = i
                        second_best_score = rf_to_nj
                        
            return candidates[best_idx][0]
        else:
            return tree_full

# EVOLVE-BLOCK-END
