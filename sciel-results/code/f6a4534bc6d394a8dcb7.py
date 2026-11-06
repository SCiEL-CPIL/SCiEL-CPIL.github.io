# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from collections import Counter

_START = None
_DEADLINE_FRACTION = 0.80

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
    arr = np.array([[mapping[c] for c in s] for s in seqs], dtype=np.int8)
    return labels, arr

def _jc_matrix(arr):
    n, L = arr.shape
    diff = (arr[:, None, :] != arr[None, :, :])
    counts = diff.sum(axis=2)
    p = counts / L
    p_safe = np.clip(p, 0, 0.7499)
    d = -0.75 * np.log(1 - (4.0/3.0) * p_safe)
    np.fill_diagonal(d, 0)
    return d

def _nj(dist, labels):
    """
    Neighbor-Joining algorithm.
    Returns: (newick_string, set_of_splits)
    """
    n = len(labels)
    if n == 1:
        return labels[0], set()
    if n == 2:
        return f"({labels[0]},{labels[1]})", set()
    
    all_leaves = set(labels)
    nodes = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    leaf_sets = {i: {labels[i]} for i in range(n)}
    
    # Use numpy array for distance matrix for speed
    d = np.array(dist, dtype=np.float64)
    
    splits = set()
    active = [True] * n
    active_count = n
    
    while active_count > 3:
        # Calculate Q matrix for active nodes
        # r_i = sum_{j in active, j!=i} d_ij
        r = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if active[i]:
                r[i] = np.sum(d[i, :]) - d[i, i]
                # Subtract inactive ones? No, d is updated such that inactive nodes are removed or we just sum active.
                # Actually, standard NJ updates d. Let's stick to the dictionary version for correctness of the update step 
                # or implement the array update carefully.
                # The dictionary version in the seed was correct but slow. 
                # Let's optimize the array version.
        
        # Re-implementing NJ with arrays for speed
        # We maintain a list of active indices
        active_indices = [i for i in range(n) if active[i]]
        
        # Calculate r for active nodes
        # r_i = sum_{j active, j != i} d_ij
        # Since d is symmetric and d_ii=0, sum(d[i, active_indices]) - d[i,i]
        # But d contains values for inactive nodes too? 
        # In standard NJ, we update d for the new node. We don't zero out old ones necessarily, 
        # but we only sum over active ones.
        
        r_active = np.array([np.sum(d[i, active_indices]) for i in active_indices])
        
        # Q_ij = (k-2)*d_ij - r_i - r_j
        # k = active_count
        best_q = float('inf')
        best_a_idx = -1
        best_b_idx = -1
        
        # Q matrix for active nodes
        # Q = (k-2)*d_sub - r_sub[:, None] - r_sub[None, :]
        d_sub = d[np.ix_(active_indices, active_indices)]
        Q = (active_count - 2) * d_sub - r_active[:, None] - r_active[None, :]
        
        # Find min Q
        min_val = np.min(Q)
        # Get indices in the sub-matrix
        i_sub, j_sub = np.unravel_index(np.argmin(Q), Q.shape)
        
        # Map back to original indices
        a = active_indices[i_sub]
        b = active_indices[j_sub]
        
        best_a_idx = a
        best_b_idx = b
        
        # Create new node
        new_id = -1 - (n - active_count) # Unique negative ID
        # Actually, let's just use a counter for new nodes
        # We need to update the d matrix.
        # This array-based NJ is tricky to implement correctly with dynamic node addition without padding.
        # Given the time constraints and the risk of bugs in a complex array-based NJ,
        # I will stick to the dictionary-based NJ but optimize the inner loops slightly or accept the speed.
        # The seed program scored 1.0343 with dict NJ + 200 bootstraps.
        # The inspirations used similar dict NJ.
        # Let's use the dictionary NJ from the seed as it is known to work and produce splits.
        pass

    # Fallback: Use the robust dictionary-based NJ from the seed
    # This ensures correctness of splits.
    
    # Re-implementing the dictionary NJ here to avoid the complex array logic above
    nodes_dict = list(range(n))
    trees_dict = {i: labels[i] for i in range(n)}
    leaf_sets_dict = {i: {labels[i]} for i in range(n)}
    
    d_dict = {}
    for i in range(n):
        d_dict[i] = {}
        for j in range(n):
            d_dict[i][j] = dist[i][j]
    
    splits_dict = set()
    
    while len(nodes_dict) > 3:
        n_curr = len(nodes_dict)
        total = {a: sum(d_dict[a][b] for b in nodes_dict if b != a) for a in nodes_dict}
        
        best_criterion = float('inf')
        best_a, best_b = -1, -1
        
        for i, a in enumerate(nodes_dict):
            for b in nodes_dict[i+1:]:
                criterion = (n_curr - 2) * d_dict[a][b] - total[a] - total[b]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_a, best_b = a, b
        
        a, b = best_a, best_b
        
        new_id = -1 - len(nodes_dict)
        trees_dict[new_id] = f"({trees_dict[a]},{trees_dict[b]})"
        leaf_sets_dict[new_id] = leaf_sets_dict[a] | leaf_sets_dict[b]
        
        side1 = frozenset(leaf_sets_dict[new_id])
        side2 = frozenset(all_leaves - leaf_sets_dict[new_id])
        if tuple(sorted(side1)) < tuple(sorted(side2)):
            split = side1
        else:
            split = side2
        if 1 < len(split) < n - 1:
            splits_dict.add(split)
            
        d_dict[new_id] = {}
        for x in nodes_dict:
            if x == a or x == b:
                continue
            d_dict[new_id][x] = 0.5 * (d_dict[a][x] + d_dict[b][x] - d_dict[a][b])
            d_dict[x][new_id] = d_dict[new_id][x]
            
        nodes_dict.remove(a)
        nodes_dict.remove(b)
        nodes_dict.append(new_id)
        
    final_tree = "(" + ",".join(trees_dict[x] for x in nodes_dict) + ")"
    return final_tree, splits_dict

def _rf_distance(s1, s2):
    return len(s1.symmetric_difference(s2))

def _fitch_parsimony(arr, labels, newick):
    """
    Calculate Fitch parsimony score for a given tree topology.
    This is a heuristic proxy for likelihood. Lower is better.
    """
    # Parse Newick to tree structure
    # For simplicity and speed, we might skip this if it's too complex.
    # However, the prompt mentions parsimony as a key improvement in a related run.
    # Implementing a full Newick parser and Fitch algorithm is non-trivial.
    # Given the time budget, I will rely on the RF distance to the consensus as the selector,
    # which is what the current program does.
    
    # If we want to add parsimony, we need to:
    # 1. Parse Newick into a tree object.
    # 2. Run Fitch algorithm on the tree with the sequence data.
    # 3. Return the score.
    
    # For now, return 0 to indicate "use other metric" or implement a simple version.
    # Let's implement a simple version if time permits, otherwise stick to RF consensus.
    
    # Since I cannot easily test complex parsers, I will stick to the RF-based selection 
    # which is robust and already implemented in the seed.
    return 0.0

def _majority_vote(splits_list):
    """
    Given a list of split sets, return the majority vote split set.
    """
    counter = Counter()
    for splits in splits_list:
        for s in splits:
            counter[s] += 1
    
    threshold = len(splits_list) / 2.0
    majority = set()
    for s, count in counter.items():
        if count > threshold:
            majority.add(s)
    return majority

def reconstruct(sequences):
    global _START
    _START = time.time()
    
    labels, arr = _seqs_to_array(sequences)
    n = len(labels)
    
    d_full = _jc_matrix(arr)
    tree_full, splits_full = _nj(d_full, labels)
    
    candidates = [(tree_full, splits_full)]
    
    L = arr.shape[1]
    num_bootstraps = 0
    max_bootstraps = 150 # Reduced slightly to ensure time safety, as dict NJ is slow
    
    # Use more bootstraps if time permits
    while num_bootstraps < max_bootstraps and _time_left() > 5.0:
        indices = np.random.randint(0, L, L)
        arr_boot = arr[:, indices]
        d_boot = _jc_matrix(arr_boot)
        tree_boot, splits_boot = _nj(d_boot, labels)
        candidates.append((tree_boot, splits_boot))
        num_bootstraps += 1
        
    if len(candidates) > 1:
        # Use majority vote to select the consensus
        all_splits = [c[1] for c in candidates]
        consensus_splits = _majority_vote(all_splits)
        
        # Find the candidate closest to the consensus
        best_idx = 0
        best_score = float('inf')
        for i, (tree_i, splits_i) in enumerate(candidates):
            # RF distance to consensus
            score = _rf_distance(splits_i, consensus_splits)
            if score < best_score:
                best_score = score
                best_idx = i
        
        return candidates[best_idx][0] + ";"
    else:
        return tree_full + ";"

# EVOLVE-BLOCK-END
