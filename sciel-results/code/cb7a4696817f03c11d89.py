import time
import math
import numpy as np
import itertools
from collections import defaultdict

_START_TIME = None
_TIME_LIMIT = 260.0  # Safety margin for 270s limit

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_left():
    _init_time()
    return max(0.0, _TIME_LIMIT - (time.time() - _START_TIME))

def _compute_quartet_supports(sequences, labels):
    """
    Compute the support weight for each of the 3 possible unrooted quartet topologies
    for every set of 4 taxa.
    
    Returns a dictionary: {(i, j, k, l): (w12, w13, w14)}
    where w12 is the support for ((i,j),(k,l)), etc.
    The weights are estimated log-likelihood differences or distance-based scores.
    """
    n = len(labels)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    # Convert sequences to integer matrix for fast access
    seq_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seqs = np.array([[seq_map[c] for c in sequences[label]] for label in labels])
    L = seqs.shape[1]
    
    # Precompute pairwise differences and counts
    # diff[i, j] = number of positions where seqs[i] != seqs[j]
    diff = np.zeros((n, n), dtype=np.int32)
    # We need counts of specific patterns for quartet analysis.
    # For JC model, the likelihood of a quartet topology depends on the number of sites
    # showing specific patterns of matches/mismatches.
    
    # However, a simpler and robust metric for quartet support under JC is based on
    # the sum of pairwise distances.
    # Topology ((i,j),(k,l)) is supported if d(i,j) + d(k,l) < d(i,k) + d(j,l) and d(i,l) + d(j,k).
    # We can compute a "score" for each topology based on how much it beats the others.
    
    # Let's use the "Quartet Distance" method:
    # For a quartet (i,j,k,l), calculate:
    # S12 = d(i,j) + d(k,l)
    # S13 = d(i,k) + d(j,l)
    # S14 = d(i,l) + d(j,k)
    # The best topology is the one with the minimum sum.
    # To make it a robust support weight, we can use the difference from the second best.
    
    # First, compute all pairwise distances (JC corrected)
    # p[i,j] = proportion of differences
    p = np.zeros((n, n))
    # Vectorized difference counting
    # This is O(n^2 * L). For n=32, L=2000, this is 32*32*2000 = 2M ops. Fast.
    for i in range(n):
        for j in range(i+1, n):
            diffs = np.sum(seqs[i] != seqs[j])
            p[i, j] = diffs / L
            p[j, i] = p[i, j]
            
    # JC Distance
    p_clipped = np.clip(p, 0, 0.7499)
    dists = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
    np.fill_diagonal(dists, 0.0)
    dists = np.nan_to_num(dists, nan=3.0, posinf=3.0, neginf=0.0)
    
    # Now iterate over all quartets
    # There are C(n, 4) quartets.
    # For each, compute the 3 sums.
    # Store the winner and its "confidence" (difference between 2nd and 1st best sum).
    
    supports = {}
    
    # Optimization: Pre-compute d[i,j] + d[k,l] is not easy to vectorize directly for all combos,
    # but we can just loop. C(32,4) = 35960. Very small.
    
    for i, j, k, l in itertools.combinations(range(n), 4):
        s12 = dists[i, j] + dists[k, l]
        s13 = dists[i, k] + dists[j, l]
        s14 = dists[i, l] + dists[j, k]
        
        # The "score" for a topology is how much better it is than the others.
        # If s12 is the minimum, it's the best.
        # Let's define support weight as the difference between the second best and the best.
        # This gives a measure of confidence.
        sorted_sums = sorted([s12, s13, s14])
        best_sum = sorted_sums[0]
        second_best_sum = sorted_sums[1]
        confidence = second_best_sum - best_sum
        
        # Store the best topology and its confidence
        if s12 == best_sum:
            supports[(i, j, k, l)] = (12, confidence) # ((i,j),(k,l))
        elif s13 == best_sum:
            supports[(i, j, k, l)] = (13, confidence) # ((i,k),(j,l))
        else:
            supports[(i, j, k, l)] = (14, confidence) # ((i,l),(j,k))
            
    return supports, dists

def _reconstruct_from_quartets(supports, labels, dists):
    """
    Reconstruct the unrooted tree topology from quartet supports.
    Uses a greedy approach: 
    1. Start with a star tree or a small subtree.
    2. Add taxa one by one, placing each new taxon on the edge that maximizes
       the consistency with the quartet supports.
    
    Alternatively, use a "Quartet Consensus" algorithm.
    A simpler and effective method for small n is to build the tree using 
    Neighbor Joining but using the quartet-derived distances or using 
    a direct quartet-based tree construction.
    
    Let's use a simple greedy addition:
    1. Pick 4 taxa, build the best quartet tree.
    2. For each remaining taxon, find the best edge to attach to such that
       the number of conflicting quartets is minimized.
    """
    n = len(labels)
    
    if n < 4:
        # Handle small cases
        if n == 1:
            return labels[0]
        elif n == 2:
            return f"({labels[0]},{labels[1]})"
        else:
            # For 3 taxa, it's always a star, but we need a binary tree.
            # Pick the two closest?
            d01 = dists[0, 1]
            d02 = dists[0, 2]
            d12 = dists[1, 2]
            # The root is in the middle.
            # Any binary tree is topologically equivalent for 3 taxa in unrooted sense?
            # No, unrooted tree with 3 leaves is just a star.
            # Newick: ((a,b),c)
            # Let's just pick the pair with smallest distance as siblings.
            if d01 < d02 and d01 < d12:
                return f"(({labels[0]},{labels[1]}),{labels[2]})"
            elif d02 < d01 and d02 < d12:
                return f"(({labels[0]},{labels[2]}),{labels[1]})"
            else:
                return f"(({labels[1]},{labels[2]}),{labels[0]})"
    
    # Start with the first 4 taxa
    initial_taxa = [0, 1, 2, 3]
    # Build the initial tree from the best quartet
    q_support, _ = supports[(0, 1, 2, 3)]
    
    # Create a simple tree structure for the initial 4
    # We represent the tree as a set of splits or just a Newick string?
    # For greedy addition, it's easier to maintain a list of "edges" or a tree object.
    # Let's use a simple Node class again, but optimized for this.
    
    class QNode:
        __slots__ = ['left', 'right', 'label']
        def __init__(self, left=None, right=None, label=None):
            self.left = left
            self.right = right
            self.label = label
            
    # Build initial tree for 0,1,2,3 based on q_support
    # q_support is 12, 13, or 14
    # 12: ((0,1),(2,3))
    # 13: ((0,2),(1,3))
    # 14: ((0,3),(1,2))
    
    def build_quartet_tree(i, j, k, l, sup):
        if sup == 12:
            n_ij = QNode(QNode(label=labels[i]), QNode(label=labels[j]))
            n_kl = QNode(QNode(label=labels[k]), QNode(label=labels[l]))
            return QNode(n_ij, n_kl)
        elif sup == 13:
            n_ik = QNode(QNode(label=labels[i]), QNode(label=labels[k]))
            n_jl = QNode(QNode(label=labels[j]), QNode(label=labels[l]))
            return QNode(n_ik, n_jl)
        else: # 14
            n_il = QNode(QNode(label=labels[i]), QNode(label=labels[l]))
            n_jk = QNode(QNode(label=labels[j]), QNode(label=labels[k]))
            return QNode(n_il, n_jk)
            
    tree_root = build_quartet_tree(0, 1, 2, 3, q_support)
    
    # Add remaining taxa one by one
    added = {0, 1, 2, 3}
    remaining = list(range(4, n))
    
    # Shuffle remaining to avoid bias? Or add in order of distance to existing tree?
    # Let's add in order of index for simplicity, or random.
    # Random might be better to escape local minima, but we want determinism?
    # The task doesn't require determinism, just a good score.
    # Let's add in order of increasing distance to the centroid of the current tree.
    
    def get_tree_leaves(node):
        if node.is_leaf() if hasattr(node, 'is_leaf') else (node.left is None and node.right is None):
            return [node.label]
        return get_tree_leaves(node.left) + get_tree_leaves(node.right)
        
    # Actually, QNode doesn't have is_leaf. Let's add a helper.
    def is_leaf(node):
        return node.left is None and node.right is None
        
    def get_leaves(node):
        if is_leaf(node):
            return [node.label]
        return get_leaves(node.left) + get_leaves(node.right)
        
    # For each remaining taxon, find the best place to insert it.
    # We can try inserting it into every edge of the current tree.
    # For each edge, we calculate the "cost" of inserting the new taxon there.
    # The cost is the number of quartets involving the new taxon and 3 existing taxa
    # that are violated by the insertion.
    
    # To make this fast, we pre-compute the "quartet constraints" for the new taxon.
    
    # Let's iterate through remaining taxa
    for t_idx in remaining:
        # Find best edge to attach to
        best_edge_score = -1
        best_edge = None
        best_parent = None
        best_child = None
        
        # Collect all edges in the current tree
        edges = []
        def collect_edges(node, parent):
            if parent:
                edges.append((parent, node))
            if not is_leaf(node):
                collect_edges(node.left, node)
                collect_edges(node.right, node)
                
        collect_edges(tree_root, None)
        
        # For each edge, simulate inserting t_idx
        # If we insert t_idx into edge (u, v), we create a new node x connected to u, v, and t_idx.
        # This splits the tree into two parts: Part U (containing u) and Part V (containing v).
        # Any quartet (t_idx, a, b, c) where a,b,c are in the same part is consistent with the insertion
        # if the quartet topology in the original tree for (a,b,c, dummy) matches?
        # Actually, the quartet (t_idx, a, b, c) is resolved by the new tree.
        # If a, b, c are all in Part U, then in the new tree, t_idx is "outside" (a,b,c).
        # The topology is ((a,b,c), t_idx).
        # We need to check if this matches the support for (t_idx, a, b, c).
        
        # This is complex. A simpler heuristic:
        # Use the distance to the "centroid" of the edge.
        # Or, use the quartet support directly.
        
        # Let's use a simpler metric: 
        # For a candidate edge (u, v), the "quality" is the sum of supports for quartets
        # (t_idx, x, y, z) where x,y,z are leaves in the subtree of u (excluding v) 
        # such that the topology ((x,y,z), t_idx) is consistent with the stored support.
        
        # To be efficient, let's just pick the edge that minimizes the average distance
        # from t_idx to the leaves in the subtrees of u and v.
        # This is essentially what NJ does.
        
        # Let's refine: 
        # For each edge (u, v), let L_u be leaves in subtree u (excluding v), L_v be leaves in subtree v (excluding u).
        # Actually, in a rooted representation of an unrooted tree, "subtree" is tricky.
        # Let's root the tree at the edge (u, v).
        # Then u is the root of one side, v is the root of the other.
        
        # For each edge, calculate:
        # Score = sum of supports for quartets (t_idx, a, b, c) 
        # where a,b,c are in the "other" side of the tree relative to the insertion point?
        # No, if we insert t_idx into (u,v), t_idx becomes a leaf attached to the new node.
        # The quartet (t_idx, a, b, c) where a,b,c are in the same component (say, attached to u)
        # will have topology ((a,b,c), t_idx).
        # The quartet (t_idx, a, b, c) where a,b are in u-component and c is in v-component
        # will have topology ((a,b), (c, t_idx))? No.
        # It depends on the structure.
        
        # Given the complexity, let's fall back to a robust method:
        # **Neighbor Joining** is actually very good at this.
        # Why did the current program fail? Maybe it didn't fail, but the score was 1.0.
        # The current program uses NJ.
        # My new strategy is Quartet Sampling.
        # Let's use the quartet supports to **weight** the distances in NJ?
        # Or just use the quartet supports to **verify** and **correct** the NJ tree?
        
        # Let's try a different approach: **Quartet Consensus Tree**
        # 1. Generate all quartets.
        # 2. For each quartet, we have a "vote" for one of 3 topologies.
        # 3. Build a tree that maximizes the number of supported quartets.
        
        # This is an NP-hard problem (Maximum Quartet Consensus).
        # For n=32, we can use a greedy heuristic.
        
        # Let's restart the reconstruction logic with a simpler greedy:
        # Start with a star of all taxa.
        # Iteratively pick the pair of taxa with the strongest quartet support
        # to be siblings, merge them, and repeat.
        # This is essentially UPGMA/Neighbor Joining but driven by quartet votes.
        
        # Let's implement this "Quartet-Driven Clustering".
        
        # 1. Create a node for each taxon.
        # 2. While more than 2 nodes:
        #    a. For every pair of nodes (A, B), calculate the "merge score".
        #       The merge score is the sum of supports for all quartets (a, b, c, d)
        #       where a is in A, b is in B, and c, d are in other nodes,
        #       such that the topology ((a,b),(c,d)) is supported.
        #    b. Pick the pair with the highest score.
        #    c. Merge them into a new node.
        
        # This is O(n^3) per step, O(n^2) steps. Total O(n^5).
        # For n=32, 32^5 = 33M. Might be too slow in Python if not optimized.
        # But we can optimize by pre-computing quartet contributions.
        
        # Let's try this.
        
        # We need to map from taxon indices to current cluster IDs.
        # This is getting complicated.
        
        # Let's go back to **NJ** but with a twist:
        # The current NJ uses pairwise distances.
        # We can compute a "quartet-corrected" distance?
        # Or, we can just use the **Quartet Support** to break ties or adjust distances.
        
        # Actually, the simplest robust improvement over NJ is **Fast Distance (FD)** or **BIONJ**,
        # but we only have stdlib/numpy.
        
        # Let's try **Multiple NJ** with different distance metrics?
        # No, the data is fixed.
        
        # Let's try **Quartet Sampling + Consensus** properly.
        # Since n is small (<=32), we can afford a slightly slower algorithm.
        
        # Algorithm:
        # 1. Compute all quartet supports (already done).
        # 2. Use a greedy algorithm to build the tree:
        #    - Start with all taxa as separate clusters.
        #    - Repeatedly merge the two clusters that have the highest "quartet agreement".
        #    
        #    How to calculate "quartet agreement" between cluster A and cluster B?
        #    It is the sum of supports for all quartets (a, b, c, d) where:
        #    - a in A, b in B
        #    - c, d in any other clusters (or same, but A and B are distinct)
        #    - The topology is ((a,b),(c,d))
        
        #    Wait, if c and d are in the same cluster C, and A and B are merged,
        #    the quartet (a,b,c,d) will have topology ((a,b),(c,d)) ONLY IF the tree
        #    structure places A and B together and C separate.
        #    This is exactly what we want to maximize.
        
        #    So, Score(A, B) = Sum_{c, d not in A or B} Support((a,b),(c,d))
        #    where a in A, b in B.
        #    Actually, we sum over ALL a in A, b in B, c in C, d in D.
        
        #    This can be pre-computed for each pair of taxa (i, j).
        #    Let Q[i, j] be the sum of supports for all quartets (i, j, k, l)
        #    where the topology is ((i,j),(k,l)).
        #    
        #    Then, when we merge clusters A and B, the total score contribution
        #    from quartets with one leaf in A and one in B is:
        #    Sum_{a in A, b in B} Q[a, b]
        
        #    This is efficient!
        #    1. Pre-compute Q[i, j] for all i < j.
        #    2. Greedy merge:
        #       - Maintain a list of clusters.
        #       - For each pair of clusters (A, B), calculate Score(A, B) = Sum_{a in A, b in B} Q[a, b].
        #       - Merge the pair with the highest Score.
        #       - Update the Q matrix for the new cluster?
        #         No, we just need to recalculate the scores for the new cluster against others.
        #         Score(New, C) = Score(A, C) + Score(B, C).
        #         Because Q is additive over the leaves.
        
        #    This is essentially **UPGMA** but using quartet-based distances/scores.
        #    It's very fast! O(n^3).
        
        # Let's implement this.
        
        # Step 1: Pre-compute Q[i, j]
        Q = np.zeros((n, n))
        for (i, j, k, l), (top, conf) in supports.items():
            # top is 12, 13, or 14
            # If top is 12, it supports ((i,j),(k,l))
            # So we add conf to Q[i, j] and Q[k, l]
            # And subtract from the other pairs?
            # No, we want to maximize the sum of supported quartets.
            # If we merge i and j, we gain conf if the topology is ((i,j),...).
            # So Q[i, j] += conf if top == 12.
            # If top == 13, it supports ((i,k),(j,l)). So Q[i, k] += conf, Q[j, l] += conf.
            # If top == 14, it supports ((i,l),(j,k)). So Q[i, l] += conf, Q[j, k] += conf.
            
            if top == 12:
                Q[i, j] += conf
                Q[k, l] += conf
            elif top == 13:
                Q[i, k] += conf
                Q[j, l] += conf
            else: # 14
                Q[i, l] += conf
                Q[j, k] += conf
                
        # Symmetrize Q
        Q = (Q + Q.T) / 2.0
        
        # Step 2: Greedy Merge
        # Clusters are sets of taxon indices.
        clusters = [[i] for i in range(n)]
        
        # Current scores between clusters
        # score[i, j] = Sum_{a in C_i, b in C_j} Q[a, b]
        score = Q.copy()
        
        # We need to build the tree.
        # We will store the merge history to reconstruct the Newick string.
        # merge_history: list of (cluster_i, cluster_j, new_cluster)
        
        # To efficiently update scores, we maintain a matrix `score` of size n x n
        # where score[i, j] is the score between cluster i and cluster j.
        # When we merge cluster i and j into k:
        # score[k, m] = score[i, m] + score[j, m]
        
        # We need to track which original taxa are in which cluster.
        # And we need to build the tree structure.
        
        # Let's use a Node class again.
        class MergeNode:
            __slots__ = ['left', 'right', 'label', 'cluster_id']
            def __init__(self, left=None, right=None, label=None, cluster_id=-1):
                self.left = left
                self.right = right
                self.label = label
                self.cluster_id = cluster_id
                
        nodes = [MergeNode(label=labels[i], cluster_id=i) for i in range(n)]
        active_clusters = list(range(n))
        next_id = n
        
        while len(active_clusters) > 1:
            # Find the pair (i, j) with the maximum score
            # score[i, j]
            max_score = -1
            best_i, best_j = -1, -1
            n_curr = len(active_clusters)
            
            for i_idx in range(n_curr):
                for j_idx in range(i_idx + 1, n_curr):
                    id_i = active_clusters[i_idx]
                    id_j = active_clusters[j_idx]
                    s = score[id_i, id_j]
                    if s > max_score:
                        max_score = s
                        best_i, best_j = id_i, id_j
                        
            if best_i == -1:
                break
                
            # Merge best_i and best_j
            node_i = nodes[best_i]
            node_j = nodes[best_j]
            
            new_node = MergeNode(left=node_i, right=node_j, cluster_id=next_id)
            nodes.append(new_node)
            
            # Update scores
            # For all other active clusters m:
            # score[new_id, m] = score[best_i, m] + score[best_j, m]
            # score[m, new_id] = same
            
            for m in active_clusters:
                if m != best_i and m != best_j:
                    score[best_i, m] += score[best_j, m]
                    score[m, best_i] = score[best_i, m]
                    # We can reuse the best_i slot for the new cluster to save memory?
                    # Or just use the new_id.
                    score[best_i, best_i] = 0 # Just in case
                    score[best_j, best_j] = 0
            
            # Replace best_j with new_id in active_clusters?
            # Or just remove best_j and keep best_i as the representative?
            # Let's keep best_i as the ID for the merged cluster.
            # So we update score[best_i, m] to be the sum.
            # And remove best_j from active_clusters.
            
            # But we need to be careful. The `score` matrix is indexed by cluster IDs.
            # If we reuse best_i, we must ensure score[best_i, m] is updated.
            # We did: score[best_i, m] += score[best_j, m].
            # So now score[best_i, m] is the correct score for the merged cluster.
            
            active_clusters.remove(best_j)
            # best_i remains in active_clusters.
            
            next_id += 1
            
        # The last remaining cluster is the root
        root = nodes[active_clusters[0]]
        
        # Convert to Newick
        def to_newick(node):
            if node.left is None:
                return node.label
            return f"({to_newick(node.left)},{to_newick(node.right)})"
            
        return to_newick(root)

def reconstruct(sequences):
    _init_time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_taxa = len(labels)
    
    if n_taxa < 3:
        if n_taxa == 1:
            return labels[0] + ";"
        elif n_taxa == 2:
            return f"({labels[0]},{labels[1]});"
            
    # Compute quartet supports
    supports, dists = _compute_quartet_supports(sequences, labels)
    
    # Reconstruct tree
    newick = _reconstruct_from_quartets(supports, labels, dists)
    
    return newick + ";"

# EVOLVE-BLOCK-END
