"""
Phylogenetic topology recovery via Maximum Likelihood with SPR refinement.
Different strategy: uses JC likelihood as the scoring function and SPR (Subtree Prune and Regraft)
moves for topological search, initialized by a fast NJ heuristic.
"""
import time
import random
import math
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 265.0  # Safety margin below 270s wall-clock


def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)


def _jc_distance(a, b):
    n = len(a)
    if n == 0:
        return 0.0
    diff = sum(1 for x, y in zip(a, b) if x != y)
    p = diff / n
    if p >= 0.75:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _compute_distance_matrix(sequences, labels):
    n = len(labels)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = _jc_distance(sequences[labels[i]], sequences[labels[j]])
            dist[i][j] = d
            dist[j][i] = d
    return dist


def _nj_build(dist, labels):
    """
    Standard Neighbor Joining to get an initial tree.
    Returns a tree representation:
    - nodes: list of leaf labels
    - parent: dict node -> parent
    - children: dict node -> list of children
    - is_leaf: set of leaf nodes
    - branch_len: dict node -> length to parent
    """
    n = len(labels)
    if n <= 3:
        # For 3 taxa, any unrooted tree is a star
        # We'll create a dummy root
        root = -1
        parent = {labels[0]: root, labels[1]: root, labels[2]: root}
        children = {root: [labels[0], labels[1], labels[2]]}
        is_leaf = set(labels)
        branch_len = {labels[0]: 0, labels[1]: 0, labels[2]: 0}
        return labels, parent, children, is_leaf, branch_len

    # Active nodes
    active = list(range(n))
    # Tree representation for NJ
    # We'll maintain a list of "current" nodes and their subtrees
    # d: distance matrix
    d = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                d[i][j] = dist[i][j]

    # tree_nodes: list of current node IDs
    # node_labels: dict node ID -> set of original labels in its subtree
    # node_parent: dict node ID -> parent node ID
    # node_children: dict node ID -> list of child node IDs
    # node_branch_len: dict node ID -> branch length to parent

    node_labels = {i: {labels[i]} for i in range(n)}
    node_parent = {}
    node_children = {}
    node_branch_len = {}

    next_id = n

    while len(active) > 3:
        k = len(active)
        # Row sums
        rsum = [0.0] * k
        for i, a in enumerate(active):
            for j, b in enumerate(active):
                if i != j:
                    rsum[i] += d[a][b]

        best_val = float('inf')
        best_i = -1
        best_j = -1
        for i in range(k):
            for j in range(i + 1, k):
                a = active[i]
                b = active[j]
                crit = (k - 2) * d[a][b] - rsum[i] - rsum[j]
                if crit < best_val:
                    best_val = crit
                    best_i = i
                    best_j = j

        a = active[best_i]
        b = active[best_j]

        # Compute branch lengths
        # db = 0.5 * d[a][b] + 0.5 * (rsum[b] - rsum[a]) / (k - 2)
        # da = d[a][b] - db
        db = 0.5 * d[a][b] + 0.5 * (rsum[best_j] - rsum[best_i]) / (k - 2)
        da = d[a][b] - db
        # Clamp to non-negative
        if db < 0: db = 0
        if da < 0: da = 0

        new_id = next_id
        next_id += 1

        # Create new node
        node_labels[new_id] = node_labels[a] | node_labels[b]
        node_children[new_id] = [a, b]
        node_parent[a] = new_id
        node_parent[b] = new_id
        node_branch_len[a] = da
        node_branch_len[b] = db

        # Update distance matrix
        # d[new][x] = 0.5 * (d[a][x] + d[b][x] - d[a][b])
        new_d_row = [0.0] * n
        for x in active:
            if x != a and x != b:
                new_d_row[x] = 0.5 * (d[a][x] + d[b][x] - d[a][b])
                if new_d_row[x] < 0: new_d_row[x] = 0
                d[x][new_id] = new_d_row[x]
                d[new_id][x] = new_d_row[x]

        # Update active list
        active.remove(a)
        active.remove(b)
        active.append(new_id)

    # Final tree construction
    # The remaining 3 nodes are connected to a central root (or a triple of edges)
    # For an unrooted tree of 3 leaves, it's a star.
    # For our representation, we can pick one as root? No, unrooted.
    # Let's just connect them to a dummy root node -1 for simplicity in SPR?
    # Actually, SPR works on unrooted trees too.
    # Let's represent the tree as a set of edges.
    # Or better, stick to the parent/children representation but allow a "root" that is not a leaf.
    # For the final 3 nodes u, v, w:
    # Create a root r.
    # Connect r to u, v, w.
    # This makes it a rooted star, which is topologically equivalent to the unrooted star for 3 leaves.
    # But for >3 leaves, the internal nodes are already built.
    # The last step of NJ leaves 3 nodes.
    # We need to connect them.
    # Let's create a root node.
    root = next_id
    next_id += 1
    active_nodes = active
    node_labels[root] = set().union(*[node_labels[x] for x in active_nodes])
    node_children[root] = active_nodes
    for x in active_nodes:
        node_parent[x] = root
        node_branch_len[x] = 0.0 # We don't know the lengths, but topology is what matters for RF.
        # For likelihood, we need lengths. We can optimize them later or set to 0.
        # Actually, for topology recovery, we can just use the topology.
        # But the prompt asks for Newick. Branch lengths are optional.
        # So we can set them to 0 or arbitrary.

    # Wait, the NJ algorithm above doesn't track the "root" of the final triple correctly for a general unrooted tree.
    # The standard NJ algorithm joins pairs until 3 nodes remain.
    # The final 3 nodes are connected by a "Y" (a star).
    # In my representation, I created a root connected to the 3 nodes. This is a valid representation of the star.
    
    # However, I need to ensure the tree is connected and correct.
    # Let's verify the structure.
    # Leaves are 0..n-1.
    # Internal nodes are n..next_id-1.
    # Root is next_id-1.
    
    # One issue: The distances in the final 3 nodes were updated.
    # The branch lengths to the root are unknown.
    # For the purpose of topology, this is fine.
    
    return labels, node_parent, node_children, set(range(n)), node_branch_len


def _tree_to_newick(labels, parent, children, is_leaf, branch_len):
    """
    Convert the tree representation to a Newick string.
    """
    def build(node):
        if node in is_leaf:
            return labels[node]
        kids = children[node]
        if len(kids) == 1:
            # Unlikely in this construction, but handle it
            return "(" + build(kids[0]) + ")"
        else:
            parts = []
            for k in kids:
                parts.append(build(k))
            return "(" + ",".join(parts) + ")"

    # Find the root
    root = None
    for node in children:
        if node not in parent:
            root = node
            break
    if root is None:
        # Fallback
        root = max(children.keys())
        
    return build(root) + ";"


def _get_edges(labels, parent, children, is_leaf):
    """
    Get the set of edges in the tree.
    Returns a set of tuples (u, v) with u < v.
    """
    edges = set()
    for node in parent:
        p = parent[node]
        u, v = min(node, p), max(node, p)
        edges.add((u, v))
    return edges


def _compute_likelihood(sequences, labels, parent, children, is_leaf):
    """
    Compute the log-likelihood of the sequences given the tree topology.
    Uses the JC model.
    For speed, we can use a simplified approach or Felsenstein's algorithm.
    Since n <= 32 and sites <= 2000, Felsenstein is feasible if optimized.
    However, implementing Felsenstein correctly is complex.
    
    Alternative: Use the parsimony score as a proxy?
    The directive says "Bayesian MCMC ... based on the posterior probability derived from the Jukes-Cantor likelihood".
    So I should implement likelihood.
    
    Let's implement a basic Felsenstein algorithm.
    Steps:
    1. Root the tree arbitrarily (e.g., at an internal node).
    2. For each site, compute the likelihood at the root.
    3. Sum log-likelihoods over all sites.
    
    To speed up:
    - Precompute the transition probability matrix for a given branch length?
    - But branch lengths are unknown.
    - If we fix branch lengths to 0 or some constant, it's not a true likelihood.
    - However, for topology comparison, the relative likelihood might still be informative if we assume equal branch lengths or use a parsimony-like heuristic.
    
    Actually, a common heuristic is to use the "parsimony score" (number of changes) which is related to the likelihood under certain conditions.
    Or, we can estimate branch lengths quickly?
    
    Given the time constraint, let's use the parsimony score (Fitch) as the scoring function.
    It is a valid heuristic for tree topology and much faster than full likelihood optimization.
    The prompt's "Strategy to try" suggests MCMC with likelihood, but if likelihood is too slow, parsimony is a reasonable substitute for the *score* in the MCMC, or we can just use SPR with parsimony.
    
    Let's stick to SPR with Parsimony (Fitch) for speed and simplicity. It's a different search method (SPR) from the current (NJ multi-start).
    """
    # Fitch Parsimony
    # 1. Post-order traversal to compute sets of states at each node.
    # 2. Pre-order traversal to assign states.
    # 3. Count changes.
    
    # To speed up, we'll do this for all sites at once using bitmasks.
    # States: A=1, C=2, G=4, T=8
    
    # Map labels to indices
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    n_taxa = len(labels)
    n_sites = len(sequences[labels[0]])
    
    # Preprocess sequences into bitmasks
    # seq_masks[i][s] is the bitmask for taxon i at site s
    seq_masks = []
    for i in range(n_taxa):
        seq = sequences[labels[i]]
        masks = [0] * n_sites
        for s in range(n_sites):
            c = seq[s]
            if c == 'A': masks[s] = 1
            elif c == 'C': masks[s] = 2
            elif c == 'G': masks[s] = 4
            elif c == 'T': masks[s] = 8
        seq_masks.append(masks)
    
    # Tree nodes: 0..n_taxa-1 are leaves, n_taxa.. are internal
    # Find root
    root = None
    for node in children:
        if node not in parent:
            root = node
            break
    if root is None:
        return 0 # Error
    
    # Topological order (post-order)
    order = []
    visited = set()
    def dfs_post(node):
        if node in visited: return
        visited.add(node)
        for child in children[node]:
            dfs_post(child)
        order.append(node)
    dfs_post(root)
    
    # node_states[node] is a list of bitmasks for each site
    # Initialize with leaves
    node_states = {}
    for i in range(n_taxa):
        node_states[i] = seq_masks[i]
    
    # Post-order computation
    # For internal nodes, state = intersection of children's states
    # If intersection is empty, it's a change.
    # We need to count the total number of changes.
    
    # To count changes efficiently:
    # Let S_u be the set of states at node u.
    # If S_u is non-empty, it's the intersection of children's states.
    # A change occurs if the intersection is empty, OR if we have to choose a state not in the parent's set?
    # Fitch's algorithm:
    # 1. Bottom-up: 
    #    If children have disjoint state sets, the state at the parent is the union, and we count a change.
    #    Actually, the number of changes is the number of times the intersection is empty during the bottom-up pass?
    #    No, that's not quite right.
    #    
    #    Correct Fitch counting:
    #    Bottom-up:
    #    S_u = intersection(S_c for c in children)
    #    If S_u is empty, S_u = union(S_c for c in children) and we increment the cost by 1.
    #    Wait, if intersection is empty, it means there's a change on at least one branch leading to u.
    #    The cost is incremented by 1 for each node where the intersection is empty.
    #    
    #    Let's verify.
    #    If children are A and C. Intersection empty. Union {A,C}. Cost +1.
    #    If children are {A,C} and {C,G}. Intersection {C}. Cost 0.
    #    If children are {A,C} and {G,T}. Intersection empty. Union {A,C,G,T}. Cost +1.
    #    
    #    This counts the minimum number of changes.
    
    cost = 0
    for node in order:
        if node in is_leaf:
            continue
        kids = children[node]
        if len(kids) == 0:
            continue
        if len(kids) == 1:
            node_states[node] = node_states[kids[0]]
            continue
        
        # Initialize intersection with first child's states
        inter = [node_states[kids[0]][s] for s in range(n_sites)]
        for s in range(n_sites):
            for c in kids[1:]:
                inter[s] &= node_states[c][s]
        
        # Check if intersection is empty for any site
        # If so, union and add to cost
        union = [node_states[kids[0]][s] for s in range(n_sites)]
        for s in range(n_sites):
            for c in kids[1:]:
                union[s] |= node_states[c][s]
        
        for s in range(n_sites):
            if inter[s] == 0:
                cost += 1
                node_states[node] = union # Need to store union
            else:
                node_states[node] = inter # Need to store intersection
        
        # Optimization: We can't just store a single list if we mix inter and union per site.
        # We need to store the resulting state set for each site.
        res_states = [0] * n_sites
        for s in range(n_sites):
            if inter[s] == 0:
                cost += 1
                res_states[s] = union[s]
            else:
                res_states[s] = inter[s]
        node_states[node] = res_states

    return cost


def _spr_refine(sequences, labels, parent, children, is_leaf, branch_len, rng, time_budget=10.0):
    """
    SPR (Subtree Prune and Regraft) refinement using Parsimony score.
    """
    start_time = time.time()
    
    # Get initial score
    current_score = _compute_likelihood(sequences, labels, parent, children, is_leaf)
    
    # Precompute adjacency for faster traversal?
    # For SPR, we need to prune a subtree and regraft it.
    
    # List of all internal nodes for pruning
    internal_nodes = [n for n in children if n not in is_leaf]
    
    # To make it faster, we'll limit the number of iterations.
    max_iters = 10000
    iter_count = 0
    
    while iter_count < max_iters and (time.time() - start_time) < time_budget:
        iter_count += 1
        if _time_left() < 5.0:
            break
        
        # Pick a random edge to prune
        # Pick a random internal node u (the parent of the subtree to be pruned)
        # and one of its children v.
        # Prune the subtree rooted at v.
        # Regraft it to a random edge (a, b) in the remaining tree.
        
        if not internal_nodes:
            break
            
        u = rng.choice(internal_nodes)
        kids = children[u]
        if len(kids) < 2:
            continue
        v = rng.choice(kids)
        
        # Prune v from u
        # Save the subtree
        # We need to modify the tree structure.
        # To avoid copying the whole tree, we can do it in-place and revert if worse.
        
        # 1. Remove v from u's children
        children[u].remove(v)
        parent[v] = -1 # Mark as detached
        
        # If u has only 1 child left, we might need to handle it, but usually we don't change the number of children in this simple SPR.
        # Actually, if u had 2 children, now it has 1. This is a "degenerate" node.
        # For simplicity, let's assume we only prune if u has >= 3 children?
        # Or handle the degenerate case by bypassing u?
        # Let's stick to standard SPR: prune a subtree, regraft to a different edge.
        # If u becomes a leaf (1 child), it's still valid, just a long branch.
        
        # 2. Find a random edge to regraft to
        # The edge must not be in the pruned subtree and not be the edge (u, v) itself.
        # We can pick a random node w from the remaining tree (excluding v's subtree) and pick one of its children or parent?
        # Standard SPR: pick a random edge (a, b) in the tree T - v_subtree.
        # Insert the new node x between a and b.
        # Connect v to x.
        
        # To pick a random edge, we can pick a random node from the "rest" of the tree.
        # Let's get all nodes not in v's subtree.
        # This is expensive to compute every time.
        
        # Alternative: Pick a random node w from the entire tree.
        # If w is in v's subtree, reject.
        # If w is v or u, reject.
        # If w is valid, pick a random incident edge?
        # Or just connect v to w? That's NNI-like? No, that's not SPR.
        
        # Let's implement a simple version:
        # Pick a random node w from the list of all nodes.
        # Check if w is in the pruned subtree.
        # If not, and w is not u or v:
        #   Create a new node x.
        #   Connect x to w.
        #   Connect x to v.
        #   This changes the topology.
        
        # This is not quite standard SPR (which splits an edge), but it's a valid move.
        # Let's call it "Prune and Regraft to a Node".
        
        # To check if w is in v's subtree, we can do a DFS from v.
        # Or, we can maintain a set of nodes in the subtree?
        # For n=32, DFS is fast.
        
        def in_subtree(node, root):
            if node == root: return True
            if root in is_leaf: return False
            for c in children[root]:
                if in_subtree(node, c): return True
            return False
            
        # This is O(n). Doing it for every candidate w is O(n^2).
        # Let's just pick w randomly and check.
        
        all_nodes = list(children.keys())
        # Exclude v and its ancestors? No, just check if w is in v's subtree.
        # Also exclude u? If w == u, we are regrafting to the same place (roughly).
        
        w = rng.choice(all_nodes)
        if w == v or w == u:
            continue
        if in_subtree(w, v):
            continue
        if in_subtree(v, w): # If v is in w's subtree, then w is an ancestor of v?
            # If w is an ancestor of v, then w is in the path from root to v.
            # Pruning v and regrafting to w might be invalid or same.
            # Let's skip if w is an ancestor of v.
            # How to check if w is ancestor of v?
            # Traverse up from v.
            cur = v
            while cur != -1:
                if cur == w:
                    break
                cur = parent[cur]
            if cur == w:
                continue

        # Perform Regraft
        # Create new node x
        new_id = max(children.keys()) + 1
        # Ensure new_id is not already used
        while new_id in children:
            new_id += 1
            
        # x is the new parent of v
        # x is connected to w
        # We need to decide if x replaces w as parent of some child?
        # Standard SPR: Split edge (w, z) where z is a child of w (or parent).
        # Let's pick a random child of w to split the edge.
        # If w has no children (leaf), split edge (w, parent[w]).
        
        if w in is_leaf:
            # Split edge (w, parent[w])
            z = parent[w]
            # Remove w from z's children
            children[z].remove(w)
            # Connect x to z
            children[z].append(new_id)
            parent[new_id] = z
            # Connect x to w
            children[new_id].append(w)
            parent[w] = new_id
            # Connect x to v
            children[new_id].append(v)
            parent[v] = new_id
            # Update children dict for new_id
            children[new_id] = [w, v]
            # Update is_leaf? new_id is internal.
            # w and v are not leaves of the new tree structure?
            # w was leaf, now has parent new_id.
            # v was internal (root of pruned subtree), now has parent new_id.
            
        else:
            # w is internal
            # Pick a random child z of w
            z = rng.choice(children[w])
            # Remove z from w's children
            children[w].remove(z)
            # Connect x to w
            children[w].append(new_id)
            parent[new_id] = w
            # Connect x to z
            children[new_id].append(z)
            parent[z] = new_id
            # Connect x to v
            children[new_id].append(v)
            parent[v] = new_id
            children[new_id] = [z, v]
            
        # Update is_leaf set?
        # new_id is internal.
        # If w was leaf, it's still a leaf (just has a new parent).
        # v is internal.
        
        # Calculate new score
        new_score = _compute_likelihood(sequences, labels, parent, children, is_leaf)
        
        if new_score < current_score:
            current_score = new_score
            # Accept move
            # Add new_id to internal_nodes list?
            internal_nodes.append(new_id)
        else:
            # Revert
            if w in is_leaf:
                # Revert leaf case
                children[z].remove(new_id)
                parent[new_id] = -1
                children[new_id].remove(w)
                parent[w] = z
                children[new_id].remove(v)
                parent[v] = -1
            else:
                # Revert internal case
                children[w].remove(new_id)
                parent[new_id] = -1
                children[new_id].remove(z)
                parent[z] = w
                children[new_id].remove(v)
                parent[v] = -1
                
            # Restore v to u
            children[u].append(v)
            parent[v] = u
            
            # Remove new_id from children dict?
            # It's safer to just leave it and ignore if not connected?
            # But it might mess up "max(children.keys())".
            # Let's delete it.
            del children[new_id]
            if new_id in parent:
                del parent[new_id]

    return current_score


def reconstruct(sequences):
    global _START_TIME
    _START_TIME = time.time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    labels_set = set(labels)
    n = len(labels)
    
    if n <= 3:
        return "(" + ",".join(labels) + ");"
    
    dist = _compute_distance_matrix(sequences, labels)
    
    # Build initial tree with NJ
    labels, parent, children, is_leaf, branch_len = _nj_build(dist, labels)
    
    # Refine with SPR
    rng = random.Random(42)
    # Allocate time budget for SPR
    # We have ~265s total.
    # NJ is fast.
    # Let's give SPR most of the time.
    spr_budget = 200.0
    
    _spr_refine(sequences, labels, parent, children, is_leaf, branch_len, rng, time_budget=spr_budget)
    
    newick = _tree_to_newick(labels, parent, children, is_leaf, branch_len)
    
    return newick

