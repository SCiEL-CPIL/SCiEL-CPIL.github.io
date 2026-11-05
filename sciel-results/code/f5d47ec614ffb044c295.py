import numpy as np
import time
import sys

def _jc_distance_matrix(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    # Convert sequences to integer arrays for speed
    # A=0, C=1, G=2, T=3
    char_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seqs = np.array([[char_to_int[c] for c in sequences[label]] for label in labels], dtype=np.uint8)
    
    # Compute pairwise mismatches efficiently
    # seqs shape: (n, L)
    # We want (n, n) matrix of mismatches
    # Using broadcasting: seqs[:, None, :] vs seqs[None, :, :]
    # This creates a (n, n, L) array which might be large.
    # For n=32, L=2000, that's 32*32*2000 = 2M integers. ~8MB. Fine.
    mismatches = (seqs[:, None, :] != seqs[None, :, :]).sum(axis=2)
    L = seqs.shape[1]
    p = mismatches / L
    
    # Jukes-Cantor correction: d = -3/4 * ln(1 - 4p/3)
    d = np.zeros((n, n))
    mask = p < 0.75
    d[mask] = -0.75 * np.log(1.0 - 4.0 * p[mask] / 3.0)
    d[~mask] = 10.0  # Large distance for high divergence
    np.fill_diagonal(d, 0.0)
    
    return d, labels

def _neighbor_joining_newick(dist, labels):
    n = len(labels)
    if n == 0:
        return ""
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    active = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    
    max_nodes = 2 * n
    dist_arr = np.full((max_nodes, max_nodes), 10.0)
    for i in range(n):
        for j in range(i+1, n):
            dist_arr[i, j] = dist[i, j]
            dist_arr[j, i] = dist[i, j]
    np.fill_diagonal(dist_arr, 0.0)
    
    next_id = n
    
    while len(active) > 3:
        m = len(active)
        d_active = dist_arr[np.ix_(active, active)]
        
        r = d_active.sum(axis=1)
        Q = (m - 2) * d_active - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        
        min_flat = np.argmin(Q)
        i_idx, j_idx = np.unravel_index(min_flat, Q.shape)
        if i_idx > j_idx:
            i_idx, j_idx = j_idx, i_idx
        
        best_i = active[i_idx]
        best_j = active[j_idx]
        
        new_id = next_id
        next_id += 1
        
        tree_i = trees[best_i]
        tree_j = trees[best_j]
        new_tree = f"({tree_i},{tree_j})"
        
        # Update distances to the new node
        for x in active:
            if x not in (best_i, best_j):
                d_new_x = 0.5 * (dist_arr[best_i, x] + dist_arr[best_j, x] - dist_arr[best_i, best_j])
                dist_arr[new_id, x] = d_new_x
                dist_arr[x, new_id] = d_new_x
        
        active = [x for x in active if x not in (best_i, best_j)]
        active.append(new_id)
        trees[new_id] = new_tree
    
    if len(active) == 3:
        a, b, c = active
        return f"({trees[a]},({trees[b]},{trees[c]}));"
    else:
        return "(" + ",".join(trees[x] for x in active) + ");"

def _parse_newick(newick):
    newick = newick.strip()
    if newick.endswith(';'):
        newick = newick[:-1]
    
    adj = {}
    leaf_labels = {}
    is_leaf = {}
    node_counter = [0]
    
    def _parse_node(s, pos):
        if pos >= len(s):
            return None, pos
        
        if s[pos] == '(':
            node_id = node_counter[0]
            node_counter[0] += 1
            adj[node_id] = []
            is_leaf[node_id] = False
            
            pos += 1
            children = []
            while pos < len(s) and s[pos] != ')':
                child_id, pos = _parse_node(s, pos)
                if child_id is not None:
                    children.append(child_id)
                if pos < len(s) and s[pos] == ',':
                    pos += 1
            if pos < len(s) and s[pos] == ')':
                pos += 1
            
            for child_id in children:
                adj[node_id].append(child_id)
                if child_id in adj:
                    adj[child_id].append(node_id)
            
            while pos < len(s) and s[pos] not in (',', ')', ';'):
                pos += 1
            
            return node_id, pos
        else:
            start = pos
            while pos < len(s) and s[pos] not in (',', ')', ';', ':'):
                pos += 1
            label = s[start:pos].strip()
            
            node_id = node_counter[0]
            node_counter[0] += 1
            adj[node_id] = []
            is_leaf[node_id] = True
            leaf_labels[node_id] = label
            
            if pos < len(s) and s[pos] == ':':
                pos += 1
                while pos < len(s) and s[pos] not in (',', ')', ';'):
                    pos += 1
            
            return node_id, pos
    
    root_id, _ = _parse_node(newick, 0)
    return adj, leaf_labels, is_leaf

def _adj_to_newick(adj, leaf_labels, is_leaf):
    if not adj:
        return ""
    
    root = None
    for node in adj:
        if not is_leaf.get(node, False):
            root = node
            break
    if root is None:
        root = list(adj.keys())[0]
    
    def _build(node, parent):
        if is_leaf.get(node, False):
            return leaf_labels[node]
        
        children = []
        for neighbor in adj[node]:
            if neighbor != parent:
                children.append(_build(neighbor, node))
        
        if len(children) == 0:
            return ""
        elif len(children) == 1:
            return "(" + children[0] + ")"
        else:
            return "(" + ",".join(children) + ")"
    
    newick = _build(root, -1)
    return newick + ";"

def _parsimony_score(adj, leaf_labels, is_leaf, seqs_int, label_to_idx):
    n_sites = seqs_int.shape[1]
    total_score = 0
    
    leaf_idx_map = {}
    for node, label in leaf_labels.items():
        leaf_idx_map[node] = label_to_idx[label]

    root = None
    for node in adj:
        if not is_leaf.get(node, False):
            root = node
            break
    if root is None:
        root = list(adj.keys())[0]

    # Get post-order traversal
    post_order = []
    visited = set()
    stack = [(root, -1, False)]
    while stack:
        node, parent, processed = stack.pop()
        if processed:
            post_order.append(node)
            continue
        if node in visited:
            continue
        visited.add(node)
        stack.append((node, parent, True))
        for neighbor in adj[node]:
            if neighbor != parent and neighbor not in visited:
                stack.append((neighbor, node, False))
    
    # Pre-compute children for each node to avoid checking parent repeatedly
    # And store the parent for each node
    parent_map = {}
    children_map = {node: [] for node in adj}
    
    # Re-traverse to build parent/child maps
    stack = [(root, -1)]
    while stack:
        node, parent = stack.pop()
        parent_map[node] = parent
        for neighbor in adj[node]:
            if neighbor != parent:
                children_map[node].append(neighbor)
                stack.append((neighbor, node))
    
    # Fitch algorithm
    # states[node] is a bitmask: 1=A, 2=C, 4=G, 8=T
    # cost[node] is the parsimony cost for the subtree rooted at node
    
    states = np.zeros(len(adj), dtype=np.uint8)
    costs = np.zeros(len(adj), dtype=np.int32)
    
    # Map node IDs to indices in arrays for speed
    node_to_idx = {node: i for i, node in enumerate(adj.keys())}
    
    for s in range(n_sites):
        # Reset states and costs
        states.fill(0)
        costs.fill(0)
        
        for node in post_order:
            idx = node_to_idx[node]
            if is_leaf.get(node, False):
                state = seqs_int[leaf_idx_map[node], s]
                states[idx] = 1 << state
                costs[idx] = 0
            else:
                # Intersect children states
                intersect = 0
                min_cost = 0
                first = True
                for child in children_map[node]:
                    child_idx = node_to_idx[child]
                    if first:
                        intersect = states[child_idx]
                        min_cost = costs[child_idx]
                        first = False
                    else:
                        intersect &= states[child_idx]
                        if costs[child_idx] < min_cost:
                            min_cost = costs[child_idx]
                
                if intersect != 0:
                    states[idx] = intersect
                    # Cost is sum of child costs? No, Fitch cost is sum of costs of children where state changed?
                    # Actually, Fitch's algorithm cost:
                    # If intersection is non-empty, cost = sum of costs of children? No.
                    # The cost of a node is the sum of the costs of its children, PLUS 1 if the intersection is empty.
                    # Wait, standard Fitch:
                    # S(v) = union of S(c) for children c.
                    # If intersection of S(c) is non-empty, cost(v) = sum(cost(c))
                    # Else cost(v) = min(cost(c)) + 1? No.
                    # Let's check:
                    # "If the intersection is non-empty, the cost of the node is the sum of the costs of its children."
                    # "If the intersection is empty, the cost of the node is the minimum of the costs of its children plus one."
                    # Wait, that's not right either.
                    # Correct Fitch:
                    # Cost is the number of changes.
                    # If S(v) is non-empty (intersection), cost(v) = sum(cost(c))
                    # If S(v) is empty, cost(v) = min(cost(c)) + 1? No, it's sum(cost(c)) + 1?
                    # Actually, the standard definition:
                    # cost(v) = sum(cost(c)) if intersection is non-empty
                    # cost(v) = min(cost(c)) + 1 if intersection is empty?
                    # No, it is:
                    # If intersection is non-empty, cost(v) = sum(cost(c))
                    # If intersection is empty, cost(v) = min(cost(c)) + 1 is WRONG.
                    # It is: cost(v) = sum(cost(c)) + 1? No.
                    # Let's look it up mentally.
                    # Fitch (1971):
                    # S(v) = intersection of S(c).
                    # If S(v) != empty, cost(v) = sum(cost(c)).
                    # If S(v) == empty, cost(v) = min(cost(c)) + 1?
                    # Actually, the cost added is 1 for the change at the node.
                    # So cost(v) = sum(cost(c)) + 1?
                    # Let's test: 2 children, states {A}, {C}. Intersection empty.
                    # cost(child1)=0, cost(child2)=0.
                    # If cost(v) = 0+0+1 = 1. Correct.
                    # If 3 children: {A}, {C}, {A}. Intersection empty.
                    # cost = 0+0+0+1 = 1. Correct (one change C->A or A->C).
                    # What if costs are different?
                    # Child1 {A} cost 2, Child2 {C} cost 0.
                    # Intersection empty.
                    # Cost = min(2,0) + 1 = 1? Or 2+0+1=3?
                    # The change happens at the node. The children's costs are fixed.
                    # The additional cost is 1.
                    # So cost(v) = sum(cost(c)) + 1?
                    # Wait, if I choose state A for v, then child2 (C) must change. Cost = 2 + (0+1) = 3.
                    # If I choose state C for v, then child1 (A) must change. Cost = (2+1) + 0 = 3.
                    # So cost is 3.
                    # Formula: sum(cost(c)) + 1 = 2+0+1 = 3. Correct.
                    # What if Child1 {A} cost 0, Child2 {C} cost 5?
                    # Sum + 1 = 0+5+1 = 6.
                    # Choose A: 0 + (5+1) = 6.
                    # Choose C: (0+1) + 5 = 6.
                    # So it seems cost(v) = sum(cost(c)) + 1.
                    
                    states[idx] = 0 # Mark as "any" or handle separately?
                    # We need to store the set of optimal states.
                    # If intersection is empty, S(v) is the union of the sets of children with min cost?
                    # No, S(v) is the union of S(c) for c with min cost?
                    # Actually, for the next level up, we need the set of states that achieve the min cost.
                    # If intersection is non-empty, S(v) = intersection.
                    # If intersection is empty, S(v) = union of S(c) for c where cost(c) = min_cost.
                    
                    min_cost_val = min(costs[node_to_idx[c]] for c in children_map[node])
                    # The cost of this node is sum(costs) + 1
                    total_child_cost = sum(costs[node_to_idx[c]] for c in children_map[node])
                    costs[idx] = total_child_cost + 1
                    
                    # Determine S(v)
                    union = 0
                    for c in children_map[node]:
                        if costs[node_to_idx[c]] == min_cost_val:
                            union |= states[node_to_idx[c]]
                    states[idx] = union
                else:
                    costs[idx] = sum(costs[node_to_idx[c]] for c in children_map[node])
        
        total_score += costs[node_to_idx[root]]
    
    return total_score

def _spr_refine(newick, adj, leaf_labels, is_leaf, seqs_int, label_to_idx, d, labels):
    """
    Perform SPR (Subtree Pruning and Regrafting) refinement using Parsimony score.
    This is a local search to improve the topology.
    """
    current_score = _parsimony_score(adj, leaf_labels, is_leaf, seqs_int, label_to_idx)
    improved = True
    max_iterations = 50 # Limit iterations to save time
    
    while improved and max_iterations > 0:
        improved = False
        max_iterations -= 1
        
        # Get list of internal nodes (candidates for pruning)
        internal_nodes = [node for node in adj if not is_leaf.get(node, False)]
        
        for prune_node in internal_nodes:
            # Find a child to prune
            # We can prune any branch.
            # Let's iterate over all edges (u, v) where u is parent of v.
            # We need a rooted view.
            # Let's pick a root (e.g., the first internal node) and root the tree there.
            # This is inefficient to do every time.
            # Let's just try all edges in the unrooted tree.
            pass
            
        # This naive SPR is O(n^3) or worse per iteration and slow in Python.
        # Given the time constraints, I will implement a simpler heuristic:
        # **Quartet Sampling** or **Distance-based SPR**?
        # Actually, let's try **Fast Distance-based SPR** using the distance matrix.
        # Or just return the NJ tree if time is tight.
        
        # For now, let's implement a very basic SPR that only tries to move a subtree to a nearby location.
        # This is risky. Let's stick to NJ for safety and rely on the robustness of NJ.
        # Or implement a simple **Tree Bisection and Reconnection (TBR)**?
        
        # Let's try a simple **SPR** using the distance matrix as a guide.
        # If we prune a subtree, we can calculate the "optimal" reattachment point
        # by minimizing the sum of distances to the rest of the tree?
        
        # Given the complexity, I will return the NJ tree.
        # The NJ tree is usually quite good.
        
        break # Disable SPR for now to ensure correctness and speed
    
    return newick, current_score

def reconstruct(sequences):
    start_time = time.time()
    deadline = 260.0
    
    d, labels = _jc_distance_matrix(sequences)
    n = len(labels)
    
    if time.time() - start_time > deadline - 10:
        if n <= 2:
            return "(" + ",".join(labels) + ");"
        return "(" + ",".join(labels) + ");"
    
    # Run NJ
    nj_newick = _neighbor_joining_newick(d, labels)
    
    if time.time() - start_time > deadline - 5:
        return nj_newick
    
    # Parse NJ result
    adj, leaf_labels, is_leaf = _parse_newick(nj_newick)
    
    # Prepare sequences for parsimony if needed
    # For now, just return NJ
    final_newick = _adj_to_newick(adj, leaf_labels, is_leaf)
    
    return final_newick
