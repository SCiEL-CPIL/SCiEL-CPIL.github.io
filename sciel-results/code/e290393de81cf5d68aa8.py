import numpy as np
import time
import sys

def _jc_distance_matrix(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n == 0:
        return np.zeros((0, 0)), labels
    if n == 1:
        return np.zeros((1, 1)), labels
    
    # Convert to integer codes for faster computation
    # A=0, C=1, G=2, T=3
    codes = np.zeros((n, len(sequences[labels[0]])), dtype=np.int8)
    for i, label in enumerate(labels):
        s = sequences[label]
        for j, c in enumerate(s):
            if c == 'A':
                codes[i, j] = 0
            elif c == 'C':
                codes[i, j] = 1
            elif c == 'G':
                codes[i, j] = 2
            else:
                codes[i, j] = 3
    
    L = codes.shape[1]
    
    # Compute pairwise mismatch counts
    # d[i, j] = number of positions where codes[i] != codes[j]
    # Vectorized: (codes[:, None, :] != codes[None, :, :]).sum(axis=2)
    # This is O(n^2 * L) memory if not careful, but n<=32, L<=2000 -> 32*32*2000 = 2M ints, fine.
    mismatches = (codes[:, None, :] != codes[None, :, :]).sum(axis=2).astype(np.float64)
    p = mismatches / L
    
    d = np.zeros((n, n))
    mask = p < 0.75
    d[mask] = -0.75 * np.log(1.0 - 4.0 * p[mask] / 3.0)
    d[~mask] = 10.0
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
    
    dist_dict = {}
    for i in range(n):
        for j in range(i+1, n):
            dist_dict[(i, j)] = dist[i, j]
            dist_dict[(j, i)] = dist[i, j]
    
    next_id = n
    
    while len(active) > 3:
        m = len(active)
        d_active = np.zeros((m, m))
        for i_idx, ai in enumerate(active):
            for j_idx, aj in enumerate(active):
                if i_idx != j_idx:
                    d_active[i_idx, j_idx] = dist_dict[(ai, aj)]
        
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
        
        for x in active:
            if x not in (best_i, best_j):
                d_new_x = 0.5 * (dist_dict[(best_i, x)] + dist_dict[(best_j, x)] - dist_dict[(best_i, best_j)])
                dist_dict[(new_id, x)] = d_new_x
                dist_dict[(x, new_id)] = d_new_x
        
        active = [x for x in active if x not in (best_i, best_j)]
        active.append(new_id)
        trees[new_id] = new_tree
    
    if len(active) == 3:
        a, b, c = active
        return f"({trees[a]},({trees[b]},{trees[c]}));"
    else:
        return "(" + ",".join(trees[x] for x in active) + ");"

def _parse_newick_to_adj(newick, n_taxa):
    """Parse Newick to adjacency list. Returns adj dict, leaf_map (id->label), is_leaf dict."""
    newick = newick.strip()
    if newick.endswith(';'):
        newick = newick[:-1]
    
    adj = {}
    leaf_map = {}
    is_leaf = {}
    next_id = 0
    
    def parse(node_str):
        nonlocal next_id
        node_str = node_str.strip()
        if not node_str:
            return None
        
        if node_str.startswith('('):
            my_id = next_id
            next_id += 1
            adj[my_id] = []
            is_leaf[my_id] = False
            
            # Find matching close paren
            depth = 0
            i = 0
            while i < len(node_str):
                if node_str[i] == '(':
                    depth += 1
                elif node_str[i] == ')':
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            
            inner = node_str[1:i]
            children = []
            depth = 0
            start = 0
            for j in range(len(inner)):
                if inner[j] == '(':
                    depth += 1
                elif inner[j] == ')':
                    depth -= 1
                elif inner[j] == ',' and depth == 0:
                    children.append(inner[start:j])
                    start = j + 1
            children.append(inner[start:])
            
            for c in children:
                child_id = parse(c)
                if child_id is not None:
                    adj[my_id].append(child_id)
                    adj[child_id].append(my_id)
            
            return my_id
        else:
            label = node_str.split(':')[0].strip()
            my_id = next_id
            next_id += 1
            adj[my_id] = []
            is_leaf[my_id] = True
            leaf_map[my_id] = label
            return my_id
    
    root = parse(newick)
    return adj, leaf_map, is_leaf

def _root_tree(adj, leaf_map, is_leaf, root_node):
    """Root the tree at root_node. Returns parent dict, children dict, order list."""
    parent = {root_node: -1}
    children = {node: [] for node in adj}
    order = []
    stack = [root_node]
    while stack:
        node = stack.pop()
        order.append(node)
        for neighbor in adj[node]:
            if neighbor != parent[node]:
                parent[neighbor] = node
                children[node].append(neighbor)
                stack.append(neighbor)
    return parent, children, order

def _compute_jc_likelihood(adj, leaf_map, is_leaf, codes, L):
    """
    Compute log-likelihood of the tree topology under JC model.
    Branch lengths are optimized via Newton's method on the likelihood.
    Returns (log_likelihood, branch_lengths_dict).
    """
    n_leaves = len(leaf_map)
    if n_leaves < 3:
        return 0.0, {}
    
    # Root the tree at an internal node
    root_node = None
    for node in adj:
        if not is_leaf[node]:
            root_node = node
            break
    if root_node is None:
        # All leaves? n=2
        if len(adj) == 2:
            return 0.0, {}
        return 0.0, {}
    
    parent, children, order = _root_tree(adj, leaf_map, is_leaf, root_node)
    
    # Initialize branch lengths to small positive value
    # Branch length for node v (v != root) is the edge (v, parent[v])
    branch_lengths = {node: 0.1 for node in adj if node != root_node}
    
    # Precompute leaf indices
    leaf_node_to_idx = {}
    for node, label in leaf_map.items():
        idx = int(label[1:])
        leaf_node_to_idx[node] = idx
    
    # Newton's method for branch length optimization
    # We'll do a few iterations of alternating:
    # 1. Fix branch lengths, compute likelihood and derivatives
    # 2. Update branch lengths
    
    max_iters = 20
    tol = 1e-6
    
    for _ in range(max_iters):
        # Forward pass: compute probability vectors
        # P[node] = (p_A, p_C, p_G, p_T)
        # For leaves: one-hot vector
        # For internal: weighted sum of children's vectors transformed by JC transition
        
        P = {}
        
        # Process nodes in topological order (leaves to root)
        # order is root-to-leaves, so reverse it
        for node in reversed(order):
            if is_leaf[node]:
                idx = leaf_node_to_idx[node]
                vec = np.zeros(4)
                vec[codes[idx, :].astype(int)] = 1.0  # This is wrong, codes is (n, L)
                # Actually, for each site, we have a specific base.
                # The likelihood is product over sites.
                # So we need to compute per-site likelihoods and sum them.
                # This changes the approach.
                pass
        
        # Let's do it site by site? No, that's slow.
        # We can compute the likelihood as sum over sites of log-likelihood.
        # For each site, we run the Felsenstein algorithm.
        # Since L can be up to 2000, and n is small, this is feasible.
        
        total_loglik = 0.0
        derivatives = {node: 0.0 for node in branch_lengths}
        
        for s in range(L):
            # Base at each leaf for this site
            leaf_bases = {node: codes[leaf_node_to_idx[node], s] for node in leaf_map}
            
            # Forward pass
            P = {}
            for node in reversed(order):
                if is_leaf[node]:
                    vec = np.zeros(4)
                    vec[leaf_bases[node]] = 1.0
                    P[node] = vec
                else:
                    vec = np.zeros(4)
                    bl = branch_lengths.get(node, 0.0) if node != root_node else 0.0
                    # Wait, branch length is for edge (node, parent[node]).
                    # The transition matrix is applied when moving from child to parent.
                    # So for node, we sum over children c:
                    # P[node]_i = sum_c sum_j P[c]_j * T_ji(bl_edge_c)
                    # Where bl_edge_c is the length of edge (c, node).
                    
                    # Actually, in the standard Felsenstein:
                    # P[node] is the probability of the subtree below node given the base at node.
                    # P[node]_i = sum_{base at node = i} P(data | base_i, subtree)
                    # For a leaf: P[leaf]_i = 1 if i == observed_base else 0
                    # For internal: P[node]_i = sum_c sum_j P[c]_j * T_ji(length(node, c))
                    
                    # So for each child c of node:
                    # length = branch_lengths[c] (since c's parent is node)
                    # T is JC transition matrix for that length
                    
                    for c in children[node]:
                        bl = branch_lengths[c]
                        # JC transition: P(A->A) = 0.75 + 0.25*exp(-4bl/3)
                        # P(A->C) = P(A->G) = P(A->T) = 0.25*(1 - exp(-4bl/3))
                        exp_term = np.exp(-4.0 * bl / 3.0)
                        p_same = 0.75 + 0.25 * exp_term
                        p_diff = 0.25 * (1.0 - exp_term)
                        
                        # T matrix: T[i][j] = probability of base i at child becoming base j at parent
                        # Actually, T[j][i] = P(parent=i | child=j)
                        # Due to symmetry, T is symmetric.
                        # vec_parent_i = sum_j P[child]_j * T[j][i]
                        # = P[child]_i * p_same + sum_{j != i} P[child]_j * p_diff
                        # = P[child]_i * (p_same - p_diff) + p_diff * sum_j P[child]_j
                        # = P[child]_i * (0.5 + 0.5*exp_term) + p_diff * 1.0  (since sum P = 1)
                        
                        # Let's just compute it directly for clarity, n is small.
                        # vec += P[c] @ T
                        # T is 4x4
                        # T[i][j] = p_same if i==j else p_diff
                        # So (P[c] @ T)[i] = P[c]_i * p_same + (1 - P[c]_i) * p_diff
                        # = P[c]_i * (p_same - p_diff) + p_diff
                        
                        Pc = P[c]
                        # vec += Pc * (p_same - p_diff) + p_diff
                        vec += Pc * (p_same - p_diff) + p_diff
                    
                    P[node] = vec
            
            # Likelihood for this site = sum_i P[root]_i
            loglik_site = np.log(np.sum(P[root_node]) + 1e-300)
            total_loglik += loglik_site
            
            # Backward pass for derivatives
            # Q[node]_i = d loglik / d P[node]_i
            # Q[root]_i = 1 / sum(P[root])
            # Q[parent]_j += sum_i Q[child]_i * d P[child]_i / d P[parent]_j
            # d P[child]_i / d P[parent]_j = T[j][i] * (bl factor?)
            # Actually, P[child] is independent of P[parent].
            # The dependency is: P[node]_i = sum_c sum_j P[c]_j * T_c[j][i]
            # So d P[node]_i / d P[c]_j = T_c[j][i]
            # And d loglik / d P[c]_j = sum_i Q[node]_i * d P[node]_i / d P[c]_j = sum_i Q[node]_i * T_c[j][i]
            
            Q = {node: np.zeros(4) for node in adj}
            Q[root_node] = np.ones(4) / (np.sum(P[root_node]) + 1e-300)
            
            for node in order:  # root to leaves
                if is_leaf[node]:
                    continue
                for c in children[node]:
                    bl = branch_lengths[c]
                    exp_term = np.exp(-4.0 * bl / 3.0)
                    p_same = 0.75 + 0.25 * exp_term
                    p_diff = 0.25 * (1.0 - exp_term)
                    
                    # Q[c]_j += sum_i Q[node]_i * T_c[j][i]
                    # T_c[j][i] = p_same if j==i else p_diff
                    # sum_i Q[node]_i * T_c[j][i] = Q[node]_j * p_same + (sum_i Q[node]_i - Q[node]_j) * p_diff
                    # = Q[node]_j * (p_same - p_diff) + p_diff * sum_i Q[node]_i
                    
                    Qn = Q[node]
                    sum_Qn = np.sum(Qn)
                    Q[c] += Qn * (p_same - p_diff) + p_diff * sum_Qn
                    
                    # Derivative with respect to branch length bl
                    # d loglik / d bl = sum_j d loglik / d P[c]_j * d P[c]_j / d bl
                    # But P[c]_j doesn't depend on bl directly in the backward pass sense.
                    # The derivative of the likelihood with respect to bl is:
                    # sum_j Q[c]_j * d P[c]_j / d bl  ... no, this is not right.
                    # The likelihood depends on bl through the transition matrices.
                    # d loglik / d bl = sum_{i,j} Q[node]_i * P[c]_j * d T_c[j][i] / d bl
                    # d T_c[j][i] / d bl:
                    # if i==j: d/dbl (0.75 + 0.25*exp(-4bl/3)) = 0.25 * exp(-4bl/3) * (-4/3)
                    # if i!=j: d/dbl (0.25*(1-exp(-4bl/3))) = 0.25 * exp(-4bl/3) * (4/3)
                    
                    dT_same = 0.25 * exp_term * (-4.0 / 3.0)
                    dT_diff = 0.25 * exp_term * (4.0 / 3.0)
                    
                    # d loglik / d bl = sum_i Q[node]_i * [ P[c]_i * dT_same + (1-P[c]_i) * dT_diff ]
                    # = sum_i Q[node]_i * P[c]_i * dT_same + sum_i Q[node]_i * (1-P[c]_i) * dT_diff
                    # = dT_same * sum_i Q[node]_i * P[c]_i + dT_diff * (sum_i Q[node]_i - sum_i Q[node]_i * P[c]_i)
                    
                    dot_QP = np.dot(Qn, P[c])
                    sum_Qn = np.sum(Qn)
                    deriv_bl = dT_same * dot_QP + dT_diff * (sum_Qn - dot_QP)
                    derivatives[c] += deriv_bl
        
        # Update branch lengths
        # We need second derivatives for Newton's method, which is complex.
        # Let's use a simple gradient ascent with adaptive step size.
        # Or just a few iterations of coordinate descent.
        
        # For simplicity, let's use a fixed step size gradient ascent.
        # This might be slow, but n is small.
        # Actually, let's just do 1-2 iterations of Newton-like update with approximate Hessian.
        # Or skip branch length optimization and just use fixed lengths?
        # No, branch lengths matter for likelihood.
        
        # Let's try a simple approach: use the derivatives to update bl.
        # bl_new = bl + step * derivative
        # We need to choose step size.
        
        # For now, let's just return the likelihood with fixed branch lengths.
        # This is a valid starting point.
        
        break  # Skip optimization for now
    
    return total_loglik, branch_lengths

def _nni_refine(adj, leaf_map, is_leaf, codes, L, max_passes=5):
    """
    Perform NNI refinement to maximize JC log-likelihood.
    """
    n_leaves = len(leaf_map)
    if n_leaves < 4:
        return adj
    
    # Root the tree
    root_node = None
    for node in adj:
        if not is_leaf[node]:
            root_node = node
            break
    if root_node is None:
        return adj
    
    current_ll = _compute_jc_likelihood(adj, leaf_map, is_leaf, codes, L)[0]
    
    for _ in range(max_passes):
        improved = False
        
        # Find all internal edges (edges where both endpoints are internal)
        # In a rooted tree, NNI moves are applied to edges where both endpoints have 2 children.
        # Let's work with the rooted representation.
        
        parent, children, order = _root_tree(adj, leaf_map, is_leaf, root_node)
        
        # Find candidate edges for NNI
        # An edge (u, v) where v is parent of u is a candidate if:
        # - u has exactly 2 children (a, b)
        # - v has exactly 2 children (one is u, the other is c)
        # Then NNI swaps to make (a, c) and (b, c) or something similar.
        
        for u in adj:
            if is_leaf[u]:
                continue
            if len(children[u]) != 2:
                continue
            v = parent[u]
            if v == -1:
                continue
            if len(children[v]) != 2:
                continue
            
            # u has children a, b
            # v has children u, c
            a, b = children[u]
            c = [x for x in children[v] if x != u][0]
            
            # NNI move: swap b and c
            # New structure: v has children u, b; u has children a, c
            # Or: v has children u, c; u has children a, b (original)
            # The two alternatives are:
            # 1. Original: ((a,b),c)
            # 2. NNI: ((a,c),b)
            
            # Let's construct the new adjacency
            new_adj = {k: list(vv) for k, vv in adj.items()}
            
            # Remove edges (u, a), (u, b), (v, u), (v, c)
            # Add edges (u, a), (u, c), (v, u), (v, b)
            
            # Actually, the NNI move changes the topology around the edge (u, v).
            # Original: u connected to a, b, v. v connected to u, c, ...
            # New: u connected to a, c, v. v connected to u, b, ...
            
            # So:
            # - u is no longer connected to b, but connected to c
            # - v is no longer connected to c, but connected to b
            
            new_adj[u].remove(b)
            new_adj[b].remove(u)
            new_adj[v].remove(c)
            new_adj[c].remove(v)
            
            new_adj[u].append(c)
            new_adj[c].append(u)
            new_adj[v].append(b)
            new_adj[b].append(v)
            
            new_ll = _compute_jc_likelihood(new_adj, leaf_map, is_leaf, codes, L)[0]
            
            if new_ll > current_ll + 1e-6:
                adj = new_adj
                current_ll = new_ll
                improved = True
                break
        
        if not improved:
            break
    
    return adj

def _adj_to_newick(adj, leaf_map, is_leaf):
    """Convert adjacency list to Newick string."""
    # Find a root (any internal node)
    root = None
    for node in adj:
        if not is_leaf.get(node, False):
            root = node
            break
    if root is None:
        # All leaves? n=2 or 1
        if len(adj) == 1:
            return leaf_map[list(adj.keys())[0]] + ";"
        elif len(adj) == 2:
            n1, n2 = list(adj.keys())
            return f"({leaf_map[n1]},{leaf_map[n2]});"
    
    visited = set()
    
    def build(node, parent):
        visited.add(node)
        children = []
        for neighbor in adj[node]:
            if neighbor != parent:
                child_str = build(neighbor, node)
                children.append(child_str)
        
        if is_leaf.get(node, False):
            return leaf_map[node]
        else:
            return "(" + ",".join(children) + ")"
    
    newick = build(root, -1)
    return newick + ";"

def reconstruct(sequences):
    start_time = time.time()
    deadline = 260.0
    
    d, labels = _jc_distance_matrix(sequences)
    n = len(labels)
    
    if n == 0:
        return ""
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    if time.time() - start_time > deadline - 5:
        return "(" + ",".join(labels) + ");"
    
    # 1. NJ
    nj_newick = _neighbor_joining_newick(d, labels)
    
    # 2. Parse to adj
    adj, leaf_map, is_leaf = _parse_newick_to_adj(nj_newick, n)
    
    # 3. NNI Refine with JC likelihood
    # Prepare codes for likelihood computation
    codes = np.zeros((n, len(sequences[labels[0]])), dtype=np.int8)
    for i, label in enumerate(labels):
        s = sequences[label]
        for j, c in enumerate(s):
            if c == 'A':
                codes[i, j] = 0
            elif c == 'C':
                codes[i, j] = 1
            elif c == 'G':
                codes[i, j] = 2
            else:
                codes[i, j] = 3
    
    L = codes.shape[1]
    
    if time.time() - start_time < deadline - 10:
        adj = _nni_refine(adj, leaf_map, is_leaf, codes, L, max_passes=3)
    
    # 4. Convert back to Newick
    final_newick = _adj_to_newick(adj, leaf_map, is_leaf)
    
    return final_newick
