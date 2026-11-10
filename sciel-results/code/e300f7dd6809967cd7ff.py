import numpy as np
import math
import time
from collections import deque

def _jc_dist(a, b):
    diffs = sum(1 for x, y in zip(a, b) if x != y)
    p = diffs / len(a)
    if p >= 0.75:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)

def _nj(dist, labels):
    nodes = list(labels)
    trees = {name: name for name in labels}
    while len(nodes) > 3:
        n = len(nodes)
        r = {a: sum(dist[(a, b)] for b in nodes if b != a) for a in nodes}
        best_crit = float('inf')
        best_pair = None
        for i in range(n):
            for j in range(i+1, n):
                a, b = nodes[i], nodes[j]
                crit = (n - 2) * dist[(a, b)] - r[a] - r[b]
                if crit < best_crit:
                    best_crit = crit
                    best_pair = (a, b)
        a, b = best_pair
        joined = f"({trees[a]},{trees[b]})"
        new = f"n{len(nodes)}_{a}_{b}"
        rest = [x for x in nodes if x not in (a, b)]
        for x in rest:
            dist[(new, x)] = dist[(x, new)] = 0.5 * (dist[(a, x)] + dist[(b, x)] - dist[(a, b)])
        nodes = rest + [new]
        trees[new] = joined
    return "(" + ",".join(trees[x] for x in nodes) + ");"

def _parse_newick(newick):
    neighbors = {}
    leaves = set()
    pos = 0
    
    def parse():
        nonlocal pos
        if newick[pos] == '(':
            pos += 1
            children = []
            while True:
                child = parse()
                children.append(child)
                if newick[pos] == ',':
                    pos += 1
                elif newick[pos] == ')':
                    pos += 1
                    break
            while pos < len(newick) and newick[pos] not in '(),;':
                pos += 1
            node = f"_int_{len(neighbors)}"
            neighbors[node] = set(children)
            for c in children:
                neighbors.setdefault(c, set()).add(node)
            return node
        else:
            name = ''
            while pos < len(newick) and newick[pos] not in '(),;':
                name += newick[pos]
                pos += 1
            while pos < len(newick) and newick[pos] not in '(),;':
                pos += 1
            neighbors.setdefault(name, set())
            leaves.add(name)
            return name
    
    newick = newick.rstrip(';')
    root = parse()
    return neighbors, leaves, root

def _get_leaf_set(neighbors, start, parent):
    """Get set of leaves in subtree rooted at start, not going through parent."""
    if start in neighbors and len(neighbors[start]) == 0:
        return {start}
    leaves = set()
    stack = [(start, parent)]
    while stack:
        node, par = stack.pop()
        if node in neighbors and len(neighbors[node]) == 0:
            leaves.add(node)
        else:
            for nb in neighbors.get(node, []):
                if nb != par:
                    stack.append((nb, node))
    return leaves

def _nni_improve(newick, sequences, labels, dist, max_time=8.0):
    neighbors, leaves, root = _parse_newick(newick)
    
    # Pre-compute leaf sets for all directed edges? 
    # Or just compute on the fly for the 4 subtrees of each NNI edge.
    
    # We need to calculate SSE.
    # To do this efficiently, we need to know the current tree distance for affected pairs.
    # We can just run BFS for the affected pairs in the current tree to get old_dist,
    # and in the new tree to get new_dist.
    
    def get_internal_edges(neighbors):
        edges = []
        seen = set()
        for node in neighbors:
            if node in leaves: continue
            for nb in neighbors[node]:
                if nb in leaves: continue
                if (node, nb) not in seen and (nb, node) not in seen:
                    edges.append((node, nb))
                    seen.add((node, nb))
                    seen.add((nb, node))
        return edges

    def bfs_dist(neighbors, start, end):
        if start == end: return 0
        q = deque([(start, 0)])
        visited = {start}
        while q:
            node, d = q.popleft()
            if node == end: return d
            for nb in neighbors[node]:
                if nb not in visited:
                    visited.add(nb)
                    q.append((nb, d + 1))
        return -1 # Should not happen

    # Initial SSE
    # We don't need total SSE, just the change.
    # But we need to know if a move is an improvement.
    # We can compute the SSE change for each move.
    
    improved = True
    while improved:
        improved = False
        edges = get_internal_edges(neighbors)
        for (u, v) in edges:
            u_nbs = list(neighbors[u])
            v_nbs = list(neighbors[v])
            a, b = [x for x in u_nbs if x != v]
            c, d = [x for x in v_nbs if x != u]
            
            # Get leaf sets
            La = _get_leaf_set(neighbors, a, u)
            Lb = _get_leaf_set(neighbors, b, u)
            Lc = _get_leaf_set(neighbors, c, v)
            Ld = _get_leaf_set(neighbors, d, v)
            
            # Moves:
            # 1: Swap b and c. 
            #    New u neighbors: a, c, v. New v neighbors: u, b, d.
            #    Affected pairs: (La, Lb), (La, Lc), (Lb, Ld), (Lc, Ld)
            # 2: Swap b and d.
            #    New u neighbors: a, d, v. New v neighbors: u, b, c.
            #    Affected pairs: (La, Lb), (La, Ld), (Lb, Lc), (Lc, Ld)
            
            best_move = None
            best_delta = 0
            
            # Helper to calculate delta SSE for a specific move
            def calc_delta(move_type):
                # move_type 1: b<->c
                # move_type 2: b<->d
                
                # Identify affected sets
                if move_type == 1:
                    # Affected: La-Lb, La-Lc, Lb-Ld, Lc-Ld
                    pairs = []
                    for x in La:
                        for y in Lb: pairs.append((x, y))
                        for y in Lc: pairs.append((x, y))
                    for x in Lb:
                        for y in Ld: pairs.append((x, y))
                    for x in Lc:
                        for y in Ld: pairs.append((x, y))
                else:
                    # Affected: La-Lb, La-Ld, Lb-Lc, Lc-Ld
                    pairs = []
                    for x in La:
                        for y in Lb: pairs.append((x, y))
                        for y in Ld: pairs.append((x, y))
                    for x in Lb:
                        for y in Lc: pairs.append((x, y))
                    for x in Lc:
                        for y in Ld: pairs.append((x, y))
                
                delta = 0.0
                for x, y in pairs:
                    d_obs = dist[(x, y)]
                    d_old = bfs_dist(neighbors, x, y)
                    
                    # Construct new neighbors for this move to get d_new
                    # We can simulate the swap locally
                    new_neighbors = {k: set(v) for k, v in neighbors.items()}
                    if move_type == 1:
                        new_neighbors[u].remove(b)
                        new_neighbors[u].add(c)
                        new_neighbors[b].remove(u)
                        new_neighbors[b].add(v)
                        new_neighbors[v].remove(c)
                        new_neighbors[v].add(b)
                        new_neighbors[c].remove(v)
                        new_neighbors[c].add(u)
                    else:
                        new_neighbors[u].remove(b)
                        new_neighbors[u].add(d)
                        new_neighbors[b].remove(u)
                        new_neighbors[b].add(v)
                        new_neighbors[v].remove(d)
                        new_neighbors[v].add(b)
                        new_neighbors[d].remove(v)
                        new_neighbors[d].add(u)
                        
                    d_new = bfs_dist(new_neighbors, x, y)
                    
                    delta += (d_obs - d_new)**2 - (d_obs - d_old)**2
                return delta

            d1 = calc_delta(1)
            if d1 < -1e-6:
                best_delta = d1
                best_move = 1
            
            d2 = calc_delta(2)
            if d2 < best_delta - 1e-6:
                best_delta = d2
                best_move = 2
            
            if best_move:
                # Apply move
                if best_move == 1:
                    neighbors[u].remove(b)
                    neighbors[u].add(c)
                    neighbors[b].remove(u)
                    neighbors[b].add(v)
                    neighbors[v].remove(c)
                    neighbors[v].add(b)
                    neighbors[c].remove(v)
                    neighbors[c].add(u)
                else:
                    neighbors[u].remove(b)
                    neighbors[u].add(d)
                    neighbors[b].remove(u)
                    neighbors[b].add(v)
                    neighbors[v].remove(d)
                    neighbors[v].add(b)
                    neighbors[d].remove(v)
                    neighbors[d].add(u)
                improved = True
                break # Restart loop with new topology

    return _to_newick(neighbors, leaves)

def _to_newick(neighbors, leaves):
    internal_nodes = [n for n in neighbors if n not in leaves]
    if not internal_nodes:
        return ""
    root = internal_nodes[0]
    
    parent = {}
    children = {}
    visited = set()
    
    def build(node, par):
        parent[node] = par
        children[node] = []
        for nb in neighbors[node]:
            if nb != par and nb not in visited:
                visited.add(nb)
                children[node].append(nb)
                build(nb, node)
    
    visited.add(root)
    build(root, None)
    
    def to_string(node):
        if node in leaves:
            return node
        if len(children[node]) == 0:
            return f"({node})"
        child_strs = [to_string(c) for c in children[node]]
        return f"({','.join(child_strs)})"
    
    return to_string(root) + ";"

def reconstruct(sequences):
    start_time = time.time()
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    
    # Compute distance matrix
    dist = {}
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            d = _jc_dist(sequences[labels[i]], sequences[labels[j]])
            dist[(labels[i], labels[j])] = d
            dist[(labels[j], labels[i])] = d
    
    # NJ
    newick = _nj(dist, labels)
    
    elapsed = time.time() - start_time
    if elapsed > 170:
        return newick
    
    # NNI refinement
    try:
        newick = _nni_improve(newick, sequences, labels, dist, max_time=170 - elapsed)
    except:
        pass
    
    return newick
