"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random, sample
from itertools import combinations

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2: trees.append([] if k == 1 else [(0, 1)]); continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code: degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    trees = instance_trees(20, 611)
    n = 20
    
    # All edges in K20
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))
    
    # For each tree, we need to find a set of edges that forms an isomorphic copy
    # Tree k has k vertices and k-1 edges
    # We need to assign distinct vertices to the tree's vertices such that edges map to available edges
    
    # Order trees by difficulty (most edges first for constraint propagation)
    tree_order = list(range(20))  # indices 0..19 for T1..T20
    # Sort by number of edges descending, then by k descending
    tree_order.sort(key=lambda idx: (-(len(trees[idx])), -idx))
    
    def get_tree_edges(k_idx):
        """Return the edge list of tree k_idx (0-indexed, so tree k_idx+1)"""
        return trees[k_idx]
    
    def try_embed(tree_idx, available_edges, rng):
        """Try to embed tree tree_idx into available_edges.
        Returns a list of (host_u, host_v) edges if successful, else None.
        Also returns the vertex mapping if successful."""
        tree_edges = get_tree_edges(tree_idx)
        k = tree_idx + 1  # number of vertices in this tree
        if k == 1:
            return []
        
        # Get the tree's vertex set and edges
        tree_verts = set()
        for u, v in tree_edges:
            tree_verts.add(u)
            tree_verts.add(v)
        tree_verts = sorted(tree_verts)
        
        # Build adjacency for the tree
        tree_adj = {v: [] for v in tree_verts}
        for u, v in tree_edges:
            tree_adj[u].append(v)
            tree_adj[v].append(u)
        
        # We need to assign distinct host vertices to tree vertices
        # such that for each tree edge (u,v), (host[u], host[v]) is in available_edges
        
        # Get candidate host vertices: those incident to available edges
        # For k=1, any single vertex works
        if k == 1:
            # Just pick any available vertex
            # Actually T1 has no edges, so we just need to pick a vertex
            # But the return should be a list of k distinct int host vertices
            # For k=1, return [v] for some v
            # We need to track which vertices are used
            return None  # handled specially
        
        # Backtracking: assign host vertices to tree vertices
        # Order tree vertices by degree descending for better pruning
        ordered_tree_verts = sorted(tree_verts, key=lambda v: -len(tree_adj[v]))
        
        host_map = {}  # tree_vertex -> host_vertex
        used_hosts = set()
        
        # Candidate hosts: all vertices 0..19 that are incident to at least one available edge
        # Actually, for a tree with k vertices, we need k distinct host vertices
        # where the induced subgraph contains the tree as a subgraph
        
        # Let's use a simpler approach: try all injective mappings from tree_verts to host vertices
        # This is too expensive for large k. Let's use backtracking with pruning.
        
        # For each tree vertex, candidates are host vertices h such that
        # for all already-assigned neighbors u of v in tree, (h, host_map[u]) is in available_edges
        
        def backtrack(pos):
            if pos == len(ordered_tree_verts):
                # Check all edges
                for u, v in tree_edges:
                    hu, hv = host_map[u], host_map[v]
                    edge = (min(hu, hv), max(hu, hv))
                    if edge not in available_edges:
                        return False
                return True
            
            tv = ordered_tree_verts[pos]
            # Try candidate host vertices
            for h in range(n):
                if h in used_hosts:
                    continue
                # Check consistency with already-assigned neighbors
                ok = True
                for nb in tree_adj[tv]:
                    if nb in host_map:
                        hn = host_map[nb]
                        edge = (min(h, hn), max(h, hn))
                        if edge not in available_edges:
                            ok = False
                            break
                if not ok:
                    continue
                host_map[tv] = h
                used_hosts.add(h)
                if backtrack(pos + 1):
                    return True
                del host_map[tv]
                used_hosts.remove(h)
            return False
        
        if backtrack(0):
            # Build the result: list of host vertices for each tree vertex in order 0..k-1
            # Wait, the task says "position i is the image of vertex i of the fixed tree T_k"
            # So we need to return a list of length k where position i = host vertex for tree vertex i
            result = []
            for i in range(k):
                result.append(host_map[i])
            return result
        return None
    
    # Main search: assign trees in order
    # We'll use randomized restarts
    
    best_result = None
    
    for attempt in range(10000):
        if time.time() > deadline:
            break
        
        rng = Random(attempt * 7919 + 13)
        
        # Shuffle the tree order slightly for variety, but keep large trees early
        # Actually, let's keep a fixed order but randomize the embedding choices
        available = set(all_edges)
        result = [None] * 20
        success = True
        
        # Process trees from largest to smallest
        for t_idx in tree_order:
            if time.time() > deadline:
                success = False
                break
            k = t_idx + 1
            if k == 1:
                # T1: no edges, just pick any vertex
                # We need to pick a vertex that doesn't conflict... but T1 uses no edges
                # So any vertex works. But we need distinct vertices across all trees?
                # No! The task says "Embedded trees must be edge-disjoint in K20"
                # Vertices can be shared! Only edges must be disjoint.
                # So for T1, we can pick any vertex.
                result[t_idx] = [rng.randint(0, n - 1)]
                continue
            
            # Try to embed tree t_idx
            # Randomize the order of host vertex trials
            embedded = None
            for trial in range(50):
                if time.time() > deadline:
                    success = False
                    break
                # Try embedding with some randomization
                # We'll use a modified backtracking that randomizes choices
                tree_edges = trees[t_idx]
                tree_verts = set()
                for u, v in tree_edges:
                    tree_verts.add(u)
                    tree_verts.add(v)
                tree_verts = sorted(tree_verts)
                
                tree_adj = {v: [] for v in tree_verts}
                for u, v in tree_edges:
                    tree_adj[u].append(v)
                    tree_adj[v].append(u)
                
                ordered_tv = sorted(tree_verts, key=lambda v: -len(tree_adj[v]))
                # Randomize the order within same degree
                deg_groups = {}
                for v in ordered_tv:
                    d = len(tree_adj[v])
                    if d not in deg_groups:
                        deg_groups[d] = []
                    deg_groups[d].append(v)
                randomized_ordered = []
                for d in sorted(deg_groups.keys(), reverse=True):
                    group = deg_groups[d]
                    rng.shuffle(group)
                    randomized_ordered.extend(group)
                
                host_map = {}
                used_hosts = set()
                
                def bt(pos, avail):
                    if pos == len(randomized_ordered):
                        for u, v in tree_edges:
                            hu, hv = host_map[u], host_map[v]
                            edge = (min(hu, hv), max(hu, hv))
                            if edge not in avail:
                                return False
                        return True
                    
                    tv = randomized_ordered[pos]
                    candidates = list(range(n))
                    rng.shuffle(candidates)
                    for h in candidates:
                        if h in used_hosts:
                            continue
                        ok = True
                        for nb in tree_adj[tv]:
                            if nb in host_map:
                                hn = host_map[nb]
                                edge = (min(h, hn), max(h, hn))
                                if edge not in avail:
                                    ok = False
                                    break
                        if not ok:
                            continue
                        host_map[tv] = h
                        used_hosts.add(h)
                        if bt(pos + 1, avail):
                            return True
                        del host_map[tv]
                        used_hosts.remove(h)
                    return False
                
                if bt(0, available):
                    embedded = [host_map[i] for i in range(k)]
                    break
            
            if not success:
                break
            
            if embedded is None:
                success = False
                break
            
            result[t_idx] = embedded
            # Remove used edges from available
            for u, v in tree_edges:
                hu, hv = embedded[u], embedded[v]
                edge = (min(hu, hv), max(hu, hv))
                available.discard(edge)
        
        if success:
            # Verify all edges are used exactly once
            total_edges = sum(len(trees[i]) for i in range(20))
            if total_edges == 190 and len(available) == 0:
                best_result = result
                break
            elif total_edges == 190:
                # Even if some edges remain, if all trees are embedded edge-disjointly, it's valid
                # But we need exactly 190 edges used. Let me check.
                # Actually, sum of (k-1) for k=1..20 = 190, so all edges must be used.
                # If available is not empty, something went wrong.
                pass
        
        if best_result is not None:
            break
    
    if best_result is None:
        # Fallback: return the seed solution
        return [None] * 19 + [list(range(20))]
    
    # Ensure types are correct: lists of ints
    final = []
    for entry in best_result:
        if entry is None:
            final.append(None)
        else:
            final.append([int(x) for x in entry])
    return final

# EVOLVE-BLOCK-END
