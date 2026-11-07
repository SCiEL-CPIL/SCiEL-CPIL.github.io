"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

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
    deadline = time.time() + 145
    trees = instance_trees(20, 611)
    n = 20
    
    # Precompute all edges in K20
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))
    
    # Precompute tree data
    tree_data = []
    for t_idx in range(20):
        tree_edges = trees[t_idx]
        k = t_idx + 1
        if k == 1:
            tree_data.append(None)
            continue
        
        tree_verts = set()
        for u, v in tree_edges:
            tree_verts.add(u)
            tree_verts.add(v)
        tree_verts = sorted(tree_verts)
        
        tree_adj = {v: [] for v in tree_verts}
        for u, v in tree_edges:
            tree_adj[u].append(v)
            tree_adj[v].append(u)
            
        # Order vertices by degree descending, then by index for determinism
        ordered_tv = sorted(tree_verts, key=lambda v: (-len(tree_adj[v]), v))
        
        tree_data.append({
            'edges': tree_edges,
            'verts': tree_verts,
            'adj': tree_adj,
            'ordered': ordered_tv,
            'k': k
        })
        
    # Sort tree indices: largest trees first (more constraints)
    tree_order = list(range(20))
    tree_order.sort(key=lambda idx: (-(len(trees[idx])), -idx))
    
    def embed_tree(t_idx, available_edges, rng, time_limit):
        """
        Attempts to embed tree t_idx into the graph defined by available_edges.
        Returns a list of host vertices for tree vertices 0..k-1, or None.
        """
        if t_idx == 0:
            return [rng.randint(0, n - 1)]
        
        data = tree_data[t_idx]
        k = data['k']
        tree_adj = data['adj']
        ordered_tv = data['ordered']
        tree_edges = data['edges']
        
        # Build adjacency of the available host graph
        host_adj = {h: set() for h in range(n)}
        for (a, b) in available_edges:
            host_adj[a].add(b)
            host_adj[b].add(a)
            
        host_map = {}
        used_hosts = set()
        node_count = [0]
        
        def backtrack(pos):
            node_count[0] += 1
            if node_count[0] % 10000 == 0 and time.time() > time_limit:
                return False
            
            if pos == len(ordered_tv):
                # All vertices assigned. Verify edges
                for u, v in tree_edges:
                    hu, hv = host_map[u], host_map[v]
                    edge = (min(hu, hv), max(hu, hv))
                    if edge not in available_edges:
                        return False
                return True
            
            tv = ordered_tv[pos]
            tv_neighbors = tree_adj[tv]
            
            # Identify assigned neighbors of tv
            assigned_neighbors = [(nb, host_map[nb]) for nb in tv_neighbors if nb in host_map]
            
            # Determine candidates
            if assigned_neighbors:
                candidates = set(host_adj[assigned_neighbors[0][1]])
                for _, hn in assigned_neighbors[1:]:
                    candidates &= host_adj[hn]
            else:
                candidates = set(range(n))
                
            candidates -= used_hosts
            
            if not candidates:
                return False
            
            # MRV: try candidates with fewer options for unassigned neighbors first
            # Score by how constrained they make remaining vertices
            scored_candidates = []
            for h in candidates:
                score = 0
                for w in tv_neighbors:
                    if w not in host_map:
                        other_assigned = [host_map[nb] for nb in tree_adj[w] if nb in host_map and nb != tv]
                        if other_assigned:
                            inter = set(host_adj[h])
                            for oh in other_assigned:
                                inter &= host_adj[oh]
                            inter -= used_hosts
                            score += max(0, 5 - len(inter))
                        else:
                            inter = set(host_adj[h]) - used_hosts
                            score += max(0, 5 - len(inter))
                scored_candidates.append((score, -len(host_adj[h]), h))
            
            scored_candidates.sort()
            candidates_list = [h for _, _, h in scored_candidates]
            
            # Add randomness to break ties
            if len(candidates_list) > 1:
                # Partial shuffle for diversity
                for i in range(len(candidates_list) - 1, 0, -1):
                    j = rng.randint(0, i)
                    candidates_list[i], candidates_list[j] = candidates_list[j], candidates_list[i]
                    if rng.random() < 0.3:
                        break
                        
            for h in candidates_list:
                host_map[tv] = h
                used_hosts.add(h)
                
                # Forward Checking with stronger pruning
                valid = True
                for w in tv_neighbors:
                    if w not in host_map:
                        other_assigned_hosts = [host_map[nb] for nb in tree_adj[w] if nb in host_map and nb != tv]
                        
                        if other_assigned_hosts:
                            inter = set(host_adj[h])
                            for oh in other_assigned_hosts:
                                inter &= host_adj[oh]
                            inter -= used_hosts
                        else:
                            inter = set(host_adj[h])
                            inter -= used_hosts
                            
                        if not inter:
                            valid = False
                            break
                            
                        # Additional check: if w has all neighbors assigned, verify edge exists
                        w_unassigned_neighbors = [nb for nb in tree_adj[w] if nb not in host_map]
                        if not w_unassigned_neighbors:
                            # All neighbors of w are assigned; check all required edges
                            for nb in tree_adj[w]:
                                hn = host_map[nb]
                                edge = (min(h, hn), max(h, hn))
                                if edge not in available_edges:
                                    valid = False
                                    break
                            if not valid:
                                break
                            
                if valid:
                    if backtrack(pos + 1):
                        return True
                else:
                    pass
                    
                del host_map[tv]
                used_hosts.remove(h)
                
            return False
            
        if backtrack(0):
            return [host_map[i] for i in range(k)]
        return None

    best_result = None
    best_partial = None
    best_partial_edges = 0
    
    # Try multiple random seeds for robustness
    max_attempts = 2000
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        
        # Adaptive time limit per tree based on remaining time
        remaining_trees = 20 - (attempt % 20)  # rough estimate
        time_per_tree = max(0.5, (deadline - time.time()) / max(1, remaining_trees))
        
        rng = Random(attempt * 104729 + 1)
        available = set(all_edges)
        result = [None] * 20
        success = True
        current_edges = 0
        
        for t_idx in tree_order:
            if time.time() > deadline:
                success = False
                break
                
            k = t_idx + 1
            if k == 1:
                result[t_idx] = [rng.randint(0, n - 1)]
                continue
                
            # Give more time to larger trees
            tree_time_limit = deadline - 0.5
            embedded = embed_tree(t_idx, available, rng, tree_time_limit)
            
            if embedded is None:
                success = False
                break
                
            result[t_idx] = embedded
            current_edges += len(trees[t_idx])
            for u, v in trees[t_idx]:
                hu, hv = embedded[u], embedded[v]
                edge = (min(hu, hv), max(hu, hv))
                available.discard(edge)
                
        if success:
            total_edges = sum(len(trees[i]) for i in range(20))
            if total_edges == 190 and len(available) == 0:
                best_result = result
                break
        else:
            # Track best partial result
            if current_edges > best_partial_edges:
                best_partial_edges = current_edges
                best_partial = result
                if best_partial_edges == 190:
                    best_result = best_partial
                    break

    if best_result is None:
        if best_partial:
            best_result = best_partial
        else:
            return [None] * 19 + [list(range(20))]

    final = []
    for entry in best_result:
        if entry is None:
            final.append(None)
        else:
            final.append([int(x) for x in entry])
    return final

# EVOLVE-BLOCK-END
