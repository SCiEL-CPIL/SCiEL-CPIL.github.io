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
    deadline = time.time() + 130  # Safety margin before 150s kill
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
    
    def embed_tree(t_idx, available_edges, rng):
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
        
        # Build adjacency of the available host graph using bitmasks for speed
        host_adj_mask = [0] * n
        for (a, b) in available_edges:
            host_adj_mask[a] |= (1 << b)
            host_adj_mask[b] |= (1 << a)
        
        all_mask = (1 << n) - 1
        host_map = {}
        used_mask = 0
        
        def backtrack(pos):
            nonlocal used_mask
            if pos == len(ordered_tv):
                # All vertices assigned. Verify edges
                for u, v in tree_edges:
                    hu, hv = host_map[u], host_map[v]
                    if not (host_adj_mask[hu] >> hv) & 1:
                        return False
                return True
            
            tv = ordered_tv[pos]
            tv_neighbors = tree_adj[tv]
            
            # Identify assigned neighbors of tv
            assigned_neighbors = [(nb, host_map[nb]) for nb in tv_neighbors if nb in host_map]
            
            # Determine candidates using bitmask intersection
            if assigned_neighbors:
                cand_mask = host_adj_mask[assigned_neighbors[0][1]]
                for _, hn in assigned_neighbors[1:]:
                    cand_mask &= host_adj_mask[hn]
            else:
                cand_mask = all_mask
                
            cand_mask &= ~used_mask
            
            if cand_mask == 0:
                return False
            
            # Extract candidate list, sorted by degree in available graph
            candidates = []
            cm = cand_mask
            while cm:
                h = (cm & (-cm)).bit_length() - 1
                cm &= cm - 1
                candidates.append(h)
            
            # Sort by degree descending (precomputed from mask)
            candidates.sort(key=lambda h: bin(host_adj_mask[h]).count('1'), reverse=True)
            
            # Light randomization: shuffle only if many candidates, and only the tail
            if len(candidates) > 3:
                # Keep top 3 in order, shuffle the rest
                rng.shuffle(candidates[3:])
            
            for h in candidates:
                host_map[tv] = h
                new_used = used_mask | (1 << h)
                
                # Forward Checking:
                valid = True
                for w in tv_neighbors:
                    if w not in host_map:
                        # Check if w still has valid candidates
                        # w must connect to h and all other assigned neighbors of w
                        other_assigned = [host_map[nb] for nb in tree_adj[w] if nb in host_map and nb != tv]
                        
                        if other_assigned:
                            inter = host_adj_mask[h]
                            for oh in other_assigned:
                                inter &= host_adj_mask[oh]
                        else:
                            inter = host_adj_mask[h]
                        
                        inter &= ~new_used
                        if inter == 0:
                            valid = False
                            break
                            
                if valid:
                    used_mask = new_used
                    if backtrack(pos + 1):
                        return True
                    used_mask = new_used & ~(1 << h)
                    
                del host_map[tv]
                
            return False
            
        if backtrack(0):
            return [host_map[i] for i in range(k)]
        return None

    best_result = None
    
    # Try multiple random seeds for robustness
    for attempt in range(2000):
        if time.time() > deadline:
            break
            
        rng = Random(attempt * 104729 + 1)
        available = set(all_edges)
        result = [None] * 20
        success = True
        
        for t_idx in tree_order:
            if time.time() > deadline:
                success = False
                break
                
            k = t_idx + 1
            if k == 1:
                result[t_idx] = [rng.randint(0, n - 1)]
                continue
                
            embedded = embed_tree(t_idx, available, rng)
            
            if embedded is None:
                success = False
                break
                
            result[t_idx] = embedded
            for u, v in trees[t_idx]:
                hu, hv = embedded[u], embedded[v]
                edge = (min(hu, hv), max(hu, hv))
                available.discard(edge)
                
        if success:
            total_edges = sum(len(trees[i]) for i in range(20))
            if total_edges == 190 and len(available) == 0:
                best_result = result
                break

    if best_result is None:
        # Fallback: try to find best partial solution
        best_partial = None
        best_partial_edges = 0
        
        for attempt in range(200):
            if time.time() > deadline:
                break
            rng = Random(attempt * 999999 + 7)
            available = set(all_edges)
            result = [None] * 20
            current_edges = 0
            
            for t_idx in tree_order:
                if time.time() > deadline:
                    break
                k = t_idx + 1
                if k == 1:
                    result[t_idx] = [rng.randint(0, n - 1)]
                    continue
                embedded = embed_tree(t_idx, available, rng)
                if embedded is None:
                    break
                result[t_idx] = embedded
                current_edges += len(trees[t_idx])
                for u, v in trees[t_idx]:
                    hu, hv = embedded[u], embedded[v]
                    edge = (min(hu, hv), max(hu, hv))
                    available.discard(edge)
                    
            if current_edges > best_partial_edges:
                best_partial_edges = current_edges
                best_partial = result
                if best_partial_edges == 190:
                    break
                    
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
