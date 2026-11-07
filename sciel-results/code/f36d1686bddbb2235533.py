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
    deadline = time.time() + 140
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
        
        # Build adjacency of the available host graph
        host_adj = {h: set() for h in range(n)}
        for (a, b) in available_edges:
            host_adj[a].add(b)
            host_adj[b].add(a)
            
        host_map = {}
        used_hosts = set()
        
        def backtrack(pos):
            if pos == len(ordered_tv):
                # All vertices assigned. Verify edges (should be guaranteed by constraints, but safe to check)
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
                # Candidate must be connected to all assigned neighbors in host graph
                # Intersection of neighbor sets of assigned hosts
                candidates = set(host_adj[assigned_neighbors[0][1]])
                for _, hn in assigned_neighbors[1:]:
                    candidates &= host_adj[hn]
            else:
                candidates = set(range(n))
                
            candidates -= used_hosts
            
            if not candidates:
                return False
                
            # Heuristic: Sort candidates by degree in available graph (descending)
            # This tends to find solutions faster as high-degree nodes are more flexible
            # or more constrained? Usually, assigning high-degree tree nodes to high-degree host nodes
            # leaves more flexibility for leaves.
            # Let's try sorting by degree in host_adj.
            candidates_list = sorted(candidates, key=lambda h: -len(host_adj[h]))
            
            # Add some randomness to break ties and explore different branches
            if len(candidates_list) > 1:
                # Shuffle a small prefix or the whole list?
                # Shuffling the whole list is expensive but N is small.
                # Better: Pick a random candidate from the top 3?
                # Or just shuffle.
                rng.shuffle(candidates_list)
                
            for h in candidates_list:
                host_map[tv] = h
                used_hosts.add(h)
                
                # Forward Checking:
                # For any unassigned neighbor w of tv, check if it still has valid candidates.
                valid = True
                for w in tv_neighbors:
                    if w not in host_map:
                        # w is unassigned. It is adjacent to tv.
                        # Its candidates must be in host_adj[h] AND not used.
                        # Also, it must be connected to any *other* assigned neighbors of w.
                        
                        w_assigned_neighbors = [host_map[nb] for nb in tree_adj[w] if nb in host_map]
                        if not w_assigned_neighbors:
                            # Only constraint is connection to h (which is assigned now)
                            # So w must be in host_adj[h] and not used
                            if h not in host_adj[w]: # Wait, host_adj is symmetric. w must be in host_adj[h]
                                pass # This check is implicit in the candidate generation for w later
                            # Actually, we need to check if there EXISTS a candidate for w.
                            # Candidates for w:
                            # 1. Must be in host_adj[h] (since h is assigned to tv, and tv-w is an edge)
                            # 2. Must be connected to all other assigned neighbors of w.
                            # 3. Must not be in used_hosts (excluding h, but h is now used).
                            
                            # Let's do a quick check:
                            # Is there any vertex x in host_adj[h] \ used_hosts 
                            # such that x is connected to all other assigned neighbors of w?
                            
                            other_assigned_hosts = [host_map[nb] for nb in tree_adj[w] if nb in host_map and nb != tv]
                            
                            # Candidates for w must be in intersection of host_adj[x] for x in [h] + other_assigned_hosts
                            # And not in used_hosts.
                            
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
                            
                if valid:
                    if backtrack(pos + 1):
                        return True
                else:
                    pass # Invalid, undo below
                    
                del host_map[tv]
                used_hosts.remove(h)
                
            return False
            
        if backtrack(0):
            return [host_map[i] for i in range(k)]
        return None

    best_result = None
    
    # Try multiple random seeds for robustness
    for attempt in range(1000):
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
            # Verify total edges
            total_edges = sum(len(trees[i]) for i in range(20))
            if total_edges == 190 and len(available) == 0:
                best_result = result
                break

    if best_result is None:
        # Fallback: return a valid partial solution or None?
        # The task says "Entry k-1 is None or a list".
        # If we fail, we should return what we have?
        # "Any broken rule makes the whole result invalid".
        # If we can't find a full packing, we return the best partial?
        # The raw value is the sum of edges of *embedded* trees.
        # If we return a partial result, the score will be lower.
        # But returning all None gives 0.
        # Let's try to return a partial result if full packing fails.
        # However, the loop above breaks on failure.
        # We should store the best partial result.
        
        # Re-run with partial tracking
        best_partial = None
        best_partial_edges = 0
        
        for attempt in range(100): # Fewer attempts for partial
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
                    continue # Skip this tree, try next? No, order is fixed.
                    # Actually, if we skip, we might not be able to embed later ones either.
                    # But maybe a different order would work?
                    # For simplicity, if one fails, we stop this attempt.
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
            # Absolute fallback
            return [None] * 19 + [list(range(20))]

    final = []
    for entry in best_result:
        if entry is None:
            final.append(None)
        else:
            final.append([int(x) for x in entry])
    return final

# EVOLVE-BLOCK-END
