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
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    trees = instance_trees(20, 611)
    n = 20
    
    # Precompute tree structures
    tree_data = []
    for t_idx in range(20):
        tree_edges = trees[t_idx]
        k = t_idx + 1
        if k == 1:
            tree_data.append((1, [], {0: []}, [0]))
            continue
        
        adj = {v: [] for v in range(k)}
        for u, v in tree_edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Order vertices by degree descending to prioritize high-degree nodes
        ordered_verts = sorted(range(k), key=lambda x: (-len(adj[x]), x))
        tree_data.append((k, tree_edges, adj, ordered_verts))

    # Total possible edges in K20
    ALL_EDGES = 1 << 190 # This is too big for a simple int bitset if we index by edge ID 0..189.
    # Actually, 20 vertices have 190 edges. An integer can hold 190 bits easily.
    # Let's map each edge (u, v) with u < v to an index.
    edge_indices = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_indices[(u, v)] = idx
            idx += 1
            
    all_edges_mask = (1 << 190) - 1

    def get_edge_mask(u, v):
        if u > v: u, v = v, u
        return 1 << edge_indices[(u, v)]

    def embed_tree(t_idx, available_mask, rng, deadline):
        """
        Embed tree t_idx into the graph defined by available_mask.
        Returns a list of host vertices or None.
        """
        if time.time() > deadline:
            return None
            
        k, tree_edges, adj, ordered_verts = tree_data[t_idx]
        
        if k == 1:
            # Find any available vertex? No, T1 is just a vertex. 
            # It consumes no edges. It can be any vertex 0..19.
            # However, the host vertex must be distinct within the tree, but T1 has only 1 vertex.
            # The constraint is edge-disjointness. T1 uses no edges.
            # So T1 can be placed anywhere.
            return [rng.randint(0, n - 1)]

        # Host adjacency as bitsets for fast intersection
        # host_adj[v] is a bitmask of neighbors u such that (u,v) is in available_mask
        host_adj = [0] * n
        # Also track which vertices have available edges?
        # Let's build the adjacency bitsets from the mask
        for u in range(n):
            for v in range(u + 1, n):
                if (available_mask >> edge_indices[(u, v)]) & 1:
                    host_adj[u] |= (1 << v)
                    host_adj[v] |= (1 << u)

        host_map = {}
        used_hosts_mask = 0 # Bitmask of used host vertices

        # We need to assign tree vertices (0..k-1) to host vertices (0..19)
        
        def backtrack(pos):
            if time.time() > deadline:
                return False
            
            if pos == k:
                # Verify all tree edges are present in available_mask
                for u, v in tree_edges:
                    hu, hv = host_map[u], host_map[v]
                    if hu > hv: hu, hv = hv, hu
                    if not ((available_mask >> edge_indices[(hu, hv)]) & 1):
                        return False
                return True

            tv = ordered_verts[pos]
            tv_neighbors = adj[tv]
            
            # Identify neighbors of tv that are already assigned
            assigned_neighbors = []
            for nb in tv_neighbors:
                if nb in host_map:
                    assigned_neighbors.append(host_map[nb])

            if assigned_neighbors:
                # Candidate hosts must be neighbors of ALL assigned neighbors
                candidates_mask = host_adj[assigned_neighbors[0]]
                for hn in assigned_neighbors[1:]:
                    candidates_mask &= host_adj[hn]
            else:
                # No assigned neighbors, any host is a candidate initially
                candidates_mask = (1 << n) - 1

            # Exclude already used hosts
            candidates_mask &= ~used_hosts_mask

            if candidates_mask == 0:
                return False

            # Generate candidate list
            candidates = []
            cm = candidates_mask
            while cm:
                lsb = cm & -cm
                h = lsb.bit_length() - 1
                candidates.append(h)
                cm ^= lsb

            # Heuristic: Sort candidates by degree in available graph (descending)
            # To get degree, we need popcount of host_adj[h]
            candidates.sort(key=lambda h: -bin(host_adj[h]).count('1'))
            
            # Add some randomness to top candidates to escape local minima
            if len(candidates) > 3:
                top3 = candidates[:3]
                rest = candidates[3:]
                rng.shuffle(rest)
                candidates = top3 + rest
            elif len(candidates) > 1:
                rng.shuffle(candidates)

            for h in candidates:
                host_map[tv] = h
                new_used_mask = used_hosts_mask | (1 << h)
                
                # Forward Checking:
                # For each unassigned neighbor w of tv, check if there is still a valid host
                # A valid host for w must be:
                # 1. Neighbor of h (host of tv) in available graph
                # 2. Neighbor of any other assigned neighbor of w
                # 3. Not used
                
                valid = True
                for w in tv_neighbors:
                    if w in host_map:
                        continue
                    
                    # Start with neighbors of h
                    possible = host_adj[h]
                    
                    # Intersect with neighbors of other assigned neighbors of w
                    for nb in adj[w]:
                        if nb in host_map and nb != tv:
                            possible &= host_adj[host_map[nb]]
                    
                    possible &= ~new_used_mask
                    
                    if possible == 0:
                        valid = False
                        break
                
                if valid:
                    if backtrack(pos + 1):
                        return True

                del host_map[tv]

            return False

        if backtrack(0):
            return [host_map[i] for i in range(k)]
        return None

    best_result = None
    best_score = -1

    # Main search loop
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        rng = Random(num_restarts * 12345 + 67890)
        
        available_mask = all_edges_mask
        result = [None] * 20
        success = True
        
        # Determine tree ordering
        # T1 (0) is trivial. T20 (19) is hardest.
        # Sort T2..T19 by size descending, then randomize ties.
        tree_order = list(range(20))
        
        # Separate T1 and T20
        t1_idx = 0
        t20_idx = 19
        
        # Create a list of indices to process: T2..T19
        mid_indices = [i for i in range(20) if i != t1_idx and i != t20_idx]
        
        # Sort by number of edges (k-1) descending
        mid_indices.sort(key=lambda i: -(len(trees[i])))
        
        # Shuffle within groups of same size
        i = 0
        while i < len(mid_indices):
            j = i
            while j < len(mid_indices) and len(trees[mid_indices[j]]) == len(trees[mid_indices[i]]):
                j += 1
            if j > i + 1:
                rng.shuffle(mid_indices[i:j])
            i = j
        
        # Place T1 and T20 in random positions in the order
        # Usually, we want to place larger trees first.
        # Let's insert T20 at the beginning or near beginning, T1 at the end or anywhere.
        # Actually, T1 doesn't consume edges, so it can be last.
        # T20 consumes 19 edges, very constraining. Put it first?
        # Let's try: [T20, ...shuffled mid..., T1]
        # Or randomize the position of T20 among the first 3?
        
        pos_t20 = rng.randint(0, 2) # Place T20 in first 3 slots
        pos_t1 = len(mid_indices) # T1 last
        
        tree_order = mid_indices[:pos_t20] + [t20_idx] + mid_indices[pos_t20:] + [t1_idx]
        
        # Shuffle T1 position too?
        # T1 is [v]. It doesn't affect edge availability.
        # So its position doesn't matter for the search, except for the result array.
        
        for t_idx in tree_order:
            if time.time() > deadline:
                success = False
                break
                
            if t_idx == t1_idx:
                # T1
                result[t_idx] = [rng.randint(0, n - 1)]
                continue
                
            embedded = embed_tree(t_idx, available_mask, rng, deadline)
            
            if embedded is None:
                success = False
                break
                
            result[t_idx] = embedded
            
            # Remove used edges from available_mask
            for u, v in trees[t_idx]:
                hu, hv = embedded[u], embedded[v]
                if hu > hv: hu, hv = hv, hu
                edge_idx = edge_indices[(hu, hv)]
                available_mask &= ~(1 << edge_idx)

        if success:
            # Check if all edges are used (should be true if sum is 190)
            # And verify no conflicts
            if available_mask == 0:
                # Perfect solution
                best_result = result
                break
            else:
                # This shouldn't happen if the sum is exactly 190 and we embedded all
                # But just in case, treat as valid partial? No, task requires specific trees.
                # If we embedded all trees, we used sum(k-1) = 190 edges.
                # So available_mask MUST be 0.
                best_result = result
                break

    if best_result is None:
        # Fallback: return None for all? Or try a greedy approach?
        # If we can't find a perfect packing, we return the best partial?
        # The task says "Entry k-1 is None or a list...".
        # "The raw value is the total number of embedded tree edges".
        # So we want to maximize embedded edges.
        
        # Let's try a few more random restarts to get a good partial solution
        best_partial = None
        best_partial_edges = 0
        
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            rng = Random(99999 + attempt)
            available_mask = all_edges_mask
            result = [None] * 20
            edges_count = 0
            
            # Order: largest first
            order = sorted(range(20), key=lambda i: -len(trees[i]))
            
            for t_idx in order:
                if t_idx == 0: # T1
                    result[0] = [rng.randint(0, n-1)]
                    continue
                embedded = embed_tree(t_idx, available_mask, rng, deadline - 5)
                if embedded is None:
                    continue
                result[t_idx] = embedded
                edges_count += len(trees[t_idx])
                for u, v in trees[t_idx]:
                    hu, hv = embedded[u], embedded[v]
                    if hu > hv: hu, hv = hv, hu
                    edge_idx = edge_indices[(hu, hv)]
                    available_mask &= ~(1 << edge_idx)
            
            if edges_count > best_partial_edges:
                best_partial_edges = edges_count
                best_partial = result
                if best_partial_edges == 190:
                    break
                    
        if best_partial:
            best_result = best_partial
        else:
            # Absolute fallback
            best_result = [None] * 20
            # Try to embed at least T20?
            # T20 is a spanning tree. K20 is connected.
            # Any spanning tree of K20 works.
            # Let's just map T20 to 0..19
            best_result[19] = list(range(20))
            # Mark edges used by T20 as unavailable (though result is just the mapping)
            # For the score, we only count embedded tree edges.
            # T20 has 19 edges.
            
    # Final formatting
    final = []
    for entry in best_result:
        if entry is None:
            final.append(None)
        else:
            final.append([int(x) for x in entry])
    return final

# EVOLVE-BLOCK-END
