import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    D = 11

    # 1. Generate all supports (combinations of 4 indices)
    supports = list(itertools.combinations(range(D), 4))
    N_S = len(supports)  # 330

    # Convert supports to bitmasks for fast intersection check
    masks = []
    for s in supports:
        m = 0
        for idx in s:
            m |= (1 << idx)
        masks.append(m)

    # 2. Build conflict graph: edge if intersection size >= 3
    adj = [set() for _ in range(N_S)]
    for i in range(N_S):
        for j in range(i + 1, N_S):
            inter = (masks[i] & masks[j]).bit_count()
            if inter >= 3:
                adj[i].add(j)
                adj[j].add(i)

    adj_list = [list(s) for s in adj]

    # 3. Branch and bound for Maximum Independent Set
    # Using greedy coloring for upper bound estimation
    
    def greedy_coloring(vertices, adj_set):
        """
        Greedy coloring of the subgraph induced by vertices.
        Returns number of colors used (upper bound on independent set size).
        vertices: list of vertex indices
        adj_set: list of sets (adjacency)
        """
        if not vertices:
            return 0
        
        # Order vertices by degree in the subgraph (descending)
        subgraph = {v: set() for v in vertices}
        vset = set(vertices)
        for v in vertices:
            for nb in adj_set[v]:
                if nb in vset:
                    subgraph[v].add(nb)
        
        # Order by degree descending
        ordered = sorted(vertices, key=lambda v: len(subgraph[v]), reverse=True)
        
        colors = {}
        num_colors = 0
        for v in ordered:
            # Find smallest available color
            used_colors = set()
            for nb in subgraph[v]:
                if nb in colors:
                    used_colors.add(colors[nb])
            
            c = 0
            while c in used_colors:
                c += 1
            colors[v] = c
            num_colors = max(num_colors, c + 1)
        
        return num_colors

    def branch_and_bound(adj_set, initial_bound=0):
        """
        Branch and bound for maximum independent set.
        Returns the best independent set found.
        """
        n = len(adj_set)
        best_set = set()
        best_size = initial_bound
        
        # Use a stack for DFS
        # State: (candidate_set, current_set, upper_bound)
        # We'll use a more efficient approach with bitmasks
        
        # Precompute adjacency as bitmasks for fast operations
        adj_mask = [0] * n
        for i in range(n):
            for j in adj_set[i]:
                adj_mask[i] |= (1 << j)
        
        # All vertices mask
        all_mask = (1 << n) - 1
        
        def popcount(x):
            return bin(x).count('1')
        
        # Recursive branch and bound with coloring bound
        # We use a simplified version that works well for 330 nodes
        
        # First, get a good initial solution using greedy
        def greedy_independent_set():
            remaining = set(range(n))
            selected = set()
            while remaining:
                # Pick vertex with minimum degree in remaining
                min_deg = float('inf')
                min_v = None
                for v in remaining:
                    deg = sum(1 for nb in adj_set[v] if nb in remaining)
                    if deg < min_deg:
                        min_deg = deg
                        min_v = v
                selected.add(min_v)
                to_remove = {min_v} | {nb for nb in adj_set[min_v] if nb in remaining}
                remaining -= to_remove
            return selected
        
        initial_set = greedy_independent_set()
        best_size = len(initial_set)
        best_set = set(initial_set)
        
        # Branch and bound
        # We'll use a recursive approach with a time limit
        
        def bb_search(candidates, current_set, depth=0):
            nonlocal best_set, best_size
            
            if time.time() > deadline - 5.0:
                return
            
            # Upper bound: |candidates| + |current_set|
            if len(candidates) + len(current_set) <= best_size:
                return
            
            if not candidates:
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
                return
            
            # Greedy coloring bound
            color_bound = greedy_coloring(list(candidates), adj_set)
            if color_bound + len(current_set) <= best_size:
                return
            
            # Branch on a vertex
            # Pick vertex with maximum degree in candidates for better pruning
            v_list = list(candidates)
            v = max(v_list, key=lambda x: sum(1 for nb in adj_set[x] if nb in candidates))
            
            # Branch 1: Include v
            new_current = current_set | {v}
            # New candidates: those compatible with v (not adjacent to v)
            new_candidates = {x for x in candidates if x != v and x not in adj_set[v]}
            bb_search(new_candidates, new_current, depth + 1)
            
            # Branch 2: Exclude v
            new_candidates2 = candidates - {v}
            bb_search(new_candidates2, current_set, depth + 1)
        
        # Start with all vertices as candidates
        all_candidates = set(range(n))
        bb_search(all_candidates, set())
        
        return best_set

    # Run branch and bound
    best_set = branch_and_bound(adj_list)
    best_size = len(best_set)
    
    # If B&B didn't finish or found something suboptimal, try local search refinement
    if time.time() < deadline - 10.0:
        # Local search refinement
        adj_bool = np.zeros((N_S, N_S), dtype=bool)
        for i in range(N_S):
            for j in adj_list[i]:
                adj_bool[i][j] = True
        
        current_mask = np.zeros(N_S, dtype=bool)
        for v in best_set:
            current_mask[v] = True
        
        rng = np.random.RandomState(42)
        
        for _ in range(500):
            if time.time() > deadline - 5.0:
                break
            
            if np.any(current_mask):
                blocked = np.any(adj_bool[current_mask], axis=0)
            else:
                blocked = np.zeros(N_S, dtype=bool)
            
            candidates = ~(current_mask | blocked)
            cand_indices = np.where(candidates)[0]
            
            if len(cand_indices) > 0:
                v_add = int(cand_indices[rng.randint(len(cand_indices))])
                current_mask[v_add] = True
                continue
            
            # Stuck: try 1-swap
            current_indices = np.where(current_mask)[0]
            if len(current_indices) == 0:
                break
            
            v_remove = int(current_indices[rng.randint(len(current_indices))])
            current_mask[v_remove] = False
            
            if np.any(current_mask):
                blocked2 = np.any(adj_bool[current_mask], axis=0)
            else:
                blocked2 = np.zeros(N_S, dtype=bool)
            
            candidates2 = ~(current_mask | blocked2)
            cand_indices2 = np.where(candidates2)[0]
            
            if len(cand_indices2) > 0:
                v_add = int(cand_indices2[rng.randint(len(cand_indices2))])
                current_mask[v_add] = True
            else:
                current_mask[v_remove] = True
        
        # Update best if improved
        if np.sum(current_mask) > best_size:
            best_size = int(np.sum(current_mask))
            best_set = set(np.where(current_mask)[0])

    # 4. Construct vectors
    vectors = []
    for s_idx in best_set:
        support = supports[s_idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, sign in zip(support, signs):
                v[idx] = sign
            vectors.append(v)
    
    if not vectors:
        return np.array([], dtype=np.int8).reshape(0, D)
        
    final_vectors = np.array(vectors, dtype=np.int8)
    
    return final_vectors
