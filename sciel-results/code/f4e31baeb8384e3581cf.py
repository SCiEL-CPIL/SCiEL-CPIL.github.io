"""
Covering design C(v,k,t) using Greedy Set Cover + Exact Branch-and-Bound Deletion.

Strategy:
1. Fast randomized greedy to get an initial covering.
2. Iterative exact deletion:
   a. Identify critical blocks (those uniquely covering at least one t-subset).
   b. Build conflict graph on non-critical blocks: edge if they don't share any
      uniquely-covered t-subset (i.e., removing both is safe).
   c. Solve Maximum Independent Set on this graph via branch-and-bound with
      greedy coloring upper bound (Tomita's algorithm).
   d. Remove the MIS blocks.
   e. Repeat until no non-critical blocks remain or time runs out.
"""
import itertools
import random
import time
import sys
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    deadline = time.time() + 115  # safety margin before 150s kill
    start_time = time.time()
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)
    
    def block_tsub_indices(block):
        """Return list of t-subset indices covered by this block."""
        indices = []
        for ts in itertools.combinations(block, t):
            idx = tsub_index.get(ts)
            if idx is not None:
                indices.append(idx)
        return indices
    
    def greedy_build(seed, randomize_order=True):
        """Randomized greedy covering - optimized for speed."""
        rng = random.Random(seed)
        uncovered = [True] * n_tsubs
        blocks = []
        
        # Precompute t-subsets for faster access
        while any(uncovered):
            # Find first uncovered
            first_uncovered = -1
            for i in range(n_tsubs):
                if uncovered[i]:
                    first_uncovered = i
                    break
            if first_uncovered == -1:
                break
            
            block = list(all_tsubs[first_uncovered])
            
            while len(block) < k:
                in_block = set(block)
                candidates = [x for x in range(v) if x not in in_block]
                
                if not candidates:
                    break
                    
                if randomize_order:
                    rng.shuffle(candidates)
                
                best_gain = -1
                best_x = candidates[0]
                # Precompute (t-1)-subsets of current block for gain calculation
                block_t1 = list(itertools.combinations(block, t-1)) if t > 1 else [()]
                
                for x in candidates:
                    gain = 0
                    for combo in block_t1:
                        ts = tuple(sorted(combo + (x,)))
                        idx = tsub_index.get(ts)
                        if idx is not None and uncovered[idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain and gain > 0:
                        if rng.random() < 0.5:
                            best_x = x
                
                block.append(best_x)
            
            block.sort()
            blocks.append(block)
            
            for idx in block_tsub_indices(block):
                uncovered[idx] = False
        
        return blocks
    
    def exact_deletion_pass(blocks, deadline):
        """
        Iteratively remove redundant blocks using exact MIS on conflict graph.
        Returns (reduced_blocks, num_removed).
        """
        blocks_list = [list(b) for b in blocks]
        total_removed = 0
        
        while time.time() < deadline - 10:
            n_blocks = len(blocks_list)
            if n_blocks == 0:
                break
            
            # Compute coverage counts and block t-subsets
            cov_count = np.zeros(n_tsubs, dtype=np.int32)
            block_tsubs = []
            for block in blocks_list:
                indices = block_tsub_indices(block)
                block_tsubs.append(indices)
                for idx in indices:
                    cov_count[idx] += 1
            
            # Identify critical blocks: those with at least one t-subset where cov_count == 1
            critical = [False] * n_blocks
            for bi in range(n_blocks):
                for idx in block_tsubs[bi]:
                    if cov_count[idx] == 1:
                        critical[bi] = True
                        break
            
            non_critical_idx = [i for i in range(n_blocks) if not critical[i]]
            
            if len(non_critical_idx) < 2:
                break
            
            # Build conflict graph on non-critical blocks
            # Edge between i and j if they don't share any t-subset with cov_count == 1
            # (i.e., removing both i and j is safe)
            m = len(non_critical_idx)
            
            # For each non-critical block, get set of t-subsets it uniquely covers
            unique_sets = []
            for bi in non_critical_idx:
                unique = set()
                for idx in block_tsubs[bi]:
                    if cov_count[idx] == 1:
                        unique.add(idx)
                unique_sets.append(unique)
            
            # Build adjacency: edge if unique_sets[i] ∩ unique_sets[j] is empty
            # This means removing both is safe
            adj = [set() for _ in range(m)]
            for i in range(m):
                for j in range(i+1, m):
                    if not (unique_sets[i] & unique_sets[j]):
                        adj[i].add(j)
                        adj[j].add(i)
            
            # Solve MIS on this graph using branch-and-bound with coloring
            # The MIS of this graph = maximum set of non-critical blocks we can remove
            mis_size, mis_nodes = solve_mis(adj, m, deadline - 15)
            
            if mis_size < 2:
                break
            
            # Remove the MIS blocks
            remove_set = set(non_critical_idx[i] for i in mis_nodes)
            blocks_list = [b for i, b in enumerate(blocks_list) if i not in remove_set]
            total_removed += len(remove_set)
        
        return blocks_list, total_removed
    
    def solve_mis(adj, n, deadline):
        """
        Solve Maximum Independent Set using branch-and-bound with greedy coloring.
        Returns (size, list_of_node_indices).
        """
        best_size = [0]
        best_nodes = [[]]
        
        # Greedy coloring for upper bound
        def greedy_coloring(subgraph_nodes, adj):
            """Color the induced subgraph, return (colors, color_classes)."""
            if not subgraph_nodes:
                return 0, []
            
            # Order by degree (descending)
            deg = {v: len(adj[v]) for v in subgraph_nodes}
            ordered = sorted(subgraph_nodes, key=lambda v: -deg[v])
            
            color_of = {}
            color_classes = []
            
            for v in ordered:
                # Find smallest available color
                used_colors = set()
                for u in adj[v]:
                    if u in color_of:
                        used_colors.add(color_of[u])
                
                c = 0
                while c in used_colors:
                    c += 1
                
                color_of[v] = c
                if c >= len(color_classes):
                    color_classes.append([])
                color_classes[c].append(v)
            
            return len(color_classes), color_classes
        
        def branch(nodes, current_selection):
            if time.time() > deadline:
                return
            
            if not nodes:
                if len(current_selection) > best_size[0]:
                    best_size[0] = len(current_selection)
                    best_nodes[0] = list(current_selection)
                return
            
            # Upper bound via coloring
            n_colors, color_classes = greedy_coloring(nodes, adj)
            
            if len(current_selection) + n_colors <= best_size[0]:
                return
            
            # Branch on a vertex
            # Pick vertex with max degree in the subgraph
            v = max(nodes, key=lambda x: len(adj[x] & set(nodes)))
            
            # Option 1: Include v
            branch([u for u in nodes if u != v and u not in adj[v]], 
                   current_selection + [v])
            
            # Option 2: Exclude v
            branch([u for u in nodes if u != v], current_selection)
        
        # Initial call
        all_nodes = list(range(n))
        branch(all_nodes, [])
        
        return best_size[0], best_nodes[0]
    
    # Phase 1: Get initial covering via randomized greedy
    best_blocks = None
    best_count = float('inf')
    
    # Deterministic greedy first
    blocks = greedy_build(0, randomize_order=False)
    best_blocks = blocks
    best_count = len(blocks)
    
    # Randomized greedy restarts
    seed = 1
    while time.time() < deadline - 40:
        blocks = greedy_build(seed, randomize_order=True)
        if len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)
        seed += 1
    
    # Phase 2: Exact deletion pass
    if time.time() < deadline - 15:
        best_blocks, removed = exact_deletion_pass(best_blocks, deadline)
        best_count = len(best_blocks)
    
    # Phase 3: Try exact deletion from a few other starting points
    for start_seed in range(3):
        if time.time() > deadline - 10:
            break
        alt_blocks = greedy_build(100 + start_seed, randomize_order=True)
        if len(alt_blocks) < best_count + 10:  # only if competitive
            alt_blocks, removed = exact_deletion_pass(alt_blocks, deadline)
            if len(alt_blocks) < best_count:
                best_blocks = alt_blocks
                best_count = len(best_blocks)
    
    return best_blocks if best_blocks else greedy_build(0, randomize_order=False)
