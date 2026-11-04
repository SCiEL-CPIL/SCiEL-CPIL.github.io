import numpy as np
import itertools
import time
import random
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    D = 11
    
    # 1. Generate all supports (combinations of 4 indices)
    supports = list(itertools.combinations(range(D), 4))
    N_S = len(supports)  # 330
    
    # Convert supports to bitmasks for fast intersection
    masks = []
    for s in supports:
        m = 0
        for idx in s:
            m |= (1 << idx)
        masks.append(m)
        
    # 2. Build conflict graph
    # Conflict if intersection size is exactly 3
    adj = [set() for _ in range(N_S)]
    for i in range(N_S):
        for j in range(i + 1, N_S):
            inter = (masks[i] & masks[j]).bit_count()
            if inter == 3:
                adj[i].add(j)
                adj[j].add(i)
    
    # Precompute adjacency as boolean matrix
    adj_bool = np.zeros((N_S, N_S), dtype=bool)
    for i in range(N_S):
        for j in adj[i]:
            adj_bool[i][j] = True
            adj_bool[j][i] = True
    
    # Degree of each node
    degrees = np.array([len(adj[i]) for i in range(N_S)])
    
    best_size = 0
    best_set = set()
    rng = np.random.RandomState(12345)
    
    def greedy_min_degree():
        """Greedy min-degree heuristic for independent set."""
        remaining = set(range(N_S))
        s = set()
        
        while remaining:
            min_deg = float('inf')
            min_node = None
            for v in remaining:
                deg = sum(1 for nb in adj[v] if nb in remaining)
                if deg < min_deg:
                    min_deg = deg
                    min_node = v
            s.add(min_node)
            to_remove = {min_node}
            for nb in adj[min_node]:
                if nb in remaining:
                    to_remove.add(nb)
            remaining -= to_remove
        
        return s
    
    def greedy_random(seed_offset=0):
        """Randomized greedy independent set with min-degree tie-breaking."""
        local_rng = np.random.RandomState(42 + seed_offset)
        remaining = set(range(N_S))
        s = set()
        
        while remaining:
            candidates = list(remaining)
            degs = [(sum(1 for nb in adj[v] if nb in remaining), v) for v in candidates]
            degs.sort()
            
            min_deg = degs[0][0]
            min_nodes = [v for d, v in degs if d == min_deg]
            
            min_node = min_nodes[local_rng.randint(len(min_nodes))]
            
            s.add(min_node)
            to_remove = {min_node}
            for nb in adj[min_node]:
                if nb in remaining:
                    to_remove.add(nb)
            remaining -= to_remove
        
        return s
    
    def local_search(init_set, max_iter=500, time_budget=None):
        """Local search with add, 1-swap, and 2-swap moves."""
        current_set = set(init_set)
        best_local = set(current_set)
        best_local_size = len(current_set)
        
        in_set = np.zeros(N_S, dtype=bool)
        for v in current_set:
            in_set[v] = True
        
        def compute_blocked(set_obj, in_set_arr):
            if not set_obj:
                return np.zeros(N_S, dtype=bool)
            return np.any(adj_bool[in_set_arr], axis=0)
        
        blocked = compute_blocked(current_set, in_set)
        
        for it in range(max_iter):
            if time.time() > deadline - 8.0:
                break
            if time_budget is not None and time.time() > time_budget:
                break
            
            candidates_mask = ~(in_set | blocked)
            cand_indices = np.where(candidates_mask)[0]
            
            if len(cand_indices) > 0:
                # Try adding a random candidate
                v_add = int(cand_indices[rng.randint(len(cand_indices))])
                current_set.add(v_add)
                in_set[v_add] = True
                blocked |= adj_bool[v_add]
                
                if len(current_set) > best_local_size:
                    best_local_size = len(current_set)
                    best_local = set(current_set)
                continue
            else:
                # Stuck: try 1-swap or 2-swap
                if not current_set:
                    break
                
                # Try 1-swap: remove v_remove, add v_add
                v_remove = list(current_set)[rng.randint(len(current_set))]
                
                # Save state
                saved_set = set(current_set)
                saved_in_set = in_set.copy()
                saved_blocked = blocked.copy()
                
                current_set.remove(v_remove)
                in_set[v_remove] = False
                blocked = compute_blocked(current_set, in_set)
                
                # Find new candidates
                candidates_mask = ~(in_set | blocked)
                cand_indices = np.where(candidates_mask)[0]
                
                improved = False
                if len(cand_indices) > 0:
                    # Try to add one candidate
                    v_add = int(cand_indices[rng.randint(len(cand_indices))])
                    current_set.add(v_add)
                    in_set[v_add] = True
                    blocked |= adj_bool[v_add]
                    
                    if len(current_set) > best_local_size:
                        best_local_size = len(current_set)
                        best_local = set(current_set)
                        improved = True
                
                if not improved:
                    # Revert
                    current_set = saved_set
                    in_set = saved_in_set
                    blocked = saved_blocked
                    
                    # Try 2-swap: remove two, add two
                    if len(current_set) >= 2:
                        items = list(current_set)
                        idx1 = rng.randint(len(items))
                        idx2 = rng.randint(len(items) - 1)
                        if idx2 >= idx1:
                            idx2 += 1
                        v_rem1, v_rem2 = items[idx1], items[idx2]
                        
                        saved_set2 = set(current_set)
                        saved_in_set2 = in_set.copy()
                        saved_blocked2 = blocked.copy()
                        
                        current_set.remove(v_rem1)
                        current_set.remove(v_rem2)
                        in_set[v_rem1] = False
                        in_set[v_rem2] = False
                        blocked = compute_blocked(current_set, in_set)
                        
                        candidates_mask = ~(in_set | blocked)
                        cand_indices = np.where(candidates_mask)[0]
                        
                        if len(cand_indices) >= 2:
                            # Pick two candidates greedily
                            chosen = []
                            temp_blocked = blocked.copy()
                            for _ in range(2):
                                avail = np.where(~(in_set | temp_blocked))[0]
                                if len(avail) == 0:
                                    break
                                v_add = int(avail[rng.randint(len(avail))])
                                chosen.append(v_add)
                                in_set[v_add] = True
                                temp_blocked |= adj_bool[v_add]
                            
                            if len(chosen) == 2:
                                current_set.update(chosen)
                                in_set[chosen[0]] = True
                                in_set[chosen[1]] = True
                                blocked = temp_blocked
                                
                                if len(current_set) > best_local_size:
                                    best_local_size = len(current_set)
                                    best_local = set(current_set)
                                    improved = True
                            
                            if not improved:
                                current_set = saved_set2
                                in_set = saved_in_set2
                                blocked = saved_blocked2
                        else:
                            current_set = saved_set2
                            in_set = saved_in_set2
                            blocked = saved_blocked2
        
        return best_local
    
    # Branch and bound for maximum independent set
    def branch_and_bound(time_budget):
        """Branch and bound with coloring upper bound."""
        best_bb_size = 0
        best_bb_set = set()
        
        # Order nodes by degree (ascending) for better pruning
        order = list(range(N_S))
        order.sort(key=lambda x: degrees[x])
        
        # Reverse so we process high-degree nodes first in the recursion
        # Actually, for independent set, we want to branch on nodes that
        # eliminate many options. Let's use the natural order.
        
        # Use a simpler approach: recursive branch and bound
        # State: (candidate_set, current_set, upper_bound)
        
        # Precompute: for each node, its neighbors as a bitmask (for 330 nodes, use Python int)
        node_masks = []
        for i in range(N_S):
            m = 0
            for j in adj[i]:
                m |= (1 << j)
            node_masks.append(m)
        
        # Upper bound: greedy coloring of the candidate subgraph
        def color_bound(candidates):
            """Return upper bound on independent set size in candidates."""
            if not candidates:
                return 0
            # Greedy coloring: assign colors to nodes
            # For upper bound, we want to find a coloring of the graph induced by candidates
            # The number of colors is an upper bound on the independence number
            
            # Simple greedy: process nodes, assign smallest available color
            # But we want a fast upper bound. Use a simple heuristic:
            # Count how many nodes can be packed if we greedily pick non-adjacent ones
            # This gives a lower bound on clique cover, hence upper bound on independence.
            
            # Actually, for upper bound on independence number:
            # Use the fact that in any independent set, at most one node from each clique.
            # Find a greedy clique cover: repeatedly find a clique, remove it.
            
            # Simpler: use the degree-based bound
            # In a graph with n nodes and max degree d, alpha >= n/(d+1)
            # But we need an upper bound, not lower bound.
            
            # For upper bound: use the fact that if we can 3-color the graph,
            # then alpha <= n/3 * max_color_class_size... no that's not right.
            
            # Let's just use a simple greedy: the independence number is at most
            # the number of nodes in the largest independent set we can greedily find
            # No, that's a lower bound.
            
            # Upper bound via greedy coloring:
            # Color the graph greedily. The independence number <= number of colors? No.
            # Actually, alpha <= n - (minimum vertex cover) = n - (maximum matching) for bipartite...
            
            # For general graphs, a simple upper bound:
            # alpha(G) <= n - tau(G) where tau is minimum vertex cover
            # And tau(G) >= maximum matching size
            # But computing maximum matching is expensive.
            
            # Simplest useful bound: alpha(G) <= n / (1 + average_degree/n) ... not great.
            
            # Let's use: alpha(G) <= number of nodes that are not in any edge? No.
            
            # Practical approach: use a greedy independent set as a lower bound,
            # and use n as a trivial upper bound, with better pruning via recursion.
            
            # For now, return len(candidates) as a weak upper bound.
            # The recursion will prune based on: len(current) + len(candidates) <= best
            return len(candidates)
        
        # Better: use a simple upper bound based on the structure
        # In our graph, each node conflicts with nodes sharing 3 indices.
        # A simple upper bound: for each coordinate, at most some number of selected supports
        # can use that coordinate. But this is complex.
        
        # Let's just do a depth-first branch and bound with the trivial bound
        # and rely on good branching order.
        
        def dfs(candidates_mask, current_size, current_set):
            nonlocal best_bb_size, best_bb_set
            
            if time.time() > time_budget:
                return
            
            # Upper bound: current_size + len(candidates)
            n_cand = candidates_mask.bit_count()
            if current_size + n_cand <= best_bb_size:
                return
            
            if n_cand == 0:
                if current_size > best_bb_size:
                    best_bb_size = current_size
                    best_bb_set = set(current_set)
                return
            
            # Pick a node to branch on
            # Choose the node with highest degree in the candidate subgraph
            # to maximize pruning
            best_node = -1
            best_deg = -1
            for v in range(N_S):
                if candidates_mask & (1 << v):
                    deg = (candidates_mask & node_masks[v]).bit_count()
                    if deg > best_deg:
                        best_deg = deg
                        best_node = v
            
            if best_node == -1:
                return
            
            # Branch 1: include best_node
            new_cand = candidates_mask & ~(1 << best_node) & ~node_masks[best_node]
            current_set.append(best_node)
            dfs(new_cand, current_size + 1, current_set)
            current_set.pop()
            
            # Branch 2: exclude best_node
            new_cand = candidates_mask & ~(1 << best_node)
            dfs(new_cand, current_size, current_set)
        
        # Initial candidates: all nodes
        init_mask = (1 << N_S) - 1
        dfs(init_mask, 0, [])
        
        return best_bb_set
    
    # Run greedy + local search first to get a good initial solution
    init_set = greedy_min_degree()
    best_set = local_search(init_set, max_iter=300)
    best_size = len(best_set)
    
    # Try multiple random restarts
    for r in range(200):
        if time.time() > deadline - 20.0:
            break
        
        init_set = greedy_random(seed_offset=r * 7)
        final_set = local_search(init_set, max_iter=200)
        size = len(final_set)
        
        if size > best_size:
            best_size = size
            best_set = set(final_set)
    
    # Try branch and bound with time limit
    if time.time() < deadline - 15.0:
        bb_set = branch_and_bound(deadline - 5.0)
        if len(bb_set) > best_size:
            best_size = len(bb_set)
            best_set = set(bb_set)
    
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
        v = np.zeros(D, dtype=np.int8)
        v[0] = 1
        return np.array([v])
        
    final_vectors = np.array(vectors, dtype=np.int8)
    
    return final_vectors
