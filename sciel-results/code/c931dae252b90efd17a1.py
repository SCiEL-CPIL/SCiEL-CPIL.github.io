import math
import time

MAX_MOD = 128
DEADLINE_MARGIN = 5.0

def construct():
    t0 = time.time()
    deadline = t0 + 60.0 - DEADLINE_MARGIN
    
    # 1. Map (m, r) to ID
    # m ranges from 2 to 128
    # Total nodes ~ 8000
    id_map = {}
    node_info = [] # (m, r)
    current_id = 0
    for m in range(2, MAX_MOD + 1):
        for r in range(m):
            id_map[(m, r)] = current_id
            node_info.append((m, r))
            current_id += 1
            
    n_nodes = current_id
    
    # 2. Build adjacency bitsets
    # adj[i] is a bitmask of nodes incompatible with i
    adj = [0] * n_nodes
    
    # Incompatibility conditions:
    # 1. Same modulus
    # 2. gcd(m1, m2) divides (r1 - r2)
    
    # Optimization: Precompute gcds for moduli pairs?
    # Or just calculate on the fly?
    # n_nodes is ~8000. Pairs are ~32 million.
    # Calculating gcd 32M times might be slow in Python.
    # Let's group by modulus.
    
    # For each pair of moduli (m1, m2), we can determine which residues are incompatible.
    # If m1 == m2: all residues are incompatible (same modulus).
    # If m1 != m2:
    #   g = gcd(m1, m2)
    #   r1 and r2 are incompatible if (r1 - r2) % g == 0.
    #   This means r1 % g == r2 % g.
    
    # We can build the graph more efficiently.
    # Initialize adj with 0.
    
    # For same modulus:
    for m in range(2, MAX_MOD + 1):
        ids = [id_map[(m, r)] for r in range(m)]
        mask = 0
        for idx in ids:
            mask |= (1 << idx)
        # All nodes in this group are incompatible with each other (except themselves, but we handle self-loop logic or just ignore self in MIS logic? 
        # Usually MIS graphs don't have self-loops, but here "incompatible with self" is not an issue for set construction, 
        # we just don't pick the same node twice. 
        # However, if we consider "same modulus" as an edge, we ensure we pick at most one from each modulus.
        # So, for each node in group, add all other nodes in group to its adj.
        for idx in ids:
            adj[idx] |= mask
            adj[idx] &= ~(1 << idx) # Remove self-loop if we added it? 
            # Actually, if mask includes idx, then adj[idx] has bit idx set.
            # In MIS, we don't want to pick v and v. 
            # The standard MIS algorithm assumes no self-loops or handles them.
            # Let's just set adj[idx] to have all OTHER nodes in the same modulus.
            # So adj[idx] |= (mask ^ (1 << idx))
            
    # Let's restart the adj construction cleanly.
    adj = [0] * n_nodes
    
    # Group nodes by modulus
    mod_groups = {}
    for (m, r), idx in id_map.items():
        if m not in mod_groups:
            mod_groups[m] = []
        mod_groups[m].append(idx)
        
    # 1. Same modulus incompatibility
    for m, ids in mod_groups.items():
        mask = 0
        for idx in ids:
            mask |= (1 << idx)
        for idx in ids:
            # Incompatible with all other nodes in this group
            adj[idx] |= (mask ^ (1 << idx))
            
    # 2. Different modulus incompatibility
    # Iterate over pairs of moduli
    moduli = list(mod_groups.keys())
    for i in range(len(moduli)):
        for j in range(i + 1, len(moduli)):
            m1 = moduli[i]
            m2 = moduli[j]
            g = math.gcd(m1, m2)
            ids1 = mod_groups[m1]
            ids2 = mod_groups[m2]
            
            # For each r1 in m1, it is incompatible with r2 in m2 if (r1 - r2) % g == 0
            # i.e., r1 % g == r2 % g
            
            # We can group residues in m1 by their value mod g
            # And residues in m2 by their value mod g
            
            # Map: remainder -> list of node IDs
            rem_map_1 = {}
            for idx in ids1:
                m, r = node_info[idx]
                rem = r % g
                if rem not in rem_map_1:
                    rem_map_1[rem] = []
                rem_map_1[rem].append(idx)
                
            rem_map_2 = {}
            for idx in ids2:
                m, r = node_info[idx]
                rem = r % g
                if rem not in rem_map_2:
                    rem_map_2[rem] = []
                rem_map_2[rem].append(idx)
                
            # For each remainder value k in 0..g-1
            for k in range(g):
                if k in rem_map_1 and k in rem_map_2:
                    # All nodes in rem_map_1[k] are incompatible with all nodes in rem_map_2[k]
                    list1 = rem_map_1[k]
                    list2 = rem_map_2[k]
                    
                    # Create masks
                    mask1 = 0
                    for idx in list1:
                        mask1 |= (1 << idx)
                    mask2 = 0
                    for idx in list2:
                        mask2 |= (1 << idx)
                    
                    # Add edges
                    for idx in list1:
                        adj[idx] |= mask2
                    for idx in list2:
                        adj[idx] |= mask1

    # 3. Solve MIS
    # We want the largest set of nodes such that no two are adjacent in `adj`.
    
    # Tomita's Algorithm (Branch and Bound with Coloring)
    # We'll use a recursive function.
    # To avoid recursion depth issues, we might need to increase limit or use iterative.
    # n_nodes ~ 8000. Depth can be up to 128 (max moduli).
    # Recursion depth 128 is safe.
    
    import sys
    sys.setrecursionlimit(10000)
    
    best_size = 0
    best_set_mask = 0
    
    # Initial ordering: Sort nodes by degree (popcount of adj)
    # High degree nodes first? 
    # Actually, for MIS, we often want to process nodes that are "central" first to prune?
    # Or low degree?
    # Tomita uses a specific ordering. Let's try sorting by degree descending.
    
    # Calculate degrees
    degrees = [bin(adj[i]).count('1') for i in range(n_nodes)]
    
    # Create an ordered list of node indices
    # We will process nodes in this order.
    # In the coloring step, we also use this order or a dynamic one.
    # Let's stick to the standard Tomita:
    # The candidate set P is a subset of nodes.
    # We iterate through P in a specific order.
    
    # Let's implement a helper to get set bits in an int mask
    def get_bits(mask):
        while mask:
            lsb = mask & -mask
            yield lsb.bit_length() - 1
            mask ^= lsb

    def color_bound(P):
        """
        Returns (max_color, color_masks) where color_masks is a list of bitmasks.
        The number of colors is the bound.
        """
        colors = []
        unvisited = P
        while unvisited:
            color_mask = 0
            new_unvisited = unvisited
            # Iterate through nodes in unvisited
            # To make it deterministic and fast, we can just iterate through bits
            # We want to pick a node v from unvisited such that it has no neighbors in color_mask
            # This is a greedy maximal independent set on the subgraph induced by unvisited, 
            # but we are assigning colors to independent sets.
            # Wait, the coloring for MIS bound is:
            # Assign color 1 to a maximal independent set in P.
            # Assign color 2 to a maximal independent set in P \ IS1.
            # ...
            # The number of colors is an upper bound on the size of MIS in P.
            
            # So, find a Maximal Independent Set (MIS) of the subgraph induced by unvisited.
            # Note: This MIS is in the *original* graph G.
            # So we pick v from unvisited, add to color_mask, remove v and N(v) from unvisited?
            # No, if we remove N(v), we are finding an Independent Set.
            # Yes.
            
            temp_unvisited = unvisited
            while temp_unvisited:
                # Pick a node. Which one?
                # To be fast, pick the first one? Or one with lowest degree in temp_unvisited?
                # Let's just pick the lowest index for simplicity/speed.
                # v = lowest set bit
                v = (temp_unvisited & -temp_unvisited).bit_length() - 1
                
                # Add v to current color
                color_mask |= (1 << v)
                
                # Remove v and its neighbors from temp_unvisited
                # Neighbors in the subgraph are adj[v] & temp_unvisited
                # We want to keep nodes NOT adjacent to v.
                # So remove (1<<v) and (adj[v] & temp_unvisited)
                temp_unvisited &= ~(1 << v)
                temp_unvisited &= ~(adj[v] & temp_unvisited)
                
            # The nodes remaining in unvisited that were NOT added to color_mask
            # are the ones we didn't pick.
            # Actually, the loop above picked a set of nodes that form an Independent Set.
            # Let's call this IS.
            # The next color will be applied to unvisited \ IS.
            # How to get IS mask?
            # IS is `color_mask`.
            # So new_unvisited = unvisited & ~color_mask
            
            new_unvisited = unvisited & ~color_mask
            colors.append(color_mask)
            unvisited = new_unvisited
            
        return len(colors), colors

    def expand(P, k):
        nonlocal best_size, best_set_mask
        
        if P == 0:
            if k > best_size:
                best_size = k
                # best_set_mask = current_set (not tracked, only size needed for score)
            return
            
        # Bound
        max_color, _ = color_bound(P)
        if k + max_color <= best_size:
            return
            
        # Order nodes in P
        # We want to process nodes in an order that helps pruning.
        # Tomita suggests ordering by the color they were assigned in the coloring step?
        # Or just degree?
        # Let's try processing nodes in P in decreasing order of their index in the color layers?
        # Simpler: Just iterate through bits of P.
        # To improve, we can sort nodes in P by degree in the subgraph induced by P.
        # But that's expensive.
        
        # Let's just iterate through nodes in P.
        # We can optimize by picking the node with the highest degree in P first?
        # Or the node that "blocks" the most other nodes?
        
        # Let's try a simple heuristic: Process nodes in P in order of decreasing global degree.
        # Precomputed `degrees`.
        # Get list of nodes in P
        nodes_in_P = list(get_bits(P))
        # Sort by degree descending
        nodes_in_P.sort(key=lambda x: degrees[x], reverse=True)
        
        for v in nodes_in_P:
            # Check if v is still in P?
            # The loop structure of MIS usually modifies P as it goes.
            # Standard algorithm:
            # for v in P (in some order):
            #   expand(P \ (N(v) U {v}), k+1)
            #   P = P \ {v}
            
            # But if we sort P, we need to be careful.
            # Let's stick to the bitset iteration which is natural.
            # However, the order matters for performance.
            # Let's try iterating through nodes_in_P.
            
            # If v is no longer in P (removed by previous steps? No, P is fixed for this level)
            # Wait, the standard algorithm removes v from P *after* the recursive call.
            # So P is the set of candidates for the *current* level.
            # We pick v, recurse on P \ (N(v) U {v}), then remove v from P for the next iteration of the loop.
            
            # So we need to maintain P as a mutable variable in the loop.
            
            pass # Placeholder

    # Let's rewrite the expand function properly with mutable P
    def expand(P, k):
        nonlocal best_size
        
        if P == 0:
            if k > best_size:
                best_size = k
            return
            
        max_color, _ = color_bound(P)
        if k + max_color <= best_size:
            return
            
        # We iterate through nodes in P.
        # To respect the order, we can extract nodes, sort them, and iterate.
        # But we must update P by removing v after each step.
        
        # Get nodes in P
        # To make it faster, we can just iterate through bits of P directly.
        # But we want a specific order.
        # Let's try: Iterate through nodes in P sorted by degree.
        
        # This requires creating a list every time, which might be slow.
        # Alternative: Use the order from the coloring?
        # The coloring returns `colors` (list of masks).
        # Nodes in `colors[-1]` (last color) are "most constrained" or something?
        # Actually, Tomita's algorithm processes nodes in the order of the coloring.
        
        # Let's try a different, simpler approach for the search:
        # Randomized Greedy with Local Search?
        # Given the time limit, a robust MIS solver is better.
        
        # Let's use the bitset iteration but pick the node with the highest degree in P first.
        # This is a "Most Constrained Variable" heuristic.
        
        # Find node v in P with max degree in P.
        # deg_in_P(v) = popcount(adj[v] & P)
        
        # This is O(|P|) per step.
        
        # Let's try this.
        
        # To avoid O(|P|) search for max degree node, we can just pick the first one?
        # Or precompute a global order and pick the first one in that order that is in P.
        
        # Let's use the global degree order.
        # `global_order` is a list of node indices sorted by degree descending.
        
        # We need to find the first v in `global_order` such that v is in P.
        
        # This is efficient if we maintain a pointer? No, P changes.
        
        # Let's just iterate through `global_order` and check if bit is set in P.
        
        for v in global_order:
            if not (P & (1 << v)):
                continue
                
            # Recurse
            new_P = P & ~(adj[v] | (1 << v))
            expand(new_P, k + 1)
            
            # Remove v from P
            P &= ~(1 << v)
            
            # Check time
            if time.time() > deadline:
                return

    global_order = list(range(n_nodes))
    global_order.sort(key=lambda x: degrees[x], reverse=True)
    
    # Initial P is all nodes
    initial_P = (1 << n_nodes) - 1
    expand(initial_P, 0)
    
    # If time ran out or we didn't find anything, we need a fallback.
    # But MIS should find at least 1.
    
    # We need to return the actual pairs, not just the size.
    # The `expand` function above only tracks `best_size`.
    # We need to track the set of nodes.
    
    # Let's modify `expand` to track the best set mask.
    
    best_size = 0
    best_set_mask = 0
    
    def expand2(P, k, current_mask):
        nonlocal best_size, best_set_mask
        
        if time.time() > deadline:
            return
            
        if P == 0:
            if k > best_size:
                best_size = k
                best_set_mask = current_mask
            return
            
        max_color, _ = color_bound(P)
        if k + max_color <= best_size:
            return
            
        for v in global_order:
            if not (P & (1 << v)):
                continue
                
            new_P = P & ~(adj[v] | (1 << v))
            expand2(new_P, k + 1, current_mask | (1 << v))
            
            P &= ~(1 << v)
            
            if time.time() > deadline:
                return

    expand2(initial_P, 0, 0)
    
    # Extract pairs from best_set_mask
    result = []
    for i in range(n_nodes):
        if best_set_mask & (1 << i):
            m, r = node_info[i]
            result.append([r, m])
            
    if not result:
        # Fallback
        result = [[0, 2]]
        
    return result

# Run construct
# print(construct())
