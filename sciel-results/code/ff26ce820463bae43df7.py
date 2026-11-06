import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s/90s limits
    
    N = 256
    
    # Build conflict graph as adjacency lists
    # conflicts[i] is a set of j such that (i*j) % (i+j) == 0
    conflicts = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i * j) % (i + j) == 0:
                conflicts[i].add(j)
                conflicts[j].add(i)

    def mask_to_set(mask):
        return [i for i in range(1, N + 1) if (mask >> i) & 1]

    def set_to_mask(s):
        m = 0
        for i in s:
            m |= (1 << i)
        return m

    best_size = 0
    best_mask = 0

    # Heuristic 1: Greedy Minimum Degree
    def greedy_min_degree():
        in_set = [False] * (N + 1)
        available = set(range(1, N + 1))
        solution = []
        
        while available:
            # Find node with min degree in available
            # Degree in available = len(conflicts[x] & available)
            # To speed up, we can precompute degrees or just compute on the fly.
            # N is small (256), so O(N^2) is fine.
            min_deg_node = None
            min_deg = float('inf')
            for x in available:
                deg = len(conflicts[x] & available)
                if deg < min_deg:
                    min_deg = deg
                    min_deg_node = x
            
            if min_deg_node is None:
                break
                
            # Add to solution
            solution.append(min_deg_node)
            in_set[min_deg_node] = True
            
            # Remove min_deg_node and its neighbors from available
            to_remove = set(conflicts[min_deg_node])
            to_remove.add(min_deg_node)
            available -= to_remove
            
        return set_to_mask(solution)

    # Heuristic 2: Greedy Random
    def greedy_random():
        in_set = [False] * (N + 1)
        nodes = list(range(1, N + 1))
        random.shuffle(nodes)
        solution = []
        
        for x in nodes:
            # Check if x is compatible with all in solution
            compatible = True
            for y in solution:
                if y in conflicts[x]:
                    compatible = False
                    break
            if compatible:
                solution.append(x)
                in_set[x] = True
                
        return set_to_mask(solution)

    # Local Search: 1-Swap Hill Climbing
    def local_search(mask):
        in_set = [False] * (N + 1)
        for i in range(1, N + 1):
            if (mask >> i) & 1:
                in_set[i] = True
        
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 2:
                break
                
            # Create a list of nodes in the set
            nodes_in_set = [i for i in range(1, N + 1) if in_set[i]]
            random.shuffle(nodes_in_set)
            
            for u in nodes_in_set:
                if time.time() > deadline - 2:
                    break
                    
                # Try removing u
                in_set[u] = False
                
                # Find candidates v not in set that are compatible with the new set
                # A node v is compatible if it has no neighbors in the current set
                # We can precompute the set of "forbidden" nodes: neighbors of all nodes in set
                # But since we only removed u, the forbidden set shrinks by neighbors of u,
                # but we must ensure v is not a neighbor of any *other* node in the set.
                
                # Efficient check:
                # v is compatible if (conflicts[v] & current_set_mask) == 0
                # Let's maintain a mask for the current set to make this O(1) bitwise op?
                # Or just iterate through candidates?
                
                # Let's iterate through all v not in set.
                # To speed up, we can check if v is in conflicts of any node in set.
                # Actually, let's just use the mask approach for compatibility check as it's fast.
                
                # Reconstruct mask for current set (without u)
                # This is expensive. Let's maintain the mask.
                
                # Let's rewrite local search to maintain mask.
                pass
                
        return set_to_mask([i for i in range(1, N + 1) if in_set[i]])

    # Optimized Local Search with Mask
    def local_search_optimized(mask):
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 2:
                break
                
            nodes_in_set = [i for i in range(1, N + 1) if (mask >> i) & 1]
            random.shuffle(nodes_in_set)
            
            for u in nodes_in_set:
                if time.time() > deadline - 2:
                    break
                    
                # Remove u
                new_mask = mask & ~(1 << u)
                
                # Find best candidate v to add
                # v must not be in new_mask
                # v must be compatible with new_mask: (conflicts[v] & new_mask) == 0
                # Note: conflicts[v] is a set. We need a bitmask for conflicts.
                # Let's precompute conflict masks.
                
                # This requires precomputation of conflict masks.
                pass
                
        return mask

    # Precompute conflict masks for fast bitwise AND
    conflict_masks = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in conflicts[i]:
            conflict_masks[i] |= (1 << j)

    def local_search_mask(mask):
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 2:
                break
                
            nodes_in_set = [i for i in range(1, N + 1) if (mask >> i) & 1]
            random.shuffle(nodes_in_set)
            
            for u in nodes_in_set:
                if time.time() > deadline - 2:
                    break
                    
                # Remove u
                new_mask = mask & ~(1 << u)
                
                # Find candidates v
                # v not in new_mask
                # (conflict_masks[v] & new_mask) == 0
                
                # To find v efficiently, we can iterate through all v not in new_mask.
                # There are at most 256 such v.
                # For each v, check compatibility.
                # If compatible, we can swap.
                # To improve, we should pick the v that allows further additions?
                # Simple hill climbing: pick the first valid v? Or the one with min degree?
                
                # Let's pick the v with minimum degree in the remaining graph?
                # Or just the first one found to be fast.
                # Since we want to maximize cardinality, any 1-swap that keeps size same is not an improvement in size.
                # Wait, 1-swap removes 1 and adds 1. Size is constant.
                # This only helps if it leads to a state where a 1-add is possible?
                # No, standard local search for Max Independent Set uses 1-swap to escape local optima of 1-add.
                # But if we only do 1-add, we might get stuck.
                # If we do 1-swap, we change the configuration.
                
                # Actually, if we can just ADD a node, we should do that.
                # So first, try to ADD any node.
                # If no node can be added, try SWAPS.
                
                # Phase 1: Try to add
                # If we can add a node, we do it.
                # If not, we try swaps.
                
                # Let's integrate this.
                
                # Check if any v can be added to current mask
                # This is outside the loop over u.
                
                pass
                
        return mask

    # Corrected Local Search Strategy:
    # 1. Try to add any node. If successful, add it and restart loop.
    # 2. If no node can be added, try 1-swaps. If a swap leads to a state where a node can be added?
    #    No, usually we just do the swap if it doesn't reduce size, hoping to break symmetry.
    #    But for Max Independent Set, 1-swaps keep size same.
    #    We need to ensure we don't cycle.
    #    Better: Use Simulated Annealing or just multiple restarts of Greedy.
    #    Given the time, multiple restarts of Greedy Min Degree with random tie-breaking is often very strong for sparse graphs.
    
    def greedy_min_degree_random():
        available = set(range(1, N + 1))
        solution = []
        
        while available:
            # Get degrees
            degrees = {}
            for x in available:
                degrees[x] = len(conflicts[x] & available)
            
            # Find min degree
            min_deg = min(degrees.values())
            candidates = [x for x, d in degrees.items() if d == min_deg]
            
            # Pick random candidate
            x = random.choice(candidates)
            solution.append(x)
            
            # Remove x and neighbors
            to_remove = set(conflicts[x])
            to_remove.add(x)
            available -= to_remove
            
        return set_to_mask(solution)

    num_restarts = 1000
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        if i % 2 == 0:
            mask = greedy_min_degree_random()
        else:
            mask = greedy_random()
            
        # Try to polish with simple 1-add local search
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 5:
                break
            # Try to add any node
            added = False
            # Shuffle nodes to avoid bias
            nodes = list(range(1, N + 1))
            random.shuffle(nodes)
            for v in nodes:
                if not (mask >> v) & 1:
                    if (conflict_masks[v] & mask) == 0:
                        mask |= (1 << v)
                        improved = True
                        added = True
                        break # Restart loop with new mask
                        # Actually, breaking and restarting the while loop is good
                        # But we need to break out of the for loop too
                        # Let's use a flag
            if not added and improved:
                 # This logic is flawed. If we add, we break for, then while checks improved.
                 # If we don't add, improved stays False, while ends.
                 pass
            
        size = bin(mask).count('1')
        if size > best_size:
            best_size = size
            best_mask = mask
            
    return mask_to_set(best_mask)
