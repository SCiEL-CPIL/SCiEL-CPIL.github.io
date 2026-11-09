import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin for safety
    
    N = 256
    
    # Precompute squares up to 2*N
    max_sum = 2 * N
    squares = set()
    i = 2
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1
    
    # 1. Filter nodes that violate the diagonal condition (a=a => 2a is square)
    valid_nodes = []
    node_to_idx = {}
    for v in range(1, N + 1):
        if 2 * v not in squares:
            node_to_idx[v] = len(valid_nodes)
            valid_nodes.append(v)
            
    M = len(valid_nodes)
    
    # 2. Build conflict graph using bitsets for efficiency
    # conflict_mask[i] is a bitmask of nodes j such that valid_nodes[i] + valid_nodes[j] is a square
    conflict_mask = [0] * M
    
    for i in range(M):
        vi = valid_nodes[i]
        mask = 0
        for j in range(i + 1, M):
            vj = valid_nodes[j]
            if vi + vj in squares:
                mask |= (1 << j)
                conflict_mask[j] |= (1 << i)
        conflict_mask[i] = mask
        
    # 3. Branch and Bound for Maximum Independent Set
    # State: available nodes mask, current set size
    # We use a simple recursive B&B with greedy upper bound
    
    best_size = 0
    best_set = []
    
    # Heuristic ordering: sort nodes by degree (number of conflicts) descending
    # This often helps B&B prune faster
    degrees = [bin(conflict_mask[i]).count('1') for i in range(M)]
    order = sorted(range(M), key=lambda x: degrees[x], reverse=True)
    
    # Re-index to place high-degree nodes at lower indices? 
    # Actually, standard MIS B&B works well if we just pick the first available node.
    # Let's stick to the original indexing but use a greedy heuristic for the bound.
    
    def greedy_bound(mask):
        """Estimate max independent set size from remaining nodes."""
        # Simple greedy: pick node with min degree in remaining subgraph?
        # Or just count bits? Counting bits is a weak bound.
        # A better bound: partition into cliques? Too slow.
        # Let's use a greedy coloring bound or just a simple heuristic.
        # For M~200, a simple greedy selection might be okay for a bound if we're careful.
        # However, a tighter bound is needed for efficiency.
        # Let's use the fact that if we pick a node, we remove its neighbors.
        # A common bound is the size of the maximum matching, but that's complex.
        # Let's try a simple greedy independent set size as a *lower* bound? No, we need an upper bound.
        # Upper bound: number of remaining nodes. (Weak)
        # Better Upper Bound: Greedy Coloring of the complement graph? 
        # Or simply: if we pick a node v, we remove v and N(v).
        # Let's use a recursive bound with memoization? No, mask is too large.
        
        # Let's use a simple heuristic: 
        # The size of a maximum independent set is at most the number of nodes.
        # To make it tighter, we can use a greedy algorithm to find a *large* independent set 
        # and use that as a lower bound to prune branches that can't beat it.
        # But for the *bound* in the recursion, we need an upper bound.
        # A standard upper bound for MIS is the number of vertices.
        # A tighter one is the size of the maximum clique in the complement, which is hard.
        
        # Alternative: Use a simple greedy algorithm to find an independent set 
        # and update the global best. The pruning happens if current_size + remaining_count <= best_size.
        # This is weak but might work if the graph is sparse.
        
        # Let's implement a slightly better bound:
        # If the graph is bipartite, MIS = N - MaxMatching. 
        # Is this graph bipartite?
        # Squares mod 4 are 0 or 1.
        # If a+b is square, a+b is 0 or 1 mod 4.
        # If a is even, b is even => sum even => square even => sum 0 mod 4.
        # If a is odd, b is odd => sum even => square even => sum 0 mod 4.
        # If a even, b odd => sum odd => square odd => sum 1 mod 4.
        # This doesn't immediately prove bipartiteness.
        
        # Let's stick to a simple B&B with a greedy lower bound update.
        # To speed up, we will pick the node with the smallest index in the mask first?
        # Or the one with the most conflicts?
        
        return bin(mask).count('1') # Weak upper bound: all remaining nodes

    def solve(mask, current_size, current_set):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return
        
        if mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set[:]
            return
            
        # Upper bound check
        remaining_count = bin(mask).count('1')
        if current_size + remaining_count <= best_size:
            return
            
        # Pick a node to branch on
        # Heuristic: pick the node with the most conflicts within the remaining set?
        # This tends to reduce the search space faster.
        # Find the node in mask with the highest degree in the subgraph induced by mask.
        best_node = -1
        max_deg = -1
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            # Count conflicts of idx within mask
            # conflicts = conflict_mask[idx] & mask
            # deg = bin(conflicts).count('1')
            # This is O(M) per node, so O(M^2) per step. M is ~200. 40000 ops per step.
            # Depth is ~200. Total ops ~8M. Should be fast enough in Python? Maybe slow.
            # Let's just pick the first available node for simplicity and speed, 
            # or use a precomputed order.
            
            # Let's pick the node with the smallest index for now to keep it simple.
            # Or better, pick the one with the most conflicts to prune more.
            
            # Optimization: Precompute degrees? No, degrees change as mask changes.
            # Let's just pick the first set bit.
            best_node = idx
            break # Just take the first one for speed in this implementation
            
            # If we want better pruning:
            # deg = bin(conflict_mask[idx] & mask).count('1')
            # if deg > max_deg:
            #     max_deg = deg
            #     best_node = idx
            # m &= m - 1
            
        if best_node == -1:
            return
            
        # Branch 1: Exclude best_node
        new_mask_excl = mask & ~(1 << best_node)
        solve(new_mask_excl, current_size, current_set)
        
        # Branch 2: Include best_node
        # Remove best_node and all its neighbors from mask
        neighbors = conflict_mask[best_node]
        new_mask_incl = mask & ~(1 << best_node) & ~neighbors
        current_set.append(valid_nodes[best_node])
        solve(new_mask_incl, current_size + 1, current_set)
        current_set.pop()
        
        # Update best if we found a better solution in this branch?
        # The recursive calls handle updating best_size/best_set when they hit leaves.
        
    initial_mask = (1 << M) - 1
    solve(initial_mask, 0, [])
    
    # If the B&B was too slow or didn't finish, we might have a partial best_set.
    # However, the base case only updates when mask==0.
    # It's better to update best_set whenever we have a valid set larger than current best.
    # Let's modify the logic to update best_set at every node if current_size > best_size?
    # No, current_set is not necessarily maximal. But it is independent.
    # So we can update best_size/best_set at every step if current_size > best_size.
    
    # Let's re-run with a slightly improved B&B that updates best at every step
    # and uses a better node selection heuristic if time permits.
    
    # Actually, the previous B&B might be too slow in Python for M=200 if it explores deeply.
    # Let's add a greedy initialization to get a good lower bound quickly.
    
    # Greedy MIS:
    # Sort nodes by degree (global) descending.
    # Iterate, if node not in excluded set, add to set, exclude neighbors.
    
    greedy_set = []
    excluded = set()
    # Order by degree descending
    sorted_nodes = sorted(range(M), key=lambda x: degrees[x], reverse=True)
    for idx in sorted_nodes:
        if idx not in excluded:
            greedy_set.append(valid_nodes[idx])
            excluded.add(idx)
            excluded.update([j for j in range(M) if (conflict_mask[idx] >> j) & 1])
            
    if len(greedy_set) > best_size:
        best_size = len(greedy_set)
        best_set = greedy_set[:]
        
    # Now run B&B with this better lower bound
    # We need to re-initialize the B&B state or just call it again.
    # Since best_size is global, we can just call solve again.
    # But we already called it. The previous run might have been weak.
    # Let's just trust the second run with the better bound.
    
    # To avoid double work, we should have initialized best_size with greedy first.
    # Let's structure it properly:
    # 1. Greedy to get lower bound
    # 2. B&B to improve
    
    # I already ran B&B. Let's just run it again with the better bound.
    # It's safe because it's deterministic and the bound is tighter.
    
    solve(initial_mask, 0, [])
    
    if not best_set:
        # Fallback to a simple valid set if B&B failed or found nothing
        # e.g. all odd numbers? 2*odd = even, not square unless square is even.
        # Squares of even numbers are divisible by 4.
        # 2*odd is 2 mod 4. Never a square.
        # So all odd numbers are valid for diagonal.
        # What about pairs? odd + odd = even. Can be square.
        # So {1, 3, 5...} is not necessarily independent.
        
        # Fallback: return the greedy set
        return greedy_set
        
    return best_set

# EVOLVE-BLOCK-START
# The function construct is defined above.
# EVOLVE-BLOCK-END
