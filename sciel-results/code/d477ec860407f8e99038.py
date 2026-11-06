import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s limit to be safe
    
    # 1. Determine valid candidates (diagonal check)
    # 2a must not be a perfect square
    squares = set(k*k for k in range(1, 23)) # max sum 512, sqrt(512) ~ 22.6
    
    candidates = []
    for i in range(1, 257):
        if 2 * i not in squares:
            candidates.append(i)
            
    N = len(candidates)
    
    # 2. Build conflict masks
    # mask[i] is a bitset where bit j is 1 if candidates[i] + candidates[j] is a square
    masks = [0] * N
    for i in range(N):
        ci = candidates[i]
        m = 0
        for j in range(N):
            if i == j:
                continue
            cj = candidates[j]
            if ci + cj in squares:
                m |= (1 << j)
        masks[i] = m
        
    # 3. Greedy initialization
    def greedy_solve():
        remaining = (1 << N) - 1
        selected = 0
        while remaining:
            # Pick node with min degree in remaining
            best_node = -1
            min_deg = float('inf')
            r = remaining
            while r:
                lsb = r & (-r)
                idx = lsb.bit_length() - 1
                deg = (masks[idx] & remaining).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_node = idx
                r ^= lsb
                
            if best_node == -1:
                break
                
            selected |= (1 << best_node)
            # Remove best_node and its neighbors
            remaining &= ~(masks[best_node] | (1 << best_node))
            
        return selected, selected.bit_count()

    best_set, best_size = greedy_solve()
    
    # 4. Local Search (Simulated Annealing style)
    # We will try to improve the current set by adding/removing nodes.
    
    def local_search(current_set, current_size):
        nonlocal best_set, best_size
        temperature = 100.0
        cooling_rate = 0.99
        min_temperature = 0.1
        
        current = current_set
        current_sz = current_size
        best_local = current
        best_local_sz = current_size
        
        while temperature > min_temperature:
            if time.time() > deadline:
                break
                
            # Randomly pick a node to toggle
            # To be efficient, we can pick from a subset of nodes
            # Let's pick a random node index
            idx = random.randint(0, N - 1)
            bit = 1 << idx
            
            if current & bit:
                # Try to remove it
                new_set = current & ~bit
                new_sz = current_sz - 1
                # Always accept removal if it allows adding more later? 
                # Actually, removing decreases size. We only accept if it leads to improvement or with prob.
                # But simple hill climbing on size:
                # If we remove, size decreases. We might want to swap.
                # Let's stick to:
                # 1. Try adding a node
                # 2. Try swapping one node for another
                # 3. Try removing a node (only if it allows adding 2+ others? Complex)
                
                # For simplicity, let's focus on ADD and SWAP.
                # If we are at a local optimum for additions, we need swaps.
                
                # Let's just try to ADD a node first.
                # If we can add, we do it.
                # If we can't, we try to SWAP.
                
                # Re-evaluate strategy:
                # 1. Check if any node can be added.
                # 2. If yes, add the one that maximizes future potential (min degree among addable).
                # 3. If no, try to swap: remove one node, see if we can add two others.
                
                # This is getting complex. Let's use a simpler SA move:
                # Move: Add a random valid node, or remove a random node, or swap.
                
                # Let's try: Pick a random node NOT in set.
                # If it's valid, add it.
                # If not, pick a random node IN set, remove it, then try to add the original node?
                
                # Let's implement a specific move:
                # 1. Identify all nodes that CAN be added to current_set.
                # 2. If any, add one (random or best).
                # 3. If none, try to swap:
                #    - Pick a node v in current_set.
                #    - Pick a node u not in current_set.
                #    - Check if (current_set - {v} + {u}) is valid.
                #    - If valid, accept.
                
                # This requires checking validity which is O(N) per check.
                # N ~ 150. 150 * 150 = 22500 ops per step.
                # 10000 steps = 225M ops. Might be too slow in Python.
                
                # Optimization:
                # Precompute for each node u, the set of nodes it conflicts with.
                # To check if u can be added to S:
                #   (masks[u] & S) == 0
                # To check if swap (remove v, add u) is valid:
                #   1. u must not conflict with any node in S \ {v}.
                #      i.e., (masks[u] & (S & ~bit_v)) == 0
                #   2. v must not conflict with u? No, we are removing v.
                #   3. But we also need to ensure u doesn't conflict with itself? Diagonal check already done.
                #   4. And we need to ensure the rest of S is independent (which it is).
                
                # So swap check is fast: bit operations.
                
                # Let's implement this.
                
                pass # Placeholder for logic below
                
            else:
                # Node not in set. Try to add it.
                if (masks[idx] & current) == 0:
                    # Can add
                    new_set = current | bit
                    new_sz = current_sz + 1
                    # Always accept improvement
                    current = new_set
                    current_sz = new_sz
                    if current_sz > best_local_sz:
                        best_local = current
                        best_local_sz = current_sz
                    # Also update global best
                    if current_sz > best_size:
                        best_size = current_sz
                        best_set = current
                else:
                    # Cannot add directly. Try to swap.
                    # Pick a random node v from current to remove.
                    # We need to iterate through bits of current.
                    r = current
                    if r:
                        # Pick random bit from current
                        # Efficient random bit selection from bitset:
                        # Generate random number, map to index?
                        # Or just iterate?
                        # For now, let's pick a random index and check if it's in current.
                        # Or better: collect indices? Too slow.
                        
                        # Let's try: Pick a random index v_idx.
                        # If v_idx is in current, try swap.
                        # If not, skip.
                        # This might be inefficient if current is sparse.
                        
                        # Alternative: Iterate through a few random nodes to find one in current.
                        # Or just pick the first one?
                        
                        # Let's try a few random swaps.
                        for _ in range(5): # Try 5 random swaps
                            v_idx = random.randint(0, N - 1)
                            v_bit = 1 << v_idx
                            if current & v_bit:
                                # Try to swap v with idx
                                # Check if idx can be added after removing v
                                # Condition: (masks[idx] & (current & ~v_bit)) == 0
                                if (masks[idx] & (current & ~v_bit)) == 0:
                                    # Valid swap
                                    new_set = (current & ~v_bit) | bit
                                    new_sz = current_sz # Size unchanged
                                    # Accept swap with some probability to escape local optima?
                                    # Since size is same, we accept with prob exp(-delta/T) where delta=0?
                                    # Or just accept to explore.
                                    # Let's accept with prob 0.5 to avoid cycling.
                                    if random.random() < 0.5:
                                        current = new_set
                                        # current_sz stays same
                                        break
                                
            temperature *= cooling_rate
        
        return best_local, best_local_sz

    # Run local search multiple times with random restarts if time permits
    # But first, let's run it on the greedy result.
    best_set, best_size = local_search(best_set, best_size)
    
    # 5. Branch and Bound (if time remains)
    if time.time() < deadline - 5.0:
        import sys
        sys.setrecursionlimit(10000)
        
        def branch_and_bound(remaining, current_size, current_set):
            nonlocal best_set, best_size
            
            if time.time() > deadline:
                return
                
            # Pruning
            if current_size + remaining.bit_count() <= best_size:
                return
                
            if remaining == 0:
                if current_size > best_size:
                    best_size = current_size
                    best_set = current_set
                return
                
            # Heuristic: Pick a node to branch on
            best_node = -1
            max_deg = -1
            r = remaining
            while r:
                lsb = r & (-r)
                idx = lsb.bit_length() - 1
                deg = (masks[idx] & remaining).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_node = idx
                r ^= lsb
                
            if best_node == -1:
                return
                
            bit = 1 << best_node
            
            # Branch 1: Include best_node
            new_remaining_1 = remaining & ~(masks[best_node] | bit)
            branch_and_bound(new_remaining_1, current_size + 1, current_set | bit)
            
            # Branch 2: Exclude best_node
            new_remaining_2 = remaining & ~bit
            branch_and_bound(new_remaining_2, current_size, current_set)
            
        initial_remaining = (1 << N) - 1
        branch_and_bound(initial_remaining, 0, 0)
    
    # Convert best_set bitset back to list of values
    result = []
    for i in range(N):
        if best_set & (1 << i):
            result.append(candidates[i])
            
    return result

# EVOLVE-BLOCK-START
# The construct function is defined above. 
# To match the interface exactly, I should wrap it or just define it here.
# The prompt asks to keep function names. The current program has `construct`.
# I will place the code inside the block or define it globally.
# The seed program defines `construct` globally. I will do the same.
# EVOLVE-BLOCK-END
