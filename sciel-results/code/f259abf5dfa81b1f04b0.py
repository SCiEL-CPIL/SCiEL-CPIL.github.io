import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin for safety

    N = 256
    
    # Squares up to 2*256 = 512
    squares = set()
    i = 2
    while i * i <= 2 * N:
        squares.add(i * i)
        i += 1
    
    # Identify valid nodes (no self-conflict: 2*v is not a square)
    valid_nodes = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_nodes.append(v)
            
    # Build conflict graph
    # conflicts[v] is a set of u such that u != v and u+v is a square
    conflicts = {v: set() for v in valid_nodes}
    for i, v in enumerate(valid_nodes):
        for u in valid_nodes[i+1:]:
            if v + u in squares:
                conflicts[v].add(u)
                conflicts[u].add(v)

    # Initial solution: Greedy MIS with randomization to get a good starting point
    def get_initial_solution():
        available = set(valid_nodes)
        S = set()
        while available:
            # Pick node with min degree in available
            min_deg = float('inf')
            candidates = []
            for v in available:
                deg = len(conflicts[v] & available)
                if deg < min_deg:
                    min_deg = deg
                    candidates = [v]
                elif deg == min_deg:
                    candidates.append(v)
            if candidates:
                v = random.choice(candidates)
                S.add(v)
                to_remove = {v}
                to_remove.update(conflicts[v] & available)
                available.difference_update(to_remove)
        return S

    best_set = get_initial_solution()
    best_size = len(best_set)
    
    # Simulated Annealing
    # State: set S of selected nodes
    # Moves: 
    # 1. Add a node v not in S if it doesn't conflict with any in S (Improvement)
    # 2. Remove a node v from S (Worsening, allows escape)
    # 3. Swap: Remove v from S, add u not in S (Neutral or Improvement if u has fewer conflicts? No, size stays same)
    #    Actually, simple add/remove is enough for cardinality maximization if we allow removals.
    #    But to increase size, we need to find a way to add. 
    #    If S is maximal, no single add is possible. We must remove some nodes to free up space for more.
    #    So the move set should be:
    #    - Add v if valid (size +1)
    #    - Remove v (size -1)
    #    This allows the system to "breathe" and find larger configurations.
    
    # To be more efficient, we can try "Swap" moves: Remove v, Add u. Size same. 
    # This helps rearrange to a better position.
    
    current_S = best_set.copy()
    current_size = len(current_S)
    
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.01
    
    # Precompute neighbors for faster checks? 
    # conflicts[v] is a set. 
    # Check if v can be added: conflicts[v] & current_S is empty.
    
    while time.time() < deadline:
        if T < min_T:
            # Restart with new temperature and maybe a new random start if stuck?
            # Or just break and return best.
            # Let's try a few restarts if time permits.
            T = 1.0
            # Optionally re-initialize current_S to best_set or a new random one
            # Re-initializing to best_set is safer to not lose progress.
            current_S = best_set.copy()
            current_size = len(best_set)
            
        # Perform a single step
        # Generate a random move
        # Probability of add vs remove vs swap?
        # If we can add, we should. 
        # Let's try:
        # 1. Find all valid additions. If any, pick one. (Always accept)
        # 2. If no valid additions, pick a random node in S to remove, or a random swap.
        
        # Optimization: Track which nodes are "blocked" by S?
        # A node v is blocked if conflicts[v] & current_S is not empty.
        
        # Try to add
        added = False
        # To avoid O(N^2) per step, we can just pick a random candidate from not in S.
        # But we need to check validity.
        
        # Let's pick a random move type
        r = random.random()
        
        if r < 0.5:
            # Try to add a random node not in S
            candidates_to_add = [v for v in valid_nodes if v not in current_S]
            if candidates_to_add:
                v = random.choice(candidates_to_add)
                # Check if v conflicts with any in current_S
                if not (conflicts[v] & current_S):
                    # Valid addition
                    current_S.add(v)
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_set = current_S.copy()
                    added = True
        elif r < 0.8:
            # Try to remove a random node from S
            if current_S:
                v = random.choice(list(current_S))
                current_S.discard(v)
                current_size -= 1
        else:
            # Try a swap: remove v from S, add u not in S
            # This keeps size constant but changes composition
            if current_S:
                v = random.choice(list(current_S))
                candidates_to_add = [u for u in valid_nodes if u not in current_S]
                if candidates_to_add:
                    u = random.choice(candidates_to_add)
                    # Check if u conflicts with S \ {v}
                    # If u conflicts with v, that's fine since v is being removed.
                    # But we must ensure u doesn't conflict with any other node in S.
                    # Also u != v (guaranteed since v in S, u not in S)
                    # Check: conflicts[u] & (current_S - {v}) is empty
                    if not (conflicts[u] & (current_S - {v})):
                        # Valid swap
                        current_S.discard(v)
                        current_S.add(u)
                        # Size unchanged
                        # Note: This move is always "neutral" in size, so accept with prob 1?
                        # Or use SA logic? Since size is same, it's a neutral move. 
                        # Accepting it always helps explore the landscape.
                        pass
        
        # If we didn't add, and we are in a "stuck" state (maximal), we rely on remove/swap to escape.
        # We don't strictly need SA acceptance for adds (always good) or removes (always bad but necessary for escape).
        # The "temperature" T isn't really used in the acceptance criterion above for add/remove.
        # Let's implement proper SA for the "Remove" move to control how often we shrink.
        # Actually, the above logic is a heuristic random walk. 
        # Let's refine:
        # If we attempt a remove, we should only do it if we are stuck or with some probability.
        # The current logic tries add 50%, remove 30%, swap 20%.
        # This seems reasonable.
        
        # Cooling
        T *= cooling_rate

    # Final polish: Greedy add on best_set
    improved = True
    while improved:
        improved = False
        for v in valid_nodes:
            if v not in best_set:
                if not (conflicts[v] & best_set):
                    best_set.add(v)
                    improved = True
                    break
                    
    return list(best_set)
