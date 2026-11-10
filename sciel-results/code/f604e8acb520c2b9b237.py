import time
from math import lcm
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin for 90s limit

    N = 256
    values = list(range(2, N + 1))
    n = len(values)

    # Precompute conflict graph: conflict[i][j] = True if lcm(values[i], values[j]) <= 256
    # We want to select a subset with NO conflicts (independent set)
    # Maximize sum of 1/values[i]
    
    # Use bitsets for fast compatibility checks
    # conflict_mask[i] is a bitset where bit j is 1 if i and j conflict
    conflict_mask = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if lcm(values[i], values[j]) <= 256:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    # Sort vertices by descending weight (1/value), so we pick small values first
    # Index 0 in sorted order is the best candidate (smallest value, highest weight)
    order = sorted(range(n), key=lambda i: -1.0 / values[i])
    # order[k] is the original index of the k-th best candidate
    weights_sorted = [1.0 / values[order[k]] for k in range(n)]
    
    # Rebuild conflict masks in sorted order
    # new_conflict_mask[k] has bit l set if order[k] and order[l] conflict
    new_conflict_mask = [0] * n
    for k in range(n):
        orig_k = order[k]
        mask = conflict_mask[orig_k]
        for l in range(n):
            if mask & (1 << order[l]):
                new_conflict_mask[k] |= (1 << l)
    
    # Precompute suffix sums of weights for upper bound
    # suffix_sum[k] = sum of weights_sorted[k:]
    suffix_sum = [0.0] * (n + 1)
    for k in range(n - 1, -1, -1):
        suffix_sum[k] = suffix_sum[k + 1] + weights_sorted[k]
    
    best_score = 0.0
    best_set = []
    
    # Branch and bound
    # State: current_index (next to consider), current_sum, current_mask (bitmask of selected vertices in sorted indices)
    # We use recursion with pruning
    
    # For efficiency, we'll use an iterative approach with a stack
    # But recursion is cleaner for B&B. Let's use recursion with a time check.
    
    # To make it faster, we can use a greedy initial solution to set a good lower bound
    # Greedy: pick vertices in order if compatible with all previously picked
    greedy_set = []
    greedy_mask = 0
    greedy_sum = 0.0
    for k in range(n):
        if not (new_conflict_mask[k] & greedy_mask):
            greedy_set.append(k)
            greedy_mask |= (1 << k)
            greedy_sum += weights_sorted[k]
    
    best_score = greedy_sum
    best_set = list(greedy_set)
    
    # Now B&B to improve
    # We'll try to add vertices that improve the score
    # Actually, B&B typically explores the space of independent sets.
    # Given the structure, a simple recursive B&B might be too slow if not pruned well.
    # Let's try a different approach: Local search from the greedy solution, but with a different neighborhood.
    # Or, since the task asks for a DIFFERENT strategy, let's stick to B&B but optimize it.
    
    # Optimized B&B:
    # We process vertices in order of decreasing weight.
    # At each step k, we decide to include or exclude vertex k.
    # If we include it, we must exclude all its neighbors.
    # Upper bound: current_sum + sum of weights of all remaining vertices that are compatible with current selection.
    # To compute this bound fast, we maintain a bitmask of "available" vertices (compatible with all selected).
    
    def solve_bnb():
        nonlocal best_score, best_set, deadline
        
        # available_mask: bitmask of vertices that can still be added (compatible with all in current_set)
        # current_set: list of indices in sorted order
        # current_sum: sum of weights in current_set
        # k: the next index to consider (from 0 to n)
        # We use a stack for iterative DFS to avoid recursion depth limits and allow early exit
        
        # State: (k, available_mask, current_sum, current_set_tuple)
        # To save memory, we can just track the set as a bitmask too, but we need the actual set for the result.
        # Let's store the set as a list in the stack.
        
        stack = [(0, (1 << n) - 1, 0.0, [])]
        
        while stack:
            if time.time() > deadline:
                break
                
            k, avail_mask, cur_sum, cur_set = stack.pop()
            
            # Prune: if current sum + upper bound <= best_score, skip
            # Upper bound: cur_sum + sum of weights of vertices in avail_mask with index >= k
            # We can precompute this? No, avail_mask changes.
            # Approximation: cur_sum + suffix_sum[k]
            if cur_sum + suffix_sum[k] <= best_score:
                continue
            
            if k == n:
                if cur_sum > best_score:
                    best_score = cur_sum
                    best_set = list(cur_set)
                continue
            
            # Option 1: Exclude vertex k
            # The available mask remains the same for the next vertex
            # But we only consider vertices >= k+1
            # So we push (k+1, avail_mask, cur_sum, cur_set)
            stack.append((k + 1, avail_mask, cur_sum, cur_set))
            
            # Option 2: Include vertex k
            # Only if k is in avail_mask
            if avail_mask & (1 << k):
                # New available mask: remove k and all neighbors of k
                new_avail = avail_mask & ~(new_conflict_mask[k] | (1 << k))
                new_sum = cur_sum + weights_sorted[k]
                new_set = cur_set + [k]
                
                # Prune check for this branch
                if new_sum + suffix_sum[k + 1] > best_score:
                    stack.append((k + 1, new_avail, new_sum, new_set))
    
    # The simple B&B above might be too slow because the state space is large.
    # Let's try a hybrid: Use the B&B logic but with a tighter bound or a different order.
    # Actually, for n=256, even with pruning, 2^256 is impossible.
    # The key is that the graph is sparse? Or dense?
    # Let's check the density.
    # If the graph is dense, the independent set is small.
    # If sparse, it's large.
    
    # Let's try a different structural approach: **Greedy with Randomized Restarts and 2-Swap Local Search**
    # But the prompt asks for a DIFFERENT strategy from the current one (which is Greedy + Local Search).
    # So B&B is a good candidate, but it needs to be efficient.
    
    # Let's refine the B&B bound.
    # Instead of suffix_sum[k], we can compute the exact sum of weights in avail_mask for indices >= k.
    # This is expensive.
    
    # Alternative: **Simulated Annealing**
    # This is a different optimization family.
    # Let's implement Simulated Annealing.
    
    def simulated_annealing():
        nonlocal best_score, best_set, deadline
        
        # Initial solution: Greedy
        current_set = set(greedy_set)
        current_score = greedy_sum
        
        # Temperature schedule
        T_start = 1.0
        T_end = 1e-6
        alpha = 0.999  # Cooling rate
        T = T_start
        
        # Precompute neighbors for fast access?
        # We need to check if a vertex is compatible with all in current_set.
        
        def is_compatible(v, S):
            # v is sorted index
            # Check if v conflicts with any in S
            mask_v = new_conflict_mask[v]
            # Create mask for S
            # This is slow. Let's maintain a bitmask for current_set.
            pass
        
        # Let's maintain current_mask
        current_mask = 0
        for v in current_set:
            current_mask |= (1 << v)
            
        def get_score(S):
            return sum(weights_sorted[v] for v in S)
            
        # Neighbor generation:
        # 1. Remove a random vertex
        # 2. Add a random compatible vertex
        # 3. Swap: remove one, add one
        
        import random
        
        while T > T_end and time.time() < deadline:
            if not current_set:
                break
                
            # Choose a random move
            move_type = random.random()
            
            if move_type < 0.3:
                # Remove a vertex
                if len(current_set) == 0:
                    continue
                v = random.choice(list(current_set))
                new_set = current_set - {v}
                new_mask = current_mask & ~(1 << v)
                new_score = current_score - weights_sorted[v]
                delta = new_score - current_score
            elif move_type < 0.6:
                # Add a vertex
                # Find all compatible vertices not in current_set
                # This is slow if done naively.
                # Let's precompute the list of compatible vertices? No, it changes.
                # We can iterate through all vertices and check compatibility.
                # n=256, so 256 checks per move is fine.
                candidates = []
                for v in range(n):
                    if not (current_mask & (1 << v)):
                        if not (new_conflict_mask[v] & current_mask):
                            candidates.append(v)
                if not candidates:
                    continue
                v = random.choice(candidates)
                new_set = current_set | {v}
                new_mask = current_mask | (1 << v)
                new_score = current_score + weights_sorted[v]
                delta = new_score - current_score
            else:
                # Swap
                if len(current_set) == 0:
                    continue
                v_remove = random.choice(list(current_set))
                temp_set = current_set - {v_remove}
                temp_mask = current_mask & ~(1 << v_remove)
                
                candidates = []
                for v in range(n):
                    if not (temp_mask & (1 << v)):
                        if not (new_conflict_mask[v] & temp_mask):
                            candidates.append(v)
                if not candidates:
                    continue
                v_add = random.choice(candidates)
                new_set = temp_set | {v_add}
                new_mask = temp_mask | (1 << v_add)
                new_score = current_score - weights_sorted[v_remove] + weights_sorted[v_add]
                delta = new_score - current_score
            
            # Acceptance
            if delta > 0:
                current_set = new_set
                current_mask = new_mask
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = list(current_set)
            else:
                # Metropolis criterion
                r = random.random()
                if r < pow(2.718281828, delta / T):
                    current_set = new_set
                    current_mask = new_mask
                    current_score = new_score
            
            T *= alpha
            
        return best_score, best_set
    
    # Let's run Simulated Annealing
    best_score, best_set = simulated_annealing()
    
    # Convert back to original values
    result = [values[order[v]] for v in best_set]
    
    return result
# EVOLVE-BLOCK-END
