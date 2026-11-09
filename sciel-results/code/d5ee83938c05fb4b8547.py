import time
import random
import math

N = 256
TIME_LIMIT = 45.0  # Safety margin for 90s limit

def construct():
    start_time = time.time()
    
    # Reciprocals
    recip = [0.0] * (N + 1)
    for i in range(1, N + 1):
        recip[i] = 1.0 / i

    # Helper to check if adding x to bitset creates a 3-AP
    def can_add(bitset, x):
        # x is not in bitset
        # Check for any v in bitset such that {x, v, u} forms a 3-AP for some u in bitset
        for v in range(1, N + 1):
            if bitset & (1 << (v - 1)):
                # Case 1: x is middle. u = 2x - v
                u = 2 * x - v
                if 1 <= u <= N:
                    if bitset & (1 << (u - 1)):
                        return False
                
                # Case 2: v is middle. u = 2v - x
                u = 2 * v - x
                if 1 <= u <= N:
                    if bitset & (1 << (u - 1)):
                        return False
                
                # Case 3: u is middle. v + x = 2u => u = (v+x)/2
                if (v + x) % 2 == 0:
                    u = (v + x) // 2
                    if 1 <= u <= N and u != v:
                        if bitset & (1 << (u - 1)):
                            return False
        return True

    def get_elements(bitset):
        return [i for i in range(1, N + 1) if bitset & (1 << (i - 1))]

    def get_non_elements(bitset):
        return [i for i in range(1, N + 1) if not (bitset & (1 << (i - 1)))]

    # 1. Greedy Initialization
    current_bitset = 0
    current_score = 0.0
    for x in range(1, N + 1):
        if can_add(current_bitset, x):
            current_bitset |= (1 << (x - 1))
            current_score += recip[x]

    best_bitset = current_bitset
    best_score = current_score

    # 2. Quick Simulated Annealing to get a strong incumbent
    temp = 0.05
    cooling_rate = 0.999
    min_temp = 1e-5
    sa_time_limit = 5.0  # Run SA for 5 seconds
    
    while time.time() - start_time < sa_time_limit:
        if temp < min_temp:
            temp = 0.01
            current_bitset = best_bitset
            current_score = best_score

        move_type = random.random()
        
        if move_type < 0.3:
            # Add move
            non_elements = get_non_elements(current_bitset)
            if non_elements:
                x_add = random.choice(non_elements)
                if can_add(current_bitset, x_add):
                    delta = recip[x_add]
                    current_bitset |= (1 << (x_add - 1))
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_bitset = current_bitset

        elif move_type < 0.5:
            # Remove move
            elements = get_elements(current_bitset)
            if elements:
                x_rem = random.choice(elements)
                delta = -recip[x_rem]
                if random.random() < math.exp(delta / temp):
                    current_bitset &= ~(1 << (x_rem - 1))
                    current_score += delta

        else:
            # Swap move
            elements = get_elements(current_bitset)
            non_elements = get_non_elements(current_bitset)
            if elements and non_elements:
                x_rem = random.choice(elements)
                x_add = random.choice(non_elements)
                
                trial_bitset = current_bitset & ~(1 << (x_rem - 1))
                if can_add(trial_bitset, x_add):
                    trial_bitset |= (1 << (x_add - 1))
                    delta = recip[x_add] - recip[x_rem]
                    
                    if delta > 0:
                        current_bitset = trial_bitset
                        current_score += delta
                        if current_score > best_score:
                            best_score = current_score
                            best_bitset = current_bitset
                    else:
                        if random.random() < math.exp(delta / temp):
                            current_bitset = trial_bitset
                            current_score += delta

        temp *= cooling_rate

    # 3. Branch and Bound
    # We will explore the space of subsets.
    # To make it efficient, we order candidates by reciprocal descending.
    # We use a DFS approach.
    
    # Pre-sort elements by reciprocal descending
    sorted_elements = list(range(1, N + 1))
    sorted_elements.sort(key=lambda x: recip[x], reverse=True)
    
    # We need to track which elements are "decided" (included or excluded)
    # And which are "remaining".
    
    # State: (current_bitset, current_score, index_in_sorted)
    # To prune, we need an upper bound on the score we can add from remaining elements.
    
    def get_compatible_remaining(bitset, remaining_elements):
        """
        Returns a list of elements from remaining_elements that can be added to bitset.
        """
        compatible = []
        for x in remaining_elements:
            if not (bitset & (1 << (x - 1))):
                if can_add(bitset, x):
                    compatible.append(x)
        return compatible

    # We will use a stack for DFS
    # Stack items: (bitset, score, remaining_list)
    # To avoid copying large lists, we can pass the index in the sorted list,
    # but the "remaining" set changes dynamically based on conflicts?
    # No, in this formulation, "remaining" is just the set of elements not yet decided.
    # The bound is calculated on the "compatible" subset of remaining.
    
    # However, if we exclude an element, it stays excluded.
    # If we include it, it's in bitset.
    
    # Let's use a recursive function with a time check.
    
    max_score = best_score
    max_bitset = best_bitset
    
    # To speed up B&B, we can use a "greedy bound" which is the sum of reciprocals
    # of all compatible remaining elements.
    
    # We process elements in order of sorted_elements.
    # At each step, we decide for sorted_elements[i].
    # If we include it, we update bitset.
    # If we exclude it, we just move to i+1.
    # The remaining elements are sorted_elements[i+1:].
    
    def branch_and_bound(bitset, score, i):
        nonlocal max_score, max_bitset
        
        if time.time() - start_time > TIME_LIMIT:
            return
            
        if i >= N:
            if score > max_score:
                max_score = score
                max_bitset = bitset
            return
        
        # Calculate bound
        # Upper bound = score + sum of reciprocals of compatible elements in sorted_elements[i:]
        # Note: We only consider elements from i onwards.
        # We must check if they are compatible with current bitset.
        
        bound = score
        # Optimization: If the sum of ALL remaining reciprocals (even incompatible) + score <= max_score, prune.
        # Precompute suffix sums of reciprocals for sorted_elements.
        
        # Let's just calculate the compatible sum.
        # To make it faster, we can break early if bound > max_score is unlikely?
        # No, we need the full sum to prune.
        
        # We can approximate the bound by just summing all remaining reciprocals.
        # If score + suffix_sum[i] <= max_score: return
        # This is a safe bound.
        
        # Then, for tighter pruning, we can check compatibility.
        # But checking compatibility for all remaining is O(N^2) per node, which is too slow.
        
        # Let's use the loose bound first:
        if score + suffix_sums[i] <= max_score:
            return
            
        x = sorted_elements[i]
        
        # Option 1: Include x
        if not (bitset & (1 << (x - 1))):
            if can_add(bitset, x):
                new_bitset = bitset | (1 << (x - 1))
                new_score = score + recip[x]
                branch_and_bound(new_bitset, new_score, i + 1)
                
        # Option 2: Exclude x
        branch_and_bound(bitset, score, i + 1)
        
    # Precompute suffix sums of reciprocals for sorted_elements
    suffix_sums = [0.0] * (N + 1)
    for i in range(N - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + recip[sorted_elements[i]]
        
    # Start B&B from the greedy solution?
    # No, start from empty set to explore the whole space?
    # That might be too slow.
    # Better: Start from the best SA solution and try to improve?
    # B&B is a global search. It should start from the root (empty set, i=0).
    # But with 256 elements, the tree is huge.
    # The loose bound `score + suffix_sums[i]` might not prune enough initially.
    
    # Let's try to start B&B.
    # To make it faster, we can use the SA result to set a high `max_score`.
    
    branch_and_bound(0, 0.0, 0)
    
    # Final check
    if max_score > best_score:
        best_bitset = max_bitset
        best_score = max_score

    # Final polish: try to add any number that can be added
    improved = True
    while improved:
        improved = False
        non_elements = get_non_elements(best_bitset)
        for x in non_elements:
            if can_add(best_bitset, x):
                best_bitset |= (1 << (x - 1))
                improved = True

    result = get_elements(best_bitset)
    return result

if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(sum(1.0/x for x in res))
