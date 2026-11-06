import time
import sys

def construct():
    n = 5040
    target = 4979
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin

    # Get divisors
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    divisors.sort(reverse=True)
    
    # Filter out divisors larger than target
    divisors = [d for d in divisors if d <= target]
    N = len(divisors)
    
    # Precompute prefix sums for max sum pruning
    # prefix_sums[i] = sum(divisors[0:i])
    prefix_sums = [0] * (N + 1)
    for i in range(N):
        prefix_sums[i+1] = prefix_sums[i] + divisors[i]
        
    # Precompute sums of smallest j divisors for min sum pruning
    # divisors is sorted descending, so smallest are at the end.
    # small_sums[j] = sum of j smallest divisors = sum(divisors[N-j:N])
    small_sums = [0] * (N + 1)
    for j in range(N + 1):
        small_sums[j] = prefix_sums[N] - prefix_sums[N-j]
        
    # Search for minimum k
    for k in range(1, N + 1):
        # Check if it's even possible
        # Max sum with k divisors is sum of largest k: prefix_sums[k]
        if prefix_sums[k] < target:
            continue
            
        # Min sum with k divisors is sum of smallest k: small_sums[k]
        if small_sums[k] > target:
            break
            
        # Iterative DFS
        # Stack items: (idx, remaining, current_count, path_list)
        # To avoid copying lists, we use a global path list and backtracking manually,
        # or we store the path in the stack. Storing path in stack is safer for iterative.
        # However, path length is small (k), so copying is cheap.
        
        stack = [(0, target, 0, [])]
        found = False
        result_path = []
        
        while stack:
            idx, remaining, current_count, path = stack.pop()
            
            if remaining == 0:
                found = True
                result_path = path
                break
                
            if current_count == k:
                continue
                
            if idx >= N:
                continue
                
            if remaining < 0:
                continue
                
            # Time check
            if time.time() > deadline:
                return [] # Timeout, fail fast or return best? Task says "invalid result... scores 0". So we must return a valid one or timeout.
                # Actually, if we timeout, we should probably return [] which is invalid? 
                # "an invalid result, an exception or a timeout scores 0".
                # So we need to ensure we finish in time.
                
            slots = k - current_count
            
            # Pruning 1: Not enough elements left
            if N - idx < slots:
                continue
                
            # Pruning 2: Max possible sum with 'slots' elements from idx
            # Largest 'slots' elements in divisors[idx:] are divisors[idx : idx+slots]
            max_possible_sum = prefix_sums[idx + slots] - prefix_sums[idx]
            if max_possible_sum < remaining:
                continue
                
            # Pruning 3: Min possible sum with 'slots' elements from idx
            # Smallest 'slots' elements in divisors[idx:] are the global smallest 'slots'
            # provided idx <= N - slots.
            min_possible_sum = small_sums[slots]
            if min_possible_sum > remaining:
                continue

            # Branching
            # We push "Exclude" first so that "Include" is popped first (LIFO)
            # This maintains the preference for including larger divisors first.
            
            # Exclude divisors[idx]
            if idx + 1 < N or remaining == 0: # Optimization: don't push if no elements left and remaining != 0 handled above? 
                 # Actually, just push. Pruning will handle it.
                 stack.append((idx + 1, remaining, current_count, path))
            
            # Include divisors[idx]
            new_remaining = remaining - divisors[idx]
            if new_remaining >= 0:
                stack.append((idx + 1, new_remaining, current_count + 1, path + [divisors[idx]]))
                
        if found:
            return result_path
            
    # Fallback (should not be reached if solution exists)
    return []
