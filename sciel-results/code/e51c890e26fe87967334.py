import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 115  # Safety margin
    
    # Precomputation
    N = 511
    factors = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        if not factors[i]: # i is prime
            for j in range(i, N + 1, i):
                factors[j].add(i)
    
    # We want to maximize sum 1/(512-a).
    # Sort candidates a from 511 down to 1.
    candidates = list(range(N, 0, -1))
    weights = [0.0] * (N + 1)
    for a in candidates:
        weights[a] = 1.0 / (512 - a)
        
    # Precompute suffix sums of weights for a quick loose upper bound
    # suffix_sum[i] = sum(weights[candidates[i]] ... weights[candidates[N-1]])
    suffix_sum = [0.0] * (N + 1)
    for i in range(N - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + weights[candidates[i]]
        
    best_score = 0.0
    best_solution = []
    
    # Branch and Bound
    # We process candidates in order of decreasing weight (which is already 511..1)
    # index: current index in the sorted candidates list (0 to N-1)
    # selected_factors: frozenset of prime factors of chosen numbers
    # current_score: sum of weights of chosen numbers
    
    def bound(idx, selected_factors, current_score):
        # Quick check: if selected_factors is empty, the max possible is suffix_sum[idx]
        if not selected_factors:
            return current_score + suffix_sum[idx]
            
        # Otherwise, sum weights of compatible items
        b = current_score
        for j in range(idx, N):
            a = candidates[j]
            if not (factors[a] & selected_factors):
                b += weights[a]
        return b

    def solve(idx, selected_factors, current_score, chosen):
        nonlocal best_score, best_solution
        
        if time.time() - start_time > DEADLINE:
            return
            
        # Pruning
        b = bound(idx, selected_factors, current_score)
        if b <= best_score:
            return
            
        if idx == N:
            if current_score > best_score:
                best_score = current_score
                best_solution = list(chosen)
            return
            
        a = candidates[idx]
        fac_a = factors[a]
        
        # Branch 1: Include a (if possible)
        # We try include first to get a good solution early
        if not (fac_a & selected_factors):
            new_selected_factors = frozenset(selected_factors | fac_a)
            new_current_score = current_score + weights[a]
            new_chosen = chosen + [a]
            solve(idx + 1, new_selected_factors, new_current_score, new_chosen)
            
        # Branch 2: Exclude a
        solve(idx + 1, selected_factors, current_score, chosen)

    solve(0, frozenset(), 0.0, [])
    
    # Local search refinement: try to improve the solution by swapping elements
    if best_solution and time.time() - start_time < DEADLINE - 5:
        def is_coprime_set(lst):
            """Check if all elements in lst are pairwise coprime."""
            for i in range(len(lst)):
                for j in range(i + 1, len(lst)):
                    if math.gcd(lst[i], lst[j]) != 1:
                        return False
            return True
        
        def compute_score(lst):
            """Compute the score for a list of integers."""
            return sum(1.0 / (512 - a) for a in lst)
        
        current_solution = list(best_solution)
        current_score = best_score
        improved = True
        
        while improved and time.time() - start_time < DEADLINE - 2:
            improved = False
            
            # Try single swaps: remove one element, add another
            for i in range(len(current_solution)):
                if time.time() - start_time > DEADLINE - 2:
                    break
                removed = current_solution.pop(i)
                removed_factors = factors[removed]
                
                # Try adding each candidate not already in the solution
                for a in candidates:
                    if a in current_solution:
                        continue
                    if factors[a] & removed_factors:
                        continue
                    
                    # Check if a is coprime to all remaining elements
                    compatible = True
                    for b in current_solution:
                        if factors[a] & factors[b]:
                            compatible = False
                            break
                    if not compatible:
                        continue
                    
                    new_score = current_score - weights[removed] + weights[a]
                    if new_score > current_score + 1e-15:
                        current_solution.append(a)
                        current_score = new_score
                        improved = True
                        break
                
                if not improved:
                    current_solution.insert(i, removed)
            
            # Try double swaps: remove two elements, add two others
            if not improved and len(current_solution) >= 2 and time.time() - start_time < DEADLINE - 2:
                for i in range(len(current_solution)):
                    for j in range(i + 1, len(current_solution)):
                        if time.time() - start_time > DEADLINE - 2:
                            break
                        removed1 = current_solution.pop(i)
                        removed2 = current_solution.pop(j - 1)  # Adjust index after pop
                        removed_factors = factors[removed1] | factors[removed2]
                        
                        # Try adding pairs of candidates
                        for a in candidates:
                            if a in current_solution or a == removed1 or a == removed2:
                                continue
                            if factors[a] & removed_factors:
                                continue
                            
                            for b in candidates:
                                if b == a or b in current_solution or b == removed1 or b == removed2:
                                    continue
                                if factors[b] & removed_factors:
                                    continue
                                if factors[a] & factors[b]:
                                    continue
                                
                                # Check if a and b are coprime to all remaining elements
                                compatible = True
                                for c in current_solution:
                                    if (factors[a] & factors[c]) or (factors[b] & factors[c]):
                                        compatible = False
                                        break
                                if not compatible:
                                    continue
                                
                                new_score = current_score - weights[removed1] - weights[removed2] + weights[a] + weights[b]
                                if new_score > current_score + 1e-15:
                                    current_solution.extend([a, b])
                                    current_score = new_score
                                    improved = True
                                    break
                            if improved:
                                break
                        
                        if not improved:
                            current_solution.insert(i, removed1)
                            current_solution.insert(j, removed2)
                    
                    if improved:
                        break
            
            # Update best if we improved
            if improved:
                if current_score > best_score:
                    best_score = current_score
                    best_solution = list(current_solution)
    
    return best_solution
