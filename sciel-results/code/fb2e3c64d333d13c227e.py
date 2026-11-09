import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    N = 256

    def count_3aps(A):
        """Count distinct 3-term APs (positive difference, counted once) in A."""
        count = 0
        A_set = set(A)
        for a in A_set:
            for b in A_set:
                if b <= a:
                    continue
                d = b - a
                c = b + d
                if c in A_set and c > b:
                    count += 1
        return count

    def has_4ap(A):
        """Check if A contains a nonconstant 4-term AP."""
        A_set = set(A)
        for a in A_set:
            for b in A_set:
                if b <= a:
                    continue
                d = b - a
                c = b + d
                if c not in A_set or c <= b:
                    continue
                e = c + d
                if e in A_set and e > c:
                    return True
        return False

    def creates_4ap(x, A):
        """Check if adding x to A creates a 4-AP."""
        A_set = set(A)
        # x as 1st: x, x+d, x+2d, x+3d
        for d in range(1, (N - x) // 3 + 1):
            if (x + d) in A_set and (x + 2 * d) in A_set and (x + 3 * d) in A_set:
                return True
        # x as 2nd: x-d, x, x+d, x+2d
        for d in range(1, min(x - 1, (N - x) // 2) + 1):
            if (x - d) in A_set and (x + d) in A_set and (x + 2 * d) in A_set:
                return True
        # x as 3rd: x-2d, x-d, x, x+d
        for d in range(1, min(x // 2, N - x) + 1):
            if (x - 2 * d) in A_set and (x - d) in A_set and (x + d) in A_set:
                return True
        # x as 4th: x-3d, x-2d, x-d, x
        for d in range(1, (x - 1) // 3 + 1):
            if (x - 3 * d) in A_set and (x - 2 * d) in A_set and (x - d) in A_set:
                return True
        return False

    def gain_3ap(x, A):
        """Count how many 3-APs are completed by adding x to A."""
        A_set = set(A)
        gain = 0
        # x as 1st: x, x+d, x+2d
        for d in range(1, (N - x) // 2 + 1):
            if (x + d) in A_set and (x + 2 * d) in A_set:
                gain += 1
        # x as 2nd: x-d, x, x+d
        for d in range(1, min(x - 1, N - x) + 1):
            if (x - d) in A_set and (x + d) in A_set:
                gain += 1
        # x as 3rd: x-2d, x-d, x
        for d in range(1, x // 2 + 1):
            if (x - 2 * d) in A_set and (x - d) in A_set:
                gain += 1
        return gain

    def greedy_weighted(A, remaining, temperature=0.0):
        """
        Weighted greedy: at each step, pick the element with highest
        score = gain_3ap(x, A) - lambda * conflict_penalty(x, A).
        conflict_penalty counts how many 4-APs would be created if we
        ignored the constraint (used as a soft penalty).
        """
        A = set(A)
        remaining = set(remaining)
        lambda_val = 0.5  # penalty weight
        
        while remaining and time.time() < deadline - 10:
            best_x = None
            best_score = -1
            
            for x in remaining:
                if creates_4ap(x, A):
                    continue  # hard constraint
                g = gain_3ap(x, A)
                # Soft penalty: count potential conflicts
                penalty = 0
                # Rough estimate: how many elements in A are "close" in AP structure
                # Use a simpler heuristic: prefer elements that create more 3-APs
                score = g - lambda_val * (1.0 / (1.0 + g))  # slight preference for higher gain
                if temperature > 0:
                    score += random.gauss(0, temperature)
                if score > best_score:
                    best_score = score
                    best_x = x
            
            if best_x is None:
                break
            A.add(best_x)
            remaining.remove(best_x)
        
        return A

    def local_search_improve(A, max_iterations=50):
        """
        Local search: try swapping elements to improve 3-AP count
        while maintaining no 4-AP.
        """
        A = set(A)
        remaining = set(range(1, N + 1)) - A
        current_score = count_3aps(A)
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 15:
                break
            
            improved = False
            # Compute loss for each element in A
            A_list = sorted(A)
            losses = {}
            for a in A_list:
                loss = 0
                A_set = set(A)
                # a as 1st
                for d in range(1, (N - a) // 2 + 1):
                    if (a + d) in A_set and (a + 2 * d) in A_set:
                        loss += 1
                # a as 2nd
                for d in range(1, min(a - 1, N - a) + 1):
                    if (a - d) in A_set and (a + d) in A_set:
                        loss += 1
                # a as 3rd
                for d in range(1, a // 2 + 1):
                    if (a - 2 * d) in A_set and (a - d) in A_set:
                        loss += 1
                losses[a] = loss
            
            # Sort by loss (ascending) - try removing low-loss elements first
            sorted_A = sorted(A_list, key=lambda x: losses[x])
            
            # Try each candidate swap
            for a in sorted_A:
                if time.time() > deadline - 15:
                    break
                if improved:
                    break
                
                A.discard(a)
                
                for b in sorted(remaining, key=lambda x: gain_3ap(x, A), reverse=True):
                    if creates_4ap(b, A):
                        continue
                    gain_b = gain_3ap(b, A)
                    if gain_b > losses[a]:
                        # Accept swap
                        A.add(b)
                        remaining.discard(b)
                        remaining.add(a)
                        current_score += gain_b - losses[a]
                        improved = True
                        break
                
                if not improved:
                    A.add(a)  # restore
            
            if not improved:
                break
        
        return A, current_score

    # Strategy: Multiple randomized greedy runs with different temperatures
    # and local search refinement
    best_A = set()
    best_score = 0
    
    # Deterministic greedy first
    A = greedy_weighted(set(), set(range(1, N + 1)), temperature=0.0)
    if not has_4ap(A):
        score = count_3aps(A)
        if score > best_score:
            best_score = score
            best_A = set(A)
    
    # Randomized greedy with local search
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        temp = 0.1 + 0.1 * (i / num_restarts)
        A = greedy_weighted(set(), set(range(1, N + 1)), temperature=temp)
        
        if has_4ap(A):
            # Fix: remove elements that cause 4-APs
            A_list = sorted(A)
            for a in A_list:
                if has_4ap(A):
                    A.discard(a)
        
        if not A:
            continue
        
        # Local search refinement
        A, score = local_search_improve(A, max_iterations=30)
        
        if not has_4ap(A):
            if score > best_score:
                best_score = score
                best_A = set(A)
    
    # Final verification
    if has_4ap(best_A):
        # Emergency fix
        A_list = sorted(best_A)
        for a in A_list:
            if has_4ap(best_A):
                best_A.discard(a)
    
    return sorted(best_A)
