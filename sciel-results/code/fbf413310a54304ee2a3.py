import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    N = 1000
    best_size = 0
    best_set = []
    
    # Helper to check if x can be added to S with current sum_counts
    # Returns True if valid, False otherwise
    def can_add(x, S, sum_counts):
        # Check 2*x
        if sum_counts.get(2 * x, 0) >= 2:
            return False
        # Check x + y for y in S
        for y in S:
            s = x + y
            if sum_counts.get(s, 0) >= 2:
                return False
        return True

    # Helper to add x to S and update sum_counts
    def add(x, S, sum_counts):
        S.append(x)
        sum_counts[2 * x] = sum_counts.get(2 * x, 0) + 1
        for y in S[:-1]: # S[:-1] is S before appending x
            s = x + y
            sum_counts[s] = sum_counts.get(s, 0) + 1

    # Helper to remove x from S and update sum_counts
    def remove(x, S, sum_counts):
        S.remove(x)
        sum_counts[2 * x] -= 1
        if sum_counts[2 * x] == 0:
            del sum_counts[2 * x]
        for y in S:
            s = x + y
            sum_counts[s] -= 1
            if sum_counts[s] == 0:
                del sum_counts[s]

    # Local search: try swapping elements to improve size
    def local_search(S, sum_counts, max_iter=50):
        S = S[:]
        sum_counts = sum_counts.copy()
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            improved = False
            # Try removing an element and adding a different one
            for i in range(len(S)):
                if time.time() > deadline:
                    break
                x_remove = S[i]
                # Temporarily remove x_remove
                old_sum_counts = {}
                for k, v in sum_counts.items():
                    old_sum_counts[k] = v
                remove(x_remove, S, sum_counts)
                
                # Try adding each candidate not in S
                for x_add in range(1, N + 1):
                    if x_add in S:
                        continue
                    if can_add(x_add, S, sum_counts):
                        # Check if this swap improves something (we just want to add if possible)
                        # But we removed one, so size stays same unless we can add without removing
                        # Actually, we want to find a swap that allows us to add MORE later?
                        # Simpler: try to add a new element without removing first.
                        # If we can add x_add to S (without removal), do it.
                        # If not, try swapping.
                        pass
                
                # Restore
                for k, v in old_sum_counts.items():
                    sum_counts[k] = v
                add(x_remove, S, sum_counts)
            
            # Simpler local search: just try to add any missing element
            # If we can add an element, do it.
            # If not, try swapping one element with another to free up space.
            # For now, let's just do a simple "try to add" pass.
            for x_add in range(1, N + 1):
                if x_add in S:
                    continue
                if can_add(x_add, S, sum_counts):
                    add(x_add, S, sum_counts)
                    improved = True
                    break
            
            if not improved:
                # Try a random swap
                if len(S) > 0:
                    i = random.randint(0, len(S) - 1)
                    x_remove = S[i]
                    old_sum_counts = sum_counts.copy()
                    remove(x_remove, S, sum_counts)
                    
                    # Try to add a random candidate
                    x_add = random.randint(1, N)
                    if x_add not in S and can_add(x_add, S, sum_counts):
                        add(x_add, S, sum_counts)
                        improved = True
                    else:
                        # Restore
                        for k, v in old_sum_counts.items():
                            sum_counts[k] = v
                        add(x_remove, S, sum_counts)
            
            if not improved:
                break
        
        return S, sum_counts

    # Run multiple random greedy attempts
    while time.time() < deadline:
        S = []
        sum_counts = {}
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if can_add(x, S, sum_counts):
                add(x, S, sum_counts)
        
        # Local search improvement
        S, sum_counts = local_search(S, sum_counts)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
        
    return best_set
