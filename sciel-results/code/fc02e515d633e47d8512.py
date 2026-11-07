import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    # U[c] is a set of sums for subsets of cardinality c
    # Initially, only empty subset: sum 0, cardinality 0
    U = [set() for _ in range(61)]
    U[0].add(0)
    
    used = set()
    max_n = 60
    
    # We want to maximize size. 
    # Greedy strategy: try to add the smallest available number that satisfies the condition.
    # This keeps the sums small, which helps in fitting more numbers.
    
    # To be robust, we can try a few different random restarts if time permits, 
    # but greedy smallest-first is usually strong for this type of packing problem.
    # Let's stick to a single deterministic greedy run first, as it's fast.
    
    # If greedy fails to reach a good size, we might need backtracking, 
    # but for n up to ~15-20, greedy is often sufficient.
    # The seed returned 8 numbers. We hope to do better.
    
    for _ in range(100): # Loop to try to add numbers
        if time.time() > deadline:
            break
            
        found = False
        # Try numbers from 1 to 256
        for x in range(1, 257):
            if x in used:
                continue
                
            # Check if x is valid
            valid = True
            # We need to check if adding x causes any collision
            # New sums for cardinality c+1 are {s + x | s in U[c]}
            # These must not intersect with U[k] for any k != c+1
            
            # Optimization: We only need to check up to current max cardinality + 1
            current_max_k = max(i for i, s in enumerate(U) if s)
            
            for c in range(current_max_k + 1):
                if not U[c]:
                    continue
                new_sums = {s + x for s in U[c]}
                
                # Check collision with all U[k] where k != c+1
                for k in range(current_max_k + 2): # k can be up to current_max_k + 1
                    if k == c + 1:
                        continue
                    if U[k] and (new_sums & U[k]):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Add x
                used.add(x)
                # Update U
                # Iterate backwards to avoid using updated U in the same step
                for c in range(current_max_k, -1, -1):
                    if U[c]:
                        new_sums = {s + x for s in U[c]}
                        U[c+1].update(new_sums)
                found = True
                break
        
        if not found:
            break

    result = list(used)
    # Ensure types are correct
    result = [int(x) for x in result]
    
    # Sort for deterministic output? The task doesn't require sorting, just a list of ints.
    # But sorting is good practice.
    result.sort()
    
    return result

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Size: {len(res)}")
    print(res)
