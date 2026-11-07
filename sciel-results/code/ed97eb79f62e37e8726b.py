"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 55  # Safety margin for 90s limit
    n = 128
    
    # Precompute all triples (a,b,c) with a<b<c and check if a+b, a+c, b+c are in 1..128
    # We need to forbid: distinct a,b,c in S such that a+b, a+c, b+c are all in S
    # Note: a+b, a+c, b+c must be in 1..128 for them to be "selected"
    
    # Precompute all potential conflicts: for each triple (a,b,c) with a<b<c,
    # the sums s1=a+b, s2=a+c, s3=b+c must all be in 1..n
    # Store as list of (a,b,c,s1,s2,s3) tuples where all sums are valid
    
    conflicts = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            for c in range(b + 1, n + 1):
                s1 = a + b
                s2 = a + c
                s3 = b + c
                if s1 <= n and s2 <= n and s3 <= n:
                    # All three sums are in range, so this is a potential conflict
                    conflicts.append((a, b, c, s1, s2, s3))
    
    # Convert to numpy arrays for fast operations
    # Each conflict is a set of 6 values that must NOT all be in S simultaneously
    # Actually, the constraint is: NOT (a in S AND b in S AND c in S AND s1 in S AND s2 in S AND s3 in S)
    # But note: a,b,c are distinct members, and s1,s2,s3 may or may not be distinct from a,b,c or each other
    
    # For fast checking, we'll use a boolean array of size n+1
    # in_set[i] = True if i is in the set
    
    def is_valid_set(in_set):
        """Check if the set (represented by boolean array in_set[1..n]) is valid."""
        for (a, b, c, s1, s2, s3) in conflicts:
            if in_set[a] and in_set[b] and in_set[c] and in_set[s1] and in_set[s2] and in_set[s3]:
                return False
        return True
    
    def count_conflicts(in_set):
        """Count how many conflicts are violated."""
        count = 0
        for (a, b, c, s1, s2, s3) in conflicts:
            if in_set[a] and in_set[b] and in_set[c] and in_set[s1] and in_set[s2] and in_set[s3]:
                count += 1
        return count
    
    # Start with upper half [65..128]
    in_set = np.zeros(n + 1, dtype=bool)
    for i in range(65, n + 1):
        in_set[i] = True
    
    best_in_set = in_set.copy()
    best_size = np.sum(in_set[1:])
    
    # Greedy: try to add elements from 1 to 64
    for x in range(1, 65):
        if time.time() > deadline:
            break
        in_set[x] = True
        if is_valid_set(in_set):
            best_in_set = in_set.copy()
            best_size = np.sum(in_set[1:])
        else:
            in_set[x] = False
    
    # Simulated Annealing
    if time.time() < deadline:
        current_in_set = best_in_set.copy()
        current_size = int(np.sum(current_in_set[1:]))
        best_sa_in_set = current_in_set.copy()
        best_sa_size = current_size
        
        # Temperature schedule
        t0 = 1.0
        t1 = 0.01
        total_iterations = 50000  # Will be limited by time
        
        for it in range(total_iterations):
            if time.time() > deadline - 2:
                break
            
            # Choose a move: add or remove a random element
            x = random.randint(1, n)
            
            if current_in_set[x]:
                # Remove x
                current_in_set[x] = False
                new_size = current_size - 1
                # Check if still valid (removing can't create new conflicts, so always valid)
                # But we might want to track conflicts for efficiency
                current_size = new_size
            else:
                # Add x
                current_in_set[x] = True
                new_size = current_size + 1
                # Check if this creates any new conflicts
                valid = True
                for (a, b, c, s1, s2, s3) in conflicts:
                    # Only need to check conflicts involving x
                    if x not in (a, b, c, s1, s2, s3):
                        continue
                    if current_in_set[a] and current_in_set[b] and current_in_set[c] and \
                       current_in_set[s1] and current_in_set[s2] and current_in_set[s3]:
                        valid = False
                        break
                
                if not valid:
                    # Revert
                    current_in_set[x] = False
                    # Accept with probability based on temperature
                    # For now, reject
                    continue
                
                current_size = new_size
            
            # Update best
            if current_size > best_sa_size:
                best_sa_size = current_size
                best_sa_in_set = current_in_set.copy()
            
            # Cooling
            progress = it / total_iterations
            temperature = t0 * (t1 / t0) ** progress
            
            # With probability, accept worse solutions
            # For simplicity, always accept (since we check validity)
            # But this is really just random walk with validity check
        
        if best_sa_size > best_size:
            best_size = best_sa_size
            best_in_set = best_sa_in_set
    
    # Local search: try to add any remaining element
    if time.time() < deadline - 1:
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for x in range(1, n + 1):
                if not best_in_set[x]:
                    best_in_set[x] = True
                    if is_valid_set(best_in_set):
                        best_size += 1
                        improved = True
                        break
                    else:
                        best_in_set[x] = False
    
    # Convert to list
    result = [i for i in range(1, n + 1) if best_in_set[i]]
    
    return result

# EVOLVE-BLOCK-END
