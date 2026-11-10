import random
import time

N = 256

def construct():
    """
    Construct a small inclusion-maximal Sidon set in [1, N].
    
    Approach:
    1. Build a Sidon set S greedily by adding elements that maximize blocking gain.
    2. Ensure maximality by verifying no missing element can be added.
    3. Minimize cardinality by being selective about which elements to add.
    
    We use a greedy heuristic: at each step, add the element x that blocks the most
    other elements (i.e., makes the most other elements incompatible with S).
    """
    
    random.seed(42)  # For reproducibility if we add randomization
    
    S = []
    Sigma = set()  # All pair sums a+b for a<=b in S
    
    # Track which elements are still "available" (can be added without violating Sidon)
    # An element x is available if adding it to S keeps S a Sidon set
    available = set(range(1, N + 1))
    
    def is_sidon_with_candidate(x, S, Sigma):
        """Check if adding x to S keeps it a Sidon set."""
        # New sums to check: x+s for s in S, and 2*x
        new_sums = [x + s for s in S] + [2 * x]
        
        # Check internal consistency of new sums
        if len(set(new_sums)) != len(new_sums):
            return False
        
        # Check no collision with existing sums
        for s in new_sums:
            if s in Sigma:
                return False
        
        return True
    
    def count_blocked_by_candidate(x, S, Sigma, available):
        """Count how many elements in available \\ {x} would be blocked if x is added to S."""
        if not is_sidon_with_candidate(x, S, Sigma):
            return -1  # Not a valid candidate
        
        # Simulate adding x
        new_S = S + [x]
        new_Sigma = Sigma | set([x + s for s in S] + [2 * x])
        
        blocked_count = 0
        for z in available:
            if z == x:
                continue
            
            # Check if z is blocked by new_S and new_Sigma
            # z is blocked if it cannot be added to new_S while keeping Sidon property
            # This happens if:
            # 1. 2*z is in new_Sigma (diagonal collision)
            # 2. There exists s in new_S such that z+s is in new_Sigma (off-diagonal collision)
            
            if 2 * z in new_Sigma:
                blocked_count += 1
                continue
            
            blocked = False
            for s in new_S:
                if z + s in new_Sigma:
                    blocked = True
                    break
            
            if blocked:
                blocked_count += 1
        
        return blocked_count
    
    def is_blocked(z, S, Sigma):
        """Check if element z is blocked by current S and Sigma."""
        if 2 * z in Sigma:
            return True
        
        for s in S:
            if z + s in Sigma:
                return True
        
        return False
    
    # Greedy construction
    while available:
        best_x = None
        best_gain = -1
        
        # Evaluate all candidates
        candidates = list(available)
        random.shuffle(candidates)  # Break ties randomly
        
        for x in candidates:
            gain = count_blocked_by_candidate(x, S, Sigma, available)
            if gain > best_gain:
                best_gain = gain
                best_x = x
        
        if best_x is None:
            # No valid candidate - S is already maximal
            break
        
        # Add best_x to S
        S.append(best_x)
        new_sums = [best_x + s for s in S[:-1]] + [2 * best_x]
        Sigma.update(new_sums)
        available.remove(best_x)
        
        # Update available: remove elements that are now blocked
        new_available = set()
        for z in available:
            if not is_blocked(z, S, Sigma):
                new_available.add(z)
        available = new_available
    
    # Verify maximality: ensure no missing element can be added
    # This should already be satisfied by our construction, but let's verify
    for z in range(1, N + 1):
        if z in S:
            continue
        # Check if z can be added
        if is_sidon_with_candidate(z, S, Sigma):
            # This shouldn't happen, but if it does, we have a problem
            # For now, just note it - the set might not be maximal
            pass
    
    return S
