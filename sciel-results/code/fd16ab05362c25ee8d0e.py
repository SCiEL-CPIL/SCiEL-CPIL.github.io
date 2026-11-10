import random
import time

N = 256

def construct():
    """
    Construct a small inclusion-maximal Sidon set in [1, N].
    
    Strategy:
    1. Greedy construction to form a maximal Sidon set.
    2. Reduction phase to remove non-essential elements while maintaining maximality.
    """
    
    S = []
    Sigma = set()
    available = set(range(1, N + 1))
    
    def is_sidon_with_candidate(x, S, Sigma):
        new_sums = [x + s for s in S] + [2 * x]
        if len(set(new_sums)) != len(new_sums):
            return False
        for s in new_sums:
            if s in Sigma:
                return False
        return True

    def is_blocked(z, S, Sigma):
        if 2 * z in Sigma:
            return True
        for s in S:
            if z + s in Sigma:
                return True
        return False

    def is_maximal(S, Sigma):
        """Check if S is inclusion-maximal."""
        for z in range(1, N + 1):
            if z in S:
                continue
            if not is_blocked(z, S, Sigma):
                return False
        return True

    # Phase 1: Greedy Construction
    # We use a heuristic: pick the element that blocks the most available elements.
    
    random.seed(42)
    
    while available:
        best_x = None
        best_gain = -1
        
        # Sample a subset of candidates to speed up if needed, or use all
        candidates = list(available)
        random.shuffle(candidates)
        
        for x in candidates:
            if not is_sidon_with_candidate(x, S, Sigma):
                continue
            
            # Estimate gain: how many available elements will be blocked?
            new_sums = set([x + s for s in S] + [2 * x])
            new_Sigma = Sigma | new_sums
            
            gain = 0
            for z in available:
                if z == x:
                    continue
                # Quick check: is z blocked by new_Sigma and S+[x]?
                if 2 * z in new_Sigma:
                    gain += 1
                    continue
                
                blocked = False
                for s in S:
                    if z + s in new_Sigma:
                        blocked = True
                        break
                if not blocked:
                    if z + x in new_Sigma:
                        blocked = True
                    
                if blocked:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_x = x
        
        if best_x is None:
            break
        
        # Add best_x
        S.append(best_x)
        new_sums = [best_x + s for s in S[:-1]] + [2 * best_x]
        Sigma.update(new_sums)
        available.remove(best_x)
        
        # Update available: remove blocked elements
        new_available = set()
        for z in available:
            if not is_blocked(z, S, Sigma):
                new_available.add(z)
        available = new_available

    # Phase 2: Reduction
    # Try to remove elements from S while keeping it maximal.
    
    # Recompute Sigma to be safe
    Sigma = set()
    for i in range(len(S)):
        for j in range(i, len(S)):
            Sigma.add(S[i] + S[j])
            
    changed = True
    while changed:
        changed = False
        # Randomize order of removal attempts
        S_list = list(S)
        random.shuffle(S_list)
        
        for x in S_list:
            if x not in S:
                continue
            
            # Try removing x
            S_test = [y for y in S if y != x]
            
            # Compute Sigma for S_test
            Sigma_test = set()
            for i in range(len(S_test)):
                for j in range(i, len(S_test)):
                    Sigma_test.add(S_test[i] + S_test[j])
            
            # Check if S_test is maximal
            if is_maximal(S_test, Sigma_test):
                S = S_test
                Sigma = Sigma_test
                changed = True
                break # Restart loop with new S
        
    return S
