import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Leave 5s margin
    
    def check_and_add(chosen, S_sets, x, deadline):
        """
        Tries to add x to chosen.
        Returns (success, new_chosen, new_S_sets)
        """
        if time.time() > deadline:
            return False, chosen, S_sets
            
        m = len(chosen)
        # S_sets is a list of sets. S_sets[k] contains sums of subsets of size k.
        # S_sets[0] = {0}
        
        # Compute new sums N_k
        # N_k = {s + x for s in S_sets[k-1]}
        # We only need to check intersections with sets of different cardinalities.
        
        new_S_sets = [set(s) for s in S_sets]
        # Update in reverse order to avoid using updated S_sets[k-1] for S_sets[k]?
        # No, N_k depends on OLD S_sets[k-1].
        # So we must compute all N_k based on OLD S_sets before updating.
        
        N = [set() for _ in range(m + 2)] # Index up to m+1
        for k in range(m + 1, 0, -1):
            # N[k] comes from S_sets[k-1]
            # S_sets[k-1] is the set of sums of size k-1
            if k-1 < len(S_sets):
                N[k] = {s + x for s in S_sets[k-1]}
            else:
                N[k] = set()
                
        # Check validity
        # For each k, N[k] must be disjoint from:
        # 1. S_sets[j] for all j != k (old sums of different size)
        # 2. N[j] for all j != k (new sums of different size)
        
        # Optimization: 
        # We can check N[k] against all S_sets[j] (j!=k) and all N[j] (j!=k).
        
        for k in range(1, m + 2):
            if not N[k]:
                continue
            # Check against old sets
            for j in range(0, m + 1):
                if j == k:
                    continue
                if j < len(S_sets) and S_sets[j]:
                    if N[k] & S_sets[j]:
                        return False, chosen, S_sets
            # Check against new sets
            for j in range(1, m + 2):
                if j == k:
                    continue
                if N[j]:
                    if N[k] & N[j]:
                        return False, chosen, S_sets
                        
        # If valid, update
        for k in range(1, m + 2):
            if N[k]:
                if k >= len(new_S_sets):
                    new_S_sets.append(set())
                new_S_sets[k].update(N[k])
                
        new_chosen = chosen + [x]
        return True, new_chosen, new_S_sets

    best_size = 0
    best_set = []
    
    # Try a few random orders
    # Also try ascending order
    orders = [
        list(range(1, 257)),
        list(range(256, 0, -1)),
        [random.randint(1, 256) for _ in range(256)]
    ]
    # Add a few more random permutations
    for _ in range(10):
        if time.time() > deadline - 5:
            break
        perm = list(range(1, 257))
        random.shuffle(perm)
        orders.append(perm)
        
    for order in orders:
        if time.time() > deadline:
            break
            
        chosen = []
        S_sets = [set([0])] # S_sets[0] = {0}
        
        for x in order:
            success, chosen, S_sets = check_and_add(chosen, S_sets, x, deadline)
            if success:
                if len(chosen) > best_size:
                    best_size = len(chosen)
                    best_set = list(chosen)
        
        # Early stop if we found a very large set? 
        # The max possible is limited by 256.
        
    return best_set
