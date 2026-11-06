"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 55  # safety margin before 90s limit
    TARGET = 200
    ALL = list(range(1, 101))
    
    def greedy(order):
        """
        Greedy construction based on a given order of elements.
        Returns a valid subset.
        """
        S = []
        # reachable[s] is True if sum s can be formed by a subset of S
        reachable = [False] * (TARGET + 1)
        reachable[0] = True
        
        for x in order:
            # Check if adding x creates a sum of TARGET
            # We need to check if any existing sum s satisfies s + x == TARGET
            # i.e., s == TARGET - x
            # If so, we cannot add x.
            if reachable[TARGET - x]:
                continue
            
            # If we can add x, update reachable sums
            # We iterate downwards to avoid using x multiple times (though 0/1 knapsack style)
            # Actually, since we are just checking existence, and we process x once,
            # we can just mark new sums.
            
            new_reachable = reachable[:]
            for s in range(TARGET - x, -1, -1):
                if reachable[s]:
                    ns = s + x
                    if ns < TARGET:
                        new_reachable[ns] = True
            
            S.append(x)
            reachable = new_reachable
            
        return S

    def local_improve(S):
        """
        Try to add back excluded elements to S if possible.
        Returns improved set.
        """
        S_set = set(S)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Sort candidates by size descending (larger elements are 'safer' to add? 
            # Actually, adding any element is binary. Let's just iterate through all missing.)
            # To maximize cardinality, we just want to add anything that fits.
            # Order might matter for which ones get added first, but any valid addition increases size.
            # We restart scan after adding to catch cascades.
            for x in sorted([i for i in ALL if i not in S_set], reverse=True):
                if time.time() > deadline - 5:
                    break
                # Check if adding x is valid
                # We need to check if x + s == TARGET for any s in reachable sums of S
                # We can recompute reachable sums for S_set, but that's expensive.
                # Better: maintain reachable sums or just do a quick check.
                # Quick check: Does S_set contain a subset summing to TARGET - x?
                # This is a subset sum problem itself.
                # However, since we are in a local improvement loop, and S is already valid,
                # we can just try to add x and check validity of S_set U {x}.
                
                trial = S_set | {x}
                if _valid_subset(trial, TARGET):
                    S_set = trial
                    improved = True
                    break # Restart scan
    
        return list(S_set)

    def _valid_subset(S, target=200):
        """Check that no subset of S sums to target. S is a set of ints."""
        reachable = {0}
        for x in S:
            new = {s + x for s in reachable if s + x <= target}
            if target in new:
                return False
            reachable |= new
        return True

    best_set = []
    best_len = 0
    
    # 1. Static orders
    static_orders = [
        list(range(100, 0, -1)),
        list(range(1, 101))
    ]
    
    for o in static_orders:
        if time.time() > deadline - 10:
            break
        S = greedy(o)
        S = local_improve(S)
        if len(S) > best_len:
            best_len = len(S)
            best_set = S
        if best_len == 100:
            return best_set

    # 2. Adaptive random search with priorities
    # Elements that were skipped in previous runs get higher priority to be tried earlier.
    priorities = {x: 1.0 for x in ALL}
    
    # Run as many iterations as possible until deadline
    while time.time() < deadline - 5:
        # Generate order based on priorities
        # Sort by priority descending, then randomize ties
        # To ensure diversity, we can also mix in pure random sometimes
        if random.random() < 0.2:
            order = ALL[:]
            random.shuffle(order)
        else:
            # Sort by priority, add small random noise to break ties and maintain exploration
            order = sorted(ALL, key=lambda x: (priorities[x] + random.uniform(0, 0.1), x), reverse=True)
        
        S = greedy(order)
        S = local_improve(S)
        
        if len(S) > best_len:
            best_len = len(S)
            best_set = S
            if best_len == 100:
                return best_set
        
        # Update priorities:
        # Elements in S: slightly decrease priority (they were easy/early)
        # Elements NOT in S: increase priority (they were skipped, try earlier next time)
        for x in ALL:
            if x in S:
                priorities[x] *= 0.95
            else:
                priorities[x] *= 1.05
                
    # Final local improvement on the best set found
    if best_set:
        best_set = local_improve(best_set)

    if not best_set:
        return [100]
        
    return best_set
# EVOLVE-BLOCK-END
