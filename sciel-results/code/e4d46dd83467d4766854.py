"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 256

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 25.0  # safety margin before 30s limit
    
    # Helper: check if a set is a valid Sidon set
    def is_sidon(S):
        sums = set()
        for i, a in enumerate(S):
            for b in S[i:]:
                s = a + b
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    # Helper: check if a Sidon set is maximal (no element can be added)
    def is_maximal(S):
        if not is_sidon(S):
            return False
        sums = set()
        for i, a in enumerate(S):
            for b in S[i:]:
                sums.add(a + b)
        
        for x in range(1, N + 1):
            if x in S:
                continue
            # Check if adding x would create a duplicate sum
            can_add = True
            for a in S:
                s = x + a
                if s in sums:
                    can_add = False
                    break
                # Also check if 2*x conflicts with existing sums
            if can_add:
                if 2 * x in sums:
                    can_add = False
            if can_add:
                return False
        return True
    
    # Helper: try to remove an element and maintain maximality by adding others
    def try_reduce(S):
        """Try to find a smaller maximal Sidon set by removing elements."""
        S = list(S)
        improved = True
        while improved:
            improved = False
            for i in range(len(S)):
                candidate = S[:i] + S[i+1:]
                if not is_sidon(candidate):
                    continue
                
                # Check if candidate is maximal
                if is_maximal(candidate):
                    S = candidate
                    improved = True
                    break
        return S
    
    # Helper: greedy construction with different orderings
    def greedy(order):
        selected = []
        sums = set()
        for x in order:
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        return selected
    
    # Helper: local search to try adding elements that maintain maximality
    def local_search_reduce(S):
        """Try to reduce S by removing elements and seeing if maximality still holds."""
        S = list(S)
        changed = True
        while changed and (time.time() < deadline - 5.0):
            changed = False
            for i in range(len(S)):
                # Try removing S[i]
                candidate = S[:i] + S[i+1:]
                if not is_sidon(candidate):
                    continue
                if is_maximal(candidate):
                    S = candidate
                    changed = True
                    break
        return S
    
    # Strategy 1: Standard greedy (ascending)
    best = greedy(list(range(1, N + 1)))
    best = local_search_reduce(best)
    
    # Strategy 2: Greedy with various random orderings
    random.seed(42)
    for trial in range(50):
        if time.time() > deadline - 10.0:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        candidate = greedy(order)
        candidate = local_search_reduce(candidate)
        if len(candidate) < len(best):
            best = candidate
    
    # Strategy 3: Greedy starting from high numbers
    if time.time() < deadline - 10.0:
        order = list(range(N, 0, -1))
        candidate = greedy(order)
        candidate = local_search_reduce(candidate)
        if len(candidate) < len(best):
            best = candidate
    
    # Strategy 4: More aggressive local search with restarts
    if time.time() < deadline - 5.0:
        for trial in range(20):
            if time.time() > deadline - 5.0:
                break
            # Start from a random subset of best
            if len(best) > 10:
                keep_prob = 0.7
                candidate = [x for x in best if random.random() < keep_prob]
            else:
                candidate = list(best)
            
            # Try to complete to a maximal set
            sums = set()
            for i, a in enumerate(candidate):
                for b in candidate[i:]:
                    sums.add(a + b)
            
            # Add elements that maintain Sidon property
            for x in range(1, N + 1):
                if x in candidate:
                    continue
                added = [x + a for a in candidate] + [2 * x]
                if len(set(added)) == len(added) and not sums.intersection(added):
                    candidate.append(x)
                    sums.update(added)
            
            candidate = local_search_reduce(candidate)
            if len(candidate) < len(best):
                best = candidate
    
    return best
# EVOLVE-BLOCK-END
