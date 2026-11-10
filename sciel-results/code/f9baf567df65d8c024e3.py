import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def construct():
    """
    Find the largest set of distinct integers in 1..256 such that
    subset sums of different cardinalities are disjoint.
    
    Strategy:
    1. Greedy construction: add elements one by one, choosing the one that
       causes the fewest future conflicts.
    2. Local search: try to swap in/out elements to improve the set.
    3. Multiple random restarts with different orderings.
    """
    deadline = time.time() + 80  # 80s safety margin before 90s limit
    best_set = []
    best_size = 0
    
    def compute_cardinality_sums(s):
        """Return list: k -> set of all k-element subset sums"""
        n = len(s)
        sums_by_k = [set() for _ in range(n + 1)]
        sums_by_k[0].add(0)
        for x in s:
            for k in range(len(sums_by_k) - 1, -1, -1):
                if sums_by_k[k]:
                    for v in sums_by_k[k]:
                        sums_by_k[k + 1].add(v + x)
        return sums_by_k
    
    def check_validity(sums_by_k):
        """Check if no two cardinalities share any sum value. Return (valid, collision_count)."""
        seen = set()
        collisions = 0
        for k in range(len(sums_by_k)):
            for v in sums_by_k[k]:
                if v in seen:
                    collisions += 1
                seen.add(v)
        return collisions == 0, collisions
    
    def greedy_construct(max_elements=256, deadline_check=None):
        """
        Greedily construct a valid set by adding elements one at a time.
        At each step, try all available elements and pick the one that
        keeps the set valid and minimizes the number of "tight" sums.
        """
        s = []
        sums_by_k = [set()]
        sums_by_k[0].add(0)
        available = set(range(1, max_elements + 1))
        
        while available:
            if deadline_check and time.time() > deadline_check:
                break
            
            best_elem = None
            best_sums = None
            best_score = float('inf')
            
            # Try each available element
            for elem in sorted(available, reverse=True):
                if deadline_check and time.time() > deadline_check:
                    break
                
                # Compute new sums if we add this element
                n = len(s)
                new_sums = [set() for _ in range(n + 2)]
                for k in range(n + 1):
                    new_sums[k] = set(sums_by_k[k])
                for k in range(n, -1, -1):
                    for v in sums_by_k[k]:
                        new_sums[k + 1].add(v + elem)
                
                # Check validity
                valid, collisions = check_validity(new_sums)
                if not valid:
                    continue
                
                # Score: prefer elements that create "spread out" sums
                # Count how many sums are "close" to existing sums from other cardinalities
                score = 0
                for k in range(n + 2):
                    for v in new_sums[k]:
                        for j in range(n + 2):
                            if j != k:
                                for w in new_sums[j]:
                                    if abs(v - w) <= 3:
                                        score += 1
                                        break
                
                if score < best_score:
                    best_score = score
                    best_elem = elem
                    best_sums = new_sums
            
            if best_elem is None:
                break
            
            s.append(best_elem)
            sums_by_k = best_sums
            available.discard(best_elem)
        
        return s
    
    def local_improve(s, deadline_check=None):
        """
        Try to improve the set by removing one element and adding another.
        Repeat until no improvement is found or deadline.
        """
        s = list(s)
        improved = True
        
        while improved and (deadline_check is None or time.time() < deadline_check):
            improved = False
            current_size = len(s)
            
            # Try removing each element
            for i in range(len(s)):
                if deadline_check and time.time() > deadline_check:
                    return s
                
                s_without = s[:i] + s[i+1:]
                sums_without = compute_cardinality_sums(s_without)
                valid, _ = check_validity(sums_without)
                if not valid:
                    continue
                
                # Try adding each available element
                available = set(range(1, 257)) - set(s_without)
                for elem in sorted(available, reverse=True):
                    if deadline_check and time.time() > deadline_check:
                        return s
                    
                    n = len(s_without)
                    new_sums = [set() for _ in range(n + 2)]
                    for k in range(n + 1):
                        new_sums[k] = set(sums_without[k])
                    for k in range(n, -1, -1):
                        for v in sums_without[k]:
                            new_sums[k + 1].add(v + elem)
                    
                    valid2, _ = check_validity(new_sums)
                    if valid2:
                        # Accept the swap
                        s = s_without + [elem]
                        improved = True
                        break
                
                if improved:
                    break
        
        return s
    
    # Phase 1: Multiple greedy constructions with different orderings
    random.seed(42)
    
    # Try greedy with different starting points
    for attempt in range(5):
        if time.time() > deadline - 10:
            break
        
        if attempt == 0:
            # Standard greedy
            candidate = greedy_construct(deadline_check=deadline - 10)
        else:
            # Perturbed greedy: start with a random subset, then greedily add
            initial_size = random.randint(10, 30)
            initial = random.sample(range(1, 257), initial_size)
            candidate = list(initial)
            sums_by_k = compute_cardinality_sums(candidate)
            valid, _ = check_validity(sums_by_k)
            
            if valid:
                # Continue greedily from here
                available = set(range(1, 257)) - set(candidate)
                while available and time.time() < deadline - 10:
                    best_elem = None
                    best_sums = None
                    best_score = float('inf')
                    
                    for elem in sorted(available, reverse=True):
                        if time.time() > deadline - 10:
                            break
                        
                        n = len(candidate)
                        new_sums = [set() for _ in range(n + 2)]
                        for k in range(n + 1):
                            new_sums[k] = set(sums_by_k[k])
                        for k in range(n, -1, -1):
                            for v in sums_by_k[k]:
                                new_sums[k + 1].add(v + elem)
                        
                        valid2, collisions = check_validity(new_sums)
                        if not valid2:
                            continue
                        
                        score = 0
                        for k in range(n + 2):
                            for v in new_sums[k]:
                                for j in range(n + 2):
                                    if j != k:
                                        for w in new_sums[j]:
                                            if abs(v - w) <= 3:
                                                score += 1
                                                break
                        
                        if score < best_score:
                            best_score = score
                            best_elem = elem
                            best_sums = new_sums
                    
                    if best_elem is None:
                        break
                    
                    candidate.append(best_elem)
                    sums_by_k = best_sums
                    available.discard(best_elem)
            else:
                candidate = greedy_construct(deadline_check=deadline - 10)
        
        if len(candidate) > best_size:
            best_size = len(candidate)
            best_set = list(candidate)
    
    # Phase 2: Local improvement on the best set
    if time.time() < deadline - 5 and best_set:
        improved = local_improve(best_set, deadline_check=deadline - 5)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = list(improved)
    
    # Ensure we return Python ints in a list
    result = [int(x) for x in best_set]
    return result

# EVOLVE-BLOCK-END
