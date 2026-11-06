"""B2[2] set construction for {1,...,1000}: maximize size with at most 2 representations per sum."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 130.0  # safety margin before 150s kill
    N = 1000
    MAX_REP = 2

    def compute_counts(S):
        """Compute sum counts for a set S."""
        S = sorted(S)
        counts = defaultdict(int)
        n = len(S)
        for i in range(n):
            for j in range(i, n):
                counts[S[i] + S[j]] += 1
        return counts

    def is_valid(S):
        """Check if set S satisfies B2[2] constraint."""
        S = sorted(S)
        n = len(S)
        counts = defaultdict(int)
        for i in range(n):
            for j in range(i, n):
                s = S[i] + S[j]
                counts[s] += 1
                if counts[s] > MAX_REP:
                    return False
        return True

    def greedy_construct(order=None):
        """Greedy construction: add elements in given order if valid."""
        if order is None:
            order = list(range(1, N + 1))
        chosen = []
        counts = defaultdict(int)
        for x in order:
            # Check if adding x would violate constraint
            ok = True
            new_sums = []
            for y in chosen:
                s = x + y
                if counts[s] >= MAX_REP:
                    ok = False
                    break
                new_sums.append(s)
            if ok:
                s = 2 * x
                if counts[s] >= MAX_REP:
                    ok = False
            if ok:
                chosen.append(x)
                for s in new_sums:
                    counts[s] += 1
                counts[2 * x] += 1
        return chosen

    def local_search_improve(S, max_iters=5000):
        """Try to improve set S by swapping elements or adding/removing."""
        S = sorted(S)
        counts = compute_counts(S)
        best_S = list(S)
        best_size = len(S)
        best_counts = dict(counts)
        
        # Precompute all possible swaps
        all_elements = set(range(1, N + 1))
        
        for iteration in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            improved = False
            
            # Try adding a random element not in S
            candidates_to_add = list(all_elements - set(S))
            random.shuffle(candidates_to_add)
            
            for x in candidates_to_add[:50]:  # Try up to 50 random additions
                # Check if x can be added
                can_add = True
                for y in S:
                    if counts.get(x + y, 0) >= MAX_REP:
                        can_add = False
                        break
                if can_add and counts.get(2 * x, 0) >= MAX_REP:
                    can_add = False
                
                if can_add:
                    new_S = S + [x]
                    new_S.sort()
                    # Verify validity
                    if is_valid(new_S):
                        S = new_S
                        counts = compute_counts(S)
                        best_S = list(S)
                        best_size = len(S)
                        best_counts = dict(counts)
                        improved = True
                        break
            
            if not improved:
                # Try swaps: remove one, add one
                if len(S) > 10:
                    removed = random.choice(S)
                    S_without = [x for x in S if x != removed]
                    candidates_to_add = list(all_elements - set(S_without))
                    random.shuffle(candidates_to_add)
                    
                    for x in candidates_to_add[:30]:
                        can_add = True
                        for y in S_without:
                            if counts.get(x + y, 0) - (1 if y == removed else 0) >= MAX_REP:
                                # Need to recompute properly
                                pass
                        
                        # Simpler: just test validity
                        new_S = S_without + [x]
                        new_S.sort()
                        if is_valid(new_S):
                            S = new_S
                            counts = compute_counts(S)
                            best_S = list(S)
                            best_size = len(S)
                            best_counts = dict(counts)
                            improved = True
                            break
            
            if not improved:
                # Try removing an element that allows adding two others
                if len(S) > 20:
                    # Find elements that are "expensive" (block many additions)
                    for _ in range(5):
                        idx = random.randint(0, len(S) - 1)
                        removed = S[idx]
                        S_without = S[:idx] + S[idx+1:]
                        
                        # Try to add two elements
                        candidates = list(all_elements - set(S_without))
                        random.shuffle(candidates)
                        
                        added_any = False
                        for x in candidates[:40]:
                            new_S = S_without + [x]
                            new_S.sort()
                            if is_valid(new_S):
                                S = new_S
                                counts = compute_counts(S)
                                best_S = list(S)
                                best_size = len(S)
                                best_counts = dict(counts)
                                added_any = True
                                break
                        
                        if not added_any:
                            # Try another removal
                            pass
                        else:
                            improved = True
                            break
            
            if not improved:
                break
        
        return best_S, best_size

    # Phase 1: Greedy construction with different orders
    best_set = []
    best_size = 0
    
    # Standard greedy
    greedy1 = greedy_construct()
    best_set = greedy1
    best_size = len(greedy1)
    
    # Greedy with random perturbations
    for restart in range(30):
        if time.time() > deadline - 60:
            break
        
        # Create a perturbed order
        order = list(range(1, N + 1))
        random.shuffle(order)
        # But bias towards smaller numbers for better density
        # Mix: 70% sorted, 30% random
        mixed_order = []
        for i in range(N):
            if random.random() < 0.7:
                mixed_order.append(i + 1)
            else:
                if i + 1 not in mixed_order:
                    mixed_order.append(i + 1)
        
        # Ensure all elements are in the order
        remaining = set(range(1, N + 1)) - set(mixed_order)
        mixed_order.extend(list(remaining))
        
        result = greedy_construct(mixed_order)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result
    
    # Phase 2: Local search improvement
    if time.time() < deadline - 30:
        improved_set, improved_size = local_search_improve(best_set, max_iters=10000)
        if improved_size > best_size:
            best_size = improved_size
            best_set = improved_set
    
    # Phase 3: More aggressive local search with multiple restarts
    if time.time() < deadline - 20:
        for _ in range(10):
            if time.time() > deadline - 15:
                break
            
            # Start from current best and do more swaps
            S = list(best_set)
            counts = compute_counts(S)
            
            for _ in range(5000):
                if time.time() > deadline - 15:
                    break
                
                # Try to add a random element
                candidates = list(set(range(1, N + 1)) - set(S))
                if not candidates:
                    break
                random.shuffle(candidates)
                
                added = False
                for x in candidates[:20]:
                    can_add = True
                    for y in S:
                        if counts.get(x + y, 0) >= MAX_REP:
                            can_add = False
                            break
                    if can_add and counts.get(2 * x, 0) >= MAX_REP:
                        can_add = False
                    
                    if can_add:
                        new_S = S + [x]
                        new_S.sort()
                        if is_valid(new_S):
                            S = new_S
                            counts = compute_counts(S)
                            added = True
                            break
                
                if not added and len(S) > 15:
                    # Try a swap
                    idx = random.randint(0, len(S) - 1)
                    removed = S[idx]
                    S_without = S[:idx] + S[idx+1:]
                    
                    candidates = list(set(range(1, N + 1)) - set(S_without))
                    random.shuffle(candidates)
                    
                    for x in candidates[:15]:
                        new_S = S_without + [x]
                        new_S.sort()
                        if is_valid(new_S):
                            S = new_S
                            counts = compute_counts(S)
                            break
                
                if not added:
                    break
            
            if len(S) > best_size:
                best_size = len(S)
                best_set = S
    
    return best_set
# EVOLVE-BLOCK-END
