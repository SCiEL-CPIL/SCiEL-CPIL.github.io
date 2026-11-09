import random
import time
import sys
from collections import defaultdict

# Precompute all 4-term arithmetic progressions in [1, 512]
N = 512
APs = []
# ap[x] will store indices of APs containing x
aps_containing = [[] for _ in range(N + 1)]

for d in range(1, (N - 1) // 3 + 1):
    for a in range(1, N - 3 * d + 1):
        ap = (a, a + d, a + 2 * d, a + 3 * d)
        idx = len(APs)
        APs.append(ap)
        for x in ap:
            aps_containing[x].append(idx)

def can_add(S_set, x):
    """Check if adding x to S_set keeps it 4-AP free."""
    for idx in aps_containing[x]:
        ap = APs[idx]
        # Check if the other 3 elements are in S_set
        count = 0
        for y in ap:
            if y != x and y in S_set:
                count += 1
        if count == 3:
            return False
    return True

def greedy_hitting_set():
    """
    Greedy hitting set / maximal independent set on the 4-AP hypergraph.
    Start with all elements, iteratively remove the element involved in the most 
    4-APs that are still 'active' (all 4 elements present).
    """
    S = set(range(1, N + 1))
    
    # Track which APs are still "active" (all 4 elements in S)
    # For efficiency, track count of elements in S for each AP
    ap_counts = [0] * len(APs)
    for idx, ap in enumerate(APs):
        ap_counts[idx] = 4  # all start in S
    
    # Count how many active APs each element is in
    elem_ap_count = [0] * (N + 1)
    for idx, ap in enumerate(APs):
        for x in ap:
            elem_ap_count[x] += 1
    
    # Iteratively remove the element in the most active APs
    while True:
        # Find element with max active AP count
        max_count = 0
        max_elem = None
        for x in range(1, N + 1):
            if x in S and elem_ap_count[x] > max_count:
                max_count = elem_ap_count[x]
                max_elem = x
        
        if max_elem is None or max_count == 0:
            break
        
        # Remove max_elem
        S.discard(max_elem)
        
        # Update AP counts
        for idx in aps_containing[max_elem]:
            ap_counts[idx] -= 1
            if ap_counts[idx] == 0:
                # This AP is no longer active, decrement counts for other elements
                for x in APs[idx]:
                    if x != max_elem and x in S:
                        elem_ap_count[x] -= 1
    
    return S

def greedy_with_backfill():
    """
    Start with greedy hitting set, then try to add back elements that no longer conflict.
    """
    S = greedy_hitting_set()
    
    # Try to add back removed elements
    for x in range(1, N + 1):
        if x not in S and can_add(S, x):
            S.add(x)
    
    return S

def local_search_improve(S, deadline):
    """
    Local search: try swaps (remove y, add x) to improve or maintain size with better structure.
    """
    S = set(S)
    best_size = len(S)
    
    while time.time() < deadline:
        # Try to add an element
        added = False
        for x in range(1, N + 1):
            if x not in S and can_add(S, x):
                S.add(x)
                best_size = len(S)
                added = True
                break
        
        if not added:
            # Try swaps: remove one element, add another
            found_swap = False
            for y in list(S):
                if time.time() > deadline:
                    return S
                S.remove(y)
                # Try to add any element that was previously blocked
                for x in range(1, N + 1):
                    if x not in S and can_add(S, x):
                        S.add(x)
                        if len(S) >= best_size:
                            # Keep the swap if size doesn't decrease
                            found_swap = True
                            break
                        else:
                            # Revert
                            S.remove(x)
                            S.add(y)
                            break
                if not found_swap:
                    S.add(y)
                else:
                    break
            
            if not found_swap:
                break
    
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s limit
    
    best_S = set()
    best_size = 0
    
    # Multiple strategies with different random seeds
    for attempt in range(20):
        if time.time() > deadline - 5:
            break
        
        if attempt % 3 == 0:
            # Strategy 1: Greedy hitting set with backfill
            S = greedy_with_backfill()
        elif attempt % 3 == 1:
            # Strategy 2: Random greedy with specific ordering
            S = set()
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
            for x in candidates:
                if time.time() > deadline:
                    break
                if can_add(S, x):
                    S.add(x)
        else:
            # Strategy 3: Greedy hitting set with random tie-breaking
            S = set(range(1, N + 1))
            ap_counts = [4] * len(APs)
            elem_ap_count = [0] * (N + 1)
            for idx, ap in enumerate(APs):
                for x in ap:
                    elem_ap_count[x] += 1
            
            while True:
                # Find elements with max active AP count, pick randomly among ties
                max_count = 0
                candidates = []
                for x in range(1, N + 1):
                    if x in S and elem_ap_count[x] > max_count:
                        max_count = elem_ap_count[x]
                        candidates = [x]
                    elif x in S and elem_ap_count[x] == max_count:
                        candidates.append(x)
                
                if not candidates or max_count == 0:
                    break
                
                max_elem = random.choice(candidates)
                S.discard(max_elem)
                
                for idx in aps_containing[max_elem]:
                    ap_counts[idx] -= 1
                    if ap_counts[idx] == 0:
                        for x in APs[idx]:
                            if x != max_elem and x in S:
                                elem_ap_count[x] -= 1
            
            # Backfill
            for x in range(1, N + 1):
                if x not in S and can_add(S, x):
                    S.add(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
        
        # Local search to improve
        if time.time() < deadline - 3:
            improved_S = local_search_improve(S, deadline - 3)
            if len(improved_S) > best_size:
                best_size = len(improved_S)
                best_S = set(improved_S)
    
    return sorted(best_S)

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    print(f"Size: {len(res)}")
    # Verify validity
    S_set = set(res)
    valid = True
    for ap in APs:
        if all(x in S_set for x in ap):
            valid = False
            break
    print(f"Valid: {valid}")
