import time
import random
import sys

N = 256
DEADLINE_MARGIN = 5.0

# Precomputation for fast access
# ap3s[x] = list of (y, z) such that (x, y, z) or (y, x, z) or (y, z, x) is a 3-AP?
# Better: For each x, store all pairs (u, v) such that {x, u, v} forms a 3-AP.
# We can just store the indices of 3-APs containing x.
# ap4s[x] = list of indices of 4-APs containing x.

def _precompute():
    all_3aps = []
    all_4aps = []
    
    # 3-APs: (a, b, c) with a < b < c and b-a = c-b
    for a in range(1, N + 1):
        for d in range(1, N + 1):
            b = a + d
            c = a + 2 * d
            if c <= N:
                all_3aps.append((a, b, c))
                
    # 4-APs: (a, b, c, d) with a < b < c < d and b-a = c-b = d-c
    for a in range(1, N + 1):
        for d in range(1, N + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= N:
                all_4aps.append((a, b, c, e))
                
    # Membership: x -> list of indices of APs containing x
    ap3_membership = [[] for _ in range(N + 1)]
    ap4_membership = [[] for _ in range(N + 1)]
    
    for i, ap in enumerate(all_3aps):
        for x in ap:
            ap3_membership[x].append(i)
            
    for i, ap in enumerate(all_4aps):
        for x in ap:
            ap4_membership[x].append(i)
            
    return all_3aps, all_4aps, ap3_membership, ap4_membership

ALL_3APS, ALL_4APS, AP3_MEM, AP4_MEM = _precompute()

def _count_3aps_bitset(mask):
    """
    Count 3-APs in the set represented by mask.
    mask is an integer where bit i is set if i is in the set.
    """
    count = 0
    # Iterate over all 3-APs and check if all three bits are set
    for a, b, c in ALL_3APS:
        if (mask >> (a-1)) & 1 and (mask >> (b-1)) & 1 and (mask >> (c-1)) & 1:
            count += 1
    return count

def _greedy_construct(deadline):
    """
    Greedily construct a set A by adding elements one by one.
    At each step, choose the element x not in A that maximizes the number of new 3-APs,
    provided that adding x does not create any 4-AP.
    """
    A_mask = 0
    A_set = set()
    
    # We need to track which 4-APs are "active" or "broken".
    # Actually, we just need to check if adding x creates a 4-AP.
    # A 4-AP is created if x is part of a 4-AP (a, b, c, d) and the other 3 are in A.
    
    # To speed up "does adding x create a 4-AP", we can precompute for each x
    # the list of 4-APs it belongs to. Then for each such 4-AP, check if the other 3 are in A.
    
    # To speed up "how many new 3-APs", we can precompute for each x
    # the list of 3-APs it belongs to. Then for each such 3-AP, check if the other 2 are in A.
    
    # Let's maintain a count of "active" 3-APs? No, just calculate gain.
    
    # Optimization:
    # For each x, new_3aps_gain(x) = number of 3-APs containing x where the other 2 elements are in A.
    # is_valid(x) = True if no 4-AP containing x has the other 3 elements in A.
    
    # We can iterate through all candidates x not in A.
    # N is 256, so this is fast.
    
    while len(A_set) < N:
        if time.time() > deadline:
            break
        
        best_x = -1
        best_gain = -1
        
        # Candidates are 1..N not in A
        for x in range(1, N + 1):
            if x in A_set:
                continue
            
            # Check validity: does x create a 4-AP?
            valid = True
            for idx in AP4_MEM[x]:
                ap = ALL_4APS[idx]
                # Check if the other 3 elements are in A
                others_in_A = 0
                for y in ap:
                    if y != x:
                        if y in A_set:
                            others_in_A += 1
                        else:
                            break
                if others_in_A == 3:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Calculate gain: how many new 3-APs does x form?
            gain = 0
            for idx in AP3_MEM[x]:
                ap = ALL_3APS[idx]
                # Check if the other 2 elements are in A
                others_in_A = 0
                for y in ap:
                    if y != x:
                        if y in A_set:
                            others_in_A += 1
                        else:
                            break
                if others_in_A == 2:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_x = x
                # If gain is 0, we might want to break early if we've found a valid 0-gain element?
                # But maybe a later element has gain > 0? Unlikely if we are adding in order?
                # Actually, gain can be 0 for many elements. We should pick the first valid one?
                # Or one with some heuristic?
                # Let's just pick the best gain. If best_gain is 0, we can stop searching for better gains
                # if we are sure no other element has gain > 0? Hard to know.
                # However, if best_gain == 0, adding any valid element with 0 gain is equivalent in terms of 3-AP count.
                # But it affects future gains.
                # For now, let's just pick the max gain.
        
        if best_x == -1:
            # No valid element found? This shouldn't happen unless A is full or all remaining create 4-APs.
            # If A is empty, any x is valid.
            # If A is not empty, it's possible that all remaining x create 4-APs?
            # No, we can always add elements in a way that avoids 4-APs (e.g. greedy removal from full set works).
            # But here we are adding. If we get stuck, we should probably just pick any valid one.
            # Wait, if best_x is -1, it means no valid element was found.
            # This implies that for every x not in A, x creates a 4-AP with A.
            # This is possible. In that case, we cannot add any more elements.
            break
            
        A_set.add(best_x)
        A_mask |= (1 << (best_x - 1))
        
        # Optional: Early termination if we've added enough? No, maximize count.
        
    return sorted(A_set), A_mask

def _local_search(A, deadline):
    """
    Try to improve A by swapping elements.
    """
    A_set = set(A)
    A_mask = 0
    for x in A_set:
        A_mask |= (1 << (x - 1))
        
    current_count = _count_3aps_bitset(A_mask)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to remove one element and add another
        # This is O(N^2) per iteration, which might be slow.
        # N=256, N^2 = 65k. 
        # If we do this for a few seconds, it's fine.
        
        # Optimization: Only consider removing elements that are in many 3-APs?
        # Or just random swaps?
        
        # Let's try a simple hill climbing:
        # For each x in A, for each y not in A:
        #   If removing x and adding y is valid (no 4-APs) and increases 3-AP count:
        #       Do it.
        
        # To make this faster, we can precompute the impact of removing x.
        # But the validity of adding y depends on the new set.
        
        # Given the time constraint, let's just do a limited number of random swaps.
        for _ in range(100): # Limit swaps per pass
            if time.time() > deadline:
                break
            
            # Pick a random x to remove
            if not A_set:
                break
            x = random.choice(list(A_set))
            
            # Pick a random y to add
            y = random.randint(1, N)
            if y in A_set:
                continue
                
            # Check if removing x and adding y is valid
            # New set: A_set - {x} + {y}
            # We need to check if y creates a 4-AP with (A_set - {x}).
            
            # Check validity of adding y to A_temp = A_set - {x}
            A_temp = A_set.copy()
            A_temp.remove(x)
            A_temp.add(y)
            
            # Check 4-APs for y
            valid = True
            for idx in AP4_MEM[y]:
                ap = ALL_4APS[idx]
                others_in_A_temp = 0
                for z in ap:
                    if z != y:
                        if z in A_temp:
                            others_in_A_temp += 1
                        else:
                            break
                if others_in_A_temp == 3:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Calculate new 3-AP count
            # This is expensive if we do it from scratch.
            # We can estimate the change.
            # But for simplicity and correctness, let's just calculate if the change is positive.
            # Actually, _count_3aps_bitset is fast enough for N=256?
            # ~2000 3-APs. 2000 checks per swap. 100 swaps = 200k checks. Very fast.
            
            new_mask = 0
            for z in A_temp:
                new_mask |= (1 << (z - 1))
            
            new_count = _count_3aps_bitset(new_mask)
            
            if new_count > current_count:
                A_set = A_temp
                A_mask = new_mask
                current_count = new_count
                improved = True
                break # Restart the search with the new set
        
    return sorted(A_set)

def construct():
    start_time = time.time()
    deadline = start_time + 60.0 - DEADLINE_MARGIN
    
    # 1. Greedy Construction
    A_greedy, _ = _greedy_construct(deadline)
    
    # 2. Local Search
    A_ls = _local_search(A_greedy, deadline)
    
    # 3. Verify and Return
    # Ensure no 4-APs
    A_set = set(A_ls)
    for a in A_ls:
        for d in range(1, N + 1):
            if (a+d) in A_set and (a+2*d) in A_set and (a+3*d) in A_set:
                # Fallback to a known valid set (e.g. just [1]) if error
                return [1]
                
    return A_ls
