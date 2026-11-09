import time
import random

N = 512
DEADLINE = 55.0  # Safety margin

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Precompute all 4-APs
    # A 4-AP is (a, a+d, a+2d, a+3d)
    # We store for each element x, the list of other elements that are in a 4-AP with x.
    # Actually, we need to update counts.
    # Let's precompute for each x, the list of (other1, other2, other3) triples such that (x, other1, other2, other3) is a 4-AP.
    # This is too big. 
    # Better: Precompute for each x, the list of all y such that x and y are in some 4-AP.
    # And for each pair (x, y), we don't need to know the specific AP, just that they are linked.
    # 
    # Let's use the "bad_count" approach.
    # bad_count[x] = number of 4-APs containing x such that the other 3 elements are in the current set S.
    # If bad_count[x] == 0, x can be added.
    
    # To update bad_count efficiently:
    # When we add u to S:
    #   For every 4-AP containing u, let the other 3 be v1, v2, v3.
    #   For each vi, we need to check if the other two in that AP are in S.
    #   This requires knowing the specific APs.
    
    # Precomputation:
    # aps_by_element[x] = list of tuples (y, z, w) such that {x, y, z, w} is a 4-AP.
    # Note: The order in the tuple doesn't matter for the count, but we need to know the set of other 3.
    
    aps_by_element = [[] for _ in range(N + 1)]
    
    # Generate all 4-APs
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            # The AP is (a, b, c, e)
            # For a: others are (b, c, e)
            aps_by_element[a].append((b, c, e))
            # For b: others are (a, c, e)
            aps_by_element[b].append((a, c, e))
            # For c: others are (a, b, e)
            aps_by_element[c].append((a, b, e))
            # For e: others are (a, b, c)
            aps_by_element[e].append((a, b, c))
            
    # State
    in_set = [False] * (N + 1)
    bad_count = [0] * (N + 1)
    S_size = 0
    
    def add(x):
        nonlocal S_size
        in_set[x] = True
        S_size += 1
        # Update bad_count for all elements in 4-APs with x
        for (y, z, w) in aps_by_element[x]:
            # If y, z, w are in S, then x completes an AP. 
            # But we are adding x, so we need to update the bad_count of y, z, w?
            # No. bad_count[y] counts APs containing y where the OTHER 3 are in S.
            # If we add x, does bad_count[y] change?
            # Yes, if x is one of the "other 3" for some AP containing y.
            # Specifically, consider an AP (y, x, z, w). If x, z, w are in S, then bad_count[y] increases by 1.
            # So, for each (y, z, w) in aps_by_element[x], we check if z and w are in S.
            # If z and w are in S, then the AP (x, y, z, w) now has 3 elements (x, z, w) in S.
            # So bad_count[y] increases by 1.
            if in_set[z] and in_set[w]:
                bad_count[y] += 1
            if in_set[y] and in_set[w]:
                bad_count[z] += 1
            if in_set[y] and in_set[z]:
                bad_count[w] += 1
            # Note: We do NOT update bad_count[x] here because bad_count[x] is the count of APs 
            # where the OTHER 3 are in S. Since we just added x, x is in S.
            # The definition is: bad_count[x] = # of APs containing x such that the other 3 are in S.
            # This value is determined by the state of the OTHER 3.
            # So bad_count[x] is already correct based on the previous state of S.
            # Wait. If I add x, bad_count[x] should be the number of APs where the other 3 are in S.
            # This number is fixed by the current S (before adding x, x was not in S, so bad_count[x] was 0? No.)
            # Let's re-define.
            # bad_count[x] is the number of 4-APs containing x where the other 3 elements are in S.
            # If x is NOT in S, bad_count[x] tells us how many APs would be formed if we added x.
            # If x IS in S, bad_count[x] should be 0 (because the set is 4-AP free, so no AP containing x can have the other 3 in S).
            # So, when we add x, we must ensure bad_count[x] was 0.
            # Then, for any y, z, w in an AP with x, we update their bad_counts.
            
    def remove(x):
        nonlocal S_size
        in_set[x] = False
        S_size -= 1
        # Reverse the updates
        for (y, z, w) in aps_by_element[x]:
            if in_set[z] and in_set[w]:
                bad_count[y] -= 1
            if in_set[y] and in_set[w]:
                bad_count[z] -= 1
            if in_set[y] and in_set[z]:
                bad_count[w] -= 1
            # bad_count[x] is not updated directly, but since x is removed, 
            # its bad_count doesn't matter for "can add" checks on x anymore.
            # However, for consistency, if we were to re-add x, we'd need to recompute or rely on it being 0.
            # Since we only add if bad_count[x]==0, and we just removed x, 
            # the bad_count[x] value is stale but irrelevant until x is considered again.
            # Actually, bad_count[x] should reflect the current S.
            # Since x is not in S, bad_count[x] is the number of APs where other 3 are in S.
            # This value was calculated before x was added? No, it was maintained.
            # When we added x, we didn't change bad_count[x].
            # When we remove x, we don't change bad_count[x].
            # So bad_count[x] remains the value it had when x was added (which was 0).
            # This is correct because if x was added, it was safe, so no AP was formed.
            # So bad_count[x] is 0.
            # After removal, is bad_count[x] still 0?
            # Yes, because removing x cannot create new APs for x. It can only break them.
            # So bad_count[x] stays 0.

    # Initialize with Greedy
    # To speed up, we can just pick elements in order or random order.
    # Let's use a fixed order for reproducibility of the seed, then local search.
    # Or random order for better initial spread?
    # The current program uses natural order. Let's stick to that for the seed, then improve.
    
    for x in range(1, N + 1):
        if bad_count[x] == 0:
            add(x)
            
    best_size = S_size
    best_set = [i for i in range(1, N + 1) if in_set[i]]
    
    rng = random.Random(42)
    
    # Local Search
    # 1. Try to add any element with bad_count 0
    # 2. If stuck, try a swap: remove r, add a.
    #    A swap is valid if after removing r, bad_count[a] becomes 0.
    #    bad_count[a] decreases by the number of APs (a, r, u, v) where u, v are in S.
    #    So if bad_count[a] == k, we need to remove r such that r is in k APs with a and the other 2 in S.
    #    Actually, we just need to find ANY r such that removing it reduces bad_count[a] to 0.
    #    This implies bad_count[a] must be small? No.
    #    If bad_count[a] = 1, and the only AP is (a, r, u, v) with u,v in S, then removing r makes bad_count[a] = 0.
    #    If bad_count[a] > 1, we might need to remove multiple elements, which is a multi-swap.
    #    For now, let's focus on 1-swap (remove 1, add 1).
    #    This only works if bad_count[a] == 1 and the blocker r is unique?
    #    Or if removing r drops bad_count[a] to 0.
    
    # Optimization: 
    # Maintain a list of candidates for addition (bad_count == 0).
    # If empty, look for (a, r) pairs.
    # To find pairs efficiently:
    # For each a with bad_count[a] > 0:
    #   For each AP (a, y, z, w) in aps_by_element[a]:
    #     If y, z, w are in S, then this is a blocking AP.
    #     The blockers are y, z, w.
    #     If we remove one of them, say y, does bad_count[a] become 0?
    #     Only if this was the ONLY blocking AP.
    #     So we should prioritize a with bad_count[a] == 1.
    
    blocked_elements = [x for x in range(1, N + 1) if not in_set[x]]
    
    while time.time() < deadline:
        # Phase 1: Add
        added = False
        # We can iterate through all elements or keep a list of zeros.
        # Iterating all is O(N). N=512 is small.
        for x in range(1, N + 1):
            if not in_set[x] and bad_count[x] == 0:
                add(x)
                if S_size > best_size:
                    best_size = S_size
                    best_set = [i for i in range(1, N + 1) if in_set[i]]
                added = True
                break # Restart scan after add? Or continue?
                # Restarting is safer to catch new zeros created? 
                # Adding x increases bad_count of others, it never decreases.
                # So no new zeros are created. 
                # So we can just continue the loop?
                # But we broke. Let's just continue the outer while loop.
                # Actually, if we add x, bad_count of others increases. 
                # So no new elements become addable.
                # So we can just scan the rest of the list?
                # But we already passed some. 
                # Since adding only increases bad_counts, any element we skipped with bad_count>0 
                # will still have bad_count>0.
                # So we can break and restart the while loop, which will scan from 1 again.
                # This is O(N^2) in worst case per full pass, but N is small.
                pass
        
        if added:
            continue
            
        # Phase 2: Swap
        # Find a swap (remove r, add a)
        # Strategy: Find a with bad_count[a] == 1.
        # Identify the blocker r.
        # Remove r. Add a.
        
        swap_found = False
        # Shuffle blocked elements to avoid bias
        # But we need to check bad_count.
        # Let's iterate through blocked elements.
        # To avoid O(N^2) scan of all blocked for bad_count==1, 
        # we can just scan all 1..N.
        
        for a in range(1, N + 1):
            if in_set[a]:
                continue
            if bad_count[a] != 1:
                continue
                
            # Find the blocker
            # bad_count[a] == 1 means there is exactly one AP (a, y, z, w) with y,z,w in S.
            # We need to find which one.
            blocker = -1
            for (y, z, w) in aps_by_element[a]:
                if in_set[y] and in_set[z] and in_set[w]:
                    # This is the blocking AP.
                    # Any of y, z, w can be the blocker?
                    # If we remove y, bad_count[a] becomes 0.
                    # If we remove z, bad_count[a] becomes 0.
                    # If we remove w, bad_count[a] becomes 0.
                    # We should choose the one that is "least useful" or random.
                    # Let's pick random among y, z, w.
                    candidates = [y, z, w]
                    r = rng.choice(candidates)
                    blocker = r
                    break
            
            if blocker != -1:
                # Perform swap: remove blocker, add a
                remove(blocker)
                add(a)
                if S_size > best_size:
                    best_size = S_size
                    best_set = [i for i in range(1, N + 1) if in_set[i]]
                swap_found = True
                break # Restart search after swap
                
        if not swap_found:
            # No simple swap found.
            # Try a random restart? Or a 2-swap?
            # 2-swap is expensive.
            # Let's try a random restart of the greedy part?
            # Or just break and return best?
            # Given the time limit, if we are stuck, we might try a few random restarts.
            
            # Random Restart
            # Clear set
            for x in range(1, N + 1):
                in_set[x] = False
                bad_count[x] = 0
            S_size = 0
            
            # Random Greedy
            candidates = list(range(1, N + 1))
            rng.shuffle(candidates)
            for x in candidates:
                if bad_count[x] == 0:
                    add(x)
            
            if S_size > best_size:
                best_size = S_size
                best_set = [i for i in range(1, N + 1) if in_set[i]]
                
            # Check time
            if time.time() > deadline - 5:
                break

    return best_set

# Quick test
if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    t1 = time.time()
    print(f"Size: {len(res)}, Time: {t1-t0:.2f}s")
