import time
import random

# Precompute all 4-term arithmetic progressions in [1, 512]
N = 512
aps = []
aps_by_elem = {i: [] for i in range(1, N + 1)}

for d in range(1, (N - 3) // 3 + 1):
    for a in range(1, N - 3 * d + 1):
        b = a + d
        c = a + 2 * d
        e = a + 3 * d
        ap = (a, b, c, e)
        aps.append(ap)
        for x in ap:
            aps_by_elem[x].append(ap)

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin
    
    # 1. Initial Greedy Construction
    S = set()
    for x in range(1, N + 1):
        valid = True
        # Check if adding x creates a 4-AP
        # We can check against existing S
        # For each AP containing x, if other 3 in S, invalid
        for ap in aps_by_elem[x]:
            others = [y for y in ap if y != x]
            if all(y in S for y in others):
                valid = False
                break
        if valid:
            S.add(x)
            
    # Initialize bad counts
    # bad[x] = number of 4-APs containing x where other 3 are in S
    bad = [0] * (N + 1)
    for x in range(1, N + 1):
        if x in S:
            continue
        count = 0
        for ap in aps_by_elem[x]:
            others = [y for y in ap if y != x]
            if all(y in S for y in others):
                count += 1
        bad[x] = count
        
    def add_element(x):
        S.add(x)
        # Update bad for neighbors
        for ap in aps_by_elem[x]:
            others = [y for y in ap if y != x]
            # For each other y, if the remaining 2 are in S, bad[y] increases
            # The AP is (x, y, u, v). For y, the others are (x, u, v).
            # Before add(x): x not in S.
            # After add(x): x in S.
            # If u, v in S, then bad[y] increases by 1.
            for y in others:
                # Check if the other two (excluding x and y) are in S
                # ap has 4 elements. x is one. y is one.
                # remaining = ap - {x, y}
                rem = [z for z in ap if z != x and z != y]
                if all(z in S for z in rem):
                    bad[y] += 1

    def remove_element(x):
        S.remove(x)
        # Update bad for neighbors (reverse of add)
        for ap in aps_by_elem[x]:
            others = [y for y in ap if y != x]
            for y in others:
                rem = [z for z in ap if z != x and z != y]
                if all(z in S for z in rem):
                    bad[y] -= 1

    # 2. Local Search
    # We want to maximize |S|.
    # Add if possible. If not, swap to unblock.
    
    # To make it faster, we can maintain a list of "available" elements (bad[x]==0)
    # But updating it dynamically is tricky. Given N=512, scanning is fast.
    
    while time.time() - start_time < time_limit:
        # Try to add
        added = False
        # Shuffle candidates to avoid bias
        candidates = [x for x in range(1, N + 1) if x not in S and bad[x] == 0]
        if candidates:
            random.shuffle(candidates)
            x = candidates[0]
            add_element(x)
            added = True
            
        if not added:
            # Try to swap
            # Pick a random a in S
            # Pick a random b not in S
            # Check if swap is valid
            # Valid swap: (S \ {a}) U {b} is 4-AP free.
            # This is equivalent to: b has no conflict with S \ {a}.
            # Conflict for b: exists ap containing b such that (ap \ {b}) subset of S \ {a}.
            
            # To speed up, we can try a few random swaps
            swapped = False
            s_list = list(S)
            random.shuffle(s_list)
            not_s_list = [x for x in range(1, N + 1) if x not in S]
            random.shuffle(not_s_list)
            
            # Limit swaps attempts per iteration to avoid wasting time
            # If no swap found after many tries, we are stuck.
            # For now, try a few.
            for _ in range(20): # Try 20 random swaps
                if time.time() - start_time > time_limit:
                    break
                a = random.choice(s_list)
                b = random.choice(not_s_list)
                
                # Check validity of swapping a -> b
                # Remove a from S temporarily? No, just check condition.
                # Condition: For all ap in aps_by_elem[b]:
                #   others = ap \ {b}
                #   if a in others:
                #       rem = others \ {a}
                #       if all(z in S for z in rem): conflict
                #   else:
                #       if all(z in S for z in others): conflict
                
                valid_swap = True
                for ap in aps_by_elem[b]:
                    others = [y for y in ap if y != b]
                    if a in others:
                        rem = [y for y in others if y != a]
                        if all(y in S for y in rem):
                            valid_swap = False
                            break
                    else:
                        if all(y in S for y in others):
                            valid_swap = False
                            break
                
                if valid_swap:
                    remove_element(a)
                    add_element(b)
                    swapped = True
                    break # Do one swap, then re-evaluate additions
                    
            if not swapped:
                # Stuck. Try a random restart or perturbation?
                # Or just break if we've been stuck for a while?
                # Let's try a random perturbation: remove a random element from S
                # This reduces size, but might allow large additions.
                # Risky. Let's just break or continue trying?
                # If we continue trying random swaps, we might find one eventually.
                # But if the local optimum is a plateau, we might loop.
                # Let's add a "stuck" counter.
                pass

    return sorted(S)
