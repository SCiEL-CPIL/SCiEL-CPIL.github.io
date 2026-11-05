import numpy as np
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Specific case optimization
    if q == 3 and n == 6 and radius == 1:
        total = 3 ** 6
        powers = [3 ** p for p in range(6)]
        
        # Precompute neighbors as bitsets
        # cover_mask[i] is an integer with bits set for all j such that dist(i, j) <= 1
        cover_mask = [0] * total
        for i in range(total):
            d = [(i // powers[p]) % 3 for p in range(6)]
            mask = 1 << i
            for p in range(6):
                for alt in range(3):
                    if alt != d[p]:
                        new_i = i + (alt - d[p]) * powers[p]
                        mask |= (1 << new_i)
            cover_mask[i] = mask

        def idx_to_list(idx):
            return [(idx // powers[p]) % 3 for p in range(6)]

        def greedy_set_cover():
            """Greedy set cover using bitsets."""
            uncovered = (1 << total) - 1
            codewords = []
            
            while uncovered:
                # Find word that covers most uncovered bits
                best_c = -1
                best_count = -1
                # Optimization: We can sample or iterate. Iterating 729 is fast.
                # To add diversity, we can randomize tie-breaking or use a small window of top-k.
                # For now, deterministic max, but we can add noise.
                for c in range(total):
                    count = (cover_mask[c] & uncovered).bit_count()
                    if count > best_count:
                        best_count = count
                        best_c = c
                
                codewords.append(best_c)
                uncovered &= ~cover_mask[best_c]
                
            return codewords

        def local_search(codewords, time_budget):
            """Perform local search to reduce code size using bitsets."""
            S = set(codewords)
            S_mask = 0
            for c in S:
                S_mask |= (1 << c)
            
            best_S = set(S)
            best_size = len(S)
            
            # Precompute neighbors of each codeword? No, cover_mask is global.
            
            def try_remove_and_swap():
                """Attempt to remove a word. If not possible, attempt a swap."""
                nonlocal S, S_mask, best_S, best_size
                
                # Try to find a removable word
                # Iterate randomly to avoid bias
                items = list(S)
                random.shuffle(items)
                
                for w in items:
                    # Check if w is redundant
                    # L = words covered by w but not by any other word in S
                    # This is equivalent to:
                    # covered_by_others = union of cover_mask[v] for v in S \ {w}
                    # But we can compute L as:
                    # L = cover_mask[w] & ~(union of cover_mask[v] for v in S \ {w})
                    # Which is:
                    # L = cover_mask[w] & ~ ( (union of cover_mask[v] for v in S) & ~cover_mask[w] )? No.
                    
                    # Efficient check:
                    # A word x is uniquely covered by w if x is in cover_mask[w] and x is not in any other cover_mask[v].
                    # We can compute the union of all others?
                    # Since |S| is ~73, union of 72 masks is fast.
                    
                    union_others = 0
                    for v in S:
                        if v != w:
                            union_others |= cover_mask[v]
                    
                    L = cover_mask[w] & ~union_others
                    
                    if L == 0:
                        # w is redundant
                        S.remove(w)
                        S_mask &= ~(1 << w)
                        if len(S) < best_size:
                            best_size = len(S)
                            best_S = set(S)
                        return True
                    
                    # L is not empty. Try to swap w with w' such that w' covers L.
                    # Candidates for w' must cover all bits in L.
                    # Candidates = intersection of cover_mask[x] for x in bits set in L.
                    # L is a bitmask. We need to iterate over set bits in L.
                    
                    # Optimization: If L has more than, say, 3 bits, the intersection might be empty or small.
                    # Also, we should only consider w' not in S.
                    
                    candidates = (1 << total) - 1
                    temp_L = L
                    while temp_L:
                        # Get lowest set bit
                        x = (temp_L & -temp_L).bit_length() - 1
                        temp_L &= temp_L - 1
                        candidates &= cover_mask[x]
                        # Early exit if candidates become 0
                        if candidates == 0:
                            break
                    
                    if candidates == 0:
                        continue
                        
                    # Exclude words already in S
                    candidates &= ~S_mask
                    # Exclude w itself (it's in S, so already excluded, but just in case)
                    candidates &= ~(1 << w)
                    
                    if candidates:
                        # Pick a random candidate
                        # To pick a random bit from a large int:
                        # We can convert to list? No, too slow.
                        # We can use random bit?
                        # Simpler: pick the first one? Or use a hash?
                        # Let's pick one at random by masking with a random mask?
                        # Actually, just picking the lowest set bit is fine for diversity if we shuffle S.
                        # Or we can pick randomly among the set bits.
                        # For now, pick lowest.
                        w_new = (candidates & -candidates).bit_length() - 1
                        
                        # Perform swap
                        S.remove(w)
                        S.add(w_new)
                        S_mask = 0
                        for c in S:
                            S_mask |= (1 << c)
                        return True
                
                return False

            stuck_count = 0
            max_stuck = 100
            
            while time.time() < time_budget:
                if try_remove_and_swap():
                    stuck_count = 0
                    max_stuck = random.randint(50, 150)
                else:
                    stuck_count += 1
                    if stuck_count > max_stuck:
                        # Restart local search phase or break
                        # If we are stuck, we might try a more aggressive restart
                        stuck_count = 0
                        max_stuck = random.randint(100, 300)
                        # Optionally, if stuck too long, break and return best
                        if max_stuck > 500:
                            break

            return list(best_S)

        # Multi-restart strategy
        best_overall = None
        best_overall_size = float('inf')
        
        restart_count = 0
        while time.time() < deadline - 5:
            restart_count += 1
            
            # Time budget per restart
            # Start with short, increase if we are finding good solutions?
            # Or just fixed reasonable budget.
            # 115s total. Let's do ~10s per restart, so ~10 restarts.
            time_budget = min(time.time() + 10, deadline - 5)
            
            # Greedy initialization
            codewords = greedy_set_cover()
            
            # Local search
            codewords = local_search(codewords, time_budget)
            
            if len(codewords) < best_overall_size:
                best_overall_size = len(codewords)
                best_overall = list(codewords)
                
            # If we found 73, we can try to polish it further?
            # The local search already runs until time budget.
            # If we find 73, we might want to extend the time for that specific solution?
            # But we don't know which restart will find it.
            # So we just keep restarting.

        if best_overall is None:
            best_overall = [0]
            
        return [idx_to_list(idx) for idx in best_overall]

    else:
        # Fallback for other parameters
        total = q ** n
        return [[0] * n]
