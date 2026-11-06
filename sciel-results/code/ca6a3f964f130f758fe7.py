import time
import random

def construct():
    """
    Finds the largest Sidon set (distinct pairwise sums a+b, a<=b) in [1, 284]
    containing {1, 2, 5, 11}.
    
    Combines beam search with bitset-based local search and multi-start greedy.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 115s limit to be safe
    
    prescribed = [1, 2, 5, 11]
    pset = set(prescribed)
    MAXV = 284
    MAX_SUM = 2 * MAXV  # 568
    
    # Bitset-based sum tracking for speed
    def sums_bitset(markers):
        bs = 0
        sl = sorted(markers)
        for i in range(len(sl)):
            x = sl[i]
            for y in sl[i:]:
                s = x + y
                bs |= (1 << s)
        return bs
    
    def can_add_bitset(x, cur_list, bs):
        # Check if adding x creates any duplicate sum
        for y in cur_list:
            s = x + y
            if (bs >> s) & 1:
                return False
        s2 = 2 * x
        if (bs >> s2) & 1:
            return False
        # Also need all new sums to be distinct from each other
        new_sums = set()
        for y in cur_list:
            new_sums.add(x + y)
        new_sums.add(2 * x)
        return len(new_sums) == len(cur_list) + 1
    
    def add_sums_bitset(x, cur_list, bs):
        for y in cur_list:
            bs |= (1 << (x + y))
        bs |= (1 << (2 * x))
        return bs
    
    def remove_sums_bitset(x, cur_list, bs):
        # cur_list does NOT include x
        for y in cur_list:
            bs &= ~(1 << (x + y))
        bs &= ~(1 << (2 * x))
        return bs
    
    def is_valid_add(x, S, used_sums):
        if x in S:
            return False
        for y in S:
            if x + y in used_sums:
                return False
        if 2 * x in used_sums:
            return False
        return True
    
    def get_sums(S):
        sums = set()
        sl = sorted(S)
        for i in range(len(sl)):
            for j in range(i, len(sl)):
                sums.add(sl[i] + sl[j])
        return sums

    # Initial state
    S0 = set(prescribed)
    sums0 = get_sums(S0)
    
    best_S = list(S0)
    best_size = len(S0)
    
    # Multi-start greedy to find good initial solutions
    rng = random.Random(123)
    allc = list(range(1, MAXV + 1))
    remaining = [x for x in range(1, MAXV + 1) if x not in pset]
    
    # Try several randomized greedy starts
    for restart in range(20):
        if time.time() > deadline - 60:
            break
        
        cur = set(prescribed)
        cur_list = list(prescribed)
        bs = sums_bitset(cur)
        order = remaining[:]
        rng.shuffle(order)
        for x in order:
            if x in cur:
                continue
            if can_add_bitset(x, cur_list, bs):
                cur.add(x)
                cur_list.append(x)
                bs = add_sums_bitset(x, cur_list[:-1], bs)
        
        if len(cur) > best_size:
            best_size = len(cur)
            best_S = list(cur)
    
    # Beam Search Phase
    beam_width = 80
    max_depth = 20
    rng_beam = random.Random(456)
    
    beam = [(S0, sums0)]
    
    for depth in range(max_depth):
        if time.time() > deadline - 40:  # Leave 40s for local search
            break
            
        next_beam = []
        seen_states = set()
        
        for S, used_sums in beam:
            valid_candidates = []
            for x in range(1, MAXV + 1):
                if x in S:
                    continue
                if is_valid_add(x, S, used_sums):
                    valid_candidates.append(x)
            
            if not valid_candidates:
                continue
                
            # Sample candidates to expand
            if len(valid_candidates) > 8:
                candidates_to_try = rng_beam.sample(valid_candidates, 8)
            else:
                candidates_to_try = valid_candidates
                
            for x in candidates_to_try:
                new_S = S | {x}
                new_sums = used_sums | {x + y for y in S} | {2 * x}
                
                key = tuple(sorted(new_S))
                if key in seen_states:
                    continue
                seen_states.add(key)
                
                next_beam.append((new_S, new_sums))
                
                if len(new_S) > best_size:
                    best_size = len(new_S)
                    best_S = list(new_S)
        
        if not next_beam:
            break
            
        if len(next_beam) > beam_width:
            # Score by number of remaining options
            def score_state(state):
                S, used_sums = state
                count = 0
                for x in range(1, MAXV + 1):
                    if x not in S and is_valid_add(x, S, used_sums):
                        count += 1
                return count
            
            next_beam.sort(key=score_state, reverse=True)
            beam = next_beam[:beam_width]
        else:
            beam = next_beam

    # Local Search Phase using bitsets for speed
    cur = set(best_S)
    cur_list = list(cur)
    bs = sums_bitset(cur)
    cur_size = len(cur)
    best = set(cur)
    best_sz = len(cur)
    
    ls_rng = random.Random(789)
    ls_deadline = deadline - 2.0
    
    # Intensive local search with add and swap operations
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < ls_deadline:
        if no_improve_count > max_no_improve:
            break
        
        improved = False
        
        # Try to add elements
        for _t in range(30):
            x = ls_rng.choice(allc)
            if x not in cur and can_add_bitset(x, cur_list, bs):
                cur.add(x)
                cur_list.append(x)
                bs = add_sums_bitset(x, cur_list[:-1], bs)
                cur_size += 1
                if cur_size > best_sz:
                    best_sz = cur_size
                    best = set(cur)
                    best_S = list(cur)
                improved = True
                break
        
        # Try swaps if no improvement from add
        if not improved:
            removable = [x for x in cur if x not in pset]
            if removable:
                y = ls_rng.choice(removable)
                temp_S = cur - {y}
                temp_list = [x for x in cur_list if x != y]
                temp_bs = remove_sums_bitset(y, temp_list, bs)
                
                # Try to find a valid addition to the reduced set
                for _t in range(20):
                    x = ls_rng.choice(allc)
                    if x in temp_S:
                        continue
                    if can_add_bitset(x, temp_list, temp_bs):
                        cur = temp_S | {x}
                        cur_list = temp_list + [x]
                        bs = add_sums_bitset(x, temp_list, temp_bs)
                        # Size remains same, but configuration changed
                        if len(cur) > best_sz:
                            best_sz = len(cur)
                            best = set(cur)
                            best_S = list(cur)
                        improved = True
                        break
        
        if not improved:
            no_improve_count += 1
        else:
            no_improve_count = 0

    result = sorted(best)
    # Final validation
    sl = result
    seen = set()
    valid = True
    for i in range(len(sl)):
        for j in range(i, len(sl)):
            s = sl[i] + sl[j]
            if s in seen:
                valid = False
                break
            seen.add(s)
        if not valid:
            break
    
    if not valid or not pset.issubset(result):
        return prescribed
    
    return result
