import time
import random

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s buffer for safety

    # Precompute conflicts for each x
    # conflicts[x] is a list of tuples (a, b, c) such that {a, b, c, x} is a 4-AP
    # Indices are 0-based for bitset convenience, so element i corresponds to bit i.
    # Problem elements are 1..512, so we map element v to bit index v-1.
    
    conflicts = [[] for _ in range(N)]
    
    # Generate all 4-APs in [1, N]
    # a, a+d, a+2d, a+3d
    for a in range(1, N + 1):
        for d in range(1, N):
            b = a + d
            if b > N:
                break
            c = b + d
            if c > N:
                break
            e = c + d
            if e > N:
                break
            
            # Found AP: a, b, c, e
            # For a: others are b, c, e
            conflicts[a - 1].append((b - 1, c - 1, e - 1))
            # For b: others are a, c, e
            conflicts[b - 1].append((a - 1, c - 1, e - 1))
            # For c: others are a, b, e
            conflicts[c - 1].append((a - 1, b - 1, e - 1))
            # For e: others are a, b, c
            conflicts[e - 1].append((a - 1, b - 1, c - 1))

    def can_add(x_idx, S_bitset):
        # x_idx is 0-based index
        # Check if any conflict triple is fully contained in S_bitset
        for a, b, c in conflicts[x_idx]:
            # Check if bits a, b, c are set in S_bitset
            if (S_bitset >> a) & 1 and (S_bitset >> b) & 1 and (S_bitset >> c) & 1:
                return False
        return True

    def greedy_build(order):
        S_bitset = 0
        S_list = [] # Keep a list of indices in S for local search
        for x_idx in order:
            if can_add(x_idx, S_bitset):
                S_bitset |= (1 << x_idx)
                S_list.append(x_idx)
        return S_bitset, S_list

    def local_search(S_bitset, S_list):
        # Try to add any element not in S
        # Then try swaps
        S_set = set(S_list)
        changed = True
        while changed:
            changed = False
            # 1. Try to add
            for x_idx in range(N):
                if x_idx not in S_set:
                    if can_add(x_idx, S_bitset):
                        S_bitset |= (1 << x_idx)
                        S_list.append(x_idx)
                        S_set.add(x_idx)
                        changed = True
                        # Restart the loop to check if more can be added
                        break
            
            if not changed:
                # 2. Try swaps: remove y, add x
                # We look for x not in S, y in S, such that:
                # - x can be added if y is removed
                # - This doesn't decrease size (it stays same, but might enable further adds)
                # Actually, standard local search for Max Independent Set:
                # If we can't add, we try to swap.
                # A swap is beneficial if it allows adding more later, or just to move to a different local optimum.
                # Simple heuristic: try all pairs (x, y) where x not in S, y in S.
                # Check if S \ {y} U {x} is valid.
                # This is O(|S| * (N-|S|)) which is ~ 200*300 = 60k checks.
                # Each check involves removing y (temporarily) and checking can_add(x).
                # can_add is O(K) ~ 170. So 60k * 170 ~ 10M ops. Might be too slow per iteration.
                # We will limit swaps to a few random attempts or a specific heuristic.
                
                # Heuristic: Only try swaps if we've been stuck for a while?
                # Or just try a few random swaps.
                # Let's try a limited number of random swaps to escape local optima.
                if random.random() < 0.5:
                    # Pick a random x not in S
                    x_candidates = [i for i in range(N) if i not in S_set]
                    if x_candidates:
                        x_idx = random.choice(x_candidates)
                        # Check if x is blocked by some y in S
                        # If we remove y, is x addable?
                        # We need to find a y in S such that S \ {y} U {x} is valid.
                        # This is equivalent to: x is blocked by some triple T in conflicts[x].
                        # If T contains y, and T \ {y} is NOT a subset of S \ {y} (which is true if y is the only one from T in S, or if removing y breaks all blocking triples), then x can be added.
                        
                        # Simpler: Just try removing each y in S one by one? No, too slow.
                        # Let's just try to remove a random y and see if x becomes addable.
                        y_idx = random.choice(S_list)
                        
                        # Temporarily remove y
                        old_bit = (1 << y_idx)
                        S_bitset_no_y = S_bitset & ~old_bit
                        
                        if can_add(x_idx, S_bitset_no_y):
                            # Swap is valid
                            S_bitset = S_bitset_no_y | (1 << x_idx)
                            S_list.remove(y_idx)
                            S_list.append(x_idx)
                            S_set.remove(y_idx)
                            S_set.add(x_idx)
                            changed = True
                            # Don't break, keep looking for more swaps/adds
                            continue

        return S_bitset, S_list

    best_size = 0
    best_set = set()
    
    # Run multiple iterations
    while time.time() < deadline:
        order = list(range(N))
        random.shuffle(order)
        
        S_bitset, S_list = greedy_build(order)
        
        # Local search
        S_bitset, S_list = local_search(S_bitset, S_list)
        
        current_size = len(S_list)
        if current_size > best_size:
            best_size = current_size
            best_set = set(S_list)
            
        # Check time
        if time.time() > deadline - 5:
            break

    # Convert 0-based indices to 1-based values
    result = sorted([x + 1 for x in best_set])
    
    # Ensure non-empty
    if not result:
        return [1]
        
    return result

