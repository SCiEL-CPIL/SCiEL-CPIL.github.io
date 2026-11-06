import random
import time
import numpy as np

def construct():
    N = 1000
    max_sum = 2 * N
    
    # Precompute bit positions for fast set operations
    # We'll use a numpy boolean array for membership and a count array
    # For O(1) validity: maintain a "blocked" set of values that cannot be added
    # A value x is blocked if counts[2*x] >= 2 OR exists y in S with counts[x+y] >= 2
    
    # Strategy: 
    # - Maintain counts array (numpy int8) of size max_sum+1
    # - Maintain S as a set and S_bits as a Python int (bitset)
    # - Maintain BadSums as a Python int (bitset) where bit k is 1 if counts[k] >= 2
    # - can_add(x): check counts[2*x] < 2 AND (S_bits & (BadSums >> x)) == 0
    # - When adding x: update counts[2*x] and counts[x+y] for all y in S
    #   But updating counts[x+y] for all y in S is O(|S|) which we want to avoid.
    
    # Alternative: Use the fact that we only need to know if counts[s] reaches 2.
    # Maintain:
    # - counts1: bitset of sums with count exactly 1
    # - counts2plus: bitset of sums with count >= 2
    # When adding x, for each y in S, sum s = x+y:
    #   if s was in counts1, move to counts2plus
    #   if s was in counts2plus, stays
    #   if s was 0, add to counts1
    # This still requires iterating over S.
    
    # Better approach: Accept O(|S|) for updates but make them very fast with numpy.
    # Or: Use a different data structure.
    
    # Let's try: maintain counts as numpy array, and for can_add, 
    # we need to check if any (x+y) for y in S has count >= 2.
    # This is equivalent to checking if the set {x+y : y in S} intersects with BadSums.
    # If we represent S as a bitset, then {x+y : y in S} is BadSums shifted... no.
    # {x+y : y in S} has bit (x+y) set iff bit y is set in S_bits.
    # So we want to know if (S_bits << x) & BadSums != 0.
    # This is O(1) with Python big integers!
    
    # So the plan:
    # - S_bits: Python int, bit i set iff i in S
    # - counts: numpy array of int8, size max_sum+1
    # - BadSums: Python int, bit k set iff counts[k] >= 2
    # 
    # can_add(x):
    #   if x in S: return False
    #   if counts[2*x] >= 2: return False
    #   if (S_bits << x) & BadSums != 0: return False
    #   return True
    #
    # add(x):
    #   S.add(x)
    #   S_bits |= (1 << x)
    #   # Update counts for 2*x
    #   if counts[2*x] == 1:
    #       BadSums |= (1 << (2*x))
    #   counts[2*x] += 1
    #   # Update counts for x+y for y in S_old
    #   # We need to iterate over S_old (S before adding x)
    #   for y in list(S):
    #       if y == x: continue
    #       s = x + y
    #       if counts[s] == 1:
    #           BadSums |= (1 << s)
    #       counts[s] += 1
    #
    # remove(x):
    #   S.remove(x)
    #   S_bits &= ~(1 << x)
    #   # Update counts for 2*x
    #   counts[2*x] -= 1
    #   if counts[2*x] == 1:
    #       BadSums &= ~(1 << (2*x))
    #   # Update counts for x+y for y in S (after removing x)
    #   for y in S:
    #       s = x + y
    #       counts[s] -= 1
    #       if counts[s] == 1:
    #           BadSums &= ~(1 << s)
    
    # The O(|S|) loop in add/remove is the bottleneck. Let's optimize it.
    # We can use numpy to update counts in bulk? 
    # For add(x): we need to increment counts[x+y] for all y in S_old.
    # If we have S as a numpy array, we can do:
    #   sums = x + S_array
    #   counts[sums] += 1
    #   # Then update BadSums for sums where counts went from 1 to 2
    #   newly_bad = sums[counts[sums] == 2]
    #   for s in newly_bad: BadSums |= (1 << s)
    # This is O(|S|) but in C, so faster.
    
    # Let's implement with numpy for the inner loop.
    
    start_time = time.time()
    time_limit = 115
    
    best_S = set()
    best_size = 0
    
    def greedy_random(order):
        S = set()
        S_array = np.array([], dtype=np.int32)
        S_bits = 0
        counts = np.zeros(max_sum + 1, dtype=np.int8)
        BadSums = 0
        
        for x in order:
            if x in S:
                continue
            # Check validity
            if counts[2 * x] >= 2:
                continue
            if (S_bits << x) & BadSums != 0:
                continue
            
            # Add x
            S.add(x)
            S_array = np.append(S_array, x)
            S_bits |= (1 << x)
            
            # Update counts
            # 2*x
            if counts[2 * x] == 1:
                BadSums |= (1 << (2 * x))
            counts[2 * x] += 1
            
            # x + y for y in S_old (S_array before adding x)
            if len(S_array) > 1:
                old_S = S_array[:-1]
                sums = x + old_S
                # Find which sums had count 1 before
                prev_counts = counts[sums]
                counts[sums] += 1
                # Update BadSums for sums that went from 1 to 2
                newly_bad = sums[prev_counts == 1]
                for s in newly_bad:
                    BadSums |= (1 << int(s))
        
        return S, len(S)
    
    # Phase 1: Greedy with random orders
    for i in range(50):
        if time.time() - start_time > time_limit - 30:
            break
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        S, size = greedy_random(candidates)
        if size > best_size:
            best_size = size
            best_S = set(S)
    
    # Phase 2: Local search
    S = set(best_S)
    S_array = np.array(list(S), dtype=np.int32)
    S_bits = 0
    for x in S:
        S_bits |= (1 << x)
    
    counts = np.zeros(max_sum + 1, dtype=np.int8)
    BadSums = 0
    # Rebuild counts
    for i in range(len(S_array)):
        a = S_array[i]
        # 2*a
        if counts[2 * a] == 1:
            BadSums |= (1 << (2 * a))
        counts[2 * a] += 1
        # a + b for b in S_array[i+1:]
        if i + 1 < len(S_array):
            sums = a + S_array[i + 1:]
            prev_counts = counts[sums]
            counts[sums] += 1
            newly_bad = sums[prev_counts == 1]
            for s in newly_bad:
                BadSums |= (1 << int(s))
    
    def can_add(x):
        if x in S:
            return False
        if counts[2 * x] >= 2:
            return False
        if (S_bits << x) & BadSums != 0:
            return False
        return True
    
    def add(x):
        nonlocal S_bits, BadSums, S_array
        S.add(x)
        S_array = np.append(S_array, x)
        S_bits |= (1 << x)
        
        if counts[2 * x] == 1:
            BadSums |= (1 << (2 * x))
        counts[2 * x] += 1
        
        if len(S_array) > 1:
            old_S = S_array[:-1]
            sums = x + old_S
            prev_counts = counts[sums]
            counts[sums] += 1
            newly_bad = sums[prev_counts == 1]
            for s in newly_bad:
                BadSums |= (1 << int(s))
    
    def remove(x):
        nonlocal S_bits, BadSums, S_array
        S.remove(x)
        # Remove x from S_array
        idx = np.where(S_array == x)[0][0]
        S_array = np.delete(S_array, idx)
        S_bits &= ~(1 << x)
        
        counts[2 * x] -= 1
        if counts[2 * x] == 1:
            BadSums &= ~(1 << (2 * x))
        
        if len(S_array) > 0:
            sums = x + S_array
            counts[sums] -= 1
            newly_good = sums[counts[sums] == 1]
            for s in newly_good:
                BadSums &= ~(1 << int(s))
    
    # Local search: try to add or swap
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try random additions
        for _ in range(100):
            x = random.randint(1, N)
            if can_add(x):
                add(x)
                improved = True
                break
        
        # Try random swaps
        if not improved and len(S) > 0:
            y = random.choice(list(S))
            x = random.randint(1, N)
            if x != y and x not in S:
                # Temporarily remove y
                remove(y)
                if can_add(x):
                    add(x)
                    improved = True
                else:
                    add(y)
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
        else:
            no_improve_count = 0
    
    return list(S)
