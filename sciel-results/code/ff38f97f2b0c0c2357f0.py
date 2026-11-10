import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Leave 5s for safety
    
    def check_disjoint(sums, max_k):
        """Check if sums[0..max_k] are pairwise disjoint."""
        for k in range(max_k + 1):
            if sums[k] == 0:
                continue
            for j in range(k + 1, max_k + 1):
                if sums[j] == 0:
                    continue
                if (sums[k] & sums[j]) != 0:
                    return False
        return True

    def try_add(sums, x, max_k):
        """
        Attempt to add x to the set represented by sums.
        Returns new sums if valid, else None.
        """
        # A_k = sums[k-1] << x
        # Check conditions:
        # 1. old_sums[k] & A_j == 0 for k != j
        # 2. A_k & old_sums[j] == 0 for k != j
        # 3. A_k & A_j == 0 for k < j
        
        new_sums = [0] * (max_k + 2)
        new_sums[0] = sums[0]
        
        # Calculate A_k
        A = [0] * (max_k + 1)
        for k in range(1, max_k + 1):
            A[k] = sums[k-1] << x
            
        # Check 1 & 2: Interactions between old and new
        # For each k (new cardinality), A_k must not intersect old_sums[j] for j != k
        # And old_sums[k] must not intersect A_j for j != k
        
        # Efficient check:
        # For each j, A_j must be disjoint from all old_sums[k] where k != j.
        # Let combined_old_excluding_j = OR of all old_sums[k] for k != j.
        # But we can just loop.
        
        valid = True
        for k in range(1, max_k + 1):
            if A[k] == 0:
                continue
            # Check A[k] against old_sums[j] for j != k
            for j in range(0, max_k + 1):
                if j == k:
                    continue
                if sums[j] == 0:
                    continue
                if (A[k] & sums[j]) != 0:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            return None
            
        # Check 3: A_k & A_j == 0 for k < j
        for k in range(1, max_k + 1):
            if A[k] == 0:
                continue
            for j in range(k + 1, max_k + 1):
                if A[j] == 0:
                    continue
                if (A[k] & A[j]) != 0:
                    valid = False
                    break
            if not valid:
                break
                
        if not valid:
            return None

        # Construct new sums
        for k in range(1, max_k + 1):
            new_sums[k] = sums[k] | A[k]
        new_sums[max_k + 1] = A[max_k + 1] if max_k + 1 <= max_k else 0 # A is size max_k+1, indices 1..max_k
        # Wait, A is calculated for k=1..max_k.
        # new_sums[k] = sums[k] | A[k] for k=1..max_k
        # new_sums[max_k+1] = A[max_k+1]? No, A[max_k+1] would be sums[max_k] << x.
        # My loop for A was 1..max_k.
        # I need A for k=1..max_k+1?
        # If we add x to a set of size max_k, the new max cardinality is max_k+1.
        # Sums of cardinality k (1..max_k) get new contributions from subsets of size k-1 (old).
        # Sums of cardinality max_k+1 come from subsets of size max_k (old).
        
        # Let's re-calculate A properly.
        # A[k] = sums[k-1] << x for k=1..max_k+1
        # Note: sums has indices 0..max_k.
        
        A = [0] * (max_k + 2)
        for k in range(1, max_k + 2):
            if k-1 <= max_k:
                A[k] = sums[k-1] << x
            else:
                A[k] = 0
                
        # Re-check with correct A size
        valid = True
        for k in range(1, max_k + 2):
            if A[k] == 0:
                continue
            for j in range(0, max_k + 1): # j is old cardinality
                if j == k:
                    continue
                if sums[j] == 0:
                    continue
                if (A[k] & sums[j]) != 0:
                    valid = False
                    break
            if not valid:
                break
            
        if not valid:
            return None
            
        for k in range(1, max_k + 2):
            if A[k] == 0:
                continue
            for j in range(k + 1, max_k + 2):
                if A[j] == 0:
                    continue
                if (A[k] & A[j]) != 0:
                    valid = False
                    break
            if not valid:
                break
                
        if not valid:
            return None

        for k in range(1, max_k + 2):
            if k <= max_k:
                new_sums[k] = sums[k] | A[k]
            else:
                new_sums[k] = A[k]
                
        return new_sums

    def backtrack(current_set, sums, max_k, target_size, deadline):
        if len(current_set) == target_size:
            return current_set[:]
        
        if time.time() > deadline:
            return None
            
        # Heuristic: Try numbers that are likely to work
        # Start from 1, but skip if already in set
        used = set(current_set)
        
        # To speed up, we can restrict the search space.
        # For large sets, numbers tend to be large.
        # But small numbers are needed for low cardinalities.
        
        for x in range(1, 257):
            if x in used:
                continue
            if time.time() > deadline:
                return None
                
            new_sums = try_add(sums, x, max_k)
            if new_sums is not None:
                res = backtrack(current_set + [x], new_sums, max_k + 1, target_size, deadline)
                if res is not None:
                    return res
        return None

    # Try to find a set of size 11
    # Initialize with empty set
    initial_sums = [0] * 12
    initial_sums[0] = 1 # sum 0
    
    best_set = []
    
    # Fallback: Powers of 2 (Size 9)
    pow2_set = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    # Verify pow2_set
    s = [0]*10
    s[0]=1
    for v in pow2_set:
       ns = [0]*10
       for k in range(9, 0, -1):
           ns[k] = s[k] | (s[k-1] << v)
       ns[0]=s[0]
       s=ns
    # Check disjoint
    valid_pow2 = True
    for k in range(10):
       for j in range(k+1, 10):
          if s[k] & s[j]:
             valid_pow2 = False
             break
    if valid_pow2:
       best_set = pow2_set
       
    # Try to find size 10
    # Backtracking might be slow. Use randomized restarts with limited depth?
    # Or just try backtracking with a time limit.
    
    if time.time() < deadline - 5:
        # Try size 10
        # To make backtracking faster, we can sort candidates?
        # Or use a heuristic order.
        # Let's try a few random restarts with shuffled order of x?
        # No, x is fixed 1..256.
        # We can shuffle the *choice* of x in the loop?
        # Backtracking explores x=1, then 2, etc.
        # If we shuffle the range, we might find a solution faster.
        
        rng = random.Random(42)
        for attempt in range(10):
            if time.time() > deadline - 5:
                break
            order = list(range(1, 257))
            rng.shuffle(order)
            
            # Modify backtrack to use custom order
            def backtrack_custom(current_set, sums, max_k, target_size, deadline, order, used_idx):
                if len(current_set) == target_size:
                    return current_set[:]
                if time.time() > deadline:
                    return None
                
                # Find next available x in order
                # This is inefficient if we scan the whole list.
                # Better: pass the list of remaining candidates.
                
                # For simplicity, stick to standard backtrack but with a time check.
                # Standard backtrack tries 1..256.
                pass
            
            # Let's just run the standard backtrack with a tight time limit
            # If it doesn't finish, we keep the best found.
            
            # Actually, let's implement a greedy with backtracking (beam search)
            pass

    # Given the complexity, I will rely on the fact that size 9 is guaranteed.
    # And I will try a greedy approach to see if we can get 10.
    
    def greedy_construct(order):
        sums = [0] * 61
        sums[0] = 1
        chosen = []
        max_k = 0
        for x in order:
            if time.time() > deadline:
                break
            new_sums = try_add(sums, x, max_k)
            if new_sums is not None:
                sums = new_sums
                max_k += 1
                chosen.append(x)
        return chosen

    # Try different greedy orders
    orders = [
        list(range(1, 257)),
        list(range(256, 0, -1)),
        [1, 2, 4, 8, 16, 32, 64, 128, 256] + [x for x in range(1, 257) if x not in {1,2,4,8,16,32,64,128,256}],
    ]
    
    rng = random.Random(42)
    for _ in range(50):
        o = list(range(1, 257))
        rng.shuffle(o)
        orders.append(o)
        
    for o in orders:
        res = greedy_construct(o)
        if len(res) > len(best_set):
            best_set = res
            
    return list(best_set)

construct()
