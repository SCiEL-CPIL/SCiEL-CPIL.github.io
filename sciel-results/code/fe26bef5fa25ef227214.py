import time
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # 1. Generate all permutations
    perms = list(permutations(range(1, n + 1)))
    N = len(perms)
    
    # Precompute prefix index for greedy initialization
    # key: tuple of length k (suffix of A), value: list of perms starting with that tuple
    # We want to find B such that B[:n-k] == A[k:]
    # So key is A[k:]
    prefix_to_perms = {}
    for p in perms:
        for k in range(1, n):
            prefix = p[:n-k] # This is the prefix of B
            # Wait, A[k:] is the suffix of A.
            # We want B to start with A[k:].
            # So key should be A[k:].
            # But we are building index for B.
            # B starts with B[:n-k].
            # So if we look up A[k:], we find B such that B[:n-k] == A[k:].
            # So we index B by its prefixes.
            prefix_to_perms.setdefault(prefix, []).append(p)
            
    # Greedy Initialization
    used = set(perms)
    seq = [random.choice(perms)]
    used.remove(seq[0])
    
    while used:
        last = seq[-1]
        best_next = None
        # Try to maximize overlap (minimize k, which is start index of suffix in A)
        # Overlap length = n - k
        # k goes from 1 to n-1
        for k in range(1, n):
            suffix = last[k:]
            candidates = prefix_to_perms.get(suffix, [])
            # Filter available
            avail = [c for c in candidates if c in used]
            if avail:
                best_next = random.choice(avail)
                break
        
        if best_next is None:
            # No overlap found, pick random
            best_next = random.choice(list(used))
        
        seq.append(best_next)
        used.remove(best_next)
        
    # Cost function
    def get_overlap(A, B):
        # Max k such that A[k:] == B[:n-k]
        # k is index in A
        for k in range(1, n):
            if A[k:] == B[:n-k]:
                return n - k
        return 0
        
    def get_length(perm_seq):
        if not perm_seq:
            return 0
        length = n
        for i in range(1, len(perm_seq)):
            ov = get_overlap(perm_seq[i-1], perm_seq[i])
            length += n - ov
        return length

    current_seq = seq
    current_len = get_length(current_seq)
    best_seq = current_seq[:]
    best_len = current_len
    
    # Simulated Annealing with Insertion Moves
    T = 10.0
    T_min = 0.01
    alpha = 0.995 # Slower cooling
    
    # Helper to get cost between two perms
    def cost(A, B):
        return n - get_overlap(A, B)
        
    while time.time() < deadline:
        # Insertion move
        i = random.randint(0, N - 1)
        j = random.randint(0, N - 1)
        
        if i == j:
            continue
            
        # Remove i
        x = current_seq[i]
        
        # Calculate delta
        # Neighbors in original
        if i == 0:
            left = None
            right = current_seq[1]
        elif i == N - 1:
            left = current_seq[N - 2]
            right = None
        else:
            left = current_seq[i - 1]
            right = current_seq[i + 1]
            
        # Cost removed
        cost_removed = 0
        if left and right:
            cost_removed += cost(left, x)
            cost_removed += cost(x, right)
            cost_removed -= cost(left, right) # This part is added back effectively
            # Actually, let's just compute full new local cost vs old local cost
            
        # Let's do it cleanly
        # Old local cost around i
        old_cost = 0
        if i > 0:
            old_cost += cost(current_seq[i-1], x)
        if i < N - 1:
            old_cost += cost(x, current_seq[i+1])
            
        # New position j
        # We insert x at j.
        # If j > i, the indices shift.
        # Let's determine the neighbors at j in the *modified* list (after removal)
        
        # This is getting complex. Let's just recompute length if N is small?
        # N=5040. get_length is O(N).
        # 5040 * 100s / (5040 * 7) ~ 14k iterations. Too few.
        # We need incremental delta.
        
        # Let's implement incremental delta carefully.
        
        # 1. Remove x from i
        # 2. Insert x at j
        
        # Case j < i:
        # Original: ... seq[j-1], seq[j], ..., seq[i-1], x, seq[i+1] ...
        # New: ... seq[j-1], x, seq[j], ..., seq[i-1], seq[i+1] ...
        
        # Case j > i:
        # Original: ... seq[i-1], x, seq[i+1], ..., seq[j-1], seq[j] ...
        # New: ... seq[i-1], seq[i+1], ..., seq[j-1], x, seq[j] ...
        
        # We need to identify neighbors at insertion point j in the list *after removal*.
        
        # Let's make a copy of the list? No, too slow.
        # Let's just use the indices.
        
        # If j > i:
        # The element at index j in the original list is now at j-1.
        # The element at index j-1 is now at j-2.
        # We are inserting x at index j.
        # So x will be between original seq[j-1] and original seq[j].
        # Wait.
        # List: [0, 1, 2, 3]
        # Remove 1. List: [0, 2, 3]
        # Insert at 2. List: [0, 2, x, 3]
        # x is between 2 (orig 2) and 3 (orig 3).
        # So neighbors are seq[j-1] and seq[j] from ORIGINAL list.
        
        # If j <= i:
        # List: [0, 1, 2, 3]
        # Remove 2. List: [0, 1, 3]
        # Insert at 1. List: [0, x, 1, 3]
        # x is between 0 (orig 0) and 1 (orig 1).
        # So neighbors are seq[j-1] and seq[j] from ORIGINAL list.
        
        # It seems neighbors are always seq[j-1] and seq[j] from the original list?
        # Let's check j=i.
        # Remove 1. [0, 2, 3]. Insert at 1. [0, x, 2, 3].
        # Neighbors 0 and 2. Original seq[0] and seq[2].
        # j=1. seq[j-1]=seq[0], seq[j]=seq[1] (which was x).
        # So we use seq[j] as the right neighbor?
        # If j < i, right neighbor is seq[j].
        # If j > i, right neighbor is seq[j] (which shifted to j-1, but value is same).
        
        # Left neighbor:
        # If j=0, no left.
        # Else, left is seq[j-1].
        
        # Let's verify j < i case again.
        # Remove i. Insert at j.
        # x is between seq[j-1] and seq[j].
        
        # So:
        # Left Neighbor L: seq[j-1] if j > 0 else None
        # Right Neighbor R: seq[j] if j < N else None (but j < N always)
        # Wait, if j == N? We pick j in 0..N-1.
        
        # However, if i == j, we skip.
        
        # What if i is adjacent to j?
        # If i == j-1 (j > i):
        # Remove i. x was before j.
        # Insert at j.
        # x is between seq[j-1] (which was x's original left? No, x was at i=j-1)
        # Let's trace: [A, x, B, C]. i=1 (x), j=2.
        # Remove x: [A, B, C].
        # Insert at 2: [A, B, x, C].
        # x is between B and C.
        # Original seq[j-1] = seq[1] = x.
        # Original seq[j] = seq[2] = B.
        # So L should be B?
        # My logic "L=seq[j-1]" gives x. That's wrong.
        
        # So we must handle the shift carefully.
        
        # Let's just compute the new local cost by looking up the actual neighbors in the *current* list state?
        # No, we don't want to modify the list until we accept.
        
        # Alternative: Just compute `get_length` for the new sequence?
        # If we can make `get_length` very fast (e.g. C-optimized or vectorized), it might be okay.
        # But 5040 elements.
        
        # Let's stick to incremental but handle neighbors correctly.
        
        # Determine L and R in the *new* sequence.
        # The new sequence is `current_seq` with `x` removed from `i` and inserted at `j`.
        
        # If j < i:
        #   The elements at indices < j are unchanged.
        #   The elements at indices j..i-1 are shifted right by 1.
        #   The element at i is removed.
        #   x is at j.
        #   Left of x: current_seq[j-1] (if j>0)
        #   Right of x: current_seq[j] (original)
        
        # If j > i:
        #   The elements at indices < i are unchanged.
        #   The element at i is removed.
        #   The elements at indices i+1..j are shifted left by 1.
        #   x is at j.
        #   Left of x: current_seq[j-1] (original) -- Wait.
        #   Original: ... seq[j-1], seq[j] ...
        #   Remove i (which is < j-1).
        #   seq[j-1] becomes index j-2.
        #   seq[j] becomes index j-1.
        #   Insert x at j.
        #   x is at index j.
        #   Left of x is index j-1, which is original seq[j].
        #   Right of x is index j+1, which is original seq[j+1].
        
        # If j == i:
        #   No change.
        
        # So:
        if j < i:
            L = current_seq[j-1] if j > 0 else None
            R = current_seq[j]
        else: # j > i
            L = current_seq[j] # Original seq[j] is now at j-1, which is left of x
            R = current_seq[j+1] if j+1 < N else None
            
        # Calculate New Local Cost
        new_cost = 0
        if L:
            new_cost += cost(L, x)
        if R:
            new_cost += cost(x, R)
        if L and R:
             # The connection L-R is replaced by L-x and x-R?
             # No, in the old sequence, L and R were connected?
             # If j < i: L=seq[j-1], R=seq[j]. They were connected.
             # Old cost included cost(L, R).
             # New cost includes cost(L, x) + cost(x, R).
             # So delta at j is (cost(L,x) + cost(x,R)) - cost(L,R).
             
             # Also, the removal of x at i affects neighbors of i.
             # Old neighbors of x: seq[i-1] and seq[i+1].
             # They are now connected.
             # Old cost included cost(seq[i-1], x) + cost(x, seq[i+1]).
             # New cost includes cost(seq[i-1], seq[i+1]).
             # So delta at i is cost(seq[i-1], seq[i+1]) - (cost(seq[i-1], x) + cost(x, seq[i+1])).
             
             # Total Delta = Delta_j + Delta_i
             
             pass # Logic below

        # Let's generalize
        # Delta = Cost(New Local) - Cost(Old Local)
        
        # Old Local:
        # 1. At i:
        #    Left_i = seq[i-1] if i>0 else None
        #    Right_i = seq[i+1] if i<N-1 else None
        #    Cost_i_old = 0
        #    if Left_i: Cost_i_old += cost(Left_i, x)
        #    if Right_i: Cost_i_old += cost(x, Right_i)
        #    if Left_i and Right_i: Cost_i_old -= cost(Left_i, Right_i) # This is NOT in old cost.
        #    Wait. The total length is sum of edges.
        #    Edge (Left_i, x) and (x, Right_i) exist.
        #    Edge (Left_i, Right_i) does NOT exist.
        #    So Old Cost contribution from this region is cost(Left_i, x) + cost(x, Right_i).
        
        # 2. At j (insertion point):
        #    We are inserting x between L and R.
        #    In Old sequence, L and R were connected by edge (L, R).
        #    So Old Cost contribution is cost(L, R).
        
        # New Local:
        # 1. At i (removal):
        #    Left_i and Right_i are now connected.
        #    New Cost contribution is cost(Left_i, Right_i).
        
        # 2. At j (insertion):
        #    L and R are now connected via x.
        #    New Cost contribution is cost(L, x) + cost(x, R).
        
        # Delta = [cost(Left_i, Right_i) + cost(L, x) + cost(x, R)] - [cost(Left_i, x) + cost(x, Right_i) + cost(L, R)]
        
        # Handle boundaries (None)
        
        def get_cost_pair(A, B):
            if A is None or B is None:
                return 0
            return cost(A, B)
            
        Left_i = current_seq[i-1] if i > 0 else None
        Right_i = current_seq[i+1] if i < N - 1 else None
        
        if j < i:
            L = current_seq[j-1] if j > 0 else None
            R = current_seq[j]
        else: # j > i
            L = current_seq[j]
            R = current_seq[j+1] if j+1 < N else None
            
        # If j == i, skip (handled earlier)
        
        old_cost = get_cost_pair(Left_i, x) + get_cost_pair(x, Right_i) + get_cost_pair(L, R)
        new_cost = get_cost_pair(Left_i, Right_i) + get_cost_pair(L, x) + get_cost_pair(x, R)
        
        delta = new_cost - old_cost
        
        if delta < 0 or random.random() < pow(2.718281828, -delta / T):
            # Execute move
            # Remove i
            current_seq.pop(i)
            # Insert at j
            # If j > i, the index j in the new list corresponds to original j-1?
            # We want to insert x at position j in the *final* list.
            # If we pop i, indices > i shift left.
            # If j > i, we insert at j-1?
            # Let's check:
            # [0, 1, 2, 3]. i=1, j=3.
            # Pop 1 -> [0, 2, 3].
            # We want x at index 3? Or index corresponding to original 3?
            # My logic for L/R assumed x is between orig seq[j] and orig seq[j+1].
            # In [0, 2, 3], orig seq[3] is at index 2.
            # So we insert at index 2?
            # If we insert at 2: [0, 2, x, 3].
            # x is between 2 and 3. Correct.
            # So if j > i, insert at j-1.
            # If j < i, insert at j.
            
            if j > i:
                current_seq.insert(j - 1, x)
            else:
                current_seq.insert(j, x)
                
            current_len += delta
            
            if current_len < best_len:
                best_len = current_len
                best_seq = current_seq[:]
                
        T *= alpha
        if T < T_min:
            T = T_min

    # Construct final string
    if not best_seq:
        return ""
        
    result = list(best_seq[0])
    for i in range(1, len(best_seq)):
        p_prev = best_seq[i-1]
        p_curr = best_seq[i]
        ov = get_overlap(p_prev, p_curr)
        result.extend(p_curr[ov:])
        
    return "".join(map(str, result))
