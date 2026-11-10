import time
import sys

def construct():
    start = time.time()
    deadline = start + 85.0  # Safety margin for 90s limit
    N = 128
    MAX_VAL = N
    
    # Bitset representation: bit i corresponds to number i+1
    # We use Python integers as bitsets.
    
    def time_left():
        return deadline - time.time()
        
    # Precomputation for fast conflict checking
    # For each x, we want to quickly check if adding x to a set S creates a forbidden triple.
    # A forbidden triple is {a, b, c, a+b, a+c, b+c} subset S.
    
    # Let's precompute for each x, the list of "witnesses" that would make x invalid.
    # If x is added, it might be:
    # 1. One of a, b, c.
    # 2. One of a+b, a+c, b+c.
    
    # Case 1: x is a "base" element (e.g., x=a).
    # We need b, c in S such that x+b, x+c, b+c are in S (or equal to x, which is impossible for positive sums unless 0 involved).
    # Actually, if x+b = x, b=0 (not in 1..128). So x+b, x+c, b+c must be in S.
    # So: exists b, c in S, b != c, b != x, c != x such that (x+b) in S and (x+c) in S and (b+c) in S.
    
    # Case 2: x is a "sum" element (e.g., x=a+b).
    # We need a, b in S such that a+b=x.
    # And we need c in S, c != a, c != b such that a+c in S and b+c in S.
    # Note: a+c cannot be x (implies c=b) and b+c cannot be x (implies c=a).
    # So a+c and b+c must be in S.
    
    # Precompute lists to speed up these checks.
    # For Case 1: For each x, list of pairs (b, c) such that b < c, b,c != x.
    #   We need to check if b, c, x+b, x+c, b+c are all in S.
    #   This is 5 membership checks.
    #   Number of such pairs is ~ N^2 / 2 = 8000. For each x, it's manageable.
    #   But we can filter: b+c must be <= N. x+b <= N, x+c <= N.
    
    # For Case 2: For each x, list of pairs (a, b) with a+b=x.
    #   For each such pair, we need to check if there exists c in S \ {a,b} such that a+c in S and b+c in S.
    #   This requires iterating c in S.
    
    # Let's optimize the "can_add" function using bitsets.
    # S is an integer bitmask.
    # (S >> v) & 1 checks if v+1 is in S.
    
    # Precompute for each x:
    # case1_pairs[x] = list of (b, c, x+b, x+c, b+c)
    # case2_pairs[x] = list of (a, b)
    
    case1_pairs = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        for b in range(1, N + 1):
            if b == x: continue
            for c in range(b + 1, N + 1):
                if c == x: continue
                xb = x + b
                xc = x + c
                bc = b + c
                if xb <= N and xc <= N and bc <= N:
                    case1_pairs[x].append((b, c, xb, xc, bc))
                    
    case2_pairs = [[] for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            s = a + b
            if s <= N:
                case2_pairs[s].append((a, b))
                
    def can_add(S, x):
        # Check Case 1: x is base
        for (b, c, xb, xc, bc) in case1_pairs[x]:
            # Check if b, c, xb, xc, bc are in S
            if ((S >> b) & 1) and ((S >> c) & 1) and ((S >> xb) & 1) and ((S >> xc) & 1) and ((S >> bc) & 1):
                return False
        
        # Check Case 2: x is sum
        for (a, b) in case2_pairs[x]:
            if not ((S >> a) & 1) or not ((S >> b) & 1):
                continue
            # Need c in S, c != a, c != b, such that a+c in S and b+c in S
            # Iterate through set bits in S
            temp = S
            while temp:
                lsb = temp & -temp
                c = lsb.bit_length() - 1
                if c == a or c == b:
                    temp ^= lsb
                    continue
                ac = a + c
                bc = b + c
                if ac <= N and bc <= N:
                    if ((S >> ac) & 1) and ((S >> bc) & 1):
                        return False
                temp ^= lsb
        return True

    # Branch and Bound
    best_size = 0
    best_mask = 0
    
    # Heuristic: Start with a good initial solution
    # Upper half is always valid
    init_mask = 0
    for i in range(N // 2 + 1, N + 1):
        init_mask |= (1 << i)
    best_size = len(bin(init_mask)) - 1
    best_mask = init_mask
    
    # DFS
    # To improve performance, we can order variables.
    # Smaller numbers have more conflicts (they form sums that stay in range).
    # Larger numbers are "safer".
    # So we should decide on smaller numbers first?
    # Actually, if we decide on small numbers, we constrain the large numbers.
    # If we decide on large numbers first, they don't constrain small numbers much (sums go out of range).
    # So, it is better to decide on LARGE numbers first?
    # If we include 128, it doesn't conflict with anything as a sum (128+1 > 128).
    # It only conflicts as a base.
    # If we include 1, it conflicts with many.
    # Let's try iterating from N down to 1.
    
    def dfs(x, S, size):
        nonlocal best_size, best_mask
        
        if time_left() < 1.0:
            return
            
        # Upper bound
        if size + (x - 1) <= best_size:
            return
            
        if x == 0:
            if size > best_size:
                best_size = size
                best_mask = S
            return
        
        # Try including x
        if can_add(S, x):
            new_S = S | (1 << x)
            dfs(x - 1, new_S, size + 1)
            
        # Try excluding x
        dfs(x - 1, S, size)
        
    # Start DFS from N down to 1
    # But wait, the initial solution is the upper half.
    # We can start the DFS from the state where all upper half elements are INCLUDED,
    # and we are deciding on the lower half elements?
    # No, because including a lower half element might force us to REMOVE an upper half element?
    # No, the constraint is "forbidden triple". It doesn't force removal, it just forbids the set.
    # If we have a valid set S, adding x might make it invalid.
    # But if we start with U (upper half) and add L (lower half), we might find that some U elements
    # are part of a forbidden triple with new L elements.
    # Example: L={1, 2}, U={3, 4, 5}.
    # 1+2=3. If 3 is in U, and we add 1,2 to L, we have {1,2,3}.
    # Need a third element c such that 1+c, 2+c in S.
    # If c=4 (in U), 1+4=5 (in U), 2+4=6 (not in U). So {1,2,3,4,5} is NOT forbidden by this triple?
    # Wait, the forbidden set is {a,b,c,a+b,a+c,b+c}.
    # If a=1, b=2, c=4. Sums: 3, 5, 6.
    # If S contains {1,2,3,4,5}, does it contain the forbidden set?
    # The set is {1,2,4,3,5,6}. 6 is not in S. So it's valid.
    # But if c=3? a=1, b=2, c=3. Sums: 3, 4, 5.
    # Forbidden set: {1,2,3,3,4,5} = {1,2,3,4,5}.
    # If S contains {1,2,3,4,5}, it IS forbidden.
    # So adding 1 and 2 to U={3..128} creates a conflict with 3,4,5.
    # Specifically, 3,4,5 are in U. 1,2 are in L.
    # Triple (1,2,3) uses 3 from U. Sums 3,4,5. 3 is base, 4,5 are sums.
    # All 5 distinct elements {1,2,3,4,5} are in S. Forbidden!
    
    # So we CANNOT just start with U fixed and add L. We must allow removing elements from U if necessary.
    # However, removing elements from U reduces size. We want to MAXIMIZE size.
    # So we should prefer keeping U elements if possible.
    
    # A better approach:
    # 1. Identify that the "hard" part is the lower half.
    # 2. Iterate over all subsets of the lower half? 2^64 is too big.
    # 3. Use B&B on the whole set, but with a strong heuristic.
    
    # Let's refine the B&B.
    # Order: 1, 2, ..., 128.
    # Why? Because small numbers create the most constraints.
    # If we decide on 1, 2, ..., 64, the valid choices for 65..128 are determined (or restricted).
    # Actually, for any fixed L, the set of valid U elements is those that don't form a forbidden triple with L or other U elements.
    # Since U elements don't sum to U elements (sum > 128), the only conflicts involving U are:
    # 1. Two U elements and one L element.
    # 2. One U element and two L elements.
    # (Three L elements: sums might be in L or U).
    
    # If we fix L, we can greedily select the largest possible U?
    # Not necessarily largest, but we want to maximize |U|.
    # Given L, a U element $u$ is valid if it doesn't form a forbidden triple with existing elements.
    # Since $u$ is large, it can only be a "base" or a "sum" (if $u = a+b$ with $a,b \in L$).
    # If $u = a+b$, and $a,b \in L$, we need to check if there is $c \in S$ such that $a+c, b+c \in S$.
    # If $c \in L$, sums are in $L \cup U$.
    # If $c \in U$, sums are $> 128$ (usually), so they can't be in S.
    # So if $c \in U$, $a+c > 128$ (since $a \ge 1, c \ge 65 \implies a+c \ge 66$, wait. $a \in L \implies a \le 64$. $c \in U \implies c \ge 65$. $a+c \ge 66$. If $a+c \le 128$, it could be in U).
    # So $c \in U$ CAN create conflicts.
    
    # However, note that if we pick $L$ first, the constraints on $U$ are local.
    # Let's try this:
    # 1. B&B to find the best L (subset of 1..64).
    # 2. For each candidate L, compute the max valid U.
    # 3. Total size = |L| + |U|.
    
    # How to compute max valid U given L?
    # Start with U = {65..128}.
    # Remove elements from U that create forbidden triples with L or other U elements.
    # This is a 2-SAT or similar? No, it's still a hypergraph independent set problem on U, with fixed L.
    # But U is "sparser" in conflicts.
    # Actually, we can just run a greedy or small B&B on U given L.
    # Since |U| = 64, and conflicts are limited, this should be fast.
    
    # Let's implement:
    # 1. B&B on L (1..64).
    # 2. For each L, run a fast check/greedy to find max U.
    
    def max_U_for_L(L_mask):
        # L_mask is bitset for 1..64
        # Start with all U elements
        U_mask = 0
        for i in range(65, 129):
            U_mask |= (1 << i)
            
        # We want to remove elements from U that cause conflicts.
        # Or, better, build U from scratch?
        # No, we want to KEEP as many as possible.
        # So start with all, and remove if invalid?
        # "Invalid" means: adding this element to current S (L + current_U) creates a forbidden triple.
        # But removing one might allow others?
        # It's a maximum independent set on the hypergraph induced by U, with L fixed.
        # Given time limits, a greedy removal or a limited B&B on U is needed.
        
        # Let's try a simple greedy: iterate U elements, keep if valid.
        # Order: maybe random or sorted?
        # Let's try sorted descending (safer sums) or ascending.
        # Actually, let's just check all subsets of U? No, 2^64.
        
        # Greedy with restarts?
        # Let's just do a simple forward checking greedy.
        S = L_mask
        valid_U = 0
        # Try adding U elements in a specific order
        # Elements in U that are sums of two L elements are more constrained.
        # Let's just iterate 65..128.
        for u in range(65, 129):
            if can_add(S, u):
                S |= (1 << u)
                valid_U |= (1 << u)
        return valid_U.bit_count()
        
    # B&B on L
    def dfs_L(x, L_mask, size_L):
        nonlocal best_size, best_mask
        
        if time_left() < 2.0:
            return
            
        # Upper bound: size_L + (x - 1) + 64 (max possible U)
        if size_L + (x - 1) + 64 <= best_size:
            return
            
        if x == 0:
            # Calculate max U for this L
            u_size = max_U_for_L(L_mask)
            total_size = size_L + u_size
            if total_size > best_size:
                best_size = total_size
                # Reconstruct best_mask
                # We need the actual U mask too
                # Let's store the best L and recompute U at the end?
                # Or store both.
                # For now, just track size.
                # We need to return the list of ints.
                # We can store the best L_mask and compute U at the very end.
                # But max_U_for_L is greedy, might not be optimal.
                # To be safe, let's store the L_mask that gave the best size.
                best_L_mask = L_mask
            return
            
        # Try including x
        # Check if x can be added to L_mask?
        # Wait, can_add checks against the FULL set S.
        # Here we are only building L.
        # Conflicts within L:
        # If we add x to L, does it create a forbidden triple WITHIN L?
        # i.e., exists a,b in L such that {x,a,b, x+a, x+b, a+b} subset L?
        # If so, we can't add x to L (if we want L to be valid on its own? No, L doesn't have to be valid on its own, the UNION S must be valid).
        # However, if L contains a forbidden triple, S will contain it, so S is invalid.
        # So L MUST be a valid set (no forbidden triples within L).
        # So we can use can_add(L_mask, x) to check if x can be added to L.
        
        if can_add(L_mask, x):
            new_L = L_mask | (1 << x)
            dfs_L(x - 1, new_L, size_L + 1)
            
        # Try excluding x
        dfs_L(x - 1, L_mask, size_L)
        
    # Initialize
    best_L_mask = 0
    # Start with empty L
    dfs_L(64, 0, 0)
    
    # Now we have the best L_mask.
    # We need to find the BEST U for this L_mask.
    # The greedy in max_U_for_L might be suboptimal.
    # Let's run a local search or B&B on U for the best L.
    
    L_mask = best_L_mask
    
    # Optimize U
    # Start with all U
    U_mask = 0
    for i in range(65, 129):
        U_mask |= (1 << i)
        
    S = L_mask | U_mask
    
    # Remove elements from U that cause conflicts.
    # Iterate and remove if invalid.
    # To maximize size, we should remove the "most conflicting" elements first?
    # Or just iterate.
    
    # Let's do a simple removal pass.
    changed = True
    while changed and time_left() > 2.0:
        changed = False
        for u in range(65, 129):
            if not ((U_mask >> u) & 1):
                continue
            # Check if u is "essential" for a conflict?
            # i.e., if we remove u, does S become valid?
            # Actually, we want to REMOVE u if it is PART of a forbidden triple.
            # If u is in a forbidden triple, we must remove at least one element from that triple.
            # Since L is fixed, we can only remove from U.
            # So if u is in a forbidden triple with other U elements and/or L elements,
            # we must remove u OR remove the other U elements.
            # This is a hitting set problem.
            
            # Simple heuristic: If u is in ANY forbidden triple in current S, remove it.
            # Check if u is in a forbidden triple.
            # This is expensive.
            
            # Alternative: Just try to add u to (S \ {u}) and see if it's valid?
            # No, we are REMOVING.
            
            # Let's stick to the greedy construction of U which is faster and likely good enough.
            # The B&B on L is the main driver.
            pass
            
    # Let's re-run the greedy U construction for the best L, but with multiple orders to find a better U.
    best_U_size = 0
    best_U_mask = 0
    
    orders = [list(range(65, 129)), list(range(128, 64, -1))]
    import random
    rng = random.Random(42)
    for _ in range(10):
        o = list(range(65, 129))
        rng.shuffle(o)
        orders.append(o)
        
    for o in orders:
        S = L_mask
        U_m = 0
        for u in o:
            if can_add(S, u):
                S |= (1 << u)
                U_m |= (1 << u)
        if U_m.bit_count() > best_U_size:
            best_U_size = U_m.bit_count()
            best_U_mask = U_m
            
    final_mask = L_mask | best_U_mask
    result = [i+1 for i in range(128) if (final_mask >> i) & 1]
    
    # Final validity check
    # Just to be safe
    s_set = set(result)
    valid = True
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            a = result[i]
            b = result[j]
            ab = a+b
            if ab not in s_set: continue
            for k in range(j+1, len(result)):
                c = result[k]
                if (a+c) in s_set and (b+c) in s_set:
                    valid = False
                    break
            if not valid: break
        if not valid: break
        
    if not valid:
        # Fallback to upper half
        return list(range(65, 129))
        
    return result

# Run
if __name__ == "__main__":
    construct()
