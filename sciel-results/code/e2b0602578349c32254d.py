import time
from math import lcm

def construct():
    deadline = time.time() + 55
    N = 255
    nums = list(range(2, 257))
    weights = [1.0 / nums[i] for i in range(N)]
    
    # compat[i] is a bitset of j such that lcm(nums[i], nums[j]) > 256
    # Note: The problem asks for subset where lcm(a,b) > 256 for ALL distinct pairs.
    # This is a clique in the graph where edge (i,j) exists if lcm > 256.
    compat = [0] * N
    for i in range(N):
        ni = nums[i]
        mask = 0
        for j in range(N):
            if i == j:
                continue
            if lcm(ni, nums[j]) > 256:
                mask |= (1 << j)
        compat[i] = mask

    # Order indices by weight descending
    order = list(range(N))
    order.sort(key=lambda i: -weights[i])
    
    best_weight = 0.0
    best_set = 0 # bitset of selected vertices

    def get_bits(mask):
        bits = []
        m = mask
        while m:
            lb = m & (-m)
            bits.append(lb.bit_length() - 1)
            m ^= lb
        return bits

    def bound(P):
        """
        Returns an upper bound on the weight of any clique in the subgraph induced by P.
        Uses greedy coloring.
        """
        if P == 0:
            return 0.0
        
        # Get vertices in P sorted by weight descending
        verts = get_bits(P)
        verts.sort(key=lambda i: -weights[i])
        
        # Greedy coloring
        # We partition verts into independent sets C1, C2, ...
        # The weight of any clique is <= sum of max weights in each Ci
        # To maximize this bound (tightest upper bound), we should pick the heaviest available
        # vertex for each color class? 
        # Actually, the standard Tomita coloring for max clique size just picks any available.
        # For weighted, picking the heaviest first for each color class ensures that the max weight
        # in that class is as large as possible, which might make the sum larger (looser bound)?
        # No, we want an UPPER bound. The true max clique weight is <= sum of max weights in ANY partition into independent sets.
        # To get the TIGHTEST (smallest) upper bound, we should minimize the sum of max weights.
        # This is the "minimum sum of max weights over independent set partitions".
        # Greedy: Pick the heaviest vertex v. It must be in some color class. 
        # If we put it in C1, max(C1) >= w(v).
        # If we put it in a new class, we add w(v) to the sum.
        # Strategy: 
        # 1. Sort verts by weight descending.
        # 2. For each vertex v in sorted order:
        #    Try to assign v to an existing color class C_k where v is compatible with all vertices in C_k.
        #    If no such class exists, create a new class C_{new} with v.
        # This is standard greedy coloring. 
        # Does this minimize the sum of max weights?
        # If we assign a heavy vertex to an existing class, the max of that class doesn't change (if it was already heavier) or becomes w(v).
        # If we create a new class, we add w(v).
        # By processing heaviest first, we ensure heavy vertices are placed in classes where they might not increase the max if a heavier one is already there?
        # No, heaviest first means the first vertex creates C1 with max w1.
        # Next vertex w2. If compatible with w1, it goes to C1. Max(C1) is still w1. Sum increases by 0.
        # If not compatible, it goes to C2. Max(C2) is w2. Sum increases by w2.
        # This seems to produce a relatively small sum, hence a tight bound.
        
        color_classes = [] # List of (max_weight, mask_of_vertices)
        # Actually we just need the max weight per class.
        # We need to check compatibility.
        # For each v, we check if it is compatible with ALL vertices in a class.
        # We can store the intersection of compat masks for the class?
        # Or just store the list of vertices in the class.
        
        classes = [] # List of lists of vertices
        for v in verts:
            placed = False
            for k, c in enumerate(classes):
                # Check if v is compatible with all u in c
                ok = True
                for u in c:
                    if not (compat[v] >> u) & 1:
                        ok = False
                        break
                if ok:
                    c.append(v)
                    placed = True
                    break
            if not placed:
                classes.append([v])
        
        # Calculate bound
        b = 0.0
        for c in classes:
            # Max weight in class c
            max_w = max(weights[u] for u in c)
            b += max_w
        return b

    def solve(P, Q_weight):
        nonlocal best_weight, best_set
        
        if time.time() > deadline:
            return

        # Bound check
        B = bound(P)
        if Q_weight + B <= best_weight:
            return

        # Get vertices in P sorted by weight descending
        verts = get_bits(P)
        verts.sort(key=lambda i: -weights[i])
        
        for v in verts:
            if time.time() > deadline:
                return
            
            # Optional: Prune if even adding v and the rest of P (simple sum) is not enough
            # But we already have a tight bound at the start.
            # We can check if Q_weight + weights[v] + bound(P & compat[v] & ~(1<<v)) <= best_weight
            # This is expensive. Let's just recurse.
            
            new_P = P & compat[v] & ~(1 << v)
            
            # If new_P is 0, this is a leaf
            if new_P == 0:
                if Q_weight + weights[v] > best_weight:
                    best_weight = Q_weight + weights[v]
                    best_set = 0 # We need to track the set. 
                    # Wait, we need to return the subset.
                    # We should track the path or reconstruct it.
                    # Let's pass the current set as a bitset.
                    # Modify solve signature: solve(P, Q_weight, Q_set)
                    pass
            
            # Recurse
            # We need to update best_set if we find a better one.
            # Let's refactor to pass Q_set.
            
            solve_helper(P, Q_weight, Q_set)
            # Remove v from P for the next iteration (skip branch)
            P &= ~(1 << v)
            
    # The above solve logic is a bit mixed. Let's rewrite cleanly with Q_set.

    best_set = 0

    def solve(P, Q_weight, Q_set):
        nonlocal best_weight, best_set
        
        if time.time() > deadline:
            return

        # Bound check
        B = bound(P)
        if Q_weight + B <= best_weight:
            return

        # Get vertices in P sorted by weight descending
        verts = get_bits(P)
        verts.sort(key=lambda i: -weights[i])
        
        for v in verts:
            if time.time() > deadline:
                return
            
            new_Q_weight = Q_weight + weights[v]
            new_Q_set = Q_set | (1 << v)
            new_P = P & compat[v] & ~(1 << v)
            
            solve(new_P, new_Q_weight, new_Q_set)
            
            # Remove v from P (skip branch)
            P &= ~(1 << v)
            
            # Update bound for the remaining P? 
            # The bound B was for the original P. 
            # As we remove v, the bound for the new P (without v) might be smaller.
            # But we recompute bound at the start of the recursive call? No, we recompute at the start of the NEXT iteration of this loop?
            # No, the loop continues with the modified P.
            # We should check if the new P can still beat best.
            # B_new = bound(P)
            # if Q_weight + B_new <= best_weight: break
            
            # Let's add this check to break early
            B_new = bound(P)
            if Q_weight + B_new <= best_weight:
                break

    # Initialize
    full_mask = (1 << N) - 1
    solve(full_mask, 0.0, 0)

    # Extract result
    if best_set == 0:
        # Fallback: return the single best element
        max_i = order[0]
        return [nums[max_i]]
    
    result_indices = get_bits(best_set)
    return [nums[i] for i in result_indices]
