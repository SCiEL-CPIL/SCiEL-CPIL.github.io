import time
import sys

def construct():
    N = 256
    start_time = time.time()
    DEADLINE = start_time + 85.0  # Safety margin before 90s kill

    # 1. Build Conflict Graph
    # conflict[i] is a bitset where bit j is set if i and j cannot both be in S.
    # i and j conflict if there exists k such that {i, j, k} is a 3-AP.
    # Cases:
    # 1) i, j are endpoints: k = (i+j)/2. Requires i+j even.
    # 2) i, j are (a, b): k = 2b - a. Requires 1 <= k <= N.
    # 3) i, j are (b, c): k = 2b - c. Requires 1 <= k <= N.
    # Note: If i, j are (a, c), they are endpoints (case 1).
    # If i, j are (a, b), they are not endpoints unless j=2a (impossible for distinct in AP) or... wait.
    # In a 3-AP (a, b, c), b is the middle.
    # Pairs: (a,b), (b,c), (a,c).
    # (a,c) -> k=b.
    # (a,b) -> k=c = 2b-a.
    # (b,c) -> k=a = 2b-c.
    
    # We use 0-indexed internally for bitsets, so node i (1..N) is index i-1.
    # Bitset size: N bits.
    
    conflicts = [0] * N
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            # Check if a, b are part of a 3-AP with some c.
            # Case 1: a, b are endpoints (a, c, b) or (b, c, a)? No, order is (x, y, z).
            # If a, b are endpoints, the middle is (a+b)/2.
            # If (a+b) is even, let m = (a+b)//2. If 1 <= m <= N and m != a and m != b (always true if a!=b),
            # then a, b, m form an AP. So a and b conflict.
            if (a + b) % 2 == 0:
                m = (a + b) // 2
                # m is automatically between a and b, so 1 <= m <= N is guaranteed if 1<=a<b<=N.
                conflicts[a-1] |= 1 << (b-1)
                conflicts[b-1] |= 1 << (a-1)
            
            # Case 2: a, b are first two terms (a, b, c).
            # c = 2b - a.
            c = 2 * b - a
            if 1 <= c <= N and c != a and c != b:
                # a and b conflict because they form an AP with c.
                conflicts[a-1] |= 1 << (b-1)
                conflicts[b-1] |= 1 << (a-1)
            
            # Case 3: b, a are first two terms? No, we iterate a < b.
            # What if b, a are (a', b') in an AP (a', b', c')?
            # That means a, b are (b', c')? i.e., b is middle, a is end?
            # Let's check if a, b can be (b, c) in an AP (a', b, c).
            # Here a is the middle? No, a < b. So a cannot be the middle of (a', b, c) if a'=a? No.
            # The AP is (x, y, z).
            # If the pair is (a, b) with a < b:
            # - They could be (x, y) -> z = 2y-x = 2b-a. (Handled in Case 2)
            # - They could be (x, z) -> y = (x+z)/2 = (a+b)/2. (Handled in Case 1)
            # - They could be (y, z) -> x = 2y-z = 2b-a? No, if pair is (y,z), then y=b, z=a? But a<b, so z<a=y? Contradiction.
            # Wait. If the pair is (y, z), then y < z. So y=a, z=b.
            # Then x = 2y - z = 2a - b.
            # Let's check this case.
            c2 = 2 * a - b
            if 1 <= c2 <= N and c2 != a and c2 != b:
                # a and b conflict because they form an AP with c2.
                conflicts[a-1] |= 1 << (b-1)
                conflicts[b-1] |= 1 << (a-1)

    # Weights: w[i] = 1.0 / (i+1)
    weights = [1.0 / (i + 1) for i in range(N)]

    # Sort nodes by weight descending to improve branch and bound efficiency
    # We want to consider high weight nodes first.
    order = sorted(range(N), key=lambda i: weights[i], reverse=True)
    
    # Re-index or just use the order in the recursion?
    # It's easier to keep original indices but iterate candidates in sorted order.
    # However, for bitsets, we need to map "candidate set" efficiently.
    # Standard MWIS B&B:
    # State: (current_weight, candidates_bitset)
    # candidates_bitset contains nodes that can be added without conflict with already selected nodes.
    
    # Precompute neighbors bitset for quick exclusion?
    # conflicts[i] already gives neighbors.
    
    # Upper Bound Calculation:
    # Greedy: iterate through candidates in order of decreasing weight.
    # Add to bound if not conflicting with previously added in this greedy pass.
    # This is a valid upper bound for the remaining weight.
    
    def greedy_upper_bound(candidates):
        if candidates == 0:
            return 0.0
        ub = 0.0
        used = 0
        # Iterate through all nodes in global weight order
        # Check if node is in candidates and not in used
        for i in order:
            if candidates & (1 << i):
                if not (used & (1 << i)):
                    # Can we add i?
                    # In the greedy bound, we just check against 'used' (nodes already taken in this greedy pass)
                    # We don't check against 'conflicts[i]' and 'candidates' for feasibility in the bound,
                    # we just greedily pick a maximal independent set from the candidates?
                    # No, the standard greedy bound for MWIS is:
                    # Sort candidates by weight. Pick first. Remove it and its neighbors. Repeat.
                    # This gives the weight of a maximal independent set, which is a LOWER bound on the optimal independent set?
                    # No, the weight of a greedy MIS is a lower bound on the optimal?
                    # Actually, we want an UPPER bound on the remaining weight.
                    # A common upper bound is the sum of weights of all candidates (too loose).
                    # A tighter upper bound: Greedy coloring bound (for graphs) or greedy packing.
                    # For MWIS, a valid upper bound is obtained by:
                    # 1. Order candidates by weight descending.
                    # 2. Iterate. If node is available, add its weight to bound.
                    #    Then, mark all its neighbors as "unavailable" for the bound calculation?
                    #    No, that calculates the weight of a specific independent set (greedy solution), which is a LOWER bound.
                    # We need an UPPER bound.
                    # Upper bound idea: 
                    # If we ignore conflicts, sum of all weights is an upper bound.
                    # Better: 
                    # Use the fact that for any independent set, no two adjacent nodes are picked.
                    # A valid upper bound can be derived from a fractional relaxation or specific graph properties.
                    # Given the time constraint, a simple and effective heuristic for B&B on small graphs:
                    # Use the sum of weights of candidates as the bound? Too loose, might explore too much.
                    
                    # Let's try a different bound:
                    # Greedy packing upper bound:
                    # Iterate candidates in descending weight.
                    # If node is not yet "covered", add its weight to bound.
                    # Mark node and all its neighbors as "covered".
                    # This effectively partitions the candidates into stars centered at the picked nodes.
                    # Any independent set can pick at most 1 node from each star?
                    # No, it can pick the center OR one of the leaves.
                    # Wait. If we pick center v, we get w(v). We cannot pick any neighbor.
                    # If we don't pick v, we might pick neighbors.
                    # The sum of weights of the centers is NOT an upper bound. 
                    # Example: Edge (1,2). w(1)=10, w(2)=10.
                    # Greedy picks 1. Bound = 10.
                    # Optimal is 10.
                    # Example: Path 1-2-3. w=10, 1, 10.
                    # Greedy picks 1 (w=10). Covers 2. Next is 3 (w=10). Picks 3. Bound = 20.
                    # Optimal is 20 (1 and 3).
                    # Example: Triangle. w=10,10,10.
                    # Greedy picks 1. Covers 2,3. Bound = 10.
                    # Optimal is 10.
                    # Is this always an upper bound?
                    # Let S be the set of centers picked by the greedy covering algorithm.
                    # For any node u not in S, it is covered by some center v in S, meaning (u,v) is an edge.
                    # In any independent set I, if u in I, then v not in I.
                    # This doesn't directly sum up to an upper bound on sum(w(I)).
                    
                    # Let's use a simpler, safe upper bound: Sum of weights of all candidates.
                    # To make it tighter, we can subtract weights of nodes that are forced to be excluded?
                    # No.
                    
                    # Alternative: Use the greedy solution weight as a LOWER bound to prune branches where current + greedy_bound < best.
                    # But we need an UPPER bound for the recursion limit.
                    
                    # Let's stick to a simple but effective strategy for N=256:
                    # The graph is sparse-ish?
                    # Number of edges:
                    # For each pair (a,b), they conflict if they form an AP.
                    # Total pairs ~ 32k.
                    # This is a relatively dense graph in terms of constraints, but N is small.
                    
                    # Given the 90s limit, and N=256, a pure recursive B&B with bitsets might be slow if the bound is loose.
                    # However, the weights are 1/x. The first few nodes (1, 2, 3...) have huge weights.
                    # 1/1 = 1.
                    # 1/2 = 0.5.
                    # The sum converges.
                    # Most weight is in the first few integers.
                    # If we include 1, we exclude 2, 3 (AP 1,2,3), 4? No, 1,2,3. 1,?,?
                    # 1, 2, 3 -> conflict (1,2), (2,3), (1,3).
                    # 1, 3, 5 -> conflict (1,3), (3,5), (1,5).
                    # 1 is very restrictive.
                    
                    # Strategy:
                    # 1. Greedy initialization to get a good lower bound (best_score).
                    # 2. Branch and Bound.
                    #    - Order nodes by weight desc.
                    #    - At each step, branch on the first available node (highest weight).
                    #    - Branch 1: Include node. Update candidates (remove node and neighbors).
                    #    - Branch 2: Exclude node. Update candidates (remove node).
                    #    - Bound: Sum of weights of remaining candidates. (Loose but fast).
                    #    - To tighten bound: If sum(remaining) + current < best, prune.
                    
                    # Since weights drop off fast, the sum of remaining candidates will drop quickly as we exclude high-weight nodes.
                    # This should prune effectively.
                    
                    pass # This function is not used in the final loop below, see implementation.
        
        # Let's implement the B&B loop directly.
        
    best_score = 0.0
    best_set = set()
    
    # Initial Greedy Solution
    # Try to build a set greedily.
    S_greedy = set()
    candidates_greedy = (1 << N) - 1
    # Iterate 1 to N? Or weight order?
    # Weight order (1, 2, 3...) is natural for 1/x.
    for i in range(N): # i is index, node is i+1
        if candidates_greedy & (1 << i):
            # Add node i
            S_greedy.add(i)
            # Remove i and its neighbors from candidates
            mask = 1 << i
            mask |= conflicts[i]
            candidates_greedy &= ~mask
    best_score = sum(weights[i] for i in S_greedy)
    best_set = set(S_greedy)
    
    # B&B State
    # We will use a stack for DFS to avoid recursion depth issues, though recursion is fine for depth 256.
    # Stack items: (current_weight, candidates_bitset)
    # We process the most promising branch first (Include high weight node).
    
    # To improve bound, we can precompute the sum of weights for any bitset? No, too expensive.
    # We will compute the bound on the fly.
    # Optimization: Precompute weight_sum for masks? No.
    # Just sum weights of bits in mask.
    
    def bit_count_weight(mask):
        s = 0.0
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            s += weights[idx]
            m ^= lsb
        return s

    # However, calculating sum every time is O(N).
    # We can pass the sum along in the stack.
    # Stack: (current_weight, candidates_mask, current_sum_of_candidates)
    
    stack = []
    initial_mask = (1 << N) - 1
    initial_sum = sum(weights)
    stack.append((0.0, initial_mask, initial_sum))
    
    nodes_explored = 0
    
    while stack:
        if time.time() > DEADLINE:
            break
        
        cur_w, mask, cand_sum = stack.pop()
        
        # Prune if potential max is not better
        if cur_w + cand_sum <= best_score:
            continue
        
        if mask == 0:
            if cur_w > best_score:
                best_score = cur_w
                # We don't easily reconstruct the set from the stack in this simple DFS without backtracking history.
                # We need to track the set or the decisions.
                # Let's modify stack to include the set of selected nodes?
                # That's memory heavy.
                # Alternative: Store the path.
                # Or, just run the B&B to find the optimal VALUE, and then reconstruct?
                # Reconstruction is hard if we don't store the tree.
                
                # Let's change strategy:
                # Store the selected nodes in the stack item?
                # Tuple: (cur_w, mask, cand_sum, selected_tuple)
                # Tuples are immutable, so we can add to them.
                # Depth 256, tuple size 256. OK.
                pass
        
        # Find the highest weight node in mask
        # Iterate order
        next_node = -1
        for i in order:
            if mask & (1 << i):
                next_node = i
                break
        
        if next_node == -1:
            if cur_w > best_score:
                best_score = cur_w
                # Need to update best_set. 
                # Since we didn't store the set, we can't update it here.
                # We must store the set.
                pass
            continue
        
        # Branch 1: Exclude next_node
        # New mask: mask without next_node
        # New cand_sum: cand_sum - weights[next_node]
        # This branch is less promising, so push it first (LIFO -> processed later)
        stack.append((cur_w, mask & ~(1 << next_node), cand_sum - weights[next_node], None)) 
        
        # Branch 2: Include next_node
        # New mask: mask without next_node and its neighbors
        # New cur_w: cur_w + weights[next_node]
        # New cand_sum: cand_sum - weights[next_node] - sum(weights of neighbors in mask)
        
        # Calculate neighbors to remove
        to_remove = (1 << next_node) | conflicts[next_node]
        new_mask = mask & ~to_remove
        
        # Calculate new cand_sum
        # We need to subtract weights of nodes in (mask & to_remove)
        removed_mask = mask & to_remove
        removed_weight = 0.0
        m = removed_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            removed_weight += weights[idx]
            m ^= lsb
            
        new_cand_sum = cand_sum - removed_weight
        new_cur_w = cur_w + weights[next_node]
        
        # Push include branch (processed next)
        stack.append((new_cur_w, new_mask, new_cand_sum, None))
        
        nodes_explored += 1

    # The above B&B doesn't track the SET, so it can't return the correct list if it finds a better score.
    # We need to track the set.
    # Let's restart with a version that tracks the set.
    
    # Given the complexity and time, let's use a simpler, more robust approach:
    # Local Search with Bitsets and Random Restarts.
    # The previous program did local search. It scored 1.0.
    # Maybe the greedy initialization was bad?
    # Let's try a better greedy: 
    # Sort by 1/x (which is just 1, 2, 3...).
    # The previous program added 1, then checked 2...
    
    # Let's try a different heuristic:
    # Start with the set of all odd numbers?
    # Odds: 1, 3, 5...
    # 1, 3, 5 is an AP. So no.
    
    # Start with powers of 2?
    # 1, 2, 4, 8, 16, 32, 64, 128.
    # Check APs:
    # 1, 2, 4: No.
    # 1, 4, 8: No.
    # 2, 4, 8: No.
    # 1, 2, 3? 3 not in set.
    # Powers of 2 are 3-AP free?
    # a, b, c in {2^k}.
    # 2^i, 2^j, 2^k.
    # 2*2^j = 2^i + 2^k.
    # 2^(j+1) = 2^i + 2^k.
    # If i < j < k, then 2^i < 2^(j+1) < 2^k?
    # If k = j+1, 2^(j+1) = 2^i + 2^(j+1) -> 2^i = 0. Impossible.
    # If k > j+1, 2^k > 2^(j+1). So 2^i + 2^k > 2^(j+1).
    # So powers of 2 are 3-AP free.
    # Weight: 1 + 1/2 + 1/4 + ... = 2.
    
    # Can we do better?
    # Add 3?
    # {1, 2, 3, 4, 8, 16, 32, 64, 128}
    # 1, 2, 3 is AP. Remove 2 or 3.
    # If we remove 2, we lose 0.5. Gain 1/3 = 0.33. Net loss.
    # If we remove 3, we gain 0.
    
    # What about {1, 3, 4, 8, 16...}?
    # 1, 3, 5? No 5.
    # 1, 4, 7? No 7.
    # 3, 4, 5? No 5.
    # 1, 2, 3 -> conflict.
    
    # Let's try a greedy that prioritizes small numbers but skips conflicts.
    # This is what the seed did.
    
    # Let's try to improve the local search in the seed.
    # The seed used `can_add` which iterates over `aps_containing[y]`.
    # This is O(APs per y).
    
    # Let's implement a bitset-based local search.
    # State: bitset S.
    # Conflict check: (S & conflicts[y]) == 0.
    # This is O(1) (or O(N/word_size)).
    
    # Algorithm:
    # 1. Greedy Init:
    #    S = 0
    #    for i in 1..N:
    #       if (S & conflicts[i-1]) == 0:
    #           S |= 1 << (i-1)
    
    # 2. Local Search:
    #    Loop:
    #       Improved = False
    #       # Try to add any y not in S
    #       for y in 1..N:
    #           if not (S & (1 << (y-1))):
    #               if (S & conflicts[y-1]) == 0:
    #                   S |= 1 << (y-1)
    #                   Improved = True
    #                   break
    #       
    #       # Try to swap x in S with y not in S, where y < x (to increase weight)
    #       # Actually, we want to maximize sum 1/x.
    #       # So we want to replace a large x (small weight) with a small y (large weight).
    #       # Iterate x in S (descending order of x, i.e. ascending weight)
    #       # Iterate y not in S (ascending order of y, i.e. descending weight)
    #       # If y < x and swap is valid, do it.
        
    #       # Validity of swap:
    #       # S' = S \ {x} U {y}
    #       # Check if y conflicts with S \ {x}.
    #       # (S \ {x}) & conflicts[y] == 0
    #       # (S & ~(1 << (x-1))) & conflicts[y-1] == 0
        
    #       if not Improved:
    #           break

    # 3. Return list from bitset.

    # Let's code this.

    S_bits = 0
    for i in range(1, N + 1):
        idx = i - 1
        if (S_bits & conflicts[idx]) == 0:
            S_bits |= 1 << idx

    best_bits = S_bits
    best_val = sum(weights[i] for i in range(N) if S_bits & (1 << i))

    start_time_ls = time.time()
    improved = True
    while improved and (time.time() - start_time_ls) < DEADLINE:
        improved = False
        
        # Phase 1: Add
        for y in range(1, N + 1):
            idx = y - 1
            if not (S_bits & (1 << idx)):
                if (S_bits & conflicts[idx]) == 0:
                    S_bits |= 1 << idx
                    improved = True
                    break

        # Phase 2: Swap
        # Get list of elements in S
        # We want to try removing large elements (low weight) and adding small elements (high weight).
        # So iterate x in S, sorted by x descending.
        # For each x, iterate y not in S, sorted by y ascending.
        
        # Optimization: Precompute list of S elements?
        # S is a bitset.
        
        # To speed up, we can just iterate y from 1 to N, and x from N to 1.
        # If y is in S, skip.
        # If x is not in S, skip.
        # If y >= x, skip (since we want y < x for weight gain, unless weights are equal which they aren't).
        
        swapped = False
        for y in range(1, N + 1):
            if time.time() > DEADLINE:
                return [i+1 for i in range(N) if best_bits & (1 << i)]
            
            idx_y = y - 1
            if S_bits & (1 << idx_y):
                continue
                
            # Try to find an x in S such that x > y and swap is valid.
            # We prefer the smallest x > y? Or the largest x (lowest weight loss)?
            # We want to minimize weight loss. So remove the smallest weight element possible?
            # No, we are adding y (high weight) and removing x (low weight).
            # The gain is 1/y - 1/x.
            # To maximize gain, we want small y and large x.
            # So for a fixed y, we want the largest x in S such that x > y and swap is valid.
            
            # Iterate x from N down to y+1
            for x in range(N, y, -1):
                idx_x = x - 1
                if not (S_bits & (1 << idx_x)):
                    continue
                
                # Check validity
                # New S = S - x + y
                # Check y against S - x
                S_minus_x = S_bits & ~(1 << idx_x)
                if (S_minus_x & conflicts[idx_y]) == 0:
                    # Valid swap
                    S_bits = S_minus_x | (1 << idx_y)
                    improved = True
                    swapped = True
                    break
            
            if swapped:
                break

        # Update best
        current_val = sum(weights[i] for i in range(N) if S_bits & (1 << i))
        if current_val > best_val:
            best_val = current_val
            best_bits = S_bits

    # Convert best_bits to list
    result = [i + 1 for i in range(N) if best_bits & (1 << i)]
    
    if not result:
        return [1]
        
    return result

