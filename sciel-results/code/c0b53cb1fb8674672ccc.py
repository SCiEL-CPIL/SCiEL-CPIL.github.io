import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 90s kill, 5s safety margin
    N = 256
    
    # Precompute weights
    weights = [0.0] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = 1.0 / i
    
    # Precompute all 3-APs: for each pair (a,b), the third element c such that a,b,c form a 3-AP
    # Actually, let's precompute conflicts differently.
    # For a given x and current set S (as bitset), x is invalid if:
    #   exists a in S such that 2*x - a in S (a != x)
    #   exists a in S such that 2*a - x in S (a != x) 
    #   exists a in S such that (x+a) is even and (x+a)//2 in S (a != x)
    
    # Better: precompute for each x, the list of "conflict pairs" - pairs (a,b) in [1,N] 
    # such that {x,a,b} is a 3-AP. Then x is invalid if any such pair is fully in S.
    
    # Even better: for each x, precompute a bitmask of "forbidden pairs" as a list of (a,b) pairs.
    # But that's too many. Let's use a different approach.
    
    # For each x, the 3-APs involving x are:
    # 1. x as first: x, x+d, x+2d for d=1..(N-x)//2
    # 2. x as middle: x-d, x, x+d for d=1..min(x-1, N-x)
    # 3. x as last: x-2d, x-d, x for d=1..(x-1)//2
    
    # So x conflicts with the set of pairs. Let's precompute for each x, 
    # a list of (a, b) pairs such that {x, a, b} is a 3-AP with a,b != x.
    
    conflict_pairs = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        # x as first element: x, x+d, x+2d
        for d in range(1, (N - x) // 2 + 1):
            a = x + d
            b = x + 2 * d
            if a <= N and b <= N:
                conflict_pairs[x].append((a, b))
        # x as middle: x-d, x, x+d
        for d in range(1, min(x - 1, N - x) + 1):
            a = x - d
            b = x + d
            if a >= 1 and b <= N:
                conflict_pairs[x].append((a, b))
        # x as last: x-2d, x-d, x
        for d in range(1, (x - 1) // 2 + 1):
            a = x - 2 * d
            b = x - d
            if a >= 1 and b >= 1:
                conflict_pairs[x].append((a, b))
    
    # For fast checking: x is valid in S iff for all (a,b) in conflict_pairs[x], not (a in S and b in S)
    # We'll use a bitset for S.
    
    def is_valid(x, S_bitset):
        """Check if x can be added to S (represented as bitset) without forming a 3-AP."""
        for (a, b) in conflict_pairs[x]:
            # Check if both a and b are in S
            if (S_bitset >> (a - 1)) & 1 and (S_bitset >> (b - 1)) & 1:
                return False
        return True
    
    def greedy_solve(order, S_init_bitset=0, S_init_score=0.0):
        """Greedy: process elements in given order, add if valid."""
        S = S_init_bitset
        score = S_init_score
        for x in order:
            if (S >> (x - 1)) & 1:
                continue
            if is_valid(x, S):
                S |= (1 << (x - 1))
                score += weights[x]
        return S, score
    
    def local_search(S, score, max_iterations=5000):
        """Hill climbing: try single swaps and additions."""
        best_S = S
        best_score = score
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            best_gain = 0
            best_move = None
            
            # Try additions
            for x in range(1, N + 1):
                if (S >> (x - 1)) & 1:
                    continue
                if is_valid(x, S):
                    gain = weights[x]
                    if gain > best_gain:
                        best_gain = gain
                        best_move = ('add', x)
            
            # Try swaps: remove u, add v
            # For efficiency, only consider v that are "nearly valid" - 
            # i.e., v conflicts with S only through u
            for v in range(1, N + 1):
                if (S >> (v - 1)) & 1:
                    continue
                # Check if v is valid in S. If yes, it's an add move (already handled).
                # If not, find which u's in S cause the conflict.
                # For each such u, check if v is valid in S \ {u}.
                # To be efficient, only try swaps where gain > 0.
                
                # Quick check: is v valid in S?
                v_valid_in_S = True
                conflicting_us = set()
                for (a, b) in conflict_pairs[v]:
                    a_in = (S >> (a - 1)) & 1
                    b_in = (S >> (b - 1)) & 1
                    if a_in and b_in:
                        v_valid_in_S = False
                        conflicting_us.add(a)
                        conflicting_us.add(b)
                        break  # One conflict is enough to know it's invalid
                
                if v_valid_in_S:
                    continue  # Already handled as add
                
                # Try removing each conflicting u
                for u in conflicting_us:
                    if not ((S >> (u - 1)) & 1):
                        continue
                    if weights[v] <= weights[u]:
                        continue  # No gain
                    # Check if v is valid in S \ {u}
                    S_temp = S & ~(1 << (u - 1))
                    if is_valid(v, S_temp):
                        gain = weights[v] - weights[u]
                        if gain > best_gain:
                            best_gain = gain
                            best_move = ('swap', u, v)
            
            if best_move:
                if best_move[0] == 'add':
                    x = best_move[1]
                    S |= (1 << (x - 1))
                    score += weights[x]
                else:
                    u, v = best_move[1], best_move[2]
                    S &= ~(1 << (u - 1))
                    S |= (1 << (v - 1))
                    score += weights[v] - weights[u]
                
                if score > best_score:
                    best_score = score
                    best_S = S
                improved = True
        
        return best_S, best_score
    
    def branch_and_bound(order, time_limit):
        """Branch and bound: process elements in order, branch on include/exclude."""
        best_S = 0
        best_score = 0.0
        current_S = 0
        current_score = 0.0
        remaining_weight = sum(weights[x] for x in order)
        
        # Precompute suffix sums for upper bound
        suffix = [0.0] * (len(order) + 1)
        for i in range(len(order) - 1, -1, -1):
            suffix[i] = suffix[i + 1] + weights[order[i]]
        
        def _bbsolve(idx):
            nonlocal best_S, best_score, current_S, current_score
            
            if time.time() > time_limit:
                return
            
            if idx == len(order):
                if current_score > best_score:
                    best_score = current_score
                    best_S = current_S
                return
            
            # Upper bound: current_score + suffix[idx]
            if current_score + suffix[idx] <= best_score:
                return
            
            x = order[idx]
            
            # Option 1: include x
            if is_valid(x, current_S):
                current_S |= (1 << (x - 1))
                current_score += weights[x]
                _bbsolve(idx + 1)
                current_S &= ~(1 << (x - 1))
                current_score -= weights[x]
            
            # Option 2: exclude x
            _bbsolve(idx + 1)
        
        _bbsolve(0)
        return best_S, best_score
    
    # Strategy: 
    # 1. Greedy with natural order (1 to N)
    # 2. Greedy with reverse order (N to 1)
    # 3. Multiple random restarts with shuffled orders
    # 4. Local search on each result
    # 5. Branch and bound on a reduced problem (top K elements by weight)
    
    best_S = 0
    best_score = 0.0
    
    # Phase 1: Greedy with different orderings
    orderings = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
    ]
    
    # Add some random orderings
    rng = random.Random(42)
    for _ in range(5):
        order = list(range(1, N + 1))
        rng.shuffle(order)
        orderings.append(order)
    
    for order in orderings:
        if time.time() > deadline - 5:
            break
        S, score = greedy_solve(order)
        S, score = local_search(S, score, max_iterations=2000)
        if score > best_score:
            best_score = score
            best_S = S
    
    # Phase 2: Branch and bound on top elements
    # Only consider elements 1..K for B&B since they have highest weights
    # K=20 or so, 2^20 is too large, but B&B prunes heavily
    K = 18
    top_order = list(range(1, K + 1))
    
    # Try B&B with different orderings
    bnb_orderings = [
        top_order,
        list(reversed(top_order)),
    ]
    rng2 = random.Random(123)
    for _ in range(3):
        order = list(top_order)
        rng2.shuffle(order)
        bnb_orderings.append(order)
    
    for order in bnb_orderings:
        if time.time() > deadline - 10:
            break
        # Initialize with greedy on full set, then B&B on top K
        S_full, score_full = greedy_solve(list(range(1, N + 1)))
        # Extract the subset of S_full that is in top K
        S_top = 0
        for x in range(1, K + 1):
            if (S_full >> (x - 1)) & 1:
                S_top |= (1 << (x - 1))
        
        S_bnb, score_bnb = branch_and_bound(order, deadline - 10)
        # Combine: S_bnb for top K, S_full for rest
        S_combined = S_bnb | (S_full & ~((1 << K) - 1))
        score_combined = score_bnb + sum(weights[x] for x in range(K + 1, N + 1) if (S_full >> (x - 1)) & 1)
        
        # Refine with local search
        S_combined, score_combined = local_search(S_combined, score_combined, max_iterations=3000)
        
        if score_combined > best_score:
            best_score = score_combined
            best_S = S_combined
    
    # Phase 3: More random restarts with local search
    rng3 = random.Random(789)
    for _ in range(20):
        if time.time() > deadline - 5:
            break
        order = list(range(1, N + 1))
        rng3.shuffle(order)
        S, score = greedy_solve(order)
        S, score = local_search(S, score, max_iterations=3000)
        if score > best_score:
            best_score = score
            best_S = S
    
    # Convert bitset to list
    result = []
    for x in range(1, N + 1):
        if (best_S >> (x - 1)) & 1:
            result.append(x)
    
    # Ensure non-empty
    if not result:
        result = [1]
    
    return result
