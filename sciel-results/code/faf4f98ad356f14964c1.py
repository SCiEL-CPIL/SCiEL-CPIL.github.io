"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s kill
    
    n = 128
    elements = list(range(1, n + 1))
    
    # 1. Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            lcm_table[i][j] = i * j // gcd(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # 2. Identify all "bad" triples (a, b, c) such that 
    # lcm(a,b) == lcm(a,c) == lcm(b,c)
    # We represent the set of elements as a bitmask.
    # Index i in bitmask corresponds to element (i+1).
    
    bad_triples = []
    # We only need to check triples a < b < c
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            # Optimization: if lab is very large, it's less likely to be matched?
            # Actually, just check.
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a - 1, b - 1, c - 1))
    
    if not bad_triples:
        return elements
        
    # 3. Build adjacency constraints for the Branch and Bound
    # For each vertex v, we need to know which other vertices u 
    # form a bad triple with v and some other vertex w.
    # Actually, for MIS in 3-uniform hypergraph, the constraint is:
    # If we pick v, we cannot pick any pair (u, w) such that {v, u, w} is a bad triple.
    # This is not a simple graph independence constraint.
    
    # Let's use a simpler recursive backtracking with pruning.
    # State: current_set (list), candidates (list of remaining elements)
    # To speed up, we can use a bitset for candidates.
    
    # Let's try a different approach:
    # Since n=128 is small, we can use a greedy algorithm first to get a good lower bound,
    # then use local search or a more sophisticated search.
    
    # However, the prompt asks for a strategy DIFFERENT from greedy.
    # The previous "greedy with local search" got ~1.84.
    # Let's try a randomized greedy with many restarts and local search (Simulated Annealing style)
    # or a Max-Cut style heuristic.
    
    # Actually, a very effective strategy for these "avoid hyperedges" problems is:
    # 1. Greedy construction to get a valid set S.
    # 2. Try to add elements one by one (in random order) if they don't create a conflict.
    # 3. Local search: try to swap an element in S with one not in S to increase size.
    
    # But let's look at the structure.
    # Bad triples are rare?
    # Let's count them.
    # If there are few bad triples, we can remove one element from each bad triple to get a valid set.
    # This is the Vertex Cover problem on the 3-uniform hypergraph.
    # Min Vertex Cover = Max Independent Set.
    # We want to find the smallest set of vertices to remove to hit all bad triples.
    # This is NP-hard, but for n=128, if the number of bad triples is small, we can use a hitting set algorithm.
    
    # Let's estimate the number of bad triples.
    # If it's small (< 1000), we can use a branch and bound on the hitting set (Vertex Cover).
    
    # Let's implement a simple Greedy Vertex Cover first to get an upper bound on the cover size (lower bound on IS size).
    # Then use a randomized local search to try to improve the IS.
    
    # Strategy:
    # 1. Generate all bad triples.
    # 2. Use a greedy algorithm to select a set of vertices to remove (Vertex Cover).
    #    - Pick a random bad triple, remove one of its vertices (randomly or by degree).
    #    - Repeat until no bad triples remain.
    # 3. The remaining vertices form a valid set.
    # 4. Try to improve by adding back vertices that were removed, if possible (i.e., if they don't form a bad triple with current set).
    
    # Let's implement this "Greedy Hitting Set + Repair" strategy.
    
    # First, let's see how many bad triples there are.
    # We already found them above.
    
    # If len(bad_triples) is very large, this might be slow.
    # But n=128, so max triples is ~349k.
    
    # Let's try a different angle:
    # The seed program is a sequential greedy.
    # A "different" strategy is a **Randomized Local Search** (Simulated Annealing) on the space of subsets.
    # State: A subset S of [1, 128].
    # Objective: Maximize |S| subject to S being valid.
    # Move: Add a random element not in S, or remove a random element in S, or swap.
    # Penalty: If S becomes invalid, apply a penalty or reject.
    
    # Better: Start with a valid set (from greedy).
    # Try to add elements. If conflict, remove one of the conflicting elements.
    # This is essentially a stochastic greedy.
    
    # Let's try **Simulated Annealing** on the validity constraint.
    # Energy E(S) = -|S| + M * (number of bad triples in S)
    # We want to minimize E.
    # If we find a state with 0 bad triples, we have a valid set.
    # We want to maximize |S| among those with 0 bad triples.
    
    # Since we have a time limit, we can run multiple restarts.
    
    # Let's implement a fast local search.
    # 1. Start with empty set.
    # 2. Greedily add elements in random order.
    # 3. Local search: 
    #    - While improvement possible:
    #      - Try to add a random element not in S.
    #      - If it creates k bad triples, try to remove one element from each bad triple (randomly) to restore validity.
    #      - If size increases, accept.
    
    # This is a known heuristic for Hypergraph Independent Set.
    
    import random
    
    def is_valid_set(S):
        # S is a set of integers
        # Check if any bad triple is fully contained in S
        # This is slow if done naively.
        # We can precompute for each pair (a,b) in S, which c's make it bad.
        # Or just iterate bad_triples and check if all 3 are in S.
        # Given n=128, |S| ~ 60-80.
        # Number of pairs in S is ~2500-3000.
        # For each pair, check if there's a c in S that forms a bad triple.
        # We can precompute a map: (a,b) -> list of c's such that {a,b,c} is bad.
        
        pass # Placeholder
    
    # Let's precompute the conflict graph more efficiently for local search.
    # For each element x, let Conflicts(x) be the set of pairs (u,v) such that {x,u,v} is bad.
    # If we add x to S, we must ensure that no (u,v) in Conflicts(x) is fully contained in S.
    
    # Let's build:
    # conflict_pairs[x] = set of (u, v) such that x < u < v and {x,u,v} is bad.
    # Note: We should normalize the triple so x is the smallest? No, x can be any.
    # Let's just store the other two elements.
    
    conflicts = {i: set() for i in range(1, n + 1)}
    for a, b, c in bad_triples:
        # a, b, c are 0-indexed in bad_triples, but 1-indexed in logic?
        # In step 2, we stored (a-1, b-1, c-1).
        # Let's convert back to 1-indexed for consistency with elements list.
        a_val, b_val, c_val = a + 1, b + 1, c + 1
        conflicts[a_val].add((b_val, c_val))
        conflicts[b_val].add((a_val, c_val))
        conflicts[c_val].add((a_val, b_val))
        
    # Function to check if adding x to current set S is valid.
    # S is a set of ints.
    def can_add(x, S):
        # Check if any pair in conflicts[x] is subset of S
        for u, v in conflicts[x]:
            if u in S and v in S:
                return False
        return True
        
    # Greedy construction with randomization
    def greedy_random():
        S = set()
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        for x in candidates:
            if can_add(x, S):
                S.add(x)
        return S
        
    # Local Search: Try to improve S
    def local_search(S, time_limit=1.0):
        S = set(S)
        current_size = len(S)
        start = time.time()
        
        while time.time() - start < time_limit:
            # Try to add a random element not in S
            if current_size == n:
                break
                
            not_in_S = [x for x in range(1, n + 1) if x not in S]
            if not not_in_S:
                break
                
            x = random.choice(not_in_S)
            
            # If we add x, which pairs become "blocked"?
            # We need to remove elements to break any conflict.
            # Conflicts are pairs (u,v) in S such that {x,u,v} is bad.
            # We need to pick one element from each such pair to remove.
            # This is a hitting set problem on the pairs.
            
            # Find all conflicting pairs
            conflicting_pairs = []
            for u, v in conflicts[x]:
                if u in S and v in S:
                    conflicting_pairs.append((u, v))
                    
            if not conflicting_pairs:
                S.add(x)
                current_size += 1
                continue
            
            # We need to remove a set of elements R from S such that R hits all pairs in conflicting_pairs.
            # And we want to minimize |R|.
            # If |R| < 1, we gain size. (i.e. if we can add x and remove 0 elements, which is the case above).
            # If we remove 1 element, size stays same.
            # If we remove 2 elements, size decreases.
            
            # So we should only do this if we can remove 0 or 1 elements?
            # Actually, if we remove 1 element, size is same, but we might enable future additions.
            # Let's try to find a minimum hitting set for conflicting_pairs.
            # Since the number of conflicting pairs is likely small, we can use a greedy hitting set.
            
            # Greedy Hitting Set:
            # Count frequency of each element in conflicting_pairs.
            freq = {}
            for u, v in conflicting_pairs:
                freq[u] = freq.get(u, 0) + 1
                freq[v] = freq.get(v, 0) + 1
                
            # Pick element with highest freq
            if freq:
                # Sort by freq desc
                sorted_elems = sorted(freq.keys(), key=lambda k: freq[k], reverse=True)
                # Try the best candidate
                best_elem = sorted_elems[0]
                
                # Check if removing best_elem hits all pairs
                if all(best_elem in pair for pair in conflicting_pairs):
                    # Remove best_elem, add x
                    S.remove(best_elem)
                    S.add(x)
                    # Size unchanged, but structure changed.
                    # We can accept this move to explore.
                    # To avoid oscillation, we might need a tabu or randomization.
                    # For now, accept.
                    pass
                else:
                    # Try to find a set of 2 elements that hits all pairs?
                    # This is more complex.
                    # If we can't hit with 1, we skip adding x for now.
                    pass
            
        return S

    best_set = set()
    
    # Run multiple restarts
    while time.time() < deadline:
        # 1. Greedy random start
        S = greedy_random()
        
        # 2. Local search
        # We give local search a small time slice
        remaining = deadline - time.time()
        if remaining < 0.1:
            break
            
        time_slice = min(1.0, remaining * 0.5)
        S = local_search(S, time_slice)
        
        if len(S) > len(best_set):
            best_set = S
            
        # Optional: Try to improve best_set specifically
        # But local search on a fresh random start is a good diversification strategy.
        
    # Final check: ensure best_set is valid
    # (It should be, by construction, but let's be safe)
    # If best_set is empty, return a single element
    if not best_set:
        return [1]
        
    return list(best_set)
# EVOLVE-BLOCK-END
