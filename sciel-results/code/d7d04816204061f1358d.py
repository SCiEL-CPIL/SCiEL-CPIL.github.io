import time
import sys
from math import gcd
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    N = 256
    candidates = list(range(2, N + 1))
    
    # Precompute conflicts
    # Conflict exists between i and j (i != j) if:
    # 1. candidates[i] divides candidates[j] * candidates[k] for some k != i, j
    # 2. candidates[j] divides candidates[i] * candidates[k] for some k != i, j
    # 3. candidates[k] divides candidates[i] * candidates[j] for some k != i, j
    # Wait, the condition is: For EVERY three distinct members a, b, c, a does not divide b*c.
    # This is a property of the SET.
    # A set S is valid if for all distinct a, b, c in S, a ∤ b*c.
    # This is equivalent to saying: There is no triple (a, b, c) in S such that a | b*c.
    # This is a 3-uniform hypergraph independent set problem.
    # However, checking validity of a set of size K requires O(K^3) checks.
    # We can use a recursive backtracking with pruning.
    
    # Let's try to construct the set by adding elements one by one.
    # State: current set S (as a tuple of indices or values), next candidate index.
    # To make it efficient, we can precompute for each pair (b, c), the set of a's that divide b*c.
    # If we have b and c in S, we cannot have any a in S such that a | b*c.
    
    # Let's define a conflict graph on pairs? No, it's a 3-way constraint.
    
    # Alternative: Branch and Bound on the set of candidates.
    # We want to maximize |S|.
    
    # Let's order candidates.
    # Large numbers are "safer" as they are less likely to divide smaller products?
    # Actually, if a is large, a | b*c is harder to satisfy.
    # So large numbers are "good" divisors (i.e., hard to violate).
    # Small numbers are "bad" divisors (easy to violate).
    # So we should prioritize including large numbers?
    # Or rather, the constraint is symmetric in the sense that any element can be the divisor.
    
    # Let's use a recursive search.
    # We maintain a set S.
    # We iterate through candidates in a specific order.
    # Order: Descending value.
    # Why? If we include large numbers, they are less likely to cause conflicts as divisors.
    # Also, if a large number is in S, it restricts which smaller numbers can be added (if small | large * other).
    
    # Let's try a different approach: Local Search / Simulated Annealing with better initialization.
    # The current program does greedy.
    # A "different" strategy could be **Genetic Algorithm** or **Simulated Annealing** starting from a random valid set.
    
    # But the prompt asks for a strategy *materially different* from the current program's.
    # Current: Greedy insertion with restarts.
    # New: **Maximum Independent Set via Branch and Bound** on the 3-uniform hypergraph.
    
    # Given N=256, B&B might be too slow if not pruned well.
    # However, the structure of divisibility is strong.
    
    # Let's try a **Greedy + Local Search (Swap)** approach, but with a different initialization and swap logic.
    # Actually, the current program *is* greedy with restarts.
    # Let's try **Simulated Annealing** on the set membership vector.
    
    # State: binary vector x of length 255 (indices 0..254 for values 2..256).
    # Objective: Maximize sum(x) subject to no violations.
    # Violation check: For all i, j, k distinct, if x[i]=x[j]=x[k]=1, then val[i] ∤ val[j]*val[k].
    
    # To speed up violation checking, we can maintain a list of active elements.
    # When adding an element, check against all pairs of existing elements.
    # When removing, no check needed.
    # When swapping (remove i, add j), check j against all pairs of existing elements (excluding i).
    
    # This is essentially what the current program does in `can_add`.
    
    # The key difference: **Simulated Annealing** allows temporary invalid states or accepts worse states to escape local optima, whereas the current program only accepts valid additions (monotonic improvement).
    
    # However, the constraint is hard: "Any broken rule makes the whole result invalid".
    # So we cannot have invalid states in the final output.
    # SA can work on valid states only (neighborhood: add/remove/swap that maintains validity).
    # But the neighborhood of valid states might be disconnected or small.
    
    # Let's try a **Stochastic Local Search** with **Restart** and **Swap** moves.
    # The current program only does ADD moves (greedy). It never removes.
    # By allowing SWAP moves (remove one, add another), we can escape local optima where no single addition is possible, but a swap is.
    
    # Strategy:
    # 1. Initialize with a large set (e.g., all primes > 128 + some composites).
    # 2. Loop:
    #    - Try all possible swaps (remove i, add j) where i in S, j not in S.
    #    - If swap maintains validity, accept it (if it increases size? No, swap keeps size same).
    #    - Wait, if we swap, size stays same. We need to increase size.
    #    - So, we should try ADD moves first. If no ADD move works, try SWAP moves to rearrange the set to allow a future ADD.
    #    - Or, use SA to move between valid sets of the same size, hoping to find a set where an ADD is possible.
    
    # Let's implement a **Hill Climbing with Swap Moves** strategy.
    # 1. Start with a valid set S.
    # 2. While time remains:
    #    a. Try to find an element x not in S such that S U {x} is valid. If found, add it and restart loop 2a.
    #    b. If no addition possible, try to find a swap (i in S, j not in S) such that (S \ {i}) U {j} is valid.
    #       - Prefer swaps that might open up room for additions? Hard to predict.
    #       - Just do random valid swaps.
    #    c. If no swap possible, restart with a different initialization.
    
    # This is different from the current program which ONLY does greedy additions and restarts. It never rearranges the existing set.
    
    import random
    
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5)+1):
            if n % i == 0:
                return False
        return True

    primes = [x for x in candidates if is_prime(x)]
    composites = [x for x in candidates if not is_prime(x)]
    
    # Precompute divisibility matrix?
    # div[i][j] = True if val[i] divides val[j]
    # But we need a | b*c.
    
    # Helper: Check if x can be added to S
    def can_add(x, S_set):
        S_list = list(S_set)
        n = len(S_list)
        for i in range(n):
            b = S_list[i]
            for j in range(i+1, n):
                c = S_list[j]
                if (b * c) % x == 0:
                    return False
                if (x * c) % b == 0:
                    return False
                if (x * b) % c == 0:
                    return False
        return True

    def is_valid_set(S_set):
        S_list = list(S_set)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i+1, n):
                b = S_list[j]
                for k in range(j+1, n):
                    c = S_list[k]
                    if (b * c) % a == 0:
                        return False
        return True

    best_S = set()
    best_size = 0
    
    # Initialization strategies
    # 1. Large Primes
    # 2. Random Composites
    # 3. Large Composites
    
    num_restarts = 100
    for r in range(num_restarts):
        if time.time() > deadline - 10: break
        
        # Randomize initialization
        S = set()
        
        # Strategy: Start with large numbers, then fill in
        # Sort candidates by value descending
        cand_desc = sorted(candidates, reverse=True)
        
        # Randomly shuffle the order to vary the greedy path
        random.shuffle(cand_desc)
        
        # Greedy Add
        for x in cand_desc:
            if can_add(x, S):
                S.add(x)
        
        # Local Search: Swap Moves
        # Try to improve by swapping elements
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            S_list = list(S)
            not_in_S = [x for x in candidates if x not in S]
            
            # Shuffle to randomize swap order
            random.shuffle(S_list)
            random.shuffle(not_in_S)
            
            for i in S_list:
                if time.time() > deadline - 5: break
                for j in not_in_S:
                    if time.time() > deadline - 5: break
                    # Try swap i with j
                    new_S = (S - {i}) | {j}
                    # Check if new_S is valid
                    # Optimization: We know S was valid.
                    # Removing i keeps it valid.
                    # Adding j: check j against all pairs in (S \ {i})
                    S_minus_i = S - {i}
                    if can_add(j, S_minus_i):
                        # Swap is valid
                        # Does it help? We can't tell immediately.
                        # But it changes the structure.
                        # Accept swap with some probability? Or always?
                        # Let's always accept to explore the space.
                        S = new_S
                        improved = True
                        # Restart the inner loop? Or continue?
                        # If we continue, we might swap back and forth.
                        # Let's break out of swap loop and try additions again.
                        break
                if improved:
                    break
            
            # After swaps, try to add more
            cand_remaining = [x for x in candidates if x not in S]
            random.shuffle(cand_remaining)
            added = False
            for x in cand_remaining:
                if can_add(x, S):
                    S.add(x)
                    added = True
                    improved = True # Signal that we made progress
            if not added and not improved:
                break # Stuck
            
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
            
    return list(best_S)

# Test
if __name__ == "__main__":
    res = construct()
    print(len(res))
