"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 55  # safety margin before 60s kill
    
    N = 256
    
    # Precompute all 3-term APs: for each pair (a, c) with a < c, the middle is (a+c)//2 if (a+c) is even
    # For each element x, precompute which other elements conflict with x (i.e., form a 3-term AP with some third element)
    # Actually, we need: for a set S, can we add candidate x?
    # x conflicts with S if there exist a, b in S such that {a, b, x} forms a 3-term AP.
    # Precompute: for each x, the set of pairs (a, b) that would form an AP with x.
    # Better: for each x, precompute conflict_mask[x] = bitmask of elements that cannot be in S along with x
    # Wait, that's not quite right. The conflict is: if x is in S, then for any a in S, we can't have c = 2x - a in S.
    # Let me think differently.
    
    # For each element x in [1, 256], precompute:
    # - left_conflicts[x]: set of (a, c) pairs where a < x < c and a, c form AP with x as middle
    #   Actually: if x is middle, then a + c = 2x, so for each a < x, c = 2x - a > x
    #   If both a and c are in S, we can't add x.
    # - Similarly for x as first or last term.
    
    # Simpler approach: precompute for each x, the list of "forbidden pairs" 
    # Actually, let's use a different representation.
    # For a given set S (as a bitmask), x is valid to add if:
    # 1. No a in S such that 2x - a is in S and 2x - a != a (x is middle)
    # 2. No b in S such that 2b - x is in S and 2b - x != b (x is first)
    # 3. No b in S such that 2b - x is in S and 2b - x != b (x is last) -- wait, this is same as 2
    
    # Let me re-derive:
    # A 3-term AP is (a, b, c) with a + c = 2b, a < b < c.
    # If we add x to S, a conflict occurs if:
    # Case 1: x is the middle: exists a, c in S with a + c = 2x, a < x < c
    # Case 2: x is the first: exists b, c in S with x + c = 2b, x < b < c => c = 2b - x
    # Case 3: x is the last: exists a, b in S with a + x = 2b, a < b < x => a = 2b - x
    
    # Precompute for each x:
    # mid_conflicts[x] = list of (a, c) where a + c = 2x, a < x < c, a >= 1, c <= 256
    # first_conflicts[x] = list of (b, c) where x + c = 2b, x < b < c, b >= 1, c <= 256 => c = 2b - x > x => b > x
    #   So for b in range(x+1, 257): c = 2*b - x, if c <= 256 and c > x, then (b, c) is a conflict pair
    # last_conflicts[x] = list of (a, b) where a + x = 2b, a < b < x, a >= 1, b >= 1
    #   So for b in range(1, x): a = 2*b - x, if a >= 1 and a < b, then (a, b) is a conflict pair
    
    # For fast checking with bitmask:
    # x is valid iff:
    # - For all (a, c) in mid_conflicts[x]: NOT (bitmask has a AND bitmask has c)
    # - For all (b, c) in first_conflicts[x]: NOT (bitmask has b AND bitmask has c)  
    # - For all (a, b) in last_conflicts[x]: NOT (bitmask has a AND bitmask has b)
    
    # Precompute these
    mid_conflicts = [[] for _ in range(N + 1)]
    first_conflicts = [[] for _ in range(N + 1)]
    last_conflicts = [[] for _ in range(N + 1)]
    
    for x in range(1, N + 1):
        # x as middle: a + c = 2x, a < x < c
        for a in range(1, x):
            c = 2 * x - a
            if c <= N and c > x:
                mid_conflicts[x].append((a, c))
        
        # x as first: x + c = 2b, x < b < c => c = 2b - x
        for b in range(x + 1, N + 1):
            c = 2 * b - x
            if c <= N and c > b:
                first_conflicts[x].append((b, c))
        
        # x as last: a + x = 2b, a < b < x => a = 2b - x
        for b in range(1, x):
            a = 2 * b - x
            if a >= 1 and a < b:
                last_conflicts[x].append((a, b))
    
    def is_valid_addition(bitmask, x):
        """Check if x can be added to set represented by bitmask."""
        if (bitmask >> (x - 1)) & 1:
            return False
        
        # Check mid conflicts
        for a, c in mid_conflicts[x]:
            if ((bitmask >> (a - 1)) & 1) and ((bitmask >> (c - 1)) & 1):
                return False
        
        # Check first conflicts
        for b, c in first_conflicts[x]:
            if ((bitmask >> (b - 1)) & 1) and ((bitmask >> (c - 1)) & 1):
                return False
        
        # Check last conflicts
        for a, b in last_conflicts[x]:
            if ((bitmask >> (a - 1)) & 1) and ((bitmask >> (b - 1)) & 1):
                return False
        
        return True
    
    def reciprocal_sum_from_bitmask(bitmask):
        """Compute exact reciprocal sum from bitmask."""
        total = Fraction(0, 1)
        for x in range(1, N + 1):
            if (bitmask >> (x - 1)) & 1:
                total += Fraction(1, x)
        return total
    
    def reciprocal_sum_float(bitmask):
        """Compute float reciprocal sum for fast comparison."""
        total = 0.0
        for x in range(1, N + 1):
            if (bitmask >> (x - 1)) & 1:
                total += 1.0 / x
        return total
    
    def greedy_from_bitmask(order):
        """Greedy: add elements in given order if valid."""
        bitmask = 0
        for x in order:
            if is_valid_addition(bitmask, x):
                bitmask |= (1 << (x - 1))
        return bitmask
    
    def local_search_fast(bitmask, max_iter=200, time_limit=None):
        """
        Fast local search using bitmask representation.
        Tries additions and swaps.
        """
        if time_limit is None:
            time_limit = DEADLINE
        
        current = bitmask
        current_float = reciprocal_sum_float(current)
        best = bitmask
        best_float = current_float
        
        # Precompute reciprocal values
        recip = [0.0] + [1.0 / i for i in range(1, N + 1)]
        
        for iteration in range(max_iter):
            if (time.time() - start_time) > time_limit:
                break
            
            improved = False
            
            # Try additions: iterate x from 1 to N, try to add if valid
            # Prioritize small x (high reciprocal)
            for x in range(1, N + 1):
                if (current >> (x - 1)) & 1:
                    continue
                if is_valid_addition(current, x):
                    new_bitmask = current | (1 << (x - 1))
                    new_float = current_float + recip[x]
                    if new_float > best_float + 1e-15:
                        best_float = new_float
                        best = new_bitmask
                        current = new_bitmask
                        current_float = new_float
                        improved = True
                        break  # Restart to re-evaluate
            
            # Try swaps: remove one element, add another
            if not improved:
                # For each element in current, try replacing with a smaller element
                for elem in range(1, N + 1):
                    if not ((current >> (elem - 1)) & 1):
                        continue
                    
                    # Try removing elem and adding cand where cand < elem
                    best_cand = None
                    for cand in range(1, elem):
                        if (current >> (cand - 1)) & 1:
                            continue
                        # Check if cand is valid in current without elem
                        temp = current & ~(1 << (elem - 1))
                        if is_valid_addition(temp, cand):
                            best_cand = cand
                            break  # Smallest cand gives best improvement
                    
                    if best_cand is not None:
                        new_bitmask = (current & ~(1 << (elem - 1))) | (1 << (best_cand - 1))
                        new_float = current_float - recip[elem] + recip[best_cand]
                        if new_float > best_float + 1e-15:
                            best_float = new_float
                            best = new_bitmask
                            current = new_bitmask
                            current_float = new_float
                            improved = True
                            break  # Restart
            
            if not improved:
                break
        
        return best
    
    # Phase 1: Generate initial solutions using different greedy orders
    orders = [
        list(range(1, N + 1)),  # Ascending
        list(range(N, 0, -1)),  # Descending
    ]
    
    # Add some random orders
    for _ in range(5):
        order = list(range(1, N + 1))
        random.shuffle(order)
        orders.append(order)
    
    best_bitmask = 0
    best_float = 0.0
    
    for order in orders:
        bm = greedy_from_bitmask(order)
        fl = reciprocal_sum_float(bm)
        if fl > best_float:
            best_float = fl
            best_bitmask = bm
    
    # Phase 2: Local search on best initial
    best_bitmask = local_search_fast(best_bitmask, max_iter=500)
    best_float = reciprocal_sum_float(best_bitmask)
    
    # Phase 3: Multiple restarts with perturbation
    restart_count = 0
    while (time.time() - start_time) < DEADLINE - 3:
        restart_count += 1
        
        # Perturb current best: remove 2-6 random elements
        current = best_bitmask
        elems = [x for x in range(1, N + 1) if (current >> (x - 1)) & 1]
        
        if not elems:
            break
        
        num_remove = random.randint(2, min(6, len(elems)))
        to_remove = random.sample(elems, num_remove)
        for r in to_remove:
            current &= ~(1 << (r - 1))
        
        # Re-greedy from this state with a random order
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        # Greedy from current state
        for x in order:
            if (current >> (x - 1)) & 1:
                continue
            if is_valid_addition(current, x):
                current |= (1 << (x - 1))
        
        # Local search
        current = local_search_fast(current, max_iter=100)
        fl = reciprocal_sum_float(current)
        
        if fl > best_float + 1e-15:
            best_float = fl
            best_bitmask = current
    
    # Final polish with more iterations
    best_bitmask = local_search_fast(best_bitmask, max_iter=1000)
    
    # Convert bitmask to list
    result = [x for x in range(1, N + 1) if (best_bitmask >> (x - 1)) & 1]
    
    return result
# EVOLVE-BLOCK-END
