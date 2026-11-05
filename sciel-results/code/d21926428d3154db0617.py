import random
import time
from collections import defaultdict

def construct():
    MAX_VAL = 128
    MIN_VAL = 2
    CANDIDATES = list(range(MIN_VAL, MAX_VAL + 1))
    
    def is_valid_extension(seq, x):
        """
        Check if adding x to the sorted sequence seq results in a valid sequence.
        seq is assumed to be sorted. x is not in seq.
        Returns True if valid, False otherwise.
        Also returns the new sorted sequence if valid.
        """
        # Determine insertion point to keep it sorted
        # Since we just need to check validity, we can simulate the insertion.
        # However, for speed, we can just check if x creates a duplicate product.
        # We need to check all subarrays that include x.
        
        n = len(seq)
        # Find insertion index
        lo, hi = 0, n
        while lo < hi:
            mid = (lo + hi) // 2
            if seq[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        ins_idx = lo
        
        # We need to check if any product involving x is already present in the set of products of seq.
        # And also if any new product involving x is duplicated with another new product involving x?
        # Actually, if we maintain a global set of products for the current seq, we just need to ensure
        # no product formed by x and a suffix/prefix of seq is already in the set.
        # But wait, the problem requires ALL consecutive blocks to have distinct products.
        # If we add x, the new blocks are:
        # 1. Blocks ending at x (starting from some i to x)
        # 2. Blocks starting at x (starting from x to some j)
        # 3. Blocks containing x in the middle? No, x is inserted.
        # The new blocks are exactly those that include the new element x.
        # Any block not including x was already valid.
        # So we just need to check:
        # A) Is any new block's product equal to an existing block's product (from the old seq)?
        # B) Is any new block's product equal to another new block's product?
        
        # To do this efficiently, we should maintain the set of products.
        # Let's assume we have a set `prods` for the current `seq`.
        # We will generate all products for blocks involving x.
        
        # Let's precompute prefix products of seq?
        # It's easier to just iterate.
        # Max length is ~20-30. N^2 is small.
        
        # Generate all products of subarrays in (seq + [x]) that include x.
        # Let new_seq = seq[:ins_idx] + [x] + seq[ins_idx:]
        
        # We need to check against the set of products of `seq`.
        # Let's pass the set of products as an argument for efficiency in the main loop.
        pass # Placeholder, will implement in the solver class

    class CSPSolver:
        def __init__(self, candidates, time_limit):
            self.candidates = candidates
            self.time_limit = time_limit
            self.best_seq = []
            self.start_time = time.time()
            self.random_seed = random.randint(0, 2**32)
            self.rng = random.Random(self.random_seed)

        def solve(self):
            # We will try to build the sequence by adding elements one by one.
            # State: current_seq (sorted list), current_prods (set of products), used_elements (set)
            # We use backtracking.
            
            # To make it faster, we can use a heuristic to order the candidates.
            # Often, smaller numbers are "denser" but create more collisions.
            # Let's try a randomized greedy backtracking.
            
            # We'll perform multiple restarts with different random seeds/orders.
            
            while time.time() - self.start_time < self.time_limit:
                self._run_single_search()
                
            return self.best_seq

        def _run_single_search(self):
            # Randomize the order of candidates to explore different paths
            candidates = self.candidates[:]
            self.rng.shuffle(candidates)
            
            # Initial state
            seq = []
            prods = set()
            used = set()
            
            # We will recursively try to add elements.
            # To avoid deep recursion limits and allow for time checking, 
            # we implement an iterative deepening or just a standard DFS with a stack.
            # Given N=127, depth can be up to 127. Python recursion limit is usually 1000, so it's fine.
            # But we need to be careful about performance.
            
            # Let's use a stack for backtracking: [(seq, prods, used, next_candidate_index)]
            # Actually, a recursive function is cleaner.
            
            def dfs(seq, prods, used, candidates_list):
                if time.time() - self.start_time > self.time_limit:
                    return
                
                if len(seq) > len(self.best_seq):
                    self.best_seq = seq[:]
                
                if not candidates_list:
                    return
                
                # Heuristic: Try candidates that are "compatible" first?
                # Or just try them in the given order.
                
                for i, x in enumerate(candidates_list):
                    if x in used:
                        continue
                    
                    # Check if adding x is valid
                    if self._is_valid_addition(seq, prods, x):
                        # Update state
                        new_seq, new_prods = self._get_new_state(seq, prods, x)
                        new_used = used | {x}
                        
                        # Recurse with remaining candidates
                        # We can filter candidates_list to only include those > x if we enforce sorted order?
                        # The problem says "Return strictly increasing integers".
                        # So the output must be sorted.
                        # If we build it in sorted order, we can just pick the next element from those > last element?
                        # No, we can pick any unused element, but we must insert it in sorted position.
                        # However, if we always pick the next element to be larger than the previous, 
                        # we restrict the search space to sorted sequences built left-to-right.
                        # This is a valid constraint because the final sequence is sorted.
                        # So, we can enforce that we only pick x > seq[-1] (if seq is not empty).
                        
                        if seq and x <= seq[-1]:
                            continue
                        
                        # Optimization: Only consider candidates > seq[-1]
                        # This reduces the branching factor significantly.
                        
                        dfs(new_seq, new_prods, new_used, candidates_list)
                        
                        # Backtrack
                        # No need to explicitly backtrack state variables if they are passed by value/copy
                        # But prods and used are mutable sets. We need to copy them or undo changes.
                        # Copying sets is O(N). N is small.
                        
                        if time.time() - self.start_time > self.time_limit:
                            return
                
            # To improve efficiency, we should filter candidates_list dynamically.
            # Let's refactor to pass the list of available candidates > last element.
            
            def dfs_optimized(seq, prods, available):
                if time.time() - self.start_time > self.time_limit:
                    return
                
                if len(seq) > len(self.best_seq):
                    self.best_seq = seq[:]
                
                if not available:
                    return
                
                # Heuristic: Try smaller numbers first? Or larger?
                # Smaller numbers create more products (denser), might lead to collisions sooner.
                # Larger numbers are "sparser".
                # Let's try a mix. We already shuffled.
                
                for i, x in enumerate(available):
                    # available is already sorted and contains only elements > seq[-1]
                    
                    if self._is_valid_addition(seq, prods, x):
                        new_seq = seq + [x]
                        new_prods = self._calculate_new_prods(seq, prods, x)
                        
                        # Filter available for next step: elements > x
                        next_available = [y for y in available[i+1:] if y > x]
                        
                        dfs_optimized(new_seq, new_prods, next_available)
                        
                        if time.time() - self.start_time > self.time_limit:
                            return

            # Initial call
            # available is all candidates
            # We can sort them to maintain the "x > seq[-1]" invariant easily
            # But we shuffled them. 
            # If we shuffle, we lose the sorted property needed for "next_available = available[i+1:]".
            # We need to handle this.
            # Option 1: Keep available sorted, but choose which one to pick next randomly from the valid set?
            # Option 2: Shuffle, then for each step, filter and sort the remaining? Too slow.
            
            # Better Approach:
            # Maintain `available` as a sorted list.
            # At each step, pick a random index from `available` to try first?
            # No, backtracking needs to explore all.
            
            # Let's stick to the sorted available list, but randomize the *order of exploration* 
            # by shuffling the initial candidates? 
            # If we shuffle, we can't use the `available[i+1:]` trick directly because the list isn't sorted by value anymore.
            
            # Revised Strategy:
            # 1. Start with sorted candidates.
            # 2. In DFS, we have a sorted list of `available`.
            # 3. We iterate through `available`.
            # 4. To introduce randomness, we can randomly select a subset of candidates to try first, or just rely on the fact that different runs might be seeded differently?
            # Actually, if we always iterate in sorted order, the search is deterministic.
            # To get diversity, we can randomly skip some candidates or reorder them?
            
            # Simplest Diversity: Randomly remove some candidates from the initial pool? No, we want max length.
            # Let's just run the deterministic search first. If it's fast enough, we can try to add randomness later.
            # Given N=127, the search tree is huge.
            
            # Let's use a heuristic: "Most Constrained First"?
            # For now, let's try the deterministic sorted search.
            
            # Wait, the previous program scored 1.0261.
            # A simple greedy might score lower.
            # Backtracking should find the optimal or near-optimal.
            
            # Let's implement the DFS with sorted available.
            # To add randomness, we can shuffle the *order in which we check* the available candidates?
            # If we shuffle `available`, we lose the sorted property for filtering.
            # We can keep `available` sorted, but pick a random element to try, then recurse, then backtrack?
            # That's not standard DFS.
            
            # Let's just do a standard DFS on sorted available.
            # It might be slow if it gets stuck.
            # We will add a "depth-first" bias.
            
            # Actually, let's try a different heuristic: 
            # Try to add elements that don't create many new products?
            
            # For now, let's implement the clean DFS.
            
            # We need `_is_valid_addition` and `_calculate_new_prods`.
            
            # To speed up, we can precompute products? No, depends on seq.
            
            # Let's optimize `_is_valid_addition`:
            # When adding x to seq (sorted), the new subarrays are:
            # 1. x alone
            # 2. seq[i] * ... * seq[k] * x  (suffixes of seq ending before x, multiplied by x? No, x is at the end if we append)
            # Wait, in `dfs_optimized`, we always append x to the end of seq.
            # So seq is always a prefix of the final sorted sequence?
            # NO. The final sequence is sorted. If we build it by appending, we are building the sorted sequence from left to right.
            # This is valid.
            # So `seq` is the prefix. `x` is the next element.
            # The new subarrays are:
            # - [x]
            # - [seq[i], ..., seq[n-1], x] for all i in 0..n-1
            
            # We need to check if any of these products are in `prods` (which contains products of all subarrays of `seq`).
            # Also, we need to check if any of these new products are duplicates of each other.
            # Since they all end with x, and seq is strictly increasing, 
            # prod(seq[i..n-1] + x) = prod(seq[i..n-1]) * x.
            # Since seq is strictly increasing, prod(seq[i..n-1]) is strictly increasing with decreasing i (i.e., longer suffixes have larger products).
            # So the new products are distinct from each other.
            # We only need to check if they are in `prods`.
            
            def _is_valid_addition_fast(seq, prods, x):
                # Check if x is in prods (singleton)
                if x in prods:
                    return False
                
                # Check suffixes
                p = x
                # We need products of seq[i..end] * x
                # Let's compute suffix products of seq.
                # suffix_prod[i] = prod(seq[i..end])
                # We can compute this on the fly or precompute.
                # Since we are in a tight loop, let's compute backwards.
                
                # Actually, we can just iterate.
                # p = x
                # for i from n-1 down to 0:
                #   p = p * seq[i]
                #   if p in prods: return False
                #   # Also add p to a temporary set to check for duplicates among new ones?
                #   # As argued, new ones are distinct.
                
                # But wait, `prods` contains products of ALL subarrays of `seq`.
                # So if `p` is in `prods`, it's a collision.
                
                p = x
                if p in prods:
                    return False
                
                # Multiply by seq elements from end to start
                for i in range(len(seq) - 1, -1, -1):
                    p *= seq[i]
                    if p in prods:
                        return False
                    # Note: p can become very large. Python handles big ints, but it might be slow.
                    # However, 128^30 is huge. But we stop if collision found.
                    # Also, if p exceeds some limit? No limit specified, but distinctness is key.
                    
                return True

            def _get_new_prods_fast(seq, prods, x):
                # Return the new set of products.
                # New products are:
                # 1. x
                # 2. seq[i..end] * x for all i
                
                new_prods = set(prods) # Copy
                p = x
                new_prods.add(p)
                
                for i in range(len(seq) - 1, -1, -1):
                    p *= seq[i]
                    new_prods.add(p)
                    
                return new_prods

            # The DFS
            # We need to be careful with time.
            # We'll run the search.
            
            # Initial state:
            seq = []
            prods = set()
            available = self.candidates[:] # Sorted
            
            dfs_optimized(seq, prods, available)

        def _is_valid_addition(self, seq, prods, x):
            # Implemented inside _run_single_search for closure, but let's move it here for clarity
            # Actually, the inner functions in _run_single_search can access self.
            pass

    # Run the solver
    solver = CSPSolver(CANDIDATES, time_limit=55.0)
    result = solver.solve()
    
    # Fallback if solver fails or returns empty (shouldn't happen)
    if not result:
        # Simple greedy fallback
        seq = []
        prods = set()
        for x in CANDIDATES:
            # Check validity
            valid = True
            if x in prods:
                valid = False
            else:
                p = x
                for i in range(len(seq) - 1, -1, -1):
                    p *= seq[i]
                    if p in prods:
                        valid = False
                        break
            if valid:
                seq.append(x)
                # Update prods
                p = x
                prods.add(p)
                for i in range(len(seq) - 2, -1, -1): # seq[-1] is x
                    p *= seq[i]
                    prods.add(p)
        result = seq
        
    return result
