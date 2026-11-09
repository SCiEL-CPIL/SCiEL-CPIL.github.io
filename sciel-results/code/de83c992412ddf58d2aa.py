"""Model-and-optimize approach for integer sum-product extremal sets."""
import time
import random
import numpy as np
from collections import Counter

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85.0  # Safety margin before 90s kill
    n = 20
    max_val = 10000
    best_set = None
    best_score = float('inf')

    # Generate all 5-smooth numbers up to max_val
    smooth_numbers = set([1])
    # 2^a * 3^b * 5^c
    for a in range(14): # 2^13 = 8192
        v2 = 2 ** a
        if v2 > max_val: break
        for b in range(9): # 3^8 = 6561
            v3 = v2 * (3 ** b)
            if v3 > max_val: break
            for c in range(6): # 5^5 = 3125
                v5 = v3 * (5 ** c)
                if v5 > max_val: break
                smooth_numbers.add(v5)
    
    smooth_list = sorted(smooth_numbers)
    # Filter to valid range
    smooth_list = [x for x in smooth_list if 1 <= x <= max_val]
    
    # If we don't have enough (unlikely, 5-smooths are dense enough), fallback to all ints
    if len(smooth_list) < n:
        smooth_list = list(range(1, max_val + 1))
    
    # Precompute index lookup for smooth_list
    smooth_idx = {x: i for i, x in enumerate(smooth_list)}

    def fast_eval(A):
        """Faster evaluation using sorted list approach."""
        a = sorted(A)
        sums = set()
        prods = set()
        for i in range(len(a)):
            for j in range(i, len(a)):
                sums.add(a[i] + a[j])
                prods.add(a[i] * a[j])
        return max(len(sums), len(prods))

    class IncrementalEvaluator:
        def __init__(self, A):
            self.A = sorted(A)
            self.A_set = set(self.A)
            self.sum_counter = Counter()
            self.prod_counter = Counter()
            self._rebuild()
            
        def _rebuild(self):
            self.sum_counter.clear()
            self.prod_counter.clear()
            a = self.A
            for i in range(len(a)):
                for j in range(i, len(a)):
                    self.sum_counter[a[i] + a[j]] += 1
                    self.prod_counter[a[i] * a[j]] += 1
        
        def score(self):
            return max(len(self.sum_counter), len(self.prod_counter))
        
        def replace(self, old_val, new_val):
            """Replace old_val with new_val in the set and update counters."""
            if old_val == new_val:
                return
            if new_val in self.A_set:
                return False # Invalid move, duplicate
            
            # Update Sum Counter
            # Remove pairs involving old_val
            for x in self.A:
                if x == old_val:
                    continue
                s = x + old_val
                self.sum_counter[s] -= 1
                if self.sum_counter[s] == 0:
                    del self.sum_counter[s]
            s2 = 2 * old_val
            self.sum_counter[s2] -= 1
            if self.sum_counter[s2] == 0:
                del self.sum_counter[s2]
                
            # Update Prod Counter
            for x in self.A:
                if x == old_val:
                    continue
                p = x * old_val
                self.prod_counter[p] -= 1
                if self.prod_counter[p] == 0:
                    del self.prod_counter[p]
            p2 = old_val * old_val
            self.prod_counter[p2] -= 1
            if self.prod_counter[p2] == 0:
                del self.prod_counter[p2]
                
            # Add new_val
            # Add pairs involving new_val
            for x in self.A:
                if x == old_val:
                    continue
                s = x + new_val
                self.sum_counter[s] += 1
            s2 = 2 * new_val
            self.sum_counter[s2] += 1
            
            for x in self.A:
                if x == old_val:
                    continue
                p = x * new_val
                self.prod_counter[p] += 1
            p2 = new_val * new_val
            self.prod_counter[p2] += 1
            
            # Update A
            self.A = [new_val if x == old_val else x for x in self.A]
            self.A.sort()
            self.A_set = set(self.A)
            return True

    # --- Strategy: Local Search on 5-Smooth Lattice ---
    
    # Initial candidates
    candidates = []
    
    # 1. Smallest 5-smooths
    candidates.append(smooth_list[:n])
    
    # 2. Largest 5-smooths (might have smaller sumset? No, larger numbers -> larger sums usually, but products same structure)
    # Actually, scaling by a constant k multiplies sums by k and products by k^2. 
    # |kA+kA| = |A+A|. |kA*kA| = |A*A|.
    # So scaling doesn't change the score! 
    # We can just pick any scaling that fits in max_val.
    # The smallest 5-smooths are the most "compressed" in value, but the structure is what matters.
    
    # 3. Random subsets of 5-smooths
    random.seed(42)
    for _ in range(20):
        cand = random.sample(smooth_list, n)
        candidates.append(cand)
        
    # 4. Geometric-like: powers of 2, then 3s, then 5s
    # 2^0..2^13 (14 nums), 3^0..3^3 (4 nums), 5^0..5^2 (3 nums) -> 21 nums
    # Let's mix
    cand_mixed = []
    for a in range(14):
        cand_mixed.append(2**a)
    for b in range(1, 4):
        cand_mixed.append(3**b)
    for c in range(1, 3):
        cand_mixed.append(5**c)
    # Remove duplicates and keep first 20
    seen = set()
    unique_mixed = []
    for x in sorted(cand_mixed):
        if x not in seen and x <= max_val:
            seen.add(x)
            unique_mixed.append(x)
    if len(unique_mixed) >= n:
        candidates.append(unique_mixed[:n])

    # Evaluate candidates and pick best
    for A in candidates:
        if len(set(A)) != n: continue
        if max(A) > max_val: continue
        A_sorted = sorted(A)
        score = fast_eval(A_sorted)
        if score < best_score:
            best_score = score
            best_set = A_sorted[:]

    # Local Search
    if best_set is None:
        best_set = smooth_list[:n]
        best_score = fast_eval(best_set)
        
    current = best_set[:]
    current_score = best_score
    evaluator = IncrementalEvaluator(current)
    
    no_improve = 0
    max_no_improve = 10000 # High limit, we are fast now
    
    # Pre-compute valid moves? No, dynamic is better.
    
    while no_improve < max_no_improve and time.time() < deadline - 2:
        # Pick a random index to replace
        i = random.randint(0, n - 1)
        old_val = current[i]
        
        # Pick a new value from smooth_list
        # Bias towards values close to old_val or random
        if random.random() < 0.5:
            # Random from smooth_list
            new_val = random.choice(smooth_list)
        else:
            # Near old_val in smooth_list
            idx = smooth_idx.get(old_val)
            if idx is not None:
                # Pick a neighbor
                offset = random.randint(-5, 5)
                new_idx = idx + offset
                if 0 <= new_idx < len(smooth_list):
                    new_val = smooth_list[new_idx]
                else:
                    new_val = random.choice(smooth_list)
            else:
                new_val = random.choice(smooth_list)
                
        if new_val == old_val or new_val in current:
            no_improve += 1
            continue
            
        # Try the move
        # We need to be able to revert if it fails
        # Since IncrementalEvaluator modifies state, we should copy or revert
        # Copying Counter is O(K) where K is size of set (~400). 
        # 10000 moves * 400 ops = 4M ops. Fast enough.
        
        # Save state
        old_sum_counter = evaluator.sum_counter.copy()
        old_prod_counter = evaluator.prod_counter.copy()
        old_A = current[:]
        
        # Apply
        success = evaluator.replace(old_val, new_val)
        if not success:
            no_improve += 1
            continue
            
        new_score = evaluator.score()
        
        if new_score < current_score:
            current = evaluator.A
            current_score = new_score
            no_improve = 0
            
            if current_score < best_score:
                best_score = current_score
                best_set = current[:]
                
        else:
            # Revert
            evaluator.sum_counter = old_sum_counter
            evaluator.prod_counter = old_prod_counter
            evaluator.A = old_A
            evaluator.A_set = set(old_A)
            no_improve += 1

    # Final Polish: Try all single replacements with all smooth numbers
    # This is expensive: 20 * 200 = 4000 full evals? No, incremental is O(n).
    # 4000 * 20 = 80k ops. Very fast.
    if time.time() < deadline - 1:
        current = best_set[:]
        current_score = best_score
        evaluator = IncrementalEvaluator(current)
        
        improved = True
        while improved and time.time() < deadline - 0.5:
            improved = False
            for i in range(n):
                old_val = current[i]
                for new_val in smooth_list:
                    if new_val == old_val or new_val in current:
                        continue
                    
                    old_sum_counter = evaluator.sum_counter.copy()
                    old_prod_counter = evaluator.prod_counter.copy()
                    old_A = current[:]
                    
                    success = evaluator.replace(old_val, new_val)
                    if not success:
                        continue
                        
                    new_score = evaluator.score()
                    
                    if new_score < current_score:
                        current = evaluator.A
                        current_score = new_score
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_set = current[:]
                        # Break inner loop to restart with new set
                        break
                    else:
                        # Revert
                        evaluator.sum_counter = old_sum_counter
                        evaluator.prod_counter = old_prod_counter
                        evaluator.A = old_A
                        evaluator.A_set = set(old_A)
                        
                if improved:
                    break # Restart outer loop

    return best_set
# EVOLVE-BLOCK-END
