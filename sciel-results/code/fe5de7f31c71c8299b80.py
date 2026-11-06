"""Rewritten program for large sum-free subsets of three fixed sets."""
# EVOLVE-BLOCK-START
import time
import itertools
from random import Random

def construct():
    # Deadline guard: evaluator kills after 150s; use 140s as hard stop
    t_start = time.time()
    DEADLINE = 140.0  # seconds

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        """Check if set S is sum-free: no a+b=c with a,b,c in S (a=b allowed)."""
        Sset = set(S)
        for a in S:
            for b in S:
                if a + b in Sset:
                    return False
        return True

    def greedy_add(S, candidates):
        """
        Given a current sum-free set S, try to add each candidate in 'candidates'
        (in the given order) if it keeps S sum-free. Returns the augmented set.
        """
        Sset = set(S)
        # Precompute: for each x in Sset, the set of forbidden additions
        # If we add x, then for any a in Sset, a+x must not be in Sset,
        # and x+a must not be in Sset (same thing), and x+x must not be in Sset.
        # More precisely: adding x is safe iff for all a in Sset: a+x not in Sset,
        # and x+x not in Sset.
        # Equivalently: (Sset + x) ∩ Sset = empty AND x+x not in Sset.
        # But we also need: for all a in Sset, a+x not in Sset.
        
        result = list(S)
        result_set = set(S)
        
        for x in candidates:
            if x in result_set:
                continue
            # Check if adding x is safe
            safe = True
            # x + x must not be in result_set
            if 2 * x in result_set:
                safe = False
            # For each a in result_set, a + x must not be in result_set
            if safe:
                for a in result_set:
                    if a + x in result_set:
                        safe = False
                        break
            if safe:
                result.append(x)
                result_set.add(x)
        return result

    def local_search_remove_and_readd(S, all_elements, max_iterations=50):
        """
        Try to improve S by removing one element and re-adding others.
        Simple 1-move local search: for each element in S, try removing it
        and then greedily re-adding from the remaining candidates.
        """
        Sset = set(S)
        candidates = [x for x in all_elements if x not in Sset]
        best = list(S)
        best_set = set(S)
        
        for _ in range(max_iterations):
            improved = False
            for i, elem in enumerate(best):
                # Try removing elem
                trial = best[:i] + best[i+1:]
                trial_set = set(trial)
                # Now greedily add from candidates
                trial_result = greedy_add(trial, candidates)
                if len(trial_result) > len(best):
                    best = trial_result
                    best_set = set(best)
                    candidates = [x for x in all_elements if x not in best_set]
                    improved = True
                    break
            if not improved:
                break
        return best

    def multi_strategy_search(all_elements, time_limit):
        """
        Try multiple initialization strategies and local search to find a large
        sum-free subset.
        """
        n = len(all_elements)
        best_result = []
        best_size = 0
        
        # Strategy 1: Odds only
        odds = [x for x in all_elements if x % 2 != 0]
        s1 = greedy_add([], odds)
        if len(s1) > best_size:
            best_size = len(s1)
            best_result = s1
        
        # Strategy 2: Positives only
        pos = [x for x in all_elements if x > 0]
        s2 = greedy_add([], pos)
        if len(s2) > best_size:
            best_size = len(s2)
            best_result = s2
        
        # Strategy 3: Negatives only
        neg = [x for x in all_elements if x < 0]
        s3 = greedy_add([], neg)
        if len(s3) > best_size:
            best_size = len(s3)
            best_result = s3
        
        # Strategy 4: Large absolute values first
        by_abs = sorted(all_elements, key=lambda x: -abs(x))
        s4 = greedy_add([], by_abs)
        if len(s4) > best_size:
            best_size = len(s4)
            best_result = s4
        
        # Strategy 5: Random orderings
        rng = Random(42)
        for _ in range(5):
            shuffled = list(all_elements)
            rng.shuffle(shuffled)
            s5 = greedy_add([], shuffled)
            if len(s5) > best_size:
                best_size = len(s5)
                best_result = s5
        
        # Local search on the best result found so far
        if best_result:
            t_local = time.time()
            improved = local_search_remove_and_readd(best_result, all_elements, max_iterations=10)
            if len(improved) > best_size:
                best_result = improved
                best_size = len(improved)
        
        return best_result

    # Process each instance
    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        t_inst = time.time()
        if t_inst - t_start > DEADLINE:
            # Emergency: return what we have plus trivial solutions
            all_elems = instance(size, seed)
            # Just use odds as fallback
            results.append([x for x in all_elems if x % 2 != 0])
            continue
        
        all_elements = instance(size, seed)
        
        # Budget per instance: proportional but capped
        remaining = DEADLINE - (time.time() - t_start)
        # Reserve 5s for remaining instances
        if len(results) == 0:
            budget = remaining - 10.0  # 2 instances left
        elif len(results) == 1:
            budget = remaining - 5.0   # 1 instance left
        else:
            budget = remaining - 1.0
        
        budget = max(budget, 5.0)  # minimum 5s per instance
        
        result = multi_strategy_search(all_elements, budget)
        results.append(result)
    
    # Final check: verify sum-free property
    for i, res in enumerate(results):
        if not is_sum_free(res):
            # Fallback to odds
            size, seed = ((60, 801), (120, 802), (240, 803))[i]
            all_elems = instance(size, seed)
            results[i] = [x for x in all_elems if x % 2 != 0]
    
    return results
# EVOLVE-BLOCK-END
