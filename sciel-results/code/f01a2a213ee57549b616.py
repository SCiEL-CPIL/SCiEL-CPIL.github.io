"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from fractions import Fraction

def construct():
    # Deadline guard: return best found before 150s wall-clock
    t0 = time.time()
    deadline = 140.0  # 10s safety margin
    
    # Candidate pool: all integers 1..511
    # We want to maximize sum 1/(512-a) subject to pairwise coprime
    
    # Key insight: 1/(512-a) is largest when a is large (close to 511).
    # So we prefer large a values. But they must be pairwise coprime.
    
    # Greedy baseline: pick from largest to smallest, add if coprime to all chosen
    def greedy_baseline():
        chosen = []
        for a in range(511, 0, -1):
            if all(gcd(a, b) == 1 for b in chosen):
                chosen.append(a)
        return chosen
    
    def score(lst):
        return sum(1.0 / (512.0 - a) for a in lst)
    
    # Get baseline
    best = greedy_baseline()
    best_score = score(best)
    
    # Strategy: use local search / simulated annealing style
    # Operations: 
    #   1. Try to swap a small a with a larger a that is coprime to all others
    #   2. Try to add a new large a (removing conflicts if needed)
    #   3. Random restart with different orderings
    
    # Precompute coprime relations for speed
    # coprime_mask[a] = set of b in 1..511 with gcd(a,b)==1
    # Actually, let's just use gcd directly but be smart about ordering
    
    # Better approach: iterative improvement
    # Start with greedy, then try to improve by replacing smaller elements with larger coprime ones
    
    def try_improve(chosen, max_time=None):
        """Try to improve the set by replacing elements with larger coprime ones."""
        if max_time is None:
            max_time = deadline - 2.0
        improved = True
        while improved:
            if time.time() > max_time:
                break
            improved = False
            # Sort chosen by value (smallest first) to try replacing small ones
            for i in range(len(chosen)):
                if time.time() > max_time:
                    break
                current = chosen[i]
                # Try to replace with a larger value that's coprime to all others
                others = [chosen[j] for j in range(len(chosen)) if j != i]
                best_replace = None
                best_val = 1.0 / (512.0 - current)
                # Search from 511 down to current+1
                for candidate in range(511, current, -1):
                    if candidate in chosen:
                        continue
                    if all(gcd(candidate, b) == 1 for b in others):
                        val = 1.0 / (512.0 - candidate)
                        if val > best_val:
                            best_val = val
                            best_replace = candidate
                            break  # first one found (largest) is best
                if best_replace is not None:
                    chosen[i] = best_replace
                    improved = True
        
        return chosen
    
    # Try multiple restarts with different random orderings
    random.seed(42)
    
    # First, do a thorough greedy with improvement
    candidate = list(best)
    candidate = try_improve(candidate, deadline - 10.0)
    if score(candidate) > best_score:
        best = candidate
        best_score = score(best)
    
    # Try different heuristics for initial selection
    # Heuristic: prioritize by 1/(512-a) / (number of coprime conflicts)
    # Or: try weighted greedy
    
    # Let's try several random restarts
    for restart in range(200):
        if time.time() > deadline - 15.0:
            break
        
        # Create a random perturbation of the ordering
        candidates_list = list(range(1, 512))
        
        # Weight by 1/(512-a) but add some randomness
        weights = [1.0 / (512.0 - a) * random.uniform(0.5, 1.5) for a in candidates_list]
        
        # Sort by weight descending
        pairs = sorted(zip(candidates_list, weights), key=lambda x: -x[1])
        ordered = [p[0] for p in pairs]
        
        # Greedy from this ordering
        chosen = []
        for a in ordered:
            if all(gcd(a, b) == 1 for b in chosen):
                chosen.append(a)
        
        # Improve
        chosen = try_improve(chosen, deadline - 10.0)
        
        s = score(chosen)
        if s > best_score:
            best = chosen
            best_score = s
    
    # Final improvement pass
    best = try_improve(list(best), deadline - 2.0)
    
    return best
# EVOLVE-BLOCK-END
