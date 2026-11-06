"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
import time
import heapq
from math import gcd

def construct():
    deadline = time.time() + 85  # 85s wall-clock guard (90s limit, 5s margin)
    MIN_DENOM = 5
    MAX_DENOM = 10000000
    MAX_TERMS = 256
    
    def heuristic(remaining, last_denom):
        """Admissible heuristic: lower bound on number of terms needed."""
        if remaining.numerator == 0:
            return 0
        
        # Greedy lower bound: count how many terms the greedy method would need
        # from this state (but we can't actually run full greedy for each node)
        # Use: remaining * max_possible_denom gives an upper bound on sum of remaining terms
        # If we use terms >= last_denom+1, the maximum sum with k terms is bounded
        
        # Simple admissible heuristic: 
        # The largest possible term we can add is 1/(last_denom+1)
        # So we need at least ceil(remaining / (1/(last_denom+1))) terms
        # But this isn't quite right because denominators must be distinct and increasing
        
        # Better: use the fact that if remaining = p/q, we need at least
        # enough terms such that their sum can reach p/q
        # With denominators >= last_denom+1, the maximum sum of k terms is bounded
        
        # Actually, a simpler admissible heuristic:
        # The greedy choice for the next term is d_min = ceil(q/p)
        # After choosing d_min, remaining becomes (q - p)/(p*d_min) or similar
        # But this gets complex. Let's use a simpler bound.
        
        # Since all remaining denominators must be > last_denom,
        # and we need sum = remaining, we need at least:
        # k such that k * (1/(last_denom+1)) >= remaining
        # i.e., k >= remaining * (last_denom+1)
        
        if remaining.numerator == 0:
            return 0
            
        # remaining = p/q, so remaining * (last_denom+1) = p*(last_denom+1)/q
        # k >= ceil(p*(last_denom+1)/q)
        p = remaining.numerator
        q = remaining.denominator
        lower_bound = (p * (last_denom + 1) + q - 1) // q
        
        # Also, we can't use more terms than the denominator cap allows
        # But for admissibility, we just need a lower bound
        return max(1, lower_bound) if remaining > 0 else 0
    
    def a_star_search():
        """A* search for minimal Egyptian fraction representation."""
        start = (Fraction(1, 1), 0, [])  # (remaining, last_denom, path)
        
        # Priority queue: (f_score, g_score, remaining, last_denom, path)
        # f = g + h, where g = number of terms used so far, h = heuristic
        h0 = heuristic(start[0], start[1])
        open_set = [(h0, 0, start[0], start[1], start[2])]
        
        # Track best solution found
        best_solution = None
        best_length = float('inf')
        
        # Track visited states to avoid re-exploration
        # State: (remaining, last_denom) -> min g_score seen
        visited = {}
        
        while open_set:
            if time.time() > deadline:
                break
            
            f, g, remaining, last_denom, path = heapq.heappop(open_set)
            
            # Goal check
            if remaining == 0:
                if g < best_length:
                    best_length = g
                    best_solution = path[:]
                continue
            
            # Prune: if we already have a solution and this path can't beat it
            if g + heuristic(remaining, last_denom) >= best_length:
                continue
            
            # Check if we've seen this state with a better or equal g
            state_key = (remaining, last_denom)
            if state_key in visited and visited[state_key] <= g:
                continue
            visited[state_key] = g
            
            # Generate successors: try different denominators
            # The next denominator d must satisfy:
            # 1. d > last_denom
            # 2. 1/d <= remaining (so d >= ceil(1/remaining))
            # 3. d <= MAX_DENOM
            # 4. After subtracting 1/d, the remainder must be achievable
            
            if remaining.numerator == 0:
                continue
                
            # Lower bound for d: d >= ceil(remaining.denominator / remaining.numerator)
            d_min = (remaining.denominator + remaining.numerator - 1) // remaining.numerator
            d_min = max(d_min, last_denom + 1, MIN_DENOM)
            
            # Upper bound for d: we can't go too large or we'll need too many terms
            # Also, d must be such that the remainder is still achievable
            # Try a limited range of d values around the greedy choice
            
            # Greedy choice is d_min. Try d_min, d_min+1, ..., up to some limit
            # But we need to be careful about the search space
            
            # Let's try a range: from d_min to min(d_min + 20, MAX_DENOM)
            # This is a heuristic to limit the branching factor
            
            max_d_to_try = min(d_min + 50, MAX_DENOM)
            
            for d in range(d_min, max_d_to_try + 1):
                if d > MAX_DENOM:
                    break
                    
                # Check: 1/d <= remaining?
                # remaining = p/q, so 1/d <= p/q iff q <= p*d
                if remaining.denominator > remaining.numerator * d:
                    continue
                
                new_remaining = remaining - Fraction(1, d)
                
                if new_remaining < 0:
                    continue
                
                # Reduce the fraction
                if new_remaining.numerator == 0:
                    # Found a solution
                    new_path = path + [d]
                    if len(new_path) < best_length:
                        best_length = len(new_path)
                        best_solution = new_path
                    continue
                
                # Check if new_remaining can be achieved with remaining terms
                # The minimum number of additional terms needed is heuristic(new_remaining, d)
                h_new = heuristic(new_remaining, d)
                g_new = g + 1
                
                # Prune if this can't beat the best solution
                if g_new + h_new >= best_length:
                    continue
                
                # Also prune if we've used too many terms
                if g_new >= MAX_TERMS:
                    continue
                
                # Check if new_remaining is achievable:
                # The largest term we can add is 1/(d+1), so we need
                # enough terms to sum to new_remaining
                # If new_remaining > (MAX_TERMS - g_new) / (d+1), it's not achievable
                # This is a loose bound
                
                # Generate new state
                new_state_key = (new_remaining, d)
                if new_state_key in visited and visited[new_state_key] <= g_new:
                    continue
                
                f_new = g_new + h_new
                new_path = path + [d]
                
                heapq.heappush(open_set, (f_new, g_new, new_remaining, d, new_path))
        
        return best_solution
    
    # Run A* search
    solution = a_star_search()
    
    if solution is not None:
        return solution
    
    # Fallback: greedy approach
    def greedy_egyptian(target_frac, min_denom, max_denom=10000000):
        """Greedy Egyptian fraction with strictly increasing denominators >= min_denom."""
        result = []
        remaining = target_frac
        last_denom = min_denom - 1
        
        while remaining > 0:
            if time.time() > deadline:
                return None
            
            if remaining.numerator == 0:
                break
                
            lower = (remaining.denominator + remaining.numerator - 1) // remaining.numerator
            d = max(lower, last_denom + 1, min_denom)
            
            if d > max_denom:
                return None
            
            result.append(d)
            remaining = remaining - Fraction(1, d)
            last_denom = d
            
            if len(result) > 256:
                return None
        
        if remaining == 0:
            return result
        return None
    
    solution = greedy_egyptian(Fraction(1, 1), MIN_DENOM)
    
    if solution is not None:
        return solution
    
    raise RuntimeError("no baseline representation")
# EVOLVE-BLOCK-END
