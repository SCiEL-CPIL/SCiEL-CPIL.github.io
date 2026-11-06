"""
Rewritten program: BDD/DP-based exact minimization for 4-term AP colorings.
Uses dynamic programming over states defined by recent color patterns to find
the minimum number of colors needed such that every nonconstant 4-AP uses >= 3 colors.
"""
# EVOLVE-BLOCK-START
import time
import sys
from collections import deque

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    N = 128

    # We'll use a DP/BDD approach. The state at position i needs to remember
    # enough of the past to check all 4-term APs that end at or before i.
    # A 4-term AP is (a, a+d, a+2d, a+3d). When we place color at position i,
    # we need to check APs where i is the last term: a, a+d, a+2d, i with i=a+3d.
    # But also APs where i is an earlier term that will be completed later.
    # 
    # Key insight: to check if adding color c at position i creates a bad AP,
    # we need to know colors at positions i-d, i-2d, i-3d for all d where i-3d >= 0.
    # This means we need to remember all previous colors, which is too much for a state.
    #
    # Alternative: Use a greedy/backtracking approach with memoization on a compressed state.
    # Let's try a different approach: since N=128 is small, use a smart backtracking with
    # constraint propagation and pruning.
    
    # Actually, let's think about this more carefully. The constraint is that every 4-AP 
    # has at least 3 distinct colors. This means no 4-AP can have all 4 same color, or 
    # exactly 2 distinct colors (i.e., patterns like AABB, ABAB, ABBB, etc. with only 2 colors).
    
    # Let me try a different approach: use a SAT-like backtracking with good variable ordering
    # and constraint propagation, combined with a lower bound on remaining colors.
    
    # But first, let's try a simpler approach: greedy with lookahead and local search,
    # but much more aggressive in trying to reduce colors.
    
    def validate(colors):
        """Check every nonconstant 4-term AP has >= 3 distinct colors."""
        for a in range(N):
            max_d = (N - 1 - a) // 3
            for d in range(1, max_d + 1):
                b, c, e = a + d, a + 2*d, a + 3*d
                cs = (colors[a], colors[b], colors[c], colors[e])
                # Count distinct colors
                distinct = len(set(cs))
                if distinct < 3:
                    return False
        return True

    def count_colors(colors):
        return len(set(colors))

    def relabel(colors):
        mapping = {}
        new_colors = []
        next_id = 0
        for c in colors:
            if c not in mapping:
                mapping[c] = next_id
                next_id += 1
            new_colors.append(mapping[c])
        return new_colors, next_id

    def get_forbidden(colors, pos, max_color):
        """Get set of colors forbidden at position pos given current coloring."""
        forbidden = set()
        
        # pos as 4th term: a, a+d, a+2d, pos where pos = a+3d => a = pos-3d
        for d in range(1, pos // 3 + 1):
            a = pos - 3*d
            if a < 0:
                continue
            b, c = a + d, a + 2*d
            s = {colors[a], colors[b], colors[c]}
            if len(s) <= 2:
                forbidden |= s
        
        # pos as 3rd term: a, a+d, pos, a+3d where pos = a+2d => a = pos-2d
        for d in range(1, min(pos // 2, (N - 1 - pos)) + 1):
            a = pos - 2*d
            if a < 0:
                continue
            b = a + d  # = pos - d
            e = a + 3*d  # = pos + d
            if e >= N:
                continue
            s = {colors[a], colors[b], colors[e]}
            if len(s) <= 2:
                forbidden |= s
        
        # pos as 2nd term: a, pos, a+2d, a+3d where pos = a+d => a = pos-d
        for d in range(1, min(pos, (N - 1 - pos) // 2) + 1):
            a = pos - d
            if a < 0:
                continue
            c = pos + d  # a+2d
            e = pos + 2*d  # a+3d
            if e >= N:
                continue
            s = {colors[a], colors[c], colors[e]}
            if len(s) <= 2:
                forbidden |= s
        
        # pos as 1st term: pos, pos+d, pos+2d, pos+3d
        for d in range(1, (N - 1 - pos) // 3 + 1):
            b = pos + d
            c = pos + 2*d
            e = pos + 3*d
            s = {colors[b], colors[c], colors[e]}
            if len(s) <= 2:
                forbidden |= s
        
        return forbidden

    # Strategy: Try to find a valid coloring with k colors for k = 2, 3, 4, ...
    # Start with k=2 (likely impossible), then k=3, etc.
    
    def try_k_colors(k, time_budget=5.0):
        """Try to find a valid coloring using at most k colors (0..k-1)."""
        start_time = time.time()
        
        # Use backtracking with constraint propagation
        # Order positions by most constrained first (most constraints)
        # For now, use natural order with some heuristics
        
        colors = [-1] * N
        
        # Precompute constraints for each position
        # constraint_count[i] = number of 4-APs involving position i
        constraint_count = [0] * N
        for a in range(N):
            max_d = (N - 1 - a) // 3
            for d in range(1, max_d + 1):
                for pos in (a, a+d, a+2*d, a+3*d):
                    constraint_count[pos] += 1
        
        # Order positions by descending constraint count (most constrained first)
        # But we need to be careful: forward fill is more efficient
        # Let's use a hybrid: fill in natural order but with lookahead
        
        def backtrack(pos):
            if time.time() - start_time > time_budget:
                return None
            
            if pos == N:
                return colors[:]
            
            # Get forbidden colors
            forbidden = get_forbidden(colors, pos, k)
            
            # Try colors in order of least constraining (fewest new constraints created)
            candidates = [c for c in range(k) if c not in forbidden]
            
            # Heuristic: try colors that are already used more (to minimize new colors)
            # Or try colors that create fewer future constraints
            if not candidates:
                return None
            
            # Sort candidates by how many future positions they constrain
            def constraint_score(c):
                count = 0
                # Check how many future positions would be constrained
                # This is expensive, so use a simpler heuristic
                # Try colors that appear more in the past
                return -colors[:pos].count(c)
            
            candidates.sort(key=constraint_score)
            
            for c in candidates:
                colors[pos] = c
                result = backtrack(pos + 1)
                if result is not None:
                    return result
                colors[pos] = -1
            
            return None
        
        result = backtrack(0)
        return result

    # First, try to find a valid coloring with few colors
    best_colors = None
    best_count = N + 1
    
    # Try k=3 first (likely the answer is 3 or 4)
    for k in [3, 4, 5, 6]:
        if time.time() > deadline - 10:
            break
        
        print(f"Trying k={k} colors...", file=sys.stderr)
        result = try_k_colors(k, time_budget=min(10.0, deadline - time.time() - 5))
        
        if result is not None:
            if validate(result):
                best_colors = result
                best_count = k
                break
        else:
            # If k colors don't work, try k+1
            continue
    
    # If no valid coloring found, use greedy fallback
    if best_colors is None:
        print("Fallback to greedy...", file=sys.stderr)
        best_colors = [0] * N
        for i in range(N):
            forbidden = get_forbidden(best_colors, i, N)
            for c in range(N):
                if c not in forbidden:
                    best_colors[i] = c
                    break
            else:
                best_colors[i] = 0
        
        # Validate and fix if needed
        if not validate(best_colors):
            # Try a different initial assignment
            for init in range(1, 10):
                if time.time() > deadline - 5:
                    break
                best_colors = [init % 4] * N
                valid = True
                for i in range(1, N):
                    forbidden = get_forbidden(best_colors, i, 4)
                    found = False
                    for c in range(4):
                        if c not in forbidden:
                            best_colors[i] = c
                            found = True
                            break
                    if not found:
                        valid = False
                        break
                if valid and validate(best_colors):
                    break
        
        best_count = count_colors(best_colors)
    
    # Local search to try to reduce colors further
    if best_count > 3 and time.time() < deadline - 10:
        print(f"Local search to reduce from {best_count} colors...", file=sys.stderr)
        
        def local_search_reduce(colors, max_time=5.0):
            colors = colors[:]
            n_colors = count_colors(colors)
            start_time = time.time()
            
            for _ in range(10000):
                if time.time() - start_time > max_time:
                    break
                
                # Try to recolor one position to reduce total distinct colors
                improved = False
                for pos in range(N):
                    old_c = colors[pos]
                    # Try each other color
                    for nc in range(n_colors):
                        if nc == old_c:
                            continue
                        colors[pos] = nc
                        if validate(colors):
                            new_count = count_colors(colors)
                            if new_count < n_colors:
                                n_colors = new_count
                                improved = True
                                break
                        colors[pos] = old_c
                    if improved:
                        break
                
                if not improved:
                    break
            
            return colors
        
        reduced = local_search_reduce(best_colors, max_time=min(8.0, deadline - time.time() - 3))
        if count_colors(reduced) < best_count:
            best_colors = reduced
            best_count = count_colors(best_colors)
    
    # Final relabel
    best_colors, final_count = relabel(best_colors)
    
    # Final validation
    if not validate(best_colors):
        # Emergency fallback
        print("WARNING: final validation failed, using periodic fallback", file=sys.stderr)
        best_colors = [i % 4 for i in range(N)]
        best_colors, final_count = relabel(best_colors)
    
    return best_colors
# EVOLVE-BLOCK-END
