import math
import time

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    # Calculate LCM
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
        
    def lcm(a, b):
        return a * b // gcd(a, b)
        
    L = 1
    for d in divisors:
        L = lcm(L, d)
        
    # We search for minimum W
    # Max W is bounded by L, but likely small.
    # We'll scan W from 1 to L (or a safe upper bound like 10000)
    
    # Precompute for speed?
    # For each S, we need to know which divisors have S as a multiple.
    # divs_at[S] = list of indices i such that S % divisors[i] == 0
    
    # This array is size L+1.
    # L = 360360.
    # divs_at = [[] for _ in range(L+1)]
    # for i, d in enumerate(divisors):
    #     for k in range(d, L+1, d):
    #         divs_at[k].append(i)
        
    # This precomputation is sum(L/d_i) ~ 200k operations. Very fast.
    
    divs_at = [[] for _ in range(L + 1)]
    for i, d in enumerate(divisors):
        for k in range(d, L + 1, d):
            divs_at[k].append(i)
            
    # Helper to check if a valid matching exists for current sets C
    def is_feasible(C):
        # C is list of sets C[0..9]
        # Check sizes
        for i in range(n):
            if len(C[i]) == 0:
                return False
                
        # Quick check for size 1 conflicts
        # If C[i] = {v} and C[j] = {v}, then i and j must map to v, conflict.
        # More generally, if union of C_i for i in J is size < |J|, fail.
        # Let's do a simple backtracking.
        
        # Order divisors by size of C[i]
        order = sorted(range(n), key=lambda i: len(C[i]))
        
        used_vals = set()
        
        def backtrack(idx):
            if idx == n:
                return True
            i = order[idx]
            for v in C[i]:
                if v not in used_vals:
                    used_vals.add(v)
                    if backtrack(idx + 1):
                        return True
                    used_vals.remove(v)
            return False
            
        return backtrack(0)

    best_range = None
    best_solution = None
    
    # We can just iterate W from 1 upwards
    # Upper bound for W: The seed solution has range 33.
    # So we expect W <= 33.
    # Let's scan up to 100 to be safe.
    
    for W in range(1, 101):
        # Initialize C for S=1
        # Window [1, 1+W]
        C = [set() for _ in range(n)]
        # Add values in [1, 1+W]
        for v in range(1, 1 + W + 1):
            for i in divs_at[v]:
                C[i].add(v)
                
        found = False
        # Slide S from 1 to L
        # Note: We only need to check S such that the window is within [1, L]?
        # Actually, the problem says positive integers <= 1000000.
        # Our search in [1, L] covers all residue classes.
        # If a solution exists in [kL, kL+L], it maps to [1, L] by subtracting (k-1)L.
        # The range is preserved.
        # So checking S in 1..L is sufficient.
        
        for S in range(1, L + 1):
            if is_feasible(C):
                # Found a window [S, S+W] that works
                # We need to construct the actual solution.
                # We can just return the values from the matching.
                # But we need to output the list of 10 ints.
                # We can re-run the matching to get the values.
                
                # Re-run matching to get values
                order = sorted(range(n), key=lambda i: len(C[i]))
                used_vals = set()
                assignment = [None] * n
                
                def backtrack_assign(idx):
                    if idx == n:
                        return True
                    i = order[idx]
                    for v in C[i]:
                        if v not in used_vals:
                            used_vals.add(v)
                            assignment[i] = v
                            if backtrack_assign(idx + 1):
                                return True
                            used_vals.remove(v)
                            assignment[i] = None
                    return False
                    
                if backtrack_assign(0):
                    # Verify distinctness and divisibility
                    # And that they are <= 1000000
                    # Since S <= L and W < 100, max val <= L + 100 < 1000000
                    # And min val >= 1
                    # So they are valid.
                    return [int(assignment[i]) for i in range(n)]
            
            # Slide window: remove S, add S+W+1
            # Remove S
            for i in divs_at[S]:
                C[i].discard(S)
            # Add S+W+1
            next_v = S + W + 1
            if next_v <= L:
                for i in divs_at[next_v]:
                    C[i].add(next_v)
            # If next_v > L, we don't add. But we are iterating S up to L.
            # The window [S, S+W] might extend beyond L.
            # However, we established that checking residues in 1..L is enough.
            # Does the window need to be contained in 1..L?
            # No, the values just need to be distinct positive integers.
            # But due to periodicity, any valid set of residues corresponds to a valid set in 1..L.
            # Wait. If the optimal window wraps around L?
            # e.g., multiples near L and 1.
            # Since we are checking all S in 1..L, and the sets C[i] are derived from values in [S, S+W],
            # if S+W > L, we are including values > L.
            # But values > L are congruent to values in 1..L.
            # The distinctness constraint is on the actual integers.
            # If we pick $x_i = L + r_i$, it is distinct from $x_j = r_j$ if $r_i \neq r_j$.
            # But if we pick $x_i = r_i$ and $x_j = r_j$, they are distinct if $r_i \neq r_j$.
            # The range is the same.
            # So we can just restrict our search to values in $[1, L]$.
            # This implies the window $[S, S+W]$ must be subset of $[1, L]$?
            # Not necessarily. But if we find a solution with values in $[1, L]$, that's fine.
            # If the optimal solution requires "wrapping", it would correspond to a window in the circular domain.
            # However, since $L$ is large and $W$ is small, wrapping is unlikely to be optimal compared to a non-wrapping window.
            # Actually, just checking $S$ from $1$ to $L$ and allowing values up to $L+W$ is safe.
            # But we must ensure values are $\le 1000000$.
            # $L+W \approx 360k$, which is fine.
            
    # If no solution found in W=1..100, fallback to seed
    return divisors[:]

# Test
if __name__ == "__main__":
    res = construct()
    print(res)
    print(max(res) - min(res))
