import sys
import time

def construct(n_rows, k):
    """
    Constructs a difference triangle set.
    Approach: Generate all valid Golomb rulers (rows) for a given scope M,
    then find n_rows disjoint rulers using backtracking.
    """
    
    def generate_rows(M):
        """Generates all Golomb rulers of length k+1 with max element <= M."""
        rows = []
        
        def rec(row, diff_mask, min_x):
            if len(row) == k + 1:
                rows.append((row, diff_mask))
                return
            
            # Pruning: 
            # The remaining elements must fit in M.
            # If we are at index i, we need k+1 - i more elements.
            # The smallest possible next element is min_x.
            # The largest possible is M.
            # We need to ensure we can pick enough distinct numbers.
            
            # Heuristic: Try smaller x first? Or larger?
            # For finding *any* solution, order doesn't matter much.
            # For minimizing M, we generate for a fixed M.
            
            for x in range(min_x, M + 1):
                # Check differences
                valid = True
                new_diffs = 0
                for a in row:
                    d = x - a
                    if (diff_mask >> d) & 1:
                        valid = False
                        break
                    new_diffs |= (1 << d)
                
                if valid:
                    rec(row + [x], diff_mask | new_diffs, x + 1)
        
        rec([0], 0, 1)
        return rows

    def find_disjoint_rows(candidates, n_needed):
        """Finds n_needed rows with disjoint difference masks."""
        # Sort candidates to help symmetry breaking
        # We can sort by the row itself (lexicographic)
        candidates.sort(key=lambda x: x[0])
        
        solution = []
        
        def backtrack(start_idx, used_mask, count):
            if count == n_needed:
                return True
            
            for i in range(start_idx, len(candidates)):
                row, mask = candidates[i]
                if used_mask & mask:
                    continue
                
                solution.append(row)
                if backtrack(i + 1, used_mask | mask, count + 1):
                    return True
                solution.pop()
            
            return False
        
        if backtrack(0, 0, 0):
            return solution
        return None

    # Lower bound for scope M
    # Total differences = n_rows * (k*(k+1)//2)
    total_diffs = n_rows * (k * (k + 1) // 2)
    
    # Search for the minimum M
    # Start from the theoretical lower bound
    # The differences are distinct integers >= 1.
    # So M must be at least total_diffs?
    # Not exactly, M is the max element. The max difference is M.
    # The differences are a subset of [1, M].
    # So M >= total_diffs.
    
    lower_bound = total_diffs
    
    # We might need to search a bit above the lower bound.
    # The current program searches downwards from greedy.
    # Greedy might be large.
    # Let's try searching upwards from lower_bound.
    
    for M in range(lower_bound, lower_bound + 50): # Search up to 50 above LB
        candidates = generate_rows(M)
        if len(candidates) < n_rows:
            continue
        
        res = find_disjoint_rows(candidates, n_rows)
        if res:
            return res
            
    # Fallback: If no solution found in the tight range, use a wider search or greedy
    # For safety, let's try a greedy approach if the tight search fails
    def greedy():
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                valid = True
                for a in row:
                    d = x - a
                    if (used >> d) & 1:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    for a in row[:-1]:
                        d = x - a
                        used |= (1 << d)
            rows.append(row)
        return rows

    return greedy()

