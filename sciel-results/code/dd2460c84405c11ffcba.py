import sys
sys.setrecursionlimit(10000)

def construct(n_rows, k):
    # k is 5, so 6 marks
    target_len = k + 1
    # We want to find n_rows disjoint Golomb rulers of target_len marks
    # Minimizing the max element
    
    # 1. Generate candidates
    # We don't know the optimal max element, so we search incrementally?
    # Or generate up to a safe limit (e.g., 150) and sort.
    
    candidates = []
    # Generate all Golomb rulers of length target_len with max element <= 150
    # Using backtracking
    
    def gen_rulers(current_row, used_diffs, max_val):
        if len(current_row) == target_len:
            # Check if valid (differences distinct)
            # They are distinct by construction if we prune correctly
            candidates.append((tuple(current_row), frozenset(used_diffs)))
            return
        
        start = current_row[-1] + 1
        # Heuristic: don't go too far
        for next_val in range(start, max_val + 1):
            new_diffs = set()
            valid = True
            for prev in current_row:
                d = next_val - prev
                if d in used_diffs or d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if valid:
                gen_rulers(current_row + [next_val], used_diffs | new_diffs, max_val)

    # This naive generation might be slow if max_val is 150 and branching is high.
    # Optimization: 
    # The number of Golomb rulers grows, but for 6 marks it's small.
    # Let's try max_val = 120 first.
    
    gen_rulers([0], set(), 120)
    
    # Sort by max element
    candidates.sort(key=lambda x: x[0][-1])
    
    # 2. Find 7 disjoint
    # Use backtracking
    solution = None
    
    def find_solution(idx, current_set, used_diffs):
        nonlocal solution
        if solution is not None:
            return
        if len(current_set) == n_rows:
            solution = current_set
            return
            
        for i in range(idx, len(candidates)):
            row, diffs = candidates[i]
            if not (diffs & used_diffs):
                find_solution(i + 1, current_set + [row], used_diffs | diffs)
                if solution:
                    return

    find_solution(0, [], set())
    
    if solution:
        return [list(r) for r in solution]
    else:
        # Fallback: return greedy if no solution found in time/space
        # Or expand max_val and retry
        # For robustness, let's just return a greedy solution if backtracking fails
        # But backtracking should find one.
        
        # Greedy fallback
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < target_len:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    target_len = k + 1
    # We assume n_rows=7, k=5 based on the prompt context
    # But make it generic
    
    # 1. Generate Candidates
    # To ensure we finish within 120s, we limit the search space.
    # We try max_val from 105 (lower bound) upwards.
    
    # Since generation is the bottleneck, we generate up to a reasonable limit (e.g., 120)
    # and then search.
    
    candidates = []
    
    def gen_rulers(current_row, used_diffs, max_val):
        if len(current_row) == target_len:
            candidates.append((tuple(current_row), frozenset(used_diffs)))
            return
        
        start = current_row[-1] + 1
        for next_val in range(start, max_val + 1):
            new_diffs = set()
            valid = True
            for prev in current_row:
                d = next_val - prev
                if d in used_diffs or d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if valid:
                gen_rulers(current_row + [next_val], used_diffs | new_diffs, max_val)

    # Generate up to 120
    gen_rulers([0], set(), 120)
    
    # Sort by max element
    candidates.sort(key=lambda x: x[0][-1])
    
    # 2. Backtrack to find n_rows disjoint
    solution = None
    
    def find_solution(idx, current_set, used_diffs):
        nonlocal solution
        if solution is not None:
            return
        if len(current_set) == n_rows:
            solution = current_set
            return
            
        # Pruning: if not enough candidates left, return
        if len(candidates) - idx < n_rows - len(current_set):
            return

        for i in range(idx, len(candidates)):
            row, diffs = candidates[i]
            if not (diffs & used_diffs):
                find_solution(i + 1, current_set + [row], used_diffs | diffs)
                if solution:
                    return

    find_solution(0, [], set())
    
    if solution:
        return [list(r) for r in solution]
    else:
        # Fallback greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < target_len:
                x += 1
                new = {x - a for a in row}
                if not (new & used):
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
# EVOLVE-BLOCK-END
