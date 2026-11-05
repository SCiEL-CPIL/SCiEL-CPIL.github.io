import time
import random

def get_diffs(row):
    return {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}

def build_row(used_diffs, max_scope, k, randomize=False, rng=None):
    """
    Builds a row of k+1 integers starting at 0, strictly increasing, max <= max_scope.
    Differences must not be in used_diffs.
    Returns the row or None.
    """
    if rng is None:
        rng = random

    row = [0]
    
    def backtrack(current_row, next_min):
        if len(current_row) == k + 1:
            return list(current_row)
        
        # Pruning: Check if remaining space is sufficient
        if max_scope - current_row[-1] < (k + 1 - len(current_row)):
            return None

        start_x = current_row[-1] + 1
        end_x = max_scope
        
        # Collect valid candidates
        candidates = []
        for x in range(start_x, end_x + 1):
            new_diffs = {x - a for a in current_row}
            if not (new_diffs & used_diffs):
                candidates.append(x)
                
        if not candidates:
            return None
            
        if randomize and len(candidates) > 1:
            # Pick a random candidate
            # To bias towards smaller, we can pick from the first half?
            # Or just uniform random.
            x = rng.choice(candidates)
            # To avoid getting stuck in bad local minima too often, 
            # maybe bias towards smaller?
            # Let's just pick random for now to ensure diversity.
        else:
            # Pick smallest
            x = candidates[0]
            
        res = backtrack(current_row + [x], x + 1)
        if res is not None:
            return res
            
        # If randomize, we might want to try other candidates if the chosen one failed?
        # The current logic only tries ONE candidate if randomize is True.
        # This makes it a heuristic, not a complete search.
        # For the "Swap" step, we want a complete search to ensure we don't miss a solution.
        # For "Initial Generation", a heuristic is fine and faster.
        
        if not randomize:
            # In non-random mode, we should try all candidates?
            # No, the loop above only tried candidates[0] in the non-random branch?
            # Wait, the logic above picks ONE x and recurses.
            # If it fails, it returns None.
            # This is NOT a complete search.
            # I need to fix this.
            pass
        
        return None

    # Corrected Backtracking for Complete Search (Non-Random)
    def backtrack_complete(current_row, next_min):
        if len(current_row) == k + 1:
            return list(current_row)
        
        if max_scope - current_row[-1] < (k + 1 - len(current_row)):
            return None

        for x in range(next_min, max_scope + 1):
            new_diffs = {x - a for a in current_row}
            if new_diffs & used_diffs:
                continue
            
            res = backtrack_complete(current_row + [x], x + 1)
            if res is not None:
                return res
                
        return None

    # Corrected Backtracking for Heuristic Search (Random)
    def backtrack_heuristic(current_row, next_min):
        if len(current_row) == k + 1:
            return list(current_row)
        
        if max_scope - current_row[-1] < (k + 1 - len(current_row)):
            return None

        candidates = []
        for x in range(next_min, max_scope + 1):
            new_diffs = {x - a for a in current_row}
            if not (new_diffs & used_diffs):
                candidates.append(x)
                
        if not candidates:
            return None
            
        # Pick a candidate
        # To improve quality, pick a random one from the first 10% or something?
        # Let's just pick a random one.
        x = rng.choice(candidates)
        
        res = backtrack_heuristic(current_row + [x], x + 1)
        if res is not None:
            return res
            
        # If it failed, try another random candidate?
        # This can be expensive. 
        # Let's try up to 5 random candidates.
        remaining = [c for c in candidates if c != x]
        rng.shuffle(remaining)
        for c in remaining[:5]:
             # Check if c is still valid (it is, by construction)
             # But we need to ensure we don't recurse too deep on bad paths
             # Let's just try them sequentially
             res = backtrack_heuristic(current_row + [c], c + 1)
             if res is not None:
                 return res
                 
        return None

    if randomize:
        return backtrack_heuristic(row, 1)
    else:
        return backtrack_complete(row, 1)

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    best_rows = None
    best_scope = float('inf')
    
    # Initial max scope guess
    # Counting bound is 105, known best is 112.
    # Let's start with 150 to ensure we find a solution.
    initial_max_scope = 150
    
    while time.time() < deadline:
        # 1. Generate Initial Solution
        rows = []
        used_diffs = set()
        success = True
        for i in range(n_rows):
            # Use heuristic builder for speed and diversity
            row = build_row(used_diffs, initial_max_scope, k, randomize=True)
            if row is None:
                success = False
                break
            rows.append(row)
            used_diffs |= get_diffs(row)
            
        if not success:
            # If heuristic failed, try complete search with a larger scope?
            # Or just restart
            continue
            
        scope = max(max(r) for r in rows)
        
        # 2. Local Search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try to reduce scope
            for i in range(n_rows):
                others = rows[:i] + rows[i+1:]
                used_others = set()
                for r in others:
                    used_others |= get_diffs(r)
                
                target_scope = scope - 1
                # Use complete search to ensure we find a solution if one exists
                new_row = build_row(used_others, target_scope, k, randomize=False)
                
                if new_row is not None:
                    rows[i] = new_row
                    scope = target_scope
                    improved = True
                    break # Restart outer loop to re-evaluate with new scope
            
            if not improved:
                # Try to shuffle (find different row with same scope)
                for i in range(n_rows):
                    others = rows[:i] + rows[i+1:]
                    used_others = set()
                    for r in others:
                        used_others |= get_diffs(r)
                    
                    # Find a row different from rows[i]
                    # We can do this by temporarily adding rows[i]'s diffs to used?
                    # No, we just need a row compatible with used_others.
                    # To ensure it's different, we can check after finding.
                    # Or, we can modify build_row to exclude a specific row?
                    # Simpler: Find a row, if it's the same, try to find another?
                    # build_row returns the first one found (smallest).
                    # So if we want a different one, we need to modify the search.
                    
                    # For now, let's just try to find ANY row.
                    # If it's the same, we might be stuck.
                    # Let's implement a "find_different" by excluding the current row's diffs?
                    # No, the current row is compatible with used_others.
                    # We want a row $r_{new}$ such that $d(r_{new}) \cap used\_others = \emptyset$.
                    # And $r_{new} \neq rows[i]$.
                    
                    # We can just call build_row. If it returns rows[i], 
                    # we can try to find the next best one?
                    # Let's skip shuffle for now, as scope reduction is the main goal.
                    pass

        if scope < best_scope:
            best_scope = scope
            best_rows = rows
            
        if best_scope <= 105:
            break
            
    return best_rows
