"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    def greedy_solution(seed_val):
        """Greedy construction with randomization for diversity."""
        rng = random.Random(seed_val)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                # Try candidates in random order, but prefer small values
                candidates = list(range(x + 1, x + 50))
                rng.shuffle(candidates)
                # Sort by value but with some randomness
                candidates.sort(key=lambda v: (v, rng.random()))
                placed = False
                for cand in candidates:
                    new = {cand - a for a in row}
                    if not new & used:
                        row.append(cand)
                        used |= new
                        x = cand
                        placed = True
                        break
                if not placed:
                    # Fallback: try larger values
                    for cand in range(x + 1, x + 500):
                        new = {cand - a for a in row}
                        if not new & used:
                            row.append(cand)
                            used |= new
                            x = cand
                            placed = True
                            break
                if not placed:
                    # Should not happen, but safety
                    row.append(row[-1] + 1)
            rows.append(row)
        return rows
    
    def get_score(rows):
        """Return the largest entry (scope)."""
        return max(max(row) for row in rows)
    
    def is_valid(rows):
        """Check if all differences are distinct."""
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diff = row[j] - row[i]
                    if diff in used:
                        return False
                    used.add(diff)
        return True
    
    def local_search(rows, max_time=60):
        """Improve solution via local search with simulated annealing."""
        current = [list(r) for r in rows]
        current_score = get_score(current)
        best = [list(r) for r in current]
        best_score = current_score
        
        # Precompute differences for fast checking
        def get_all_diffs(rows):
            diffs = []
            for row in rows:
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        diffs.append((row[j] - row[i], i, j, row))
            return diffs
        
        # Try to reduce the maximum value
        while time.time() < max_time:
            # Find the row and position with the maximum value
            max_val = -1
            max_row_idx = -1
            max_pos_idx = -1
            for ri, row in enumerate(current):
                for pi, val in enumerate(row):
                    if val > max_val:
                        max_val = val
                        max_row_idx = ri
                        max_pos_idx = pi
            
            if max_val <= best_score:
                break
            
            # Try to move this element to a smaller value
            row = current[max_row_idx]
            old_val = row[max_pos_idx]
            
            # Collect all currently used differences
            used_diffs = set()
            for r in current:
                for i in range(len(r)):
                    for j in range(i + 1, len(r)):
                        used_diffs.add(r[j] - r[i])
            
            # Remove differences involving the element we're moving
            for i in range(len(row)):
                if i != max_pos_idx:
                    if max_pos_idx > i:
                        used_diffs.discard(row[max_pos_idx] - row[i])
                    else:
                        used_diffs.discard(row[i] - row[max_pos_idx])
            
            # Try smaller values
            found = False
            for new_val in range(1, old_val):
                # Check if new_val maintains increasing order
                if max_pos_idx > 0 and new_val <= row[max_pos_idx - 1]:
                    continue
                if max_pos_idx < len(row) - 1 and new_val >= row[max_pos_idx + 1]:
                    continue
                
                # Check if new differences conflict
                conflicts = False
                for i in range(len(row)):
                    if i != max_pos_idx:
                        diff = abs(new_val - row[i])
                        if diff in used_diffs:
                            conflicts = True
                            break
                if not conflicts:
                    row[max_pos_idx] = new_val
                    found = True
                    break
            
            if found:
                new_score = get_score(current)
                if new_score < best_score:
                    best = [list(r) for r in current]
                    best_score = new_score
            else:
                # Try random perturbations
                ri = random.randint(0, len(current) - 1)
                pi = random.randint(1, len(current[ri]) - 1)
                old_val = current[ri][pi]
                
                # Remove its differences
                row = current[ri]
                for i in range(len(row)):
                    if i != pi:
                        used_diffs.discard(abs(row[pi] - row[i]))
                
                # Try a nearby value
                for delta in [1, -1, 2, -2, 3, -3]:
                    new_val = old_val + delta
                    if new_val <= 0:
                        continue
                    if pi > 0 and new_val <= row[pi - 1]:
                        continue
                    if pi < len(row) - 1 and new_val >= row[pi + 1]:
                        continue
                    
                    conflicts = False
                    for i in range(len(row)):
                        if i != pi:
                            diff = abs(new_val - row[i])
                            if diff in used_diffs:
                                conflicts = True
                                break
                    if not conflicts:
                        row[pi] = new_val
                        new_score = get_score(current)
                        if new_score < best_score:
                            best = [list(r) for r in current]
                            best_score = new_score
                        break
        
        return best, best_score
    
    # Phase 1: Generate multiple greedy solutions
    best_rows = None
    best_score = float('inf')
    
    num_greedy = 20
    for i in range(num_greedy):
        if time.time() > deadline - 30:
            break
        rows = greedy_solution(i * 12345 + 678)
        score = get_score(rows)
        if score < best_score:
            best_score = score
            best_rows = [list(r) for r in rows]
    
    if best_rows is None:
        best_rows = greedy_solution(0)
        best_score = get_score(best_rows)
    
    # Phase 2: Local search on the best solution
    if time.time() < deadline - 10:
        improved_rows, improved_score = local_search(best_rows, max_time=deadline - time.time() - 5)
        if improved_score < best_score:
            best_rows = improved_rows
            best_score = improved_score
    
    # Phase 3: More local search iterations if time permits
    while time.time() < deadline - 5:
        improved_rows, improved_score = local_search(best_rows, max_time=min(10, deadline - time.time() - 5))
        if improved_score < best_score:
            best_rows = improved_rows
            best_score = improved_score
        else:
            break
    
    return best_rows
# EVOLVE-BLOCK-END
