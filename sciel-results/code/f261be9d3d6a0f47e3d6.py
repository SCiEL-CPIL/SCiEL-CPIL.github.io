import time
from itertools import combinations

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = P['n']
    adj = [set() for _ in range(n)]
    for a, b in P['edges']:
        adj[a].add(b)
        adj[b].add(a)
    
    def is_clique(part):
        for i in range(len(part)):
            for j in range(i + 1, len(part)):
                if part[j] not in adj[part[i]]:
                    return False
        return True
    
    def is_independent(part):
        for i in range(len(part)):
            for j in range(i + 1, len(part)):
                if part[j] in adj[part[i]]:
                    return False
        return True
    
    def is_valid_part(part):
        if not part:
            return True
        return is_clique(part) or is_independent(part)
    
    # Greedy construction: place each vertex into first compatible part or new part
    def greedy_construct():
        parts = []
        for v in range(n):
            placed = False
            for part in parts:
                trial = part + [v]
                if is_valid_part(trial):
                    part.append(v)
                    placed = True
                    break
            if not placed:
                parts.append([v])
        return parts
    
    best_parts = greedy_construct()
    best_count = len(best_parts)
    
    if time.time() > deadline - 5.0:
        return [[int(v) for v in part] for part in best_parts]
    
    # Try to improve via local search: swap vertices between parts
    def try_improve(parts):
        """Try local search to reduce number of parts"""
        current_parts = [list(p) for p in parts]
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            # Try removing a vertex from one part and adding to another
            for i, part_i in enumerate(current_parts):
                if len(part_i) <= 1:
                    continue
                for vi_idx, vi in enumerate(part_i):
                    new_part_i = part_i[:vi_idx] + part_i[vi_idx+1:]
                    if not is_valid_part(new_part_i):
                        continue
                    for j, part_j in enumerate(current_parts):
                        if i == j:
                            continue
                        trial_j = part_j + [vi]
                        if is_valid_part(trial_j):
                            # Check if this reduces total parts (if part_i becomes empty)
                            current_parts[i] = new_part_i
                            current_parts[j] = trial_j
                            # Remove empty parts
                            current_parts = [p for p in current_parts if p]
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            # Also try moving a vertex from a 1-element part to another part
            for i, part_i in enumerate(current_parts):
                if len(part_i) != 1:
                    continue
                vi = part_i[0]
                for j, part_j in enumerate(current_parts):
                    if i == j:
                        continue
                    trial_j = part_j + [vi]
                    if is_valid_part(trial_j):
                        current_parts = [p for k, p in enumerate(current_parts) if k != i]
                        if j > i:
                            j -= 1
                        current_parts[j] = trial_j
                        improved = True
                        break
                if improved:
                    break
        return current_parts
    
    improved_parts = try_improve(best_parts)
    if len(improved_parts) < best_count:
        best_parts = improved_parts
        best_count = len(best_parts)
    
    if time.time() > deadline - 15.0:
        return [[int(v) for v in part] for part in best_parts]
    
    # Branch and bound to find optimal solution
    def branch_and_bound(unassigned, current_parts, depth, best_count_ref):
        nonlocal best_parts, best_count
        if time.time() > deadline - 10.0:
            return False
        if not unassigned:
            if len(current_parts) < best_count:
                best_parts = [list(p) for p in current_parts]
                best_count = len(current_parts)
            return len(current_parts) < best_count_ref
        
        # Lower bound: at least ceil(len(unassigned) / max_part_size) more parts needed
        # Simple bound: if current parts + 1 > best, prune
        if len(current_parts) + 1 >= best_count:
            return False
        
        v = unassigned[0]
        remaining = unassigned[1:]
        
        # Try adding v to each existing part
        for i, part in enumerate(current_parts):
            trial = part + [v]
            if is_valid_part(trial):
                current_parts[i] = trial
                if branch_and_bound(remaining, current_parts, depth + 1, best_count):
                    current_parts[i] = part
                    return True
                current_parts[i] = part
        
        # Try creating a new part
        if len(current_parts) + 1 < best_count:
            current_parts.append([v])
            if branch_and_bound(remaining, current_parts, depth + 1, best_count):
                current_parts.pop()
                return True
            current_parts.pop()
        
        return False
    
    # Search for solutions with fewer parts
    for target in range(best_count - 1, 0, -1):
        if time.time() > deadline - 10.0:
            break
        if best_count <= target:
            break
        initial_parts = []
        found = branch_and_bound(list(range(n)), initial_parts, 0, best_count)
        if best_count <= target:
            break
    
    # Final validation and type conversion
    result = []
    seen = set()
    for part in best_parts:
        int_part = [int(v) for v in part]
        for v in int_part:
            if v in seen:
                return [[int(v) for v in p] for p in greedy_construct()]  # Fallback
            seen.add(v)
        result.append(int_part)
    
    if len(seen) != n:
        return [[int(v) for v in p] for p in greedy_construct()]
    
    return result
