import time
import random
import math

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 295  # 5s safety margin
    
    all_points = [(x, y) for x in range(N) for y in range(N)]
    
    best_set = []
    best_size = 0
    
    def can_insert(px, py, selected, dist_from):
        """
        Check if point (px, py) can be added to the selected set.
        Returns True if valid.
        """
        d_list = []
        for i in range(len(selected)):
            ax, ay = selected[i]
            d_sq = (px - ax)**2 + (py - ay)**2
            d_list.append(d_sq)
        
        # Check 1: P is apex. All distances from P must be unique.
        if len(set(d_list)) != len(d_list):
            return False
        
        # Check 2: For each selected point A, d(P,A) must not equal d(A,B) for any other B.
        for i in range(len(selected)):
            d_pa = d_list[i]
            if d_pa in dist_from[i]:
                return False
        
        return True

    def insert_point(p, selected, dist_from):
        """Insert point p into selected, updating dist_from. Assumes can_insert returned True."""
        px, py = p
        d_list = []
        for i in range(len(selected)):
            ax, ay = selected[i]
            d_sq = (px - ax)**2 + (py - ay)**2
            d_list.append(d_sq)
        
        selected.append(p)
        dist_from.append(set(d_list))
        for i in range(len(selected) - 1):
            dist_from[i].add(d_list[i])
            
    def remove_point(idx, selected, dist_from):
        """Remove point at index idx from selected, updating dist_from efficiently."""
        p_remove = selected[idx]
        
        # Remove distances from other points' dist_from sets
        for i in range(len(selected)):
            if i == idx:
                continue
            ax, ay = selected[i]
            rx, ry = p_remove
            d_sq = (ax - rx)**2 + (ay - ry)**2
            if d_sq in dist_from[i]:
                dist_from[i].discard(d_sq)
                
        # Remove the point and its dist_from entry
        selected.pop(idx)
        dist_from.pop(idx)

    def greedy_with_order(order, time_limit=None):
        """
        Greedy selection following the given order.
        Returns (selected, dist_from).
        """
        selected = []
        dist_from = []
        
        for p in order:
            if time_limit is not None and time.time() > time_limit:
                break
            if can_insert(p[0], p[1], selected, dist_from):
                insert_point(p, selected, dist_from)
        
        return selected, dist_from

    def local_search(selected, dist_from, time_limit):
        """
        Local search: Try to grow the set, or swap points to enable growth.
        """
        selected_set = set(selected)
        candidates = [p for p in all_points if p not in selected_set]
        
        no_growth_count = 0
        max_no_growth = 5000 # Threshold to force swaps
        
        while time.time() < time_limit:
            # 1. Try to grow
            # Shuffle candidates occasionally to avoid bias
            if no_growth_count > 1000:
                random.shuffle(candidates)
                
            growth_found = False
            # Try a batch of candidates
            batch_size = min(100, len(candidates))
            # Random sample of candidates
            sample = random.sample(candidates, batch_size) if len(candidates) >= batch_size else candidates
            
            for p in sample:
                if time.time() > time_limit:
                    return selected, dist_from, selected_set
                
                if can_insert(p[0], p[1], selected, dist_from):
                    # Insert
                    insert_point(p, selected, dist_from)
                    selected_set.add(p)
                    # Remove from candidates
                    candidates.remove(p)
                    growth_found = True
                    no_growth_count = 0
                    break
            
            if growth_found:
                continue
                
            no_growth_count += 1
            
            # 2. If stuck, try a swap
            if no_growth_count > max_no_growth:
                no_growth_count = 0
                if not selected or not candidates:
                    break
                
                # Pick random A from selected
                idx_a = random.randint(0, len(selected) - 1)
                p_a = selected[idx_a]
                
                # Pick random B from candidates
                idx_b = random.randint(0, len(candidates) - 1)
                p_b = candidates[idx_b]
                
                # Remove A
                remove_point(idx_a, selected, dist_from)
                selected_set.remove(p_a)
                
                # Try to insert B
                if can_insert(p_b[0], p_b[1], selected, dist_from):
                    # Swap successful
                    insert_point(p_b, selected, dist_from)
                    selected_set.add(p_b)
                    
                    # Update candidates: remove B, add A
                    candidates.remove(p_b)
                    candidates.append(p_a)
                else:
                    # Swap failed, restore A
                    insert_point(p_a, selected, dist_from)
                    selected_set.add(p_a)
                    
        return selected, dist_from, selected_set

    # Main Loop
    restart_count = 0
    while time.time() < deadline - 20:
        restart_count += 1
        
        # 1. Greedy Initialization
        shuffled = list(all_points)
        random.shuffle(shuffled)
        
        # Time limit for greedy: leave time for local search
        greedy_limit = deadline - 10
        selected, dist_from = greedy_with_order(shuffled, time_limit=greedy_limit)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = list(selected)
            
        # 2. Local Search Improvement
        if time.time() < deadline - 5:
            # Copy state for local search
            sel_copy = list(selected)
            dist_copy = [set(s) for s in dist_from]
            
            new_selected, new_dist, new_set = local_search(
                sel_copy, dist_copy, deadline - 5
            )
            
            if len(new_selected) > best_size:
                best_size = len(new_selected)
                best_set = list(new_selected)
        
        # Occasionally try a different greedy strategy:
        # Sort by x+y parity or something to break symmetry?
        # Random shuffle is usually best for this type of constraint.
        
    return best_set
