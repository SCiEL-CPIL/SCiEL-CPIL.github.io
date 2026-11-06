import time
import random
from collections import defaultdict

n = 128
_DEADLINE = 90.0
_START_TIME = None

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _time_left():
    return _DEADLINE - _elapsed() - 5.0

# Precompute progressions
_AP_LIST = []
for a in range(n):
    for d in range(1, (n - a) // 3 + 1):
        _AP_LIST.append((a, a+d, a+2*d, a+3*d))

# Map position to (other1, other2, other3) for each AP containing it
_POS_TO_OTHERS = [[] for _ in range(n)]
for (a, b, c, d) in _AP_LIST:
    _POS_TO_OTHERS[a].append((b, c, d))
    _POS_TO_OTHERS[b].append((a, c, d))
    _POS_TO_OTHERS[c].append((a, b, d))
    _POS_TO_OTHERS[d].append((a, b, c))

def _is_valid_assignment(pos, new_color, colors):
    """Check if assigning new_color to pos is valid given current colors."""
    for (o1, o2, o3) in _POS_TO_OTHERS[pos]:
        # If any of the others are unassigned (-1), we can't check fully yet?
        # In elimination, all are assigned. In greedy, some might be -1.
        # This function is used in both.
        # If any is -1, we assume it's "safe" for now (optimistic) or 
        # we only check if all are assigned?
        # For Greedy, we only check if all 3 are assigned.
        # For Elimination, all are assigned.
        
        c1, c2, c3 = colors[o1], colors[o2], colors[o3]
        
        if c1 == -1 or c2 == -1 or c3 == -1:
            continue # Can't determine violation yet
            
        # Check distinct count of {new_color, c1, c2, c3}
        # Invalid if distinct count < 3
        # This happens if:
        # 1. c1==c2==c3 (then distinct is 1 or 2, always <3) -> Wait.
        #    If c1==c2==c3=x, and new_color=x, distinct=1.
        #    If c1==c2==c3=x, and new_color!=x, distinct=2.
        #    So if c1==c2==c3, NO color is valid.
        if c1 == c2 == c3:
            return False
            
        # 2. Two are same, one is different.
        #    e.g., c1==c2=x, c3=y (x!=y)
        #    If new_color == x, distinct={x,y} (2). Invalid.
        #    If new_color == y, distinct={x,y} (2). Invalid.
        #    If new_color == z (z!=x,y), distinct={x,y,z} (3). Valid.
        
        if c1 == c2:
            if new_color == c1 or new_color == c3:
                return False
        elif c1 == c3:
            if new_color == c1 or new_color == c2:
                return False
        elif c2 == c3:
            if new_color == c2 or new_color == c1:
                return False
        # else all distinct, always valid
        
    return True

def _greedy_construct(k, seed):
    rng = random.Random(seed)
    colors = [-1] * n
    
    for i in range(n):
        forbidden = [False] * k
        for (o1, o2, o3) in _POS_TO_OTHERS[i]:
            c1, c2, c3 = colors[o1], colors[o2], colors[o3]
            if c1 != -1 and c2 != -1 and c3 != -1:
                if c1 == c2 == c3:
                    return None # Dead end
                elif c1 == c2:
                    forbidden[c1] = True
                    forbidden[c3] = True
                elif c1 == c3:
                    forbidden[c1] = True
                    forbidden[c2] = True
                elif c2 == c3:
                    forbidden[c2] = True
                    forbidden[c1] = True
        available = [c for c in range(k) if not forbidden[c]]
        if not available:
            return None
        colors[i] = rng.choice(available)
        
    return colors

def _eliminate_colors(colors, k, seed):
    rng = random.Random(seed)
    current_colors = list(colors)
    current_nc = len(set(current_colors))
    
    max_steps = 5000
    for _ in range(max_steps):
        if _time_left() < 1.0:
            break
            
        if current_nc <= 3:
            break
            
        # Count colors
        counts = defaultdict(int)
        for c in current_colors:
            counts[c] += 1
            
        # Pick a color to eliminate (least frequent)
        target_color = min(counts, key=counts.get)
        positions = [i for i, c in enumerate(current_colors) if c == target_color]
        rng.shuffle(positions)
        
        success = True
        for p in positions:
            # Try to change p to a different color
            found = False
            # Try other existing colors first
            other_colors = [c for c in counts.keys() if c != target_color]
            rng.shuffle(other_colors)
            
            for new_c in other_colors:
                if _is_valid_assignment(p, new_c, current_colors):
                    current_colors[p] = new_c
                    found = True
                    break
            
            if not found:
                # Try any color in 0..k-1?
                # If we use a color not in 'counts', we increase nc.
                # But maybe it allows eliminating target_color?
                # Let's try colors 0..k-1
                all_colors = list(range(k))
                rng.shuffle(all_colors)
                for new_c in all_colors:
                    if new_c == target_color:
                        continue
                    if _is_valid_assignment(p, new_c, current_colors):
                        current_colors[p] = new_c
                        found = True
                        break
                        
            if not found:
                success = False
                break
                
        if success:
            # Recount
            new_counts = defaultdict(int)
            for c in current_colors:
                new_counts[c] += 1
            current_nc = len(new_counts)
            # Update counts for next iteration
            counts = new_counts
        else:
            # Failed to eliminate. 
            # To avoid getting stuck in a local minimum, we can 
            # try to swap colors of two random positions to "shake" the state.
            # Or just break and let outer loop try another seed.
            # Let's try a few random swaps to see if we can unblock.
            for _ in range(10):
                if _time_left() < 0.5: break
                p1, p2 = rng.sample(range(n), 2)
                c1, c2 = current_colors[p1], current_colors[p2]
                if c1 == c2: continue
                # Check if swap is valid
                # Temporarily swap
                current_colors[p1], current_colors[p2] = c2, c1
                if all(_is_valid_assignment(i, current_colors[i], current_colors) for i in [p1, p2]):
                    # Valid swap
                    break
                else:
                    # Revert
                    current_colors[p1], current_colors[p2] = c1, c2
            # If still stuck, break
            break

    return current_colors

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_colors = None
    best_nc = n + 1
    
    # Try K from 3 upwards
    for k in range(3, 10):
        if _time_left() < 2.0:
            break
            
        # Try multiple seeds for this K
        for seed in range(100):
            if _time_left() < 0.5:
                break
                
            colors = _greedy_construct(k, seed)
            if colors is None:
                continue
                
            nc = len(set(colors))
            
            # If we found a valid coloring, try to reduce colors
            # Even if nc == k, we might be able to reduce it.
            # But if nc < k, we definitely want to try.
            
            # Run elimination
            improved_colors = _eliminate_colors(colors, k, seed)
            improved_nc = len(set(improved_colors))
            
            if improved_nc < best_nc:
                best_nc = improved_nc
                best_colors = improved_colors
                
            # If we found a 3-coloring, we can't do better (theoretically min is 3? 
            # Actually, can it be 2? 
            # If 2 colors, any 4-AP must use >=3 distinct. Impossible.
            # So min is 3.
            if best_nc == 3:
                return best_colors

    # Fallback: if no valid coloring found (should not happen), return a random one?
    # Or a known valid one?
    # Greedy with K=128 always works.
    if best_colors is None:
        # Use a simple pattern
        best_colors = [i % 4 for i in range(n)]
        
    return best_colors
