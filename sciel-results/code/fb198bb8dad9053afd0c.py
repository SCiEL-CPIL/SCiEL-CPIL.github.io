import time
import random
import numpy as np

_DEADLINE = 85.0
_START_TIME = None

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _time_left():
    return _DEADLINE - _elapsed()

def _generate_progressions(n=128):
    """Generate all nonconstant 4-term progressions (a, a+d, a+2d, a+3d) with 1-indexed positions."""
    progs = []
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            progs.append((a, a + d, a + 2 * d, a + 3 * d))
    return progs

# Precompute progressions once at module level for speed
_PROGRESSIONS = _generate_progressions(128)
_NUM_PROGS = len(_PROGRESSIONS)

# Precompute for each position, which progressions it belongs to
_POS_TO_PROGS = {i: [] for i in range(1, 129)}
for idx, (a, b, c, d) in enumerate(_PROGRESSIONS):
    _POS_TO_PROGS[a].append(idx)
    _POS_TO_PROGS[b].append(idx)
    _POS_TO_PROGS[c].append(idx)
    _POS_TO_PROGS[d].append(idx)

def _count_violations(colors):
    """Count 4-term progressions that use fewer than 3 distinct colors."""
    violations = 0
    for (a, b, c, d) in _PROGRESSIONS:
        ca, cb, cc, cd = colors[a-1], colors[b-1], colors[c-1], colors[d-1]
        if ca == cb == cc == cd:
            violations += 1
        elif ca == cb == cc or ca == cb == cd or ca == cc == cd or cb == cc == cd:
            violations += 1
    return violations

def _is_valid(colors):
    """Check if coloring is valid (all 4-APs use >= 3 distinct colors)."""
    for (a, b, c, d) in _PROGRESSIONS:
        ca, cb, cc, cd = colors[a-1], colors[b-1], colors[c-1], colors[d-1]
        if ca == cb == cc == cd:
            return False
        if ca == cb == cc or ca == cb == cd or ca == cc == cd or cb == cc == cd:
            return False
    return True

def _num_colors(colors):
    return len(set(colors))

def _energy(colors, violation_weight=1000):
    """Energy = violation_count * violation_weight + num_colors"""
    v = _count_violations(colors)
    nc = _num_colors(colors)
    return v * violation_weight + nc

def _greedy_construct(k, seed=None):
    """Greedy construction using at most k colors. Returns coloring or None."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    n = 128
    colors = [-1] * n
    
    for i in range(n):
        bad_colors = [False] * k
        
        for (a, b, c_pos, d) in _PROGRESSIONS:
            pos = i + 1
            if pos == a:
                others = [b, c_pos, d]
            elif pos == b:
                others = [a, c_pos, d]
            elif pos == c_pos:
                others = [a, b, d]
            elif pos == d:
                others = [a, b, c_pos]
            else:
                continue
            
            other_colors = []
            all_assigned = True
            for p in others:
                if colors[p-1] == -1:
                    all_assigned = False
                    break
                other_colors.append(colors[p-1])
            
            if not all_assigned:
                continue
            
            for cc in range(k):
                s = {cc} | set(other_colors)
                if len(s) < 3:
                    bad_colors[cc] = True
        
        available = [c for c in range(k) if not bad_colors[c]]
        if not available:
            return None
        colors[i] = rng.choice(available)
    
    if _is_valid(colors):
        return colors
    return None

def _simulated_annealing(initial_colors, max_iters=50000, t0=1.0, t1=0.001, seed=None):
    """Simulated annealing to minimize energy (violations + colors)."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    n = 128
    colors = list(initial_colors)
    current_energy = _energy(colors)
    best_colors = list(colors)
    best_energy = current_energy
    
    for it in range(max_iters):
        if _time_left() < 5.0:
            break
        
        # Temperature schedule: exponential cooling
        alpha = t1 / t0
        T = t0 * (alpha ** (it / max_iters))
        
        # Generate a random move
        move_type = rng.random()
        new_colors = list(colors)
        
        if move_type < 0.6:
            # Recolor a random position
            pos = rng.randint(0, n - 1)
            old_color = new_colors[pos]
            new_color = rng.randint(0, 127)
            if new_color == old_color:
                continue
            new_colors[pos] = new_color
        elif move_type < 0.85:
            # Swap colors at two positions
            pos1 = rng.randint(0, n - 1)
            pos2 = rng.randint(0, n - 1)
            if pos1 == pos2:
                continue
            new_colors[pos1], new_colors[pos2] = new_colors[pos2], new_colors[pos1]
        else:
            # Try to eliminate a color: pick a position with a rare color and recolor it
            color_counts = {}
            for c in new_colors:
                color_counts[c] = color_counts.get(c, 0) + 1
            
            # Find the rarest color
            rarest_color = min(color_counts, key=lambda c: color_counts[c])
            positions_with_rarest = [i for i in range(n) if new_colors[i] == rarest_color]
            pos = rng.choice(positions_with_rarest)
            old_color = new_colors[pos]
            # Pick a new color from existing colors
            other_colors = [c for c in color_counts if c != rarest_color]
            if not other_colors:
                continue
            new_color = rng.choice(other_colors)
            new_colors[pos] = new_color
        
        new_energy = _energy(new_colors)
        delta = new_energy - current_energy
        
        # Accept if better, or with probability exp(-delta/T)
        if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
            colors = new_colors
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_colors = list(colors)
    
    return best_colors, best_energy

def _targeted_color_elimination(colors, max_iters=10000, seed=None):
    """Try to eliminate colors by recoloring positions with rare colors."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    n = 128
    colors = list(colors)
    best_colors = list(colors)
    best_nc = _num_colors(colors)
    
    for it in range(max_iters):
        if _time_left() < 5.0:
            break
        
        current_nc = _num_colors(colors)
        if current_nc <= 3:
            break
        
        # Count colors
        color_counts = {}
        for c in colors:
            color_counts[c] = color_counts.get(c, 0) + 1
        
        # Find rarest color
        rarest_color = min(color_counts, key=lambda c: color_counts[c])
        positions_with_rarest = [i for i in range(n) if colors[i] == rarest_color]
        
        # Try to recolor all positions with rarest color to other colors
        success = True
        new_colors = list(colors)
        
        for pos in rng.sample(positions_with_rarest, len(positions_with_rarest)):
            old_color = new_colors[pos]
            other_colors = [c for c in color_counts if c != rarest_color]
            
            found = False
            for new_color in rng.sample(other_colors, len(other_colors)):
                new_colors[pos] = new_color
                if _is_valid(new_colors):
                    found = True
                    break
            
            if not found:
                new_colors[pos] = old_color
                success = False
                break
        
        if success:
            colors = new_colors
            nc = _num_colors(colors)
            if nc < best_nc:
                best_nc = nc
                best_colors = list(colors)
    
    return best_colors, best_nc

def _fallback_greedy():
    """Original greedy fallback."""
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 
                    for d in range(1, (x-1)//3 + 1))
                and all(any(colors[x-1-j*d] != color for j in (1,2)) 
                        for d in range(1, (x-1)//2 + 1))):
                colors.append(color)
                break
    return colors

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_colors = None
    best_nc = 128
    
    # Phase 1: Quick greedy to get a valid baseline
    for k in range(3, 8):
        if _time_left() < 1.0:
            break
        for seed in range(5):
            if _time_left() < 0.5:
                break
            colors = _greedy_construct(k, seed=seed)
            if colors is not None:
                nc = _num_colors(colors)
                if nc < best_nc:
                    best_nc = nc
                    best_colors = list(colors)
                break
    
    # Fallback if greedy failed
    if best_colors is None:
        best_colors = _fallback_greedy()
        best_nc = _num_colors(best_colors)
    
    # Phase 2: Simulated annealing with multiple restarts
    if _time_left() > 10.0:
        for restart in range(10):
            if _time_left() < 5.0:
                break
            
            seed = restart * 100 + 42
            improved, improved_energy = _simulated_annealing(
                best_colors, 
                max_iters=20000, 
                t0=1.0, 
                t1=0.0001, 
                seed=seed
            )
            improved_nc = _num_colors(improved)
            if improved_nc < best_nc:
                best_nc = improved_nc
                best_colors = improved
    
    # Phase 3: Targeted color elimination
    if _time_left() > 5.0 and best_nc > 3:
        improved, improved_nc = _targeted_color_elimination(
            best_colors, 
            max_iters=5000, 
            seed=42
        )
        if improved_nc < best_nc:
            best_nc = improved_nc
            best_colors = improved
    
    # Phase 4: Final simulated annealing polish
    if _time_left() > 3.0 and best_nc > 3:
        improved, improved_energy = _simulated_annealing(
            best_colors, 
            max_iters=10000, 
            t0=0.5, 
            t1=0.00001, 
            seed=999
        )
        improved_nc = _num_colors(improved)
        if improved_nc < best_nc:
            best_nc = improved_nc
            best_colors = improved
    
    return best_colors
