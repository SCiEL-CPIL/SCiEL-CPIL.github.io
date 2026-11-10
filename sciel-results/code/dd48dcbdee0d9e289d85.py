"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _edge_index(i, j):
    """Map pair (i,j) with 0<=i<j<16 to flat index in [0,120)."""
    return 15 * i - i * (i - 1) // 2 + (j - i - 1)

def _all_k4s():
    """Precompute all K4s as lists of 6 edge indices."""
    k4s = []
    for quad in combinations(range(16), 4):
        edges = []
        for a, b in combinations(quad, 2):
            edges.append(_edge_index(a, b))
        k4s.append(tuple(edges))
    return k4s

_K4S = _all_k4s()
_N_K4S = len(_K4S)

def _count_violations(colours):
    """Count K4s with fewer than 5 distinct colours."""
    violations = 0
    for k4 in _K4S:
        if len(set(colours[e] for e in k4)) < 5:
            violations += 1
    return violations

def _count_colours(colours):
    """Count number of distinct colours used."""
    return len(set(colours))

def _try_merge_colour(colours, target, source, k4s):
    """Try to merge all edges of colour 'source' into colour 'target'.
    Returns (new_colours, valid) where valid is True if constraint holds."""
    new_colours = list(colours)
    changed = False
    for idx, c in enumerate(new_colours):
        if c == source:
            new_colours[idx] = target
            changed = True
    if not changed:
        return colours, False
    # Check all K4s
    for k4 in k4s:
        if len(set(new_colours[e] for e in k4)) < 5:
            return colours, False
    return new_colours, True

def _normalise_colours(colours):
    """Relabel colours to 0..k-1 preserving order of first appearance."""
    mapping = {}
    next_c = 0
    for c in colours:
        if c not in mapping:
            mapping[c] = next_c
            next_c += 1
    return [mapping[c] for c in colours]

def _try_recolour(colours, edge_idx, new_colour, k4s):
    """Try to recolour edge_idx to new_colour. Returns (new_colours, valid)."""
    new_colours = list(colours)
    new_colours[edge_idx] = new_colour
    for k4 in k4s:
        if len(set(new_colours[e] for e in k4)) < 5:
            return colours, False
    return new_colours, True

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Start with the seed colouring: (i+j) mod 17
    colours = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    colours = _normalise_colours(colours)
    best_colours = list(colours)
    best_count = _count_colours(colours)
    
    # Try to reduce colour count by merging colours
    # Strategy: try to merge colour c into colour d where c > d
    # Work from highest colour down
    current_count = best_count
    
    # First, try systematic merges
    for trial in range(300):
        if time.time() > deadline:
            break
        
        # Get current colour set
        used_colours = sorted(set(colours))
        if len(used_colours) <= best_count:
            break
        
        # Try to merge the highest colour into another
        # Try merging each colour into each lower colour
        found_merge = False
        for src in reversed(used_colours):
            if time.time() > deadline:
                break
            for tgt in used_colours:
                if tgt >= src:
                    continue
                new_colours, valid = _try_merge_colour(colours, tgt, src, _K4S)
                if valid:
                    colours = new_colours
                    current_count = _count_colours(colours)
                    if current_count < best_count:
                        best_colours = list(colours)
                        best_count = current_count
                    found_merge = True
                    break
            if found_merge:
                break
        
        if not found_merge:
            # No more systematic merges possible; try random perturbations
            # Randomly change a few edges and see if we can then merge
            if trial < 200:
                # Pick random edges and try changing them to enable merges
                for _ in range(80):
                    if time.time() > deadline:
                        break
                    # Pick a random edge and a random colour
                    edge_idx = random.randint(0, 119)
                    new_colour = random.randint(0, best_count - 1)
                    old_colour = colours[edge_idx]
                    if new_colour == old_colour:
                        continue
                    colours[edge_idx] = new_colour
                    # Check if this enables a merge
                    used = sorted(set(colours))
                    if len(used) <= best_count:
                        # Try to merge
                        for src in reversed(used):
                            for tgt in used:
                                if tgt >= src:
                                    continue
                                nc, v = _try_merge_colour(colours, tgt, src, _K4S)
                                if v:
                                    colours = nc
                                    current_count = _count_colours(colours)
                                    if current_count < best_count:
                                        best_colours = list(colours)
                                        best_count = current_count
                                    break
                    else:
                        # Revert if we increased colour count
                        colours[edge_idx] = old_colour
    
    # Local search: try to reduce violations if we had any, then reduce colours
    # Final polish: try more aggressive merges
    if time.time() < deadline - 10:
        used_colours = sorted(set(best_colours))
        for src in reversed(used_colours):
            if time.time() > deadline - 5:
                break
            for tgt in used_colours:
                if tgt >= src:
                    continue
                nc, v = _try_merge_colour(best_colours, tgt, src, _K4S)
                if v:
                    best_colours = nc
                    best_count = _count_colours(best_colours)
                    break
    
    return best_colours

# EVOLVE-BLOCK-END
