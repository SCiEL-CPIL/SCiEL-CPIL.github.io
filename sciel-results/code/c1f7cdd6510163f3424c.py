"""Improved program: NumPy-optimized greedy + local search."""
import time
import random
import numpy as np

BASE = 51975

def _get_moduli():
    mods = []
    for d in range(3, BASE + 1, 2):
        if BASE % d == 0:
            mods.append(d)
    return sorted(mods, reverse=True)

MODULI = _get_moduli()

# Cache for masks to avoid repeated np.arange calls
_MASK_CACHE = {}

def _get_mask(m, r):
    key = (m, r)
    mask = _MASK_CACHE.get(key)
    if mask is None:
        mask = np.arange(r, BASE, m)
        _MASK_CACHE[key] = mask
    return mask

def _compute_uncovered(classes):
    """Compute the number of uncovered residues given a list of [m, r] classes."""
    coverage = np.zeros(BASE, dtype=np.int32)
    for m, r in classes:
        mask = _get_mask(m, r)
        coverage[mask] += 1
    return np.sum(coverage == 0)

def _greedy_init():
    """Greedy: process moduli in descending order, pick residue covering most uncovered."""
    coverage = np.zeros(BASE, dtype=np.int32)
    classes = []
    
    for m in MODULI:
        best_r = 0
        best_gain = -1
        for r in range(m):
            mask = _get_mask(m, r)
            # Count how many residues in mask are currently uncovered (coverage == 0)
            gain = np.sum(coverage[mask] == 0)
            if gain > best_gain:
                best_gain = gain
                best_r = r
        
        classes.append([m, best_r])
        mask = _get_mask(m, best_r)
        coverage[mask] += 1
        
    return classes, _compute_uncovered(classes)

def _local_search(classes, uncovered_count, time_limit):
    """Hill-climbing: try changing residues of individual moduli to reduce uncovered."""
    best_classes = [c[:] for c in classes]
    best_uncovered = uncovered_count
    deadline = time.time() + time_limit
    
    while time.time() < deadline:
        improved = False
        indices = list(range(len(best_classes)))
        random.shuffle(indices)
        
        # Recompute coverage counts and masks for current state
        coverage = np.zeros(BASE, dtype=np.int32)
        masks = []
        for m, r in best_classes:
            mask = _get_mask(m, r)
            masks.append(mask)
            coverage[mask] += 1
        
        for idx in indices:
            if time.time() >= deadline:
                break
            
            m = best_classes[idx][0]
            r_old = best_classes[idx][1]
            mask_old = masks[idx]
            
            best_r = r_old
            best_delta = 0 # We want delta < 0 (decrease in uncovered)
            
            for r_new in range(m):
                if r_new == r_old:
                    continue
                mask_new = _get_mask(m, r_new)
                
                # Temporarily remove old mask
                coverage[mask_old] -= 1
                
                # Calculate newly uncovered: residues in mask_old that are now 0
                newly_uncovered = np.sum(coverage[mask_old] == 0)
                
                # Calculate newly covered: residues in mask_new that are now 0
                newly_covered = np.sum(coverage[mask_new] == 0)
                
                # Restore old mask
                coverage[mask_old] += 1
                
                delta = newly_uncovered - newly_covered
                
                if delta < best_delta:
                    best_delta = delta
                    best_r = r_new
            
            if best_r != r_old:
                # Apply change
                coverage[mask_old] -= 1
                mask_new = _get_mask(m, best_r)
                coverage[mask_new] += 1
                
                best_classes[idx][1] = best_r
                masks[idx] = mask_new
                best_uncovered += best_delta
                improved = True
        
        if not improved:
            break
            
    return best_classes, best_uncovered

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Greedy initialization
    classes, uncovered = _greedy_init()
    
    # Local search refinement
    remaining = deadline - time.time()
    if remaining > 5:
        classes, uncovered = _local_search(classes, uncovered, remaining - 5)
    
    return classes
