"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
import numpy as np
from collections import Counter

N = 64
TIME_LIMIT = 320  # safety margin before 330s kill

# EVOLVE-BLOCK-START

def _is_valid_point(p, points, dist_counter):
    """
    Check if point p can be added to the set 'points' given the current dist_counter.
    Returns True if valid, False otherwise.
    Also returns the list of squared distances from p to all points in points if valid,
    to avoid recomputation.
    """
    px, py = p
    if not points:
        return True, []
    
    # Compute squared distances from p to all existing points
    d2_list = []
    for q in points:
        dx = px - q[0]
        dy = py - q[1]
        d2_list.append(dx * dx + dy * dy)
    
    # Check 1: Distances from p must be distinct
    if len(set(d2_list)) != len(d2_list):
        return False, None
        
    # Check 2: No distance from p is already in dist_counter
    for d2 in d2_list:
        if dist_counter[d2] > 0:
            return False, None
            
    return True, d2_list

def _add_point(p, points, dist_counter, d2_list=None):
    """Add point p to points and update dist_counter."""
    points.append(p)
    if d2_list is None:
        px, py = p
        for q in points[:-1]:
            dx = px - q[0]
            dy = py - q[1]
            dist_counter[dx * dx + dy * dy] += 1
    else:
        for d2 in d2_list:
            dist_counter[d2] += 1

def _remove_point(p, points, dist_counter):
    """Remove point p from points and update dist_counter."""
    points.remove(p)
    px, py = p
    for q in points:
        dx = px - q[0]
        dy = py - q[1]
        d2 = dx * dx + dy * dy
        dist_counter[d2] -= 1
        if dist_counter[d2] == 0:
            del dist_counter[d2] # Optional cleanup

def _greedy_init(seed, deadline):
    """Fast greedy initialization."""
    random.seed(seed)
    points = []
    dist_counter = Counter()
    
    # Generate all points and shuffle
    all_points = [(x, y) for x in range(N) for y in range(N)]
    random.shuffle(all_points)
    
    for p in all_points:
        if time.time() > deadline:
            break
        valid, d2_list = _is_valid_point(p, points, dist_counter)
        if valid:
            _add_point(p, points, dist_counter, d2_list)
            
    return points, dist_counter

def _local_search(points, dist_counter, deadline):
    """
    Local search to improve the set.
    Tries to add points. If blocked, tries swaps (remove one, add one) to find a better configuration.
    """
    current_points = points[:]
    current_dist = dist_counter.copy()
    
    max_iters = 5000
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        # 1. Try to add a random point
        added = False
        for _ in range(5): # Try a few random candidates
            px = random.randint(0, N-1)
            py = random.randint(0, N-1)
            p = (px, py)
            if p in current_points:
                continue
            
            valid, d2_list = _is_valid_point(p, current_points, current_dist)
            if valid:
                _add_point(p, current_points, current_dist, d2_list)
                added = True
                break
        
        if added:
            continue
        
        # 2. If couldn't add, try a swap
        if len(current_points) > 0:
            # Pick a random point to remove
            idx = random.randint(0, len(current_points) - 1)
            p_rem = current_points[idx]
            
            # Save state to revert if we don't want to keep the swap?
            # For now, we accept swaps to explore.
            # We remove p_rem
            _remove_point(p_rem, current_points, current_dist)
            
            # Try to add a random point
            swapped = False
            for _ in range(5):
                px = random.randint(0, N-1)
                py = random.randint(0, N-1)
                p_add = (px, py)
                if p_add in current_points:
                    continue
                
                valid, d2_list = _is_valid_point(p_add, current_points, current_dist)
                if valid:
                    _add_point(p_add, current_points, current_dist, d2_list)
                    swapped = True
                    break
            
            if not swapped:
                # Revert removal if we couldn't add anything?
                # If we don't revert, we lose a point. We only want to keep size or grow.
                # So if swap fails, revert.
                _add_point(p_rem, current_points, current_dist)
                # Note: _add_point will recompute distances if d2_list is None, which is correct.
                # But we need to ensure we don't double count if we don't pass d2_list.
                # _add_point handles it.
            
            # After a successful swap (size constant), try to add again immediately
            # This is handled by the top of the loop in the next iteration.
            
    return current_points, current_dist

def construct():
    """Construct a set of points with no three forming an isosceles triangle."""
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_points = []
    best_size = 0
    
    # 1. Greedy Initialization with multiple seeds
    for seed in range(100):
        if time.time() > deadline - 20:
            break
        
        points, dist_counter = _greedy_init(seed, deadline)
        
        if len(points) > best_size:
            best_size = len(points)
            best_points = points[:]
            best_dist = dist_counter.copy()
            
    # 2. Local Search on the best configuration
    if best_points and time.time() < deadline - 5:
        # Run local search on the best one
        improved_points, improved_dist = _local_search(best_points, best_dist, deadline)
        if len(improved_points) > best_size:
            best_points = improved_points
            best_size = len(improved_points)
            
    # 3. If time remains, try local search on other good configurations?
    # Or just return the best.
    
    return best_points

# EVOLVE-BLOCK-END
