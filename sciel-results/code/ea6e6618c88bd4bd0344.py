# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import concurrent.futures
import os

# Global variables for worker processes
_GRID = None
_N = 8
_SQ_NORMS = None  # Pre-computed squared norms for grid points

def _worker_init(grid_data, sq_norms_data):
    """
    Initialize global variables in the worker process.
    """
    global _GRID, _SQ_NORMS
    _GRID = grid_data
    _SQ_NORMS = sq_norms_data

def _check_can_add(points, p, sq_norms_points, sq_norm_p):
    """
    Check if point p can be added to the set of points.
    points: (m, 3) array
    p: (3,) array
    sq_norms_points: (m,) array of squared norms of points
    sq_norm_p: scalar squared norm of p
    """
    m = len(points)
    if m < 4:
        return True
        
    diffs = points - p
    W = np.hstack((diffs, (sq_norms_points - sq_norm_p)[:, np.newaxis]))
    
    indices = np.array(list(itertools.combinations(range(m), 4)))
    if len(indices) == 0:
        return True
        
    sub_matrices = W[indices]
    dets = np.linalg.det(sub_matrices)
    
    return not np.any(np.abs(dets) < 0.5)

def _run_greedy(seed):
    """
    Run a single greedy pass with a specific random seed.
    Returns the best set of points found.
    """
    global _GRID, _SQ_NORMS
    grid = _GRID
    sq_norms_all = _SQ_NORMS
    
    rng = np.random.RandomState(seed)
    order = grid[rng.permutation(len(grid))]
    
    arr = np.empty((0, 3), dtype=np.int64)
    sq_norms_arr = np.empty(0, dtype=np.int64)
    
    for p in order:
        sq_norm_p = (p ** 2).sum()
        
        if _check_can_add(arr, p, sq_norms_arr, sq_norm_p):
            if len(arr) == 0:
                arr = p[None, :]
                sq_norms_arr = np.array([sq_norm_p], dtype=np.int64)
            else:
                arr = np.vstack([arr, p])
                sq_norms_arr = np.append(sq_norms_arr, sq_norm_p)
                
    return arr

def _local_search(points, deadline):
    """
    Perform local search to improve the set of points.
    Strategy:
    1. Try to add any remaining point that fits.
    2. If stuck, try 1-for-1 swaps: remove one point, try to add two.
    3. If stuck, try 1-for-2 swaps: remove one, add one, then try to add another.
    """
    if points is None or len(points) == 0:
        return points
        
    improved = points.copy()
    point_set = set(map(tuple, improved))
    sq_norms_improved = (improved ** 2).sum(axis=1)
    
    # Phase 1: Greedy addition of any fitting point
    added = True
    while added and time.time() < deadline - 2.0:
        added = False
        for p in _GRID:
            if tuple(p) in point_set:
                continue
            sq_norm_p = (p ** 2).sum()
            if _check_can_add(improved, p, sq_norms_improved, sq_norm_p):
                improved = np.vstack([improved, p])
                sq_norms_improved = np.append(sq_norms_improved, sq_norm_p)
                point_set.add(tuple(p))
                added = True
    
    # Phase 2: 1-for-1 swaps (remove one, add one)
    # This is expensive, so we limit it.
    # We'll try a few random swaps.
    rng = np.random.RandomState(42)
    n = len(improved)
    if n > 0:
        # Try to improve by swapping
        for _ in range(100):  # Limit number of swap attempts
            if time.time() > deadline - 2.0:
                break
            i = rng.randint(n)
            p_remove = improved[i]
            # Try to add a random point not in the set
            p_add = _GRID[rng.randint(len(_GRID))]
            if tuple(p_add) in point_set:
                continue
            
            # Remove p_remove
            new_improved = np.delete(improved, i, axis=0)
            new_sq_norms = np.delete(sq_norms_improved, i)
            new_point_set = point_set - {tuple(p_remove)}
            
            # Try to add p_add
            sq_norm_add = (p_add ** 2).sum()
            if _check_can_add(new_improved, p_add, new_sq_norms, sq_norm_add):
                # Swap succeeded
                improved = np.vstack([new_improved, p_add])
                sq_norms_improved = np.append(new_sq_norms, sq_norm_add)
                point_set = new_point_set | {tuple(p_add)}
    
    return improved

def no_5_on_a_sphere8() -> np.ndarray:
    """
    Returns a numpy array of points from the 8x8x8 grid such that no 5 are co-spherical.
    """
    global _GRID, _N, _SQ_NORMS
    _N = 8
    _GRID = np.array(list(itertools.product(range(_N), range(_N), range(_N))), dtype=np.int64)
    _SQ_NORMS = (_GRID ** 2).sum(axis=1)
    
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    # Determine number of CPUs
    n_cpus = os.cpu_count() or 4
    n_workers = min(n_cpus, 8)  # Cap at 8 to avoid overhead
    
    best_set = None
    best_len = 0
    
    # Phase 1: Parallel greedy passes
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=n_workers, 
        initializer=_worker_init, 
        initargs=(_GRID, _SQ_NORMS)
    ) as executor:
        
        # We'll generate an infinite stream of seeds
        seed_gen = iter(range(100000))
        
        # Keep track of futures
        running_futures = []
        
        while time.time() < deadline - 60.0:  # Leave 60s for local search
            # Submit new tasks if we have capacity
            while len(running_futures) < n_workers * 2 and time.time() < deadline - 60.0:
                try:
                    seed = next(seed_gen)
                except StopIteration:
                    break
                future = executor.submit(_run_greedy, seed)
                running_futures.append(future)
            
            if not running_futures:
                break
            
            # Check for completed futures
            done, not_done = concurrent.futures.wait(running_futures, timeout=1.0)
            
            for f in done:
                running_futures.remove(f)
                try:
                    result = f.result()
                    if result is not None and len(result) > best_len:
                        best_len = len(result)
                        best_set = result.copy()
                except Exception as e:
                    pass
            
            if not running_futures and time.time() >= deadline - 60.0:
                 break
                 
            if not running_futures:
                 time.sleep(0.1)
    
    # Phase 2: Local search on the best set found
    if best_set is not None and time.time() < deadline - 10.0:
        improved = _local_search(best_set, deadline - 10.0)
        if len(improved) > len(best_set):
            best_set = improved
    
    if best_set is None:
        return np.array([[0, 0, 0]], dtype=np.int64)
        
    return best_set.astype(np.int64)

# EVOLVE-BLOCK-END
