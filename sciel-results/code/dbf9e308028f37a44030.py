"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry.
    """
    if n_rows == 0:
        return []
    if k == 0:
        return [[0] for _ in range(n_rows)]
    
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    # Total number of differences to place: n_rows * k*(k+1)//2
    total_diffs = n_rows * k * (k + 1) // 2
    # Simple counting bound: m(m+1)/2 >= total_diffs => m >= ~sqrt(2*total_diffs)
    # More precise: m*(m+1)//2 >= total_diffs
    import math
    m_count = int(math.ceil((-1 + math.sqrt(1 + 8 * total_diffs)) / 2))
    
    best_rows = None
    best_max = float('inf')
    
    def valid_construction(rows):
        """Check if a set of rows is valid (all diffs distinct globally)."""
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    def max_entry(rows):
        return max(max(row) for row in rows)
    
    def greedy_build(n_rows, k, rng=None, randomize_prob=0.0, candidate_limit=5):
        """
        Build rows greedily. For each position, pick a value such that all
        new differences are unused. With randomize_prob, pick randomly among
        the first few valid candidates instead of always the smallest.
        Returns (rows, max_val) or (None, inf) on failure.
        """
        if rng is None:
            rng = random.Random()
        
        used = set()
        rows = []
        
        for r in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                # Find valid candidates starting from row[-1] + 1
                candidates = []
                x = row[-1] + 1
                # We need to search up to some reasonable bound
                # The max possible value is bounded by best_max - 1 if we have one, else some heuristic
                search_bound = best_max - 1 if best_max < float('inf') else (m_count + 50)
                
                while x <= search_bound:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        candidates.append(x)
                        if len(candidates) >= candidate_limit:
                            break
                    x += 1
                
                if not candidates:
                    return None, float('inf')
                
                if rng is not None and rng.random() < randomize_prob:
                    choice = rng.choice(candidates[:min(3, len(candidates))])
                else:
                    choice = candidates[0]
                
                row.append(choice)
                for a in row[:-1]:
                    used.add(choice - a)
            
            rows.append(row)
        
        return rows, max_entry(rows)
    
    # Phase 1: Deterministic greedy to get an initial solution
    rows_det, max_det = greedy_build(n_rows, k, rng=None, randomize_prob=0.0)
    if rows_det is not None:
        best_rows = rows_det
        best_max = max_det
    
    # Phase 2: Randomized restarts with different strategies
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        seed = attempt
        rng = random.Random(seed)
        
        # Vary parameters per attempt
        randomize_prob = min(0.5, 0.05 + attempt * 0.01)
        candidate_limit = min(10, 3 + attempt % 8)
        
        # Occasionally try a different row order or approach
        rows_try, max_try = greedy_build(n_rows, k, rng=rng, randomize_prob=randomize_prob, 
                                         candidate_limit=candidate_limit)
        if rows_try is not None and max_try < best_max:
            best_rows = rows_try
            best_max = max_try
    
    # Phase 3: Local improvement - try to reduce the maximum entry
    if best_rows is not None and time.time() < deadline - 5:
        # Try to trim: for the last row, try smaller values
        # Actually, let's try a different approach: rebuild with tighter bounds
        for trim_attempt in range(100):
            if time.time() > deadline - 2:
                break
            if trim_attempt >= best_max - m_count:
                break
            target_max = best_max - trim_attempt - 1
            rng = random.Random(1000 + trim_attempt)
            rows_try, max_try = greedy_build(n_rows, k, rng=rng, randomize_prob=0.3,
                                             candidate_limit=5)
            if rows_try is not None and max_try < best_max:
                best_rows = rows_try
                best_max = max_try
                trim_attempt = 0  # Reset since we found something better
    
    if best_rows is None:
        # Fallback: pure greedy
        rows_det, max_det = greedy_build(n_rows, k, rng=None, randomize_prob=0.0)
        best_rows = rows_det
        best_max = max_det
    
    return best_rows
# EVOLVE-BLOCK-END
