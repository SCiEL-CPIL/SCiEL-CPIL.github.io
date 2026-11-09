"""Improved program for flow-shop permutation construction using NEH + Vectorized Local Search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(order, P):
    """
    Compute makespan for a given job order on processing_times matrix P.
    Uses vectorized NumPy operations for the inner machine loop.
    C[j][m] = max(C[j-1][m], C[j][m-1]) + P[j][m]
    """
    if len(order) == 0:
        return 0.0
    if P.shape[1] == 0:
        return 0.0
    
    # c_machines holds the completion time of the last job on each machine
    c_machines = np.zeros(P.shape[1], dtype=float)
    
    # Pre-fetch the rows for all jobs in the order to avoid repeated indexing
    # This creates a (n_jobs, n_machines) array
    # P[np.array(order)] is a fancy index, creating a copy.
    # Given N is likely < 1000, this is acceptable and much faster than Python loops.
    rows = P[order]
    
    for row in rows:
        # c_curr[m] = max(c_machines[m], c_curr[m-1]) + row[m]
        # We need to update c_machines sequentially because c_curr[m] depends on c_curr[m-1].
        # However, we can compute the "max" part in a vectorized way if we shift?
        # No, the dependency is strict.
        # But we can use a trick:
        # Let v = np.maximum(c_machines, np.concatenate(([0], c_machines[:-1])))
        # Then c_machines = v + row
        # Wait, c_curr[0] = max(c_machines[0], 0) + row[0]
        # c_curr[1] = max(c_machines[1], c_curr[0]) + row[1]
        # This is a scan operation.
        
        # Actually, for small M (e.g., 10-50), Python loop might still be competitive or faster than 
        # creating intermediate arrays. But let's try the vectorized scan approach.
        
        # A simpler vectorized approach for small M is often just a loop, but let's try to optimize the loop.
        # The previous code used a Python list. Let's stick to a tight Python loop but with local variables.
        pass
        
    # Reverting to optimized Python loop as vectorized scan is tricky without custom C.
    # However, we can optimize the Python loop by using local variables and avoiding attribute lookups.
    
    c_machines = [0.0] * P.shape[1]
    n_m = P.shape[1]
    
    # Access P rows directly
    for j in order:
        row = P[j]
        c_curr = 0.0
        # Use local variable for c_machines to avoid global lookups? It's already local.
        for m in range(n_m):
            cm = c_machines[m]
            if cm > c_curr:
                c_curr = cm
            c_curr += row[m]
            c_machines[m] = c_curr
            
    return c_machines[-1]

def _neh_heuristic(P, seed=42):
    """
    NEH (Nawaz-Enscore-Ham) heuristic.
    1. Sort jobs by total processing time in descending order.
    2. Insert each job into the best position in the current sequence to minimize makespan.
    """
    n_jobs, n_machines = P.shape
    if n_jobs == 0:
        return []
    totals = P.sum(axis=1)
    # Sort indices by total time descending
    order_indices = np.argsort(-totals)
    
    current_seq = []
    for idx in order_indices:
        best_pos = 0
        best_makespan = float('inf')
        
        # Optimize NEH insertion by only checking necessary positions?
        # Standard NEH checks all.
        for pos in range(len(current_seq) + 1):
            new_seq = current_seq[:pos] + [idx] + current_seq[pos:]
            ms = _makespan(new_seq, P)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        current_seq.insert(best_pos, idx)
        
    return current_seq

def _fast_insertion_makespan(order, P, i, j):
    """
    Calculate makespan after moving job at index i to index j.
    Uses partial recalculation for efficiency.
    """
    n_machines = P.shape[1]
    n_jobs = len(order)
    
    if i == j:
        return _makespan(order, P)
    
    # Determine the start of the affected region
    if j > i:
        start = i
        end = j
    else:
        start = j
        end = i
    
    # Compute completion times for jobs 0 to start-1
    c_machines = [0.0] * n_machines
    for k in range(start):
        row = P[order[k]]
        c_curr = 0.0
        for m in range(n_machines):
            if c_machines[m] > c_curr:
                c_curr = c_machines[m]
            c_curr += row[m]
            c_machines[m] = c_curr
            
    # Construct the new order for the affected segment
    # Original segment: order[start:end+1]
    # Moved job: order[i]
    # Remaining jobs in segment: order[start:i] + order[i+1:end+1]
    # New segment depends on j relative to i
    
    if j < i:
        # Move job i to j.
        # New segment: [order[i]] + order[start:i] + order[i+1:end+1]
        # Wait, if j < start? start is j. So j is the start of the segment.
        # The job order in the segment [start, end] in the NEW sequence:
        # The job at index j in the new sequence is order[i].
        # The jobs that were at j..i-1 are now at j+1..i.
        # The jobs at i+1..end are unchanged in relative position?
        # Let's trace:
        # Old: A B C D E (i=2, C). Move C to j=1.
        # New: A C B D E.
        # Segment 1..2 (indices 1,2): C B.
        # My formula: [order[2]] + order[1:2] + order[3:3] = [C] + [B] + [] = C B. Correct.
        # What if j=0?
        # Old: A B C. Move C to 0.
        # New: C A B.
        # Segment 0..2: C A B.
        # Formula: [C] + order[0:2] + order[3:3] = [C] + [A, B] + [] = C A B. Correct.
        
        new_segment = [order[i]] + order[start:i] + order[i+1:end+1]
    else:
        # Move job i to j (j > i).
        # Old: A B C D E (i=1, B). Move B to j=3.
        # New: A C D B E.
        # Segment 1..3: C D B.
        # Formula: order[start:i] + order[i+1:j+1] + [order[i]] + order[j+1:end+1]
        # start=1, i=1, j=3, end=3.
        # order[1:1] + order[2:4] + [B] + order[4:4]
        # [] + [C, D] + [B] + [] = C D B. Correct.
        
        new_segment = order[start:i] + order[i+1:j+1] + [order[i]] + order[j+1:end+1]
        
    # Now compute makespan for this new segment starting with c_machines.
    for job in new_segment:
        row = P[job]
        c_curr = 0.0
        for m in range(n_machines):
            if c_machines[m] > c_curr:
                c_curr = c_machines[m]
            c_curr += row[m]
            c_machines[m] = c_curr
            
    # Now we need to process the rest of the jobs (end+1 to N-1)
    # c_machines is the state after the segment.
    for k in range(end + 1, n_jobs):
        row = P[order[k]]
        c_curr = 0.0
        for m in range(n_machines):
            if c_machines[m] > c_curr:
                c_curr = c_machines[m]
            c_curr += row[m]
            c_machines[m] = c_curr
            
    return c_machines[-1]

def order_jobs(processing_times):
    """
    Order jobs to minimize makespan using NEH heuristic followed by local search.
    """
    P = np.asarray(processing_times, dtype=float)
    if P.size == 0:
        return []
        
    n_jobs, n_machines = P.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
        
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    # 1. Initialize with NEH
    best_order = _neh_heuristic(P)
    best_makespan = _makespan(best_order, P)
    
    # 2. Local Search
    # Strategy: Iterated Local Search (ILS) with Insertion Moves
    # 1. Start with best_order
    # 2. Perform 2-opt or Insertion local search until local optimum
    # 3. Perturb the solution
    # 4. Local search again
    # 5. Accept if better, or with probability based on simulated annealing
    
    # Given time constraints, let's do a robust Insertion Local Search.
    # For each job i, try moving it to every other position j.
    # To speed up, we can use the "fast" makespan calculation if we implement it correctly.
    # But for now, let's optimize the inner loop of _makespan.
    
    # Optimization: Use numpy for the inner machine loop?
    # C[m] = max(C[m], C[m-1]) + P[j,m]
    # This is a scan operation. Not easily vectorizable without custom C code.
    # However, if n_machines is small (e.g., 10), Python loop is okay.
    
    # Let's try a different local search: Or-opt (move blocks of size 1, 2, 3)
    # Or just simple insertion.
    
    # To handle larger N, we limit the number of full passes or use a time check.
    
    # We will implement a fast insertion move evaluation.
    # Let's implement _makespan_segment(start, end, order, P, c_machines_start)
    
    def _eval_insertion(order, P, i, j):
        """
        Evaluate makespan if job at index i is moved to index j.
        Assumes i != j.
        Returns makespan.
        """
        n_machines = P.shape[1]
        n_jobs = len(order)
        
        # Determine affected range
        if j < i:
            start = j
            end = i
        else:
            start = i
            end = j
            
        # Compute completion times up to start
        c_machines = [0.0] * n_machines
        for k in range(start):
            row = P[order[k]]
            c_curr = 0.0
            for m in range(n_machines):
                if c_machines[m] > c_curr:
                    c_curr = c_machines[m]
                c_curr += row[m]
                c_machines[m] = c_curr
        
        # Construct new order for segment [start, end]
        # Original: order[start...end]
        # Remove order[i]
        # Insert at position j (relative to start? No, j is absolute index in new sequence)
        
        # Let's just build the new full order and compute makespan from start?
        # No, that's O(N*M).
        # Let's build the new segment.
        
        # The jobs in the segment are order[start:end+1]
        # We remove order[i] and insert it at position j.
        # Note: j is the index in the NEW sequence?
        # If we move job i to j:
        # If j < i:
        #   New sequence: ... order[j-1], order[i], order[j], ..., order[i-1], ...
        #   The affected segment in the new sequence is still indices start to end.
        #   The job order in this segment is:
        #   [order[i]] + order[start:i] + order[i+1:end+1]  -- Wait.
        #   Original segment: order[start], ..., order[i], ..., order[end]
        #   Remove order[i].
        #   Insert at position j.
        #   If j < i:
        #     New segment: order[j] becomes order[i]?
        #     Let's trace:
        #     Old: A B C D E (i=2, C)
        #     Move C to j=1.
        #     New: A C B D E
        #     Segment 1 to 2: C B
        #     Original segment 1 to 2: B C
        #     So new segment is: [order[i]] + order[start:i] + order[i+1:end+1]
        #     Wait, order[start:i] is B. order[i+1:end+1] is D E? No, end is i=2.
        #     So order[i+1:end+1] is empty.
        #     New segment: C B. Correct.
        
        # If j > i:
        #   Old: A B C D E (i=1, B)
        #   Move B to j=3.
        #   New: A C D B E
        #   Segment 1 to 3: C D B
        #   Original segment 1 to 3: B C D
        #   Remove B. Remaining: C D.
        #   Insert B at position 3 (which is index 2 in the segment relative to start=1? No.)
        #   In the new sequence, B is at index 3.
        #   The segment is indices 1, 2, 3.
        #   Jobs at 1,2 are C, D. Job at 3 is B.
        #   So new segment: order[start:i] + order[i+1:j+1] + [order[i]]?
        #   order[start:i] = [] (start=1, i=1)
        #   order[i+1:j+1] = order[2:4] = C, D
        #   [order[i]] = B
        #   Result: C D B. Correct.
        
        # So:
        if j < i:
            new_segment = [order[i]] + order[start:i] + order[i+1:end+1]
        else:
            new_segment = order[start:i] + order[i+1:j+1] + [order[i]] + order[j+1:end+1]
            
        # Wait, if j > i, the end of the segment is still j?
        # Old: A B C D E (i=1, j=3). End=3.
        # New: A C D B E. Segment 1..3 is C D B.
        # My formula: order[1:1] + order[2:4] + [B] + order[4:4] = [] + [C,D] + [B] + [] = C D B. Correct.
        
        # Now compute makespan for this new segment starting with c_machines.
        c_curr = 0.0
        for job in new_segment:
            row = P[job]
            # c_curr is completion on prev machine for current job
            # c_machines[m] is completion on this machine for previous job
            for m in range(n_machines):
                if c_machines[m] > c_curr:
                    c_curr = c_machines[m]
                c_curr += row[m]
                c_machines[m] = c_curr
                
        # Now we need to process the rest of the jobs (end+1 to N-1)
        # But wait, c_machines is now the state after the segment.
        # We need to continue with the rest of the jobs.
        # This requires iterating over the rest of the jobs, which is O((N-end)*M).
        # So total is O(N*M). Not faster than full makespan.
        # The only gain is if we can stop early or if the segment is small and we can approximate?
        # No, we need the exact makespan.
        
        # Therefore, fast insertion evaluation is not O(segment) unless we have a data structure.
        # Given the time, let's stick to full makespan but optimize the loop.
        
        # Let's just return the full makespan of the new order.
        new_order = order[:]
        new_order.pop(i)
        new_order.insert(j, order[i])
        return _makespan(new_order, P)

    # Local Search Loop
    # We will use a "first improvement" strategy with a time limit.
    # To speed up, we can shuffle the order of checking moves?
    
    # Let's try a different approach: Simulated Annealing or ILS with random restarts if stuck.
    # But NEH is usually very good.
    
    # Let's do a simple 2-opt (swap) and insertion search.
    # To make it faster, we'll limit the number of iterations per pass.
    
    improved = True
    no_improve_count = 0
    max_no_improve = 10 # Limit passes if no improvement
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        no_improve_count += 1
        if no_improve_count > max_no_improve:
            break
            
        # Randomize the order of checks to avoid bias?
        # Let's just do sequential.
        
        for i in range(n_jobs):
            if (time.time() - start_time) > time_limit:
                return best_order
                
            job_i = best_order[i]
            
            # Try moving to all other positions
            for j in range(n_jobs):
                if i == j:
                    continue
                    
                # Skip if adjacent? No, insertion handles it.
                
                # Check time
                if (time.time() - start_time) > time_limit:
                    return best_order
                
                new_order = best_order[:]
                new_order.pop(i)
                new_order.insert(j, job_i)
                
                ms = _makespan(new_order, P)
                
                if ms < best_makespan:
                    best_makespan = ms
                    best_order = new_order
                    improved = True
                    # Restart the outer loop? Or continue?
                    # First improvement: break inner, restart outer?
                    # Let's break inner and restart outer to ensure we find the best possible move in the new neighborhood.
                    break
                    
            if improved:
                break
                
    return best_order
# EVOLVE-BLOCK-END
