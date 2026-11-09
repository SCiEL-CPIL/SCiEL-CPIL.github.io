"""Improved program for flow-shop permutation construction using NEH + Fast Local Search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan(order, P):
    """
    Compute makespan for a given job order on processing_times matrix P.
    Uses the standard dynamic programming recurrence:
    C[j][m] = max(C[j-1][m], C[j][m-1]) + P[j][m]
    """
    n_machines = P.shape[1]
    if n_machines == 0:
        return 0.0
    
    # c_machines[m] holds the completion time of the *last* job processed on machine m
    c_machines = [0.0] * n_machines
    
    for j in order:
        row = P[j]
        c_curr = 0.0
        for m in range(n_machines):
            # c_curr is completion on previous machine for current job
            # c_machines[m] is completion on this machine for previous job
            if c_machines[m] > c_curr:
                c_curr = c_machines[m]
            c_curr += row[m]
            c_machines[m] = c_curr
            
    return c_machines[-1]

def _neh_heuristic(P):
    """
    NEH (Nawaz-Enscore-Ham) heuristic.
    1. Sort jobs by total processing time in descending order.
    2. Insert each job into the best position in the current sequence to minimize makespan.
    """
    n_jobs, n_machines = P.shape
    totals = P.sum(axis=1)
    # Sort indices by total time descending
    order_indices = np.argsort(-totals)
    
    current_seq = []
    for idx in order_indices:
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(current_seq) + 1):
            new_seq = current_seq[:pos] + [idx] + current_seq[pos:]
            ms = _makespan(new_seq, P)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        current_seq.insert(best_pos, idx)
        
    return current_seq

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
    # To speed up evaluation, we pre-compute prefix and suffix completion states?
    # No, that's complex. Instead, we use a "delta" evaluation or just fast full recalculation
    # with a tight time check.
    
    # Optimization: Pre-extract rows of P into a list of lists for faster access?
    # P is a numpy array. P[j] is a view. 
    # Let's convert P to a list of lists if n_jobs is small?
    # Actually, accessing P[j][m] is fast enough.
    
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
    # To speed up, we can shuffle the order of checks?
    
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
