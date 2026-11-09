"""Improved program for flow-shop permutation construction using NEH + Fast Local Search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(order, P):
    """
    Compute makespan for a given job order on processing_times matrix P.
    Uses the standard dynamic programming recurrence:
    C[j][m] = max(C[j-1][m], C[j][m-1]) + P[j][m]
    Optimized with pre-extracted rows to minimize attribute lookups.
    """
    n_machines = P.shape[1]
    if n_machines == 0:
        return 0.0
    
    # c_machines[m] holds the completion time of the *last* job processed on machine m
    c_machines = [0.0] * n_machines
    
    # Pre-extract rows? No, P is a numpy array. Accessing P[j] returns a view/array.
    # It's faster to loop over machines in Python if n_machines is small, 
    # but vectorizing the inner loop is hard due to sequential dependency.
    # However, we can use a list for c_machines for faster index access than numpy array?
    # Actually, numpy scalar access is slow. Python list of floats is faster for small loops.
    
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

def _neh_heuristic(P, seed=42):
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

def _fast_insertion_makespan(order, P, i, j):
    """
    Calculate makespan after moving job at index i to index j (j < i).
    This is O(N*M) but only recalculates the affected segment.
    Actually, full recalculation is O(N*M).
    The segment from min(i,j) to max(i,j) is affected.
    Jobs before min(i,j) have same completion times.
    We can compute completion times up to min(i,j)-1, then simulate the moved segment.
    """
    n_machines = P.shape[1]
    n_jobs = len(order)
    
    # Determine the start of the affected region
    if j > i:
        # Move job i to j. Jobs i..j-1 shift left. Job i moves to j.
        # Affected region: i to j
        start = i
        end = j
    else:
        # Move job i to j (j < i). Jobs j..i-1 shift right. Job i moves to j.
        # Affected region: j to i
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
    # New segment: remaining[:j-start] + [moved] + remaining[j-start:]
    # Wait, j is the index in the NEW sequence? Or old?
    # In our loop, we are considering moving job at position i to position j in the NEW sequence?
    # Usually, insertion move: remove job at i, insert at j.
    # If j < i: new pos is j.
    # If j > i: new pos is j.
    
    # Let's stick to a simpler approach: just rebuild the whole makespan if N is small,
    # or use a full recalculation. Given the complexity of partial recalculation bugs,
    # and N is likely < 1000, full recalculation is safer and fast enough in C-optimized loops?
    # No, Python loops are slow.
    # Let's optimize _makespan first.
    
    # For now, return None to indicate not implemented/used, and use full makespan in local search
    # but with a smarter neighborhood or fewer iterations.
    return None

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
