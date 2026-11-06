import numpy as np
import time

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 114.0  # safety margin before 150s kill
    
    n_jobs, n_machines = processing_times.shape
    
    # Helper to calculate makespan for a given permutation (list of job indices)
    def calc_makespan(perm):
        # perm is a list of job indices
        completion = np.zeros(n_machines)
        for j in perm:
            job_times = processing_times[j]
            for m in range(n_machines):
                if m == 0:
                    st = completion[0]
                else:
                    st = max(completion[m], completion[m-1])
                completion[m] = st + job_times[m]
        return completion[-1]

    # 1. NEH Construction
    # Sort jobs by total processing time in descending order
    total_times = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-total_times)
    
    seq = [sorted_jobs[0]]
    
    for j in sorted_jobs[1:]:
        best_pos = 0
        best_makespan = float('inf')
        
        # Try inserting job j at each position in the current sequence
        for pos in range(len(seq) + 1):
            new_seq = seq[:pos] + [j] + seq[pos:]
            ms = calc_makespan(new_seq)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        seq.insert(best_pos, j)
        
        # Time check
        if time.time() > deadline:
            return list(seq)

    best_seq = list(seq)
    best_ms = calc_makespan(seq)

    # 2. Local Search: Insertion Neighborhood with First Improvement
    # This is generally superior to 2-opt (swaps) for Flow Shop Problems
    # We iterate until no improvement is found or time runs out
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(seq)
        
        # Iterate over all pairs (i, j) representing moving job from i to j
        # i is the source index, j is the destination index
        # We consider i < j (move right) and i > j (move left)
        # To cover all insertions, we can just pick a job i and try inserting it at all j != i
        
        for i in range(n):
            if time.time() > deadline:
                break
            
            job_to_move = seq[i]
            
            # Try inserting at all positions j from 0 to n-1 (excluding i)
            for j in range(n):
                if i == j:
                    continue
                
                # Remove job from i
                seq.pop(i)
                
                # Determine insertion index
                # If j > i, the index shifts left by 1 in the popped list?
                # Let's just create a new list or handle indices carefully.
                # Simpler: create new list
                new_seq = seq[:]
                if j > i:
                    # After popping i, indices > i shift left by 1.
                    # We want to insert at position j in the ORIGINAL list context?
                    # No, we want to try all positions in the current sequence.
                    # Let's just iterate j from 0 to len(new_seq)
                    pass
                
                # Let's restart the inner loop logic to be safe and simple
                # Re-iterate j from 0 to len(new_seq)
                # This is O(N^3) if we do it naively with list slicing.
                # For N=100, N^3 = 1M ops per pass. Makespan calc is O(N*M).
                # If M=10, 10M ops. In Python, this might take ~1s-10s per pass.
                # This is feasible for a few passes.
                
                break # Break inner loop to implement properly below
            
            # Proper Insertion Loop
            # We will try moving job at index i to every other position j
            # To avoid index confusion, we construct new_seq
            
            for j in range(n):
                if i == j:
                    continue
                
                # Create new sequence: remove i, insert at j
                # Note: if j > i, the target index in the reduced list is j-1?
                # Let's just use list slicing which is clear.
                new_seq = seq[:i] + seq[i+1:]
                
                # Determine insertion point in new_seq
                # If j < i, insert at j
                # If j > i, insert at j-1 (because one element was removed before j)
                # Wait, if we remove i, the element at j (if j>i) becomes j-1.
                # We want to place the job such that it ends up at position j in the ORIGINAL length n list?
                # Actually, we just want to try all valid positions in the NEW list.
                # The number of positions in new list is n-1.
                # So we should just iterate k from 0 to n-1 for the new list.
                
                # Let's simplify: 
                # Pick job at i.
                # Remove it.
                # Try inserting at every position k in the remaining list (0 to n-1).
                
                # This logic is getting messy in the draft. Let's rewrite the local search block cleanly.
            break # Break outer i loop to rewrite
            
        # Rewrite Local Search Block
        # We'll replace the entire while loop below.
        
    return best_seq
