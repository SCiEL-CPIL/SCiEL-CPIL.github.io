"""Rewritten program for flow-shop permutation; uses a discrete PSO with local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(perm, pt):
    """Compute makespan for a given permutation on the given processing times."""
    if len(perm) == 0:
        return 0.0
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    # completion[job_index] tracks the completion time on the last machine
    # We simulate the flow shop
    # M[i] = completion time on machine i for the current job
    # Actually, let's use the standard O(n*m) simulation
    # machine_times[j] = time when machine j becomes free
    machine_times = np.zeros(n_machines)
    for job in perm:
        # For this job, go through each machine
        # The start time on machine 0 is machine_times[0]
        # The start time on machine k is max(machine_times[k], finish_time_on_prev_machine)
        # We can track the current job's finish time on the previous machine
        curr_finish = 0.0
        for m in range(n_machines):
            start = max(machine_times[m], curr_finish)
            curr_finish = start + pt[job, m]
            machine_times[m] = curr_finish
    return machine_times[-1]


def _discrete_swap(perm1, perm2):
    """
    Discrete velocity operator for permutations.
    This is a simple crossover-like operator.
    For each position, if the job in perm1 is in the set of jobs in perm2 that are 
    'ahead' in some sense, we pick it.
    A common discrete PSO operator:
    1. Identify positions where perm1 and perm2 differ.
    2. With probability w, pick from perm1, else from perm2?
    Actually, a better one:
    Create a binary mask of length n.
    For each position i, if perm1[i] != perm2[i], we have a 'conflict'.
    Let's use the 'swap mutation' approach or a specific crossover.
    
    Standard Discrete PSO for TSP/Flowshop:
    v_i is a set of indices to swap? No, velocity is often defined as a vector of 0s and 1s or a set of positions.
    
    Let's use the operator from Kennedy & Eberhart (2001) "Discrete Particle Swarm Optimization":
    Velocity v_i is a binary vector indicating whether to take the job from personal best or global best.
    Actually, a simpler effective operator:
    1. Compute the 'difference' between personal best p and global best g.
    2. Create a new permutation by taking elements from p and g.
    
    Let's try this:
    1. Start with an empty list.
    2. Iterate through positions 0 to n-1.
    3. At each position, decide whether to take the job from personal_best[pos] or global_best[pos].
    4. The decision is influenced by the 'velocity' (random probability) and the 'inertia'.
    
    A robust simple approach:
    1. Create a list of indices 0..n-1.
    2. Shuffle this list? No.
    
    Let's use the 'Crossover' approach:
    1. Pick two random cut points.
    2. Take the segment from personal best.
    3. Fill the rest with jobs from global best in the order they appear in global best.
    This is a standard Partially Mapped Crossover (PMX) or Order Crossover (OX).
    
    Let's implement Order Crossover (OX):
    1. Pick two random cut points i, j.
    2. Copy the segment [i:j] from personal_best to the new permutation.
    3. Fill the remaining positions with jobs from global_best, skipping those already in the segment, in the order they appear in global_best.
    """
    n = len(perm1)
    if n <= 1:
        return list(perm1)
    
    # Random cut points
    i = np.random.randint(0, n)
    j = np.random.randint(0, n)
    if i > j:
        i, j = j, i
    
    new_perm = [None] * n
    # Copy segment from perm1 (personal best)
    for k in range(i, j + 1):
        new_perm[k] = perm1[k]
    
    # Fill rest from perm2 (global best)
    used = set(new_perm[i:j+1])
    # We need to iterate perm2 and pick unused jobs in order
    # But we must respect the order in perm2
    # And we must fill positions 0..i-1 and j+1..n-1
    # Let's collect the available jobs from perm2 in order
    available = [job for job in perm2 if job not in used]
    
    # Fill the positions
    # Positions to fill: 0..i-1 and j+1..n-1
    # We can just iterate through the positions and pick the next available job
    # But wait, OX usually fills left-to-right or right-to-left?
    # Standard OX:
    # 1. Copy segment.
    # 2. Starting from position j+1 (wrapping around), fill with jobs from perm2 that are not in segment, in the order they appear in perm2.
    
    # Let's implement standard OX properly
    # 1. Copy segment [i:j] from perm1
    # 2. Start from position j+1 in the new permutation (wrapping to 0 if j+1 >= n)
    # 3. Iterate through perm2, and for each job not in the segment, place it in the next available position in the new permutation (starting from j+1, wrapping)
    
    # This is a bit complex to implement correctly in Python without bugs.
    # Let's use a simpler 'Blend' operator:
    # For each position, with probability p, take from perm1, else from perm2.
    # Then repair duplicates.
    
    # Actually, let's use a very simple and effective operator:
    # 1. Create a list of all jobs.
    # 2. For each position, if the job in perm1 is different from perm2, 
    #    with probability 0.5, keep perm1's job, else take perm2's job.
    # 3. This doesn't guarantee a valid permutation.
    
    # Let's stick to a known valid operator: Swap Mutation on the best of the two?
    # No, we need a crossover.
    
    # Let's implement a simple 'Position-based' crossover:
    # 1. Pick a random subset of positions to take from perm1.
    # 2. The rest are taken from perm2 in the order they appear in perm2.
    
    # Here is a robust implementation of Order Crossover (OX):
    # 1. Copy the segment from perm1.
    # 2. Fill the remaining positions with jobs from perm2, in the order they appear in perm2, 
    #    starting from position j+1 (wrapping around).
    
    # Let's try this:
    # 1. Copy segment [i:j] from perm1 to new_perm.
    # 2. Create a list of jobs in perm2 that are not in the segment.
    # 3. Fill the empty positions in new_perm (starting from j+1, wrapping) with these jobs in order.
    
    # This is correct OX.
    
    # Implementation:
    # 1. new_perm = [None] * n
    # 2. for k in range(i, j+1): new_perm[k] = perm1[k]
    # 3. used = set(perm1[i:j+1])
    # 4. available = [job for job in perm2 if job not in used]
    # 5. pos = (j + 1) % n
    # 6. for job in available:
    #        while new_perm[pos] is not None:
    #            pos = (pos + 1) % n
    #        new_perm[pos] = job
    #        pos = (pos + 1) % n
    
    # This works.
    
    # Let's code it.
    new_perm = [None] * n
    for k in range(i, j + 1):
        new_perm[k] = perm1[k]
    
    used = set(new_perm[i:j+1])
    available = [job for job in perm2 if job not in used]
    
    pos = (j + 1) % n
    for job in available:
        while new_perm[pos] is not None:
            pos = (pos + 1) % n
        new_perm[pos] = job
        pos = (pos + 1) % n
        
    return new_perm


def _local_search(perm, pt, max_swaps=10):
    """
    Perform a simple 2-opt local search (swap adjacent jobs).
    """
    n = len(perm)
    if n <= 1:
        return list(perm)
    
    best_perm = list(perm)
    best_makespan = _makespan(best_perm, pt)
    
    # Try swapping adjacent jobs
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            # Swap i and i+1
            new_perm = list(best_perm)
            new_perm[i], new_perm[i+1] = new_perm[i+1], new_perm[i]
            new_makespan = _makespan(new_perm, pt)
            if new_makespan < best_makespan:
                best_perm = new_perm
                best_makespan = new_makespan
                improved = True
                break  # Restart from the beginning after an improvement
    
    return best_perm


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on local synthetic flow-shop instances.
    Uses a discrete particle swarm optimization (PSO) framework.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    pt = np.asarray(processing_times)
    n_jobs, n_machines = pt.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Initialize particles
    n_particles = 20
    particles = []
    personal_bests = []
    personal_best_scores = []
    
    # Initialize with random permutations and some heuristics
    # 1. Random
    # 2. SPT (Shortest Processing Time)
    # 3. LPT (Longest Processing Time)
    # 4. EDD (Earliest Due Date - not applicable, no due dates)
    # 5. NEH-like
    
    # Let's use a mix
    init_perms = []
    init_perms.append(np.random.permutation(n_jobs))
    init_perms.append(np.argsort(pt.sum(axis=1)))  # SPT
    init_perms.append(np.argsort(-pt.sum(axis=1)))  # LPT
    
    # Add random permutations to fill up to n_particles
    while len(init_perms) < n_particles:
        init_perms.append(np.random.permutation(n_jobs))
    
    for perm in init_perms:
        perm = list(perm)
        particles.append(perm)
        score = _makespan(perm, pt)
        personal_bests.append(list(perm))
        personal_best_scores.append(score)
    
    # Find initial global best
    global_best_idx = np.argmin(personal_best_scores)
    global_best = list(personal_bests[global_best_idx])
    global_best_score = personal_best_scores[global_best_idx]
    
    # PSO Parameters
    w = 0.7  # Inertia weight (not directly used in discrete OX, but we can use it for probability)
    c1 = 1.5  # Cognitive parameter
    c2 = 1.5  # Social parameter
    
    # Main loop
    while time.time() < deadline:
        for i in range(n_particles):
            if time.time() > deadline:
                break
            
            # Update velocity (conceptually) and position
            # In discrete PSO, we use the OX crossover between personal best and global best
            # The 'velocity' can be simulated by a random probability of using personal best vs global best
            # Here, we use a simple approach:
            # 1. With probability w, keep the current position (inertia)
            # 2. Otherwise, perform OX crossover between personal best and global best
            
            if np.random.rand() < w:
                new_perm = list(particles[i])
            else:
                # OX crossover between personal best and global best
                new_perm = _discrete_swap(personal_bests[i], global_best)
            
            # Apply local search to the new permutation
            # To save time, only do local search if the permutation is promising or randomly
            # Let's do local search on every new permutation, but limit the swaps
            new_perm = _local_search(new_perm, pt, max_swaps=5)
            
            # Evaluate
            score = _makespan(new_perm, pt)
            
            # Update particle
            particles[i] = new_perm
            
            # Update personal best
            if score < personal_best_scores[i]:
                personal_bests[i] = list(new_perm)
                personal_best_scores[i] = score
                
                # Update global best
                if score < global_best_score:
                    global_best = list(new_perm)
                    global_best_score = score
    
    # Return the global best permutation
    return global_best
# EVOLVE-BLOCK-END
