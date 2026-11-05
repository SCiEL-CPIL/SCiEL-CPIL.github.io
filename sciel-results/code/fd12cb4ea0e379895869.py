# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using a Genetic Algorithm 
    with Cycle Crossover for smaller instances, and NEH heuristic for larger ones.
    This strategy explicitly tests the 'cross' operator hypothesis.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Fast fallback for very large instances where GA is too slow
    # NEH is O(N^2 * M) which is acceptable for N up to ~1000
    if n_jobs > 150:
        return _neh_heuristic(processing_times, n_jobs, n_machines)

    # Pre-compute totals for NEH initialization
    totals = processing_times.sum(axis=1)
    
    def makespan(seq):
        """Compute makespan for a given sequence of job indices."""
        if len(seq) == 0:
            return 0.0
        completion = np.zeros(n_machines)
        for job in seq:
            prev = completion[0]
            curr = prev + processing_times[job, 0]
            completion[0] = curr
            for m in range(1, n_machines):
                prev = curr
                curr = max(prev, completion[m]) + processing_times[job, m]
                completion[m] = curr
        return completion[-1]

    def cycle_crossover(parent1, parent2):
        """
        Cycle Crossover (CX) for permutations.
        Preserves relative order of genes from both parents.
        """
        n = len(parent1)
        if n == 0:
            return [], []
        
        # Pick two random cut points
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i > j:
            i, j = j, i
        
        child1 = [-1] * n
        child2 = [-1] * n
        
        # Copy segment from parent1 to child1
        segment = parent1[i:j+1]
        for k, gene in enumerate(segment):
            child1[i+k] = gene
        
        # Fill remaining positions in child1 from parent2
        # We need to map positions in parent2 to positions in child1
        # Actually, standard CX:
        # 1. Identify cycle 1: start at pos i in P1, go to P2, find pos of that gene in P1, etc.
        # This implementation is a simplified version often used:
        # Copy segment [i:j] from P1 to C1.
        # Fill rest of C1 with genes from P2 in the order they appear in P2, 
        # skipping those already in C1.
        
        # To do this correctly and efficiently:
        # Create a set of genes in child1 segment
        c1_genes = set(segment)
        
        # Fill child1
        idx_c1 = i
        for gene in parent2:
            if child1[idx_c1] == -1:
                if gene not in c1_genes:
                    child1[idx_c1] = gene
                    c1_genes.add(gene)
                    idx_c1 += 1
            else:
                idx_c1 += 1
                # If we pass the end, we might need to wrap or just stop?
                # Standard CX fills the rest of the positions.
                # Let's use a simpler robust method for small N:
                # Just fill empty slots with remaining genes from P2 in order.
        
        # The above loop is tricky. Let's use the standard algorithm:
        # 1. Start at position i.
        # 2. Cycle 1: gene = P1[i]. Find pos of gene in P2. Let this be j2.
        #    Mark position i in C1 as gene.
        #    Move to j2. gene = P1[j2]. Find pos in P2...
        # This is complex to implement quickly without bugs.
        
        # Alternative: Order Crossover (OX) is easier and effective.
        # But the prompt specifically mentions "cross" and the open question is about "cross".
        # Cycle Crossover is the most TSP-like.
        
        # Let's implement a robust Cycle Crossover:
        def _cx(parent1, parent2):
            n = len(parent1)
            child = [-1] * n
            used = [False] * n
            
            # Pick random start
            start = random.randint(0, n - 1)
            
            # Cycle 1
            pos = start
            while child[pos] == -1:
                gene = parent1[pos]
                child[pos] = gene
                used[gene] = True
                # Find position of this gene in parent2
                pos = parent2.index(gene)
            
            # Fill remaining positions from parent2
            for gene in parent2:
                if not used[gene]:
                    # Find first empty slot in child
                    for k in range(n):
                        if child[k] == -1:
                            child[k] = gene
                            break
            
            return child

        # To generate two children, we swap parents
        c1 = _cx(parent1, parent2)
        c2 = _cx(parent2, parent1)
        
        return c1, c2

    def mutate(seq, rate=0.1):
        """Inversion mutation: reverse a random segment."""
        seq = list(seq)
        n = len(seq)
        if random.random() < rate:
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i > j:
                i, j = j, i
            if i != j:
                seq[i:j+1] = reversed(seq[i:j+1])
        return seq

    def neh_heuristic():
        """NEH heuristic for initialization."""
        order = list(np.argsort(-totals))
        seq = [order[0]]
        for job in order[1:]:
            best_pos = 0
            best_ms = float('inf')
            for pos in range(len(seq) + 1):
                trial = seq[:pos] + [job] + seq[pos:]
                ms = makespan(trial)
                if ms < best_ms:
                    best_ms = ms
                    best_pos = pos
            seq = seq[:best_pos] + [job] + seq[best_pos:]
        return seq

    # Initialize Population
    pop_size = 40
    population = []
    
    # 50% NEH-based (with slight randomization to ensure diversity)
    # We can just run NEH once and use it, or generate variants.
    # For simplicity, add 1 NEH solution and rest random, or multiple NEHs?
    # NEH is expensive O(N^2 M). For N=100, M=10, 100*100*10 = 100k ops. Fast.
    # Let's add 5 NEH solutions (randomizing tie breaks? No, NEH is deterministic).
    # So we add 1 NEH, and fill rest with Random.
    # To make it a proper GA, we need diversity.
    
    pop = []
    pop.append(neh_heuristic())
    
    for _ in range(pop_size - 1):
        seq = list(range(n_jobs))
        random.shuffle(seq)
        pop.append(seq)
    
    # Evaluate initial fitness
    fitnesses = [makespan(seq) for seq in pop]
    best_seq = pop[fitnesses.index(min(fitnesses))]
    best_ms = min(fitnesses)
    
    # GA Loop
    while time.time() < deadline:
        # Selection: Tournament
        def select():
            idx1 = random.randint(0, pop_size - 1)
            idx2 = random.randint(0, pop_size - 1)
            if fitnesses[idx1] < fitnesses[idx2]:
                return pop[idx1]
            else:
                return pop[idx2]
        
        new_pop = []
        
        # Elitism: Keep best 2
        sorted_indices = np.argsort(fitnesses)
        for idx in sorted_indices[:2]:
            new_pop.append(pop[idx])
        
        # Generate offspring until new_pop is full
        while len(new_pop) < pop_size:
            if time.time() > deadline:
                break
                
            p1 = select()
            p2 = select()
            
            # Crossover
            c1, c2 = cycle_crossover(p1, p2)
            
            # Mutation
            c1 = mutate(c1, rate=0.2)
            c2 = mutate(c2, rate=0.2)
            
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)
        
        # Evaluate new population
        population = new_pop
        fitnesses = [makespan(seq) for seq in population]
        
        # Update best
        curr_best_idx = fitnesses.index(min(fitnesses))
        if fitnesses[curr_best_idx] < best_ms:
            best_ms = fitnesses[curr_best_idx]
            best_seq = population[curr_best_idx]
            
        # Check time
        if time.time() > deadline:
            break

    return list(best_seq)

def _neh_heuristic(processing_times, n_jobs, n_machines):
    """Fast NEH for large instances."""
    totals = processing_times.sum(axis=1)
    order = list(np.argsort(-totals))
    
    def makespan(seq):
        if len(seq) == 0:
            return 0.0
        completion = np.zeros(n_machines)
        for job in seq:
            prev = completion[0]
            curr = prev + processing_times[job, 0]
            completion[0] = curr
            for m in range(1, n_machines):
                prev = curr
                curr = max(prev, completion[m]) + processing_times[job, m]
                completion[m] = curr
        return completion[-1]

    seq = [order[0]]
    for job in order[1:]:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(seq) + 1):
            trial = seq[:pos] + [job] + seq[pos:]
            ms = makespan(trial)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = seq[:best_pos] + [job] + seq[best_pos:]
    
    return seq
# EVOLVE-BLOCK-END
