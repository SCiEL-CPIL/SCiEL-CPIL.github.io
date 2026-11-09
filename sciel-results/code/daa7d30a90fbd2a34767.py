"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    # WalkSAT-style stochastic local search for Max-Cut
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=np.int8)
    
    # Deadline guard: 150s wall-clock limit, return best with safety margin
    deadline = time.time() + 140.0
    rng = np.random.default_rng(42)
    
    def cut_size(labels):
        diff = 1 - (labels[None, :] == labels[:, None]).astype(np.int8)
        return np.sum(adj * diff) // 2
    
    def local_improve(labels):
        improved = True
        while improved:
            improved = False
            for v in range(n):
                cur_cross = adj[v][labels != labels[v]].sum()
                new_cross = adj[v][labels == labels[v]].sum()
                if new_cross > cur_cross:
                    labels[v] = 1 - labels[v]
                    improved = True
        return labels
    
    def walksat(labels, steps=3000):
        """WalkSAT-style: flip variable biased by unsatisfied clauses (uncut edges)."""
        # For Max-Cut as Max-2-SAT: each edge (i,j) is a clause (li != lj)
        # Unsatisfied clause = uncut edge (same label on both endpoints)
        # Flip a variable: count how many unsatisfied clauses it satisfies vs creates
        
        # Precompute neighbor lists for efficiency
        # For each vertex v, neighbors are where adj[v] == 1
        
        current_cut = cut_size(labels)
        best_local = current_cut
        best_local_labels = labels.copy()
        
        for step in range(steps):
            if time.time() > deadline:
                break
            
            # Find vertices with uncut edges (part of unsatisfied clauses)
            # Vertex v has an uncut edge to neighbor u if labels[v] == labels[u]
            # Count unsatisfied clauses per vertex
            unsat_count = np.zeros(n, dtype=np.int32)
            for v in range(n):
                unsat_count[v] = adj[v][labels == labels[v]].sum()
            
            if unsat_count.sum() == 0:
                break  # All clauses satisfied (perfect cut)
            
            # Choose a variable to flip
            # With probability p (bias), choose from variables that appear in unsatisfied clauses
            # Weighted by number of unsatisfied clauses they're in
            p = 0.7  # bias parameter
            if rng.random() < p:
                # Choose weighted by unsat_count
                total_unsat = unsat_count.sum()
                if total_unsat == 0:
                    v = rng.integers(0, n)
                else:
                    probs = unsat_count / total_unsat
                    v = rng.choice(n, p=probs)
            else:
                v = rng.integers(0, n)
            
            # Compute delta if we flip v
            cur_cross_v = adj[v][labels != labels[v]].sum()
            new_cross_v = adj[v][labels == labels[v]].sum()
            delta = new_cross_v - cur_cross_v
            
            # Flip the variable
            labels[v] = 1 - labels[v]
            current_cut += delta
            
            if current_cut > best_local:
                best_local = current_cut
                best_local_labels = labels.copy()
        
        return best_local_labels, best_local
    
    # Multiple restarts
    best_labels = None
    best_cut = -1
    
    # Initial solutions
    initial_solutions = []
    initial_solutions.append(np.arange(n) % 2)
    initial_solutions.append(rng.integers(0, 2, n).astype(np.int8))
    initial_solutions.append(rng.integers(0, 2, n).astype(np.int8))
    degrees = adj.sum(axis=1)
    initial_solutions.append((degrees > degrees.mean()).astype(np.int8))
    
    for init_labels in initial_solutions:
        if time.time() > deadline:
            break
        
        labels = init_labels.copy().astype(np.int8)
        labels = local_improve(labels)
        cur_cut = cut_size(labels)
        
        if cur_cut > best_cut:
            best_cut = cur_cut
            best_labels = labels.copy()
        
        # WalkSAT
        ws_labels, ws_cut = walksat(labels, steps=4000)
        
        if ws_cut > best_cut:
            best_cut = ws_cut
            best_labels = ws_labels.copy()
        
        # Final local improvement
        if time.time() < deadline:
            polished = local_improve(ws_labels)
            pol_cut = cut_size(polished)
            if pol_cut > best_cut:
                best_cut = pol_cut
                best_labels = polished
    
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
