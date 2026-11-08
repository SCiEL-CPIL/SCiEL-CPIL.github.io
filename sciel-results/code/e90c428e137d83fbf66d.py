"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    # Use vectorized computation
    diff = labels[None, :] != labels[:, None]
    return (adjacency & diff).sum() // 2

def _parallel_tempering(adjacency, num_chains=20, deadline=None):
    """
    Parallel tempering (replica exchange simulated annealing) for Max-Cut.
    
    Maintains multiple chains at different temperatures.
    - High temperature chains explore broadly
    - Low temperature chains exploit local optima
    - Periodic swaps between chains allow information exchange
    """
    n = len(adjacency)
    
    # Precompute adjacency as float for speed
    adj = adjacency.astype(np.float32)
    
    # Temperature schedule: from high to low
    # Chain 0 is coldest (exploitation), chain num_chains-1 is hottest (exploration)
    temps = np.linspace(0.01, 10.0, num_chains)
    
    # Initialize each chain with random labels
    labels = np.random.randint(0, 2, size=(num_chains, n)).astype(np.int8)
    
    # Track best solution found across all chains
    best_labels = labels[0].copy()
    best_cut = _cut_size(adjacency, best_labels)
    
    # Precompute degree for gain calculation
    degrees = adj.sum(axis=1)
    
    start_time = time.time()
    swap_interval = 5  # Swap every 5 steps
    
    step = 0
    while True:
        if deadline is not None and time.time() >= deadline:
            break
        
        # For each chain, propose a flip
        for c in range(num_chains):
            # Randomly select a vertex to flip
            v = np.random.randint(0, n)
            
            # Calculate gain of flipping vertex v in chain c
            # Gain = (neighbors with opposite label) - (neighbors with same label)
            # = 2 * (neighbors with opposite label) - degree[v]
            # But more precisely for cut:
            # Current contribution of v to cut: number of neighbors with opposite label
            # After flip: number of neighbors with same label (which becomes opposite)
            # So gain = (neighbors with same label) - (neighbors with opposite label)
            # = 2*(neighbors with same label) - degree[v]
            
            lv = labels[c, v]
            # Count neighbors with same label
            same_count = (adjacency[v] & (labels[c] == lv)).sum()
            # Count neighbors with opposite label  
            other_count = (adjacency[v] & (labels[c] != lv)).sum()
            
            # If we flip v, the new cut contribution from v's edges:
            # Edges that were crossing (other_count) become non-crossing
            # Edges that were not crossing (same_count) become crossing
            # Net change = same_count - other_count
            delta_cut = same_count - other_count
            
            # Acceptance criterion
            if delta_cut > 0:
                # Always accept improving moves
                labels[c, v] = 1 - lv
            else:
                # Accept worsening moves with probability exp(delta_cut / temperature)
                # delta_cut is negative here
                if temps[c] > 0:
                    prob = np.exp(delta_cut / temps[c])
                    if np.random.random() < prob:
                        labels[c, v] = 1 - lv
            
            # Update best if this chain improved
            if c == 0:  # Only check the coldest chain for efficiency, or check all
                pass
        
        # Periodically swap adjacent chains (Metropolis criterion for swap)
        if step % swap_interval == 0:
            for c in range(num_chains - 1):
                # Try to swap chains c and c+1
                # Accept if it doesn't break the Metropolis distribution
                # Simple approach: always swap for simplicity (works well in practice)
                labels[c], labels[c + 1] = labels[c + 1].copy(), labels[c].copy()
        
        step += 1
        
        # Check best cut periodically
        if step % 10 == 0:
            # Check all chains for best solution
            for c in range(num_chains):
                cut = _cut_size(adjacency, labels[c])
                if cut > best_cut:
                    best_cut = cut
                    best_labels = labels[c].copy()
        
        # Time check
        if deadline is not None and time.time() >= deadline - 2.0:
            break
    
    return best_labels, best_cut

def _local_search(adjacency, labels, max_time=None):
    """Fast local search via single-flip improvements (for polishing)."""
    n = len(labels)
    labels = labels.copy()
    adj = adjacency.astype(np.int8)
    
    changed = True
    start = time.time()
    while changed:
        changed = False
        for v in range(n):
            lv = labels[v]
            same = (adj[v] & (labels == lv)).sum()
            other = (adj[v] & (labels != lv)).sum()
            if other > same:
                labels[v] = 1 - lv
                changed = True
            if max_time is not None and time.time() - start > max_time:
                return labels
    return labels

def partition(adjacency):
    """
    Main partition function using parallel tempering with local search polishing.
    
    Strategy:
    1. Run parallel tempering for most of the time budget to explore solution space
    2. Polish the best found solution with local search
    3. Return the best label assignment
    """
    n = len(adjacency)
    deadline = time.time() + 140.0  # 140s out of 150s wall-clock budget
    
    # Phase 1: Parallel tempering (exploration)
    pt_deadline = deadline - 10.0  # Leave 10s for polishing and safety
    best_labels, best_cut = _parallel_tempering(
        adjacency, 
        num_chains=30, 
        deadline=pt_deadline
    )
    
    # Phase 2: Polish with local search
    if time.time() < deadline - 5.0:
        polished = _local_search(adjacency, best_labels, max_time=min(5.0, deadline - time.time() - 2.0))
        polished_cut = _cut_size(adjacency, polished)
        if polished_cut > best_cut:
            best_labels = polished
            best_cut = polished_cut
    
    # Phase 3: If time remains, try a few more restarts
    remaining = deadline - time.time() - 3.0
    if remaining > 5.0:
        # Try a few more parallel tempering runs with different seeds
        for restart in range(3):
            if time.time() >= deadline - 3.0:
                break
            sub_deadline = min(deadline - 3.0, time.time() + remaining / 3.0)
            labels_r, cut_r = _parallel_tempering(
                adjacency, 
                num_chains=15,
                deadline=sub_deadline
            )
            if cut_r > best_cut:
                best_labels = labels_r
                best_cut = cut_r
    
    return best_labels
# EVOLVE-BLOCK-END
