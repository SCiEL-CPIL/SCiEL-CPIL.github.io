"""
Constraint-Satisfaction Backtracking Search for HP Lattice Protein Folding.
Strategy:
1. Pre-compute a contact map of high-priority H-H pairs (long-range interactions).
2. Use backtracking to place residues one by one.
3. At each step, prioritize positions that satisfy pending contact constraints.
4. Use pruning to avoid dead-ends (no free neighbors) and enforce self-avoidance.
5. Since exact backtracking can explode, we use a bounded depth/branching factor 
   with multiple random restarts and a time budget.
"""

# EVOLVE-BLOCK-START
import time
import random
import math
import sys

# Global deadline: evaluator kills after 210s wall-clock. Use a safety margin.
_DEADLINE = time.time() + 190.0  # leave 20s buffer for overhead/sandbox


def _remaining():
    return _DEADLINE - time.time()


def _neighbours(point):
    """Yield the 4 neighbours of a lattice point."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)


def _energy(sequence, chain):
    """Compute the HP energy: -1 for each non-consecutive H-H contact."""
    if chain is None:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts


def _fold_backtrack(sequence, dimension, rng, time_limit=5.0, max_depth=50000):
    """
    Backtracking search for a low-energy conformation.
    Returns (chain, energy) or (None, 0).
    """
    n = len(sequence)
    if n == 0:
        return [], 0
    if n == 1:
        return [(0, 0)], 0

    # Identify H residues
    h_indices = [i for i, c in enumerate(sequence) if c == 'H']
    
    # Pre-compute potential contacts: pairs of H residues that are far apart in sequence
    # but could potentially be neighbors in space.
    # We prioritize contacts between H residues with index difference > 1.
    # For efficiency, we focus on "high value" contacts, e.g., those that are 
    # likely to form in compact structures.
    
    # We will track "satisfied" contacts dynamically.
    # State:
    # chain: list of points
    # pos_to_idx: dict point -> index
    # current_energy: int
    
    best_chain = None
    best_energy = 0 # We want to minimize energy (make it more negative)
    
    # Heuristic: prioritize contacts between H residues.
    # We can score a move by how many new H-H contacts it creates.
    
    start_time = time.time()
    
    # Initial position
    chain = [(0, 0)]
    pos_to_idx = {(0, 0): 0}
    current_energy = 0
    
    # We use a simple recursive backtracking with a depth limit and time limit.
    # To avoid deep recursion limits, we implement it iteratively or with sys.setrecursionlimit.
    # Given n up to 100, recursion depth is manageable but branching is huge.
    # We will use a "beam search" style backtracking or just plain backtracking with strong pruning.
    
    # Actually, for HP folding, plain backtracking is too slow for n>20 without strong heuristics.
    # We will use a "look-ahead" heuristic: at each step, evaluate all valid next positions
    # and pick the best one (greedy) but keep a backup of the second best to allow backtracking
    # if a dead end is reached. This is essentially a "depth-first search with pruning".
    
    # Let's implement a simple DFS with pruning.
    # Pruning:
    # 1. Self-avoidance (checked)
    # 2. Dead-end check: if the new position has no free neighbors (except the one we came from),
    #    and we haven't finished the chain, it's a dead end.
    
    def dfs(idx, chain, pos_to_idx, energy):
        nonlocal best_chain, best_energy
        
        if time.time() - start_time > time_limit:
            return False
        
        if idx == n:
            if energy < best_energy:
                best_energy = energy
                best_chain = list(chain)
            return True
        
        last = chain[-1]
        # Get neighbors of last
        neighbors = list(_neighbours(last))
        
        # Score each neighbor
        scored_neighbors = []
        for nb in neighbors:
            if nb in pos_to_idx:
                continue
            
            # Check if it's a dead end (only if we are not at the end)
            if idx < n - 1:
                free_neighbors = 0
                for nn in _neighbours(nb):
                    if nn not in pos_to_idx and nn != last:
                        free_neighbors += 1
                if free_neighbors == 0:
                    continue
            
            # Calculate energy gain
            gain = 0
            if sequence[idx] == 'H':
                for nn in _neighbours(nb):
                    j = pos_to_idx.get(nn)
                    if j is not None and j < idx - 1 and sequence[j] == 'H':
                        gain += 1
            
            # Heuristic: prefer moves that create contacts
            # Also add a small random noise to diversify
            score = gain * 10 + rng.random() * 0.1
            scored_neighbors.append((score, nb))
        
        if not scored_neighbors:
            return False
        
        # Sort by score descending
        scored_neighbors.sort(key=lambda x: -x[0])
        
        # Try the best neighbor first
        # To allow backtracking, we would need to try others if this fails.
        # For simplicity and speed, we do a greedy step but with randomization in scoring.
        # If we hit a dead end, we return False, and the caller (restart logic) will try again.
        # This is effectively a "randomized greedy" with backtracking on failure.
        
        # Let's try the top candidate
        _, next_pos = scored_neighbors[0]
        
        chain.append(next_pos)
        pos_to_idx[next_pos] = idx
        
        new_energy = energy - gain # Energy decreases by gain (since gain is count of contacts)
        
        if dfs(idx + 1, chain, pos_to_idx, new_energy):
            return True
        
        # Backtrack
        chain.pop()
        del pos_to_idx[next_pos]
        
        # Try next best if time permits? 
        # For now, just fail this branch. The restart loop will handle diversification.
        # To make it more robust, we could try the next best, but that increases complexity.
        # Let's try the next best if we have time.
        if len(scored_neighbors) > 1 and time.time() - start_time < time_limit:
            _, next_pos2 = scored_neighbors[1]
            chain.append(next_pos2)
            pos_to_idx[next_pos2] = idx
            
            # Recalculate gain for next_pos2
            gain2 = 0
            if sequence[idx] == 'H':
                for nn in _neighbours(next_pos2):
                    j = pos_to_idx.get(nn)
                    if j is not None and j < idx - 1 and sequence[j] == 'H':
                        gain2 += 1
            # Note: pos_to_idx still has old entries, but we removed next_pos, so it's consistent
            # Wait, we need to be careful. The gain calculation for next_pos2 assumes the chain is as it was before adding next_pos.
            # Since we popped next_pos, the state is correct.
            
            new_energy2 = energy - gain2
            
            if dfs(idx + 1, chain, pos_to_idx, new_energy2):
                return True
            
            # Backtrack again
            chain.pop()
            del pos_to_idx[next_pos2]
            
        return False

    dfs(1, chain, pos_to_idx, current_energy)
    
    if best_chain is not None:
        return best_chain, best_energy
    return None, 0


def fold(sequence, dimension):
    """
    Find a low-energy conformation for the given HP sequence on a square lattice.
    Uses multiple randomized backtracking searches.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    rng = random.Random(42 + len(sequence))

    best_chain = None
    best_energy = 0 # Worst case is 0 (no contacts), we want negative

    # Allocate time budget
    # 10 sequences, 120s total. ~12s per sequence.
    # We will run multiple restarts within this budget.
    
    # For small sequences, we can be more exhaustive.
    # For large sequences, we need to be faster per restart.
    
    if n <= 25:
        max_restarts = 200
        time_per_restart = 0.05
    elif n <= 40:
        max_restarts = 100
        time_per_restart = 0.1
    elif n <= 60:
        max_restarts = 50
        time_per_restart = 0.2
    else:
        max_restarts = 20
        time_per_restart = 0.5

    start_time = time.time()
    
    for restart in range(max_restarts):
        if _remaining() < 1.0:
            break
        
        # Use a small time limit per restart to allow many restarts
        # The dfs function will stop if it exceeds time_limit
        chain, energy = _fold_backtrack(sequence, dimension, rng, time_limit=time_per_restart)
        
        if chain is not None and energy < best_energy:
            best_chain = chain
            best_energy = energy
            
            # If we reached a very low energy, we might stop early
            # Theoretical minimum is roughly -n/2 for random HP, but best known is better.
            # We'll just keep going until time runs out or restarts done.

    # If no chain was found (shouldn't happen, but just in case), return a straight line
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
        best_energy = _energy(sequence, best_chain)

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
