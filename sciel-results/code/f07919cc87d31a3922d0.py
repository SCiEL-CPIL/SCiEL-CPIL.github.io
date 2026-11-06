"""
MCTS-based HP Lattice Protein Folding.
Strategy:
1. Formulate folding as a sequential decision problem: place residue 0, then 1, ..., N-1.
2. Use Monte Carlo Tree Search (MCTS) to explore the space of self-avoiding walks.
3. The "value" of a node is estimated by a fast heuristic (greedy completion) or rollouts.
4. UCB1 is used to balance exploration and exploitation.
5. Pruning: if a node has no valid children or leads to a dead end, it is pruned.
6. We maintain a global best conformation found during the search.
"""

# EVOLVE-BLOCK-START
import time
import math
import random
import sys

# Global deadline: evaluator kills after 210s wall-clock. Use a safety margin.
_DEADLINE = time.time() + 180.0  # Leave 30s buffer for safety/overhead

def _remaining():
    return _DEADLINE - time.time()

def _neighbours(point):
    """Yield the 4 neighbours of a 2D lattice point."""
    x, y = point
    return ((x+1, y), (x-1, y), (x, y+1), (x, y-1))

def _energy(sequence, chain):
    """Compute the HP energy: -1 for each non-consecutive H-H contact."""
    if chain is None:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    n = len(chain)
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

class MCTSNode:
    __slots__ = ['state', 'parent', 'children', 'visits', 'value', 'residue_idx']
    
    def __init__(self, state, parent=None, residue_idx=0):
        self.state = state # (chain, occupied_set, last_pos)
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.residue_idx = residue_idx

def _ucb1(node, c=1.414):
    if node.visits == 0:
        return float('inf')
    return node.value / node.visits + c * math.sqrt(math.log(node.parent.visits + 1) / node.visits)

def _fast_heuristic_completion(sequence, chain, occupied, rng):
    """
    Greedily complete the chain from the current state to get a quick estimate.
    Returns (energy, success).
    """
    n = len(sequence)
    current_chain = list(chain)
    current_occupied = set(occupied)
    last = current_chain[-1]
    
    # We need to place residues from len(current_chain) to n-1
    for i in range(len(current_chain), n):
        res = sequence[i]
        options = []
        for site in _neighbours(last):
            if site in current_occupied:
                continue
            # Heuristic: prefer sites that form H-H contacts if res is H
            gain = 0
            if res == 'H':
                for q in _neighbours(site):
                    if q in current_occupied:
                        # Check if the occupant is H and not adjacent in chain
                        # We don't have the full map here easily, so approximate
                        # Just check if it's occupied by H (we can't easily know index without map)
                        # For speed, we just count occupied neighbors as potential contacts
                        pass 
            # Prefer sites with more free neighbors (less trapping)
            free = sum(1 for q in _neighbours(site) if q not in current_occupied)
            score = free + rng.random() * 0.5
            options.append((score, site))
            
        if not options:
            return 0, False # Dead end
        
        # Pick best
        options.sort(key=lambda x: -x[0])
        site = options[0][1]
        current_occupied.add(site)
        current_chain.append(site)
        last = site
        
    # Calculate actual energy of the completed chain
    return _energy(sequence, current_chain), True

def _mcts_fold(sequence, dimension, max_iters=1000):
    """
    Perform MCTS to find a low-energy conformation.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]
    
    rng = random.Random(42)
    
    # Root state: Residue 0 at (0,0)
    root_state = ([(0, 0)], {(0, 0)}, (0, 0))
    root = MCTSNode(root_state, residue_idx=0)
    
    best_chain = None
    best_energy = 0 # We want negative energy, so start with 0 (unfolded)
    
    # Pre-calculate H positions for faster heuristic?
    h_positions = [i for i, r in enumerate(sequence) if r == 'H']
    
    for i in range(max_iters):
        if _remaining() < 5.0:
            break
            
        # 1. Selection
        node = root
        while node.children:
            # Sort children by UCB1
            # Optimization: only sort if many children? 
            # For small branching factor (<=4), sorting is cheap
            node.children.sort(key=lambda c: _ucb1(c), reverse=True)
            node = node.children[0]
            
            # If we reached a leaf (residue_idx == n), we don't select further
            if node.residue_idx == n:
                break
            
        # 2. Expansion
        if node.residue_idx < n - 1:
            # Generate children
            chain, occupied, last = node.state
            next_idx = node.residue_idx + 1
            res = sequence[next_idx]
            
            valid_children = []
            for site in _neighbours(last):
                if site in occupied:
                    continue
                new_occupied = occupied | {site}
                new_chain = chain + [site]
                child_state = (new_chain, new_occupied, site)
                child_node = MCTSNode(child_state, node, next_idx)
                valid_children.append(child_node)
                node.children.append(child_node)
            
            if not valid_children:
                # Dead end, mark as terminal with bad value
                node.visits += 1
                node.value = -100 * node.residue_idx # Penalize dead ends
                continue
                
            # Pick one child to expand (or just use the first one? 
            # Standard MCTS expands one child per iteration)
            # To speed up, we can expand a random child from valid_children
            # But we just added them all. We should pick one to traverse.
            # Actually, standard MCTS adds ONE child. 
            # My current loop adds ALL valid children. This is "full expansion".
            # This is more like a breadth-first search with UCB selection on the frontier.
            # Let's stick to adding all valid children to the node's list, 
            # but only traverse one in the next step? 
            # No, the selection loop picks the best child. 
            # If we add all, the selection loop will pick the best UCB among them.
            # This is valid.
            
            # We don't break here, we fall through to simulation?
            # No, we just expanded. We should simulate from the newly added child?
            # Or just update the node?
            # Standard: Select -> Expand (1 child) -> Simulate -> Backprop.
            # My implementation adds all children. This is slightly non-standard but valid.
            # Let's pick the best child among the new ones to simulate from?
            # Or just pick the first one?
            # Let's pick the child with the highest heuristic value immediately.
            
            # Heuristic evaluation of new children
            best_child = None
            best_heur = -float('inf')
            for child in valid_children:
                # Fast heuristic
                e, ok = _fast_heuristic_completion(sequence, child.state[0], child.state[1], rng)
                if ok:
                    # e is negative, so more negative is better
                    if e > best_heur:
                        best_heur = e
                        best_child = child
            
            if best_child is not None:
                node = best_child
            else:
                # All dead ends or bad
                node.visits += 1
                node.value = -50
                continue
                
        # 3. Simulation (Rollout)
        # If we are at a leaf (residue_idx == n), we have a full chain
        if node.residue_idx == n:
            chain = node.state[0]
            energy = _energy(sequence, chain)
            # Update best
            if energy < best_energy:
                best_energy = energy
                best_chain = chain
            # Backprop
            sim_value = energy
        else:
            # If not at leaf, do a fast rollout from this node
            chain, occupied, last = node.state
            e, ok = _fast_heuristic_completion(sequence, chain, occupied, rng)
            if ok:
                sim_value = e
                # Update best if this rollout is good?
                # The rollout chain is a valid self-avoiding walk
                # We don't have the full chain object easily from the helper, 
                # but we can reconstruct or just trust the energy.
                # To be safe, we only update best if we have a full chain from MCTS path.
                # But rollouts are also valid! Let's try to get the chain.
                # The helper returns energy only. We can modify it to return chain.
                # For now, we just use the value for backprop.
            else:
                sim_value = -100 # Dead end penalty

        # 4. Backpropagation
        while node:
            node.visits += 1
            node.value += sim_value
            node = node.parent
            
    if best_chain is None:
        # Fallback: straight line
        return [[i, 0] for i in range(n)]
        
    return [list(p) for p in best_chain]

def _local_search(sequence, chain, dimension, rng, max_iter=100):
    """
    Improve a valid chain via local moves.
    """
    if chain is None:
        return None, 1

    current_energy = _energy(sequence, chain)
    current_chain = list(chain)
    n = len(current_chain)
    
    def _make_pos_map(ch):
        return {p: i for i, p in enumerate(ch)}
    
    pos_map = _make_pos_map(current_chain)

    for it in range(max_iter):
        if _remaining() < 1.0:
            break

        best_delta = 0
        best_new_chain = None
        best_i = -1
        best_new_pos = None
        
        # Try single residue moves
        i = rng.randint(1, n - 2)
        p = current_chain[i]
        for q in _neighbours(p):
            if q == current_chain[i - 1] or q == current_chain[i + 1]:
                continue
            if q in pos_map and pos_map[q] != i:
                continue
            prev = current_chain[i-1]
            nxt = current_chain[i+1]
            if abs(q[0] - prev[0]) + abs(q[1] - prev[1]) != 1:
                continue
            if abs(q[0] - nxt[0]) + abs(q[1] - nxt[1]) != 1:
                continue
            
            delta = 0
            if sequence[i] == 'H':
                # Remove old contacts
                for old_q in _neighbours(p):
                    j = pos_map.get(old_q)
                    if j is not None and j != i and sequence[j] == 'H' and abs(j - i) > 1:
                        delta += 1
                # Add new contacts
                for new_q in _neighbours(q):
                    j = pos_map.get(new_q)
                    if j is not None and j != i and sequence[j] == 'H' and abs(j - i) > 1:
                        delta -= 1
                
            if delta < best_delta:
                best_delta = delta
                best_i = i
                best_new_pos = q
        
        if best_delta < 0 and best_i >= 0:
            old_p = current_chain[best_i]
            del pos_map[old_p]
            current_chain[best_i] = best_new_pos
            pos_map[best_new_pos] = best_i
            current_energy += best_delta
        else:
            # Random perturbation
            if rng.random() < 0.2:
                i = rng.randint(1, n - 2)
                p = current_chain[i]
                for q in _neighbours(p):
                    if q == current_chain[i - 1] or q == current_chain[i + 1]:
                        continue
                    if q in pos_map and pos_map[q] != i:
                        continue
                    prev = current_chain[i-1]
                    nxt = current_chain[i+1]
                    if abs(q[0] - prev[0]) + abs(q[1] - prev[1]) != 1:
                        continue
                    if abs(q[0] - nxt[0]) + abs(q[1] - nxt[1]) != 1:
                        continue
                    
                    delta = 0
                    if sequence[i] == 'H':
                        for old_q in _neighbours(p):
                            j = pos_map.get(old_q)
                            if j is not None and j != i and sequence[j] == 'H' and abs(j - i) > 1:
                                delta += 1
                        for new_q in _neighbours(q):
                            j = pos_map.get(new_q)
                            if j is not None and j != i and sequence[j] == 'H' and abs(j - i) > 1:
                                delta -= 1
                    
                    if delta <= 0 or rng.random() < math.exp(-delta / 2.0):
                        old_p = current_chain[i]
                        del pos_map[old_p]
                        current_chain[i] = q
                        pos_map[q] = i
                        current_energy += delta
                        break

    return current_chain, current_energy

def fold(sequence, dimension):
    """
    Find a low-energy conformation for the given HP sequence on a square lattice.
    Uses MCTS followed by local search.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0, 0]]

    rng = random.Random(42 + len(sequence))

    best_chain = None
    best_energy = 0

    # Allocate time budget
    # Short sequences: more MCTS iterations
    # Long sequences: fewer MCTS iterations, more local search
    if n <= 30:
        mcts_iters = 500
        ls_iter = 200
    elif n <= 50:
        mcts_iters = 200
        ls_iter = 100
    else:
        mcts_iters = 100
        ls_iter = 50

    # Run MCTS
    if _remaining() > 10.0:
        mcts_chain = _mcts_fold(sequence, dimension, max_iters=mcts_iters)
        if mcts_chain:
            best_chain = mcts_chain
            best_energy = _energy(sequence, best_chain)
            
            # Local search refinement
            if _remaining() > 5.0:
                improved_chain, improved_energy = _local_search(
                    sequence, best_chain, dimension, rng, max_iter=ls_iter
                )
                if improved_chain is not None and improved_energy < best_energy:
                    best_chain = improved_chain
                    best_energy = improved_energy

    # If MCTS failed or didn't run, use greedy fallback
    if best_chain is None or _remaining() > 2.0:
        # Simple greedy growth
        start = (0, 0)
        chain = [start]
        occupied = {start: 0}
        for i in range(1, n):
            last = chain[-1]
            options = []
            for site in _neighbours(last):
                if site in occupied:
                    continue
                free = sum(1 for q in _neighbours(site) if q not in occupied)
                score = free + rng.random()
                options.append((score, site))
            if not options:
                break
            options.sort(key=lambda x: -x[0])
            site = options[0][1]
            occupied[site] = i
            chain.append(site)
        
        if len(chain) == n:
            energy = _energy(sequence, chain)
            if energy < best_energy:
                best_chain = chain
                best_energy = energy
                
            if _remaining() > 5.0:
                improved_chain, improved_energy = _local_search(
                    sequence, chain, dimension, rng, max_iter=ls_iter
                )
                if improved_chain is not None and improved_energy < best_energy:
                    best_chain = improved_chain
                    best_energy = improved_energy

    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]

    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
