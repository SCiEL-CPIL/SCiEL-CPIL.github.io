"""HP lattice folding with Monte Carlo Tree Search and time management."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Track global start time for deadline management
_START_TIME = None

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_remaining():
    _init_time()
    # Leave 15s safety margin before 210s wall-clock limit
    return 195.0 - (time.time() - _START_TIME)

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _is_adjacent(p, q):
    return sum(abs(a - b) for a, b in zip(p, q)) == 1

def _energy(sequence, chain):
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts

def _grow(sequence, dimension, rng):
    """One randomized greedy chain growth: prefer sites with new H-H contacts, then free space."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == "H":
                gain = sum(1 for q in _neighbours(site)
                           if q in occupied and occupied[q] < i - 1 and sequence[occupied[q]] == "H")
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            options.append((gain + 0.1 * free + rng.random(), site))
        if not options:
            return None
        site = max(options)[1]
        occupied[site] = i
        chain.append(site)
    return chain

class MCTSNode:
    __slots__ = ['index', 'position', 'parent', 'children', 'visits', 'value', 'unvisited_moves', 'energy_delta']
    
    def __init__(self, index, position, parent=None):
        self.index = index
        self.position = position
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.unvisited_moves = []
        self.energy_delta = 0

def _mcts_fold(sequence, dimension, rng, max_iterations=5000, c=1.414):
    """
    Monte Carlo Tree Search for HP folding.
    Each node represents a partial fold. Moves are next positions.
    """
    n = len(sequence)
    if n <= 1:
        return [(0,)] * n
    
    is_h = [c == 'H' for c in sequence]
    
    # Root: start at origin
    root = MCTSNode(0, (0,) * dimension)
    root.unvisited_moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    
    best_chain = None
    best_energy = 0
    
    # Precompute for fast energy delta
    # We'll track the chain implicitly via parent pointers
    
    def _get_chain(node):
        chain = []
        while node is not None:
            chain.append(node.position)
            node = node.parent
        chain.reverse()
        return chain
    
    def _select(node):
        """Select a leaf node using UCB1."""
        while node.children:
            # Choose child with max UCB
            best_child = None
            best_ucb = float('-inf')
            for move, child in node.children.items():
                if child.visits == 0:
                    ucb = float('inf')
                else:
                    # Value is negative energy (higher is better)
                    # We want to maximize, so use +value
                    ucb = child.value / child.visits + c * math.sqrt(math.log(node.visits + 1) / child.visits)
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_child = child
            node = best_child
        return node
    
    def _expand(node):
        """Expand a leaf node by adding one random unvisited move."""
        if not node.unvisited_moves:
            return None
        
        # Pick a random move
        idx = rng.randint(0, len(node.unvisited_moves) - 1)
        move = node.unvisited_moves.pop(idx)
        
        new_pos = (node.position[0] + move[0], node.position[1] + move[1])
        new_index = node.index + 1
        
        # Check if valid (not visited)
        # We need to check against the entire path
        # This is expensive, so we'll do it here
        # For MCTS, we can store visited positions in the node or path
        # Let's store a set of visited positions in the node for efficiency? No, that's memory heavy.
        # Instead, we check by traversing up? That's O(N).
        # Given N<=100, O(N) per expansion is acceptable if iterations are limited.
        
        # Actually, let's store the chain in the node for faster checking? 
        # Or just check the parent's chain.
        
        # Let's check if new_pos is in the current path
        # We can store a set of positions in the node? 
        # Let's add a 'pos_set' attribute or just check the path.
        # For now, check path.
        
        # Optimization: Store the set of positions in the node? 
        # Let's just check the path for now.
        
        # Traverse up to check if new_pos is already used
        # This is O(N). 
        # Alternatively, we can maintain a global set of positions for the current path during selection/expansion?
        # No, MCTS branches.
        
        # Let's just check the path.
        # To make it faster, we can store the path in the node? No.
        
        # Check if new_pos is adjacent to the last position? It is by construction.
        # Check if new_pos is self-avoiding.
        
        # We'll build the chain for the new node to check self-avoidance.
        # This is expensive. 
        # Let's try a different approach: 
        # During selection, we can maintain the path.
        # But MCTS selection is tree-based.
        
        # Let's just check the path.
        # If it's too slow, we'll limit iterations.
        
        # Check self-avoidance
        # We need to ensure new_pos is not in the path from root to node.
        # We can do this by walking up the parent chain.
        # This is O(N).
        
        # Let's implement it.
        
        # Actually, a better way: 
        # Store the chain in the node? No.
        # Store a set of positions? 
        # Let's just walk up.
        
        # For N=100, walking up 100 steps is fast.
        
        # Check if new_pos is in the path
        # We'll do it in the node creation.
        
        # Create child node
        child = MCTSNode(new_index, new_pos, parent=node)
        
        # Check self-avoidance
        # Walk up from node to root
        # If new_pos is found, it's invalid
        # But we already popped the move, so if invalid, we should try another?
        # Or just return None if invalid.
        
        # Let's check validity before creating child.
        # Walk up from node
        current = node
        while current is not None:
            if current.position == new_pos:
                # Invalid move
                return None
            current = current.parent
        
        # Valid move
        # Calculate energy delta
        # This requires knowing which positions are H and their neighbors.
        # We need the chain to compute contacts.
        # This is expensive.
        
        # Let's simplify: 
        # We'll use a simulation-based approach where we evaluate the full chain at the leaf.
        # This is standard MCTS for combinatorial problems.
        
        node.children[move] = child
        return child
    
    def _simulate(node):
        """Simulate a random rollout from the leaf node."""
        # Build the current chain
        chain = _get_chain(node)
        occupied = {p: i for i, p in enumerate(chain)}
        
        # Continue growing randomly
        for i in range(node.index + 1, n):
            options = []
            for site in _neighbours(chain[-1]):
                if site in occupied:
                    continue
                options.append(site)
            if not options:
                return None # Dead end
            site = rng.choice(options)
            occupied[site] = i
            chain.append(site)
        
        # Compute energy
        energy = _energy(sequence, chain)
        return energy, chain
    
    def _backpropagate(node, energy):
        """Backpropagate the result up the tree."""
        while node is not None:
            node.visits += 1
            # We want to maximize energy (which is negative, so more negative is better)
            # So we add the energy to the value
            node.value += energy
            node = node.parent
    
    # Main MCTS loop
    iterations = 0
    while iterations < max_iterations and _time_remaining() > 5:
        # Selection
        node = _select(root)
        
        # Expansion
        child = _expand(node)
        if child is None:
            # No valid moves, try another path?
            # For now, just skip
            continue
        
        # Simulation
        result = _simulate(child)
        if result is None:
            # Dead end, backpropagate a bad value
            _backpropagate(child, 0) # 0 is worse than any negative energy
            iterations += 1
            continue
        
        energy, chain = result
        
        # Update best
        if best_chain is None or energy < best_energy:
            best_chain = chain
            best_energy = energy
        
        # Backpropagation
        _backpropagate(child, energy)
        
        iterations += 1
    
    if best_chain is None:
        return [(i,) + (0,) * (dimension - 1) for i in range(n)]
    
    return best_chain

def _simulated_annealing(sequence, chain, rng, max_iters=2000, start_temp=1.0, end_temp=0.01):
    """Simulated annealing with single-residue moves."""
    current = list(chain)
    n = len(current)
    where = {p: i for i, p in enumerate(current)}
    
    # Compute initial contacts
    current_contacts = set()
    is_h = [c == 'H' for c in sequence]
    for i, p in enumerate(current):
        if is_h[i]:
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and is_h[j]:
                    current_contacts.add(frozenset((i, j)))
    
    current_energy = -len(current_contacts)
    
    for it in range(max_iters):
        if _time_remaining() < 5:
            break
        
        temp = start_temp * (end_temp / start_temp) ** (it / max(1, max_iters))
        
        # Pick a random internal residue to move
        i = rng.randint(1, n - 2)
        old_pos = current[i]
        
        best_move = None
        best_delta = 0
        
        for new_pos in _neighbours(old_pos):
            if new_pos in where:
                continue
            if not _is_adjacent(new_pos, current[i-1]):
                continue
            if not _is_adjacent(new_pos, current[i+1]):
                continue
            
            # Compute energy delta
            delta = 0
            if is_h[i]:
                # Remove old contacts involving i
                for q in _neighbours(old_pos):
                    j = where.get(q)
                    if j is not None and j != i:
                        pair = frozenset((i, j))
                        if pair in current_contacts:
                            delta += 1
                # Add new contacts involving i
                for q in _neighbours(new_pos):
                    j = where.get(q)
                    if j is not None and j != i and is_h[j]:
                        pair = frozenset((i, j))
                        if pair not in current_contacts:
                            delta -= 1
            
            if best_move is None or delta < best_delta:
                best_delta = delta
                best_move = new_pos
        
        if best_move is None:
            continue
        
        new_energy = current_energy + best_delta
        
        # Accept if better, or with probability exp(-delta/temp)
        accept = False
        if new_energy < current_energy:
            accept = True
        elif temp > 0:
            prob = math.exp(-best_delta / temp)
            accept = rng.random() < prob
        
        if accept:
            # Apply the move
            del where[old_pos]
            where[best_move] = i
            current[i] = best_move
            
            # Update contacts
            if is_h[i]:
                # Remove old contacts
                old_contact_pairs = set()
                for q in _neighbours(old_pos):
                    j = where.get(q)
                    if j is not None and j != i and is_h[j]:
                        old_contact_pairs.add(frozenset((i, j)))
                for pair in old_contact_pairs:
                    if pair in current_contacts:
                        current_contacts.remove(pair)
                
                # Add new contacts
                for q in _neighbours(best_move):
                    j = where.get(q)
                    if j is not None and j != i and is_h[j]:
                        pair = frozenset((i, j))
                        current_contacts.add(pair)
            
            current_energy = new_energy
    
    return current

def fold(sequence, dimension):
    """Best conformation using MCTS + simulated annealing with time management."""
    _init_time()
    rng = random.Random(12345 + len(sequence))
    best, best_energy = None, 0
    
    n = len(sequence)
    
    # Use MCTS for initial folding
    # Limit iterations based on time
    mcts_iters = 2000 if n <= 50 else 1000
    chain = _mcts_fold(sequence, dimension, rng, max_iterations=mcts_iters)
    
    if chain is not None:
        # Apply simulated annealing for fine-tuning
        sa_iters = max(200, 2000 - n * 10)
        chain = _simulated_annealing(sequence, chain, rng, max_iters=sa_iters)
        
        value = _energy(sequence, chain)
        if value < best_energy:
            best, best_energy = chain, value
    
    # If MCTS didn't find a good solution, try greedy growth + SA
    if best is None or _time_remaining() > 20:
        num_restarts = 20 if n <= 30 else 10
        for restart in range(num_restarts):
            if _time_remaining() < 10:
                break
            chain = _grow(sequence, dimension, rng)
            if chain is None:
                continue
            
            sa_iters = max(200, 2000 - n * 10)
            chain = _simulated_annealing(sequence, chain, rng, max_iters=sa_iters)
            
            value = _energy(sequence, chain)
            if value < best_energy:
                best, best_energy = chain, value
    
    if best is None:
        # Fallback: straight line
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]
    
    return [list(p) for p in best]
# EVOLVE-BLOCK-END
