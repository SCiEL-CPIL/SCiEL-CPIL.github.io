import random
import time
import math

def fold(sequence, dimension):
    """
    Fold an HP chain on the simple cubic lattice using a contact-map-guided
    search strategy.
    
    Strategy:
    1. Estimate a "good" contact map using a fast greedy heuristic that identifies
       likely H-H contacts based on sequence proximity and a simple relaxation.
    2. Use a constraint-satisfaction-inspired growth that tries to satisfy the
       predicted contact map.
    3. Refine using local search (pivot moves) to fix any geometric inconsistencies
       and improve the energy.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]
    
    # Global deadline management
    start_time = time.time()
    deadline = start_time + 325.0  # 5s margin before 330s kill
    
    # Pre-compute H positions
    h_positions = [i for i, c in enumerate(sequence) if c == 'H']
    
    # Phase 1: Predict a contact map
    # We want to find a set of pairs (i, j) with i < j, sequence[i]==sequence[j]=='H',
    # and j-i > 1, that can be realized as non-bonded contacts.
    # A simple heuristic: prioritize pairs that are "close" in sequence but not bonded,
    # and that don't conflict with each other (e.g., can't have i contact j and i contact k
    # if j and k are too close to be on opposite sides of i in 3D? Actually in 3D, a residue
    # can have up to 6 contacts, so conflicts are less severe than in 2D).
    
    # Let's use a greedy approach to build a "target" contact list.
    # We'll iterate through H residues and try to assign contacts to previous H residues.
    # To keep it fast, we'll limit the number of contacts per residue.
    
    target_contacts = []
    # Sort H positions
    # For each H, look back at previous Hs that are not bonded (distance > 1)
    # and pick the closest few? Or use a scoring function.
    
    # A better heuristic: The number of H-H contacts in the ground state is roughly
    # proportional to the number of Hs. Let's aim for a specific number of contacts.
    # Let's just generate a list of "candidate" contacts sorted by sequence distance
    # (shorter distance is easier to satisfy geometrically? Not necessarily, but often).
    
    candidates = []
    for idx1, i in enumerate(h_positions):
        for idx2, j in enumerate(h_positions):
            if idx1 >= idx2:
                continue
            if j - i > 1:
                # Score: prefer closer in sequence? Or random?
                # Let's prefer closer in sequence as they are easier to bring together.
                score = 1.0 / (j - i)
                candidates.append((score, i, j))
                
    # Sort by score descending (closest first)
    candidates.sort(reverse=True)
    
    # Select a subset of contacts that are "compatible".
    # Compatibility: A residue can have at most 6 contacts, but in practice,
    # let's limit to 4 to be safe. Also, we can't have two contacts that "cross"
    # in a way that's impossible? In 3D, it's hard to define crossing.
    # Let's just greedily pick contacts, ensuring each residue is involved in
    # at most 3-4 contacts.
    
    contact_count = {i: 0 for i in h_positions}
    max_contacts_per_residue = 4
    selected_contacts = []
    
    for score, i, j in candidates:
        if contact_count[i] < max_contacts_per_residue and contact_count[j] < max_contacts_per_residue:
            selected_contacts.append((i, j))
            contact_count[i] += 1
            contact_count[j] += 1
            # Optional: stop if we have enough contacts?
            # Let's cap the total number of target contacts to something reasonable.
            if len(selected_contacts) >= 2 * len(h_positions):
                break

    # Phase 2: Constraint-Satisfaction-Inspired Growth
    # We will grow the chain residue by residue.
    # At each step, we prefer sites that satisfy a predicted contact with a previous residue.
    
    def _grow_with_constraints(sequence, dimension, rng, target_contacts, deadline):
        n = len(sequence)
        # Map residue index to list of target partners
        target_partners = {i: set() for i in range(n)}
        for i, j in target_contacts:
            target_partners[i].add(j)
            target_partners[j].add(i)
            
        start = (0,) * dimension
        chain = [start]
        occupied = {start: 0}
        
        # Pre-compute which previous residues are "H" and are targets for the current one?
        # We'll check on the fly.
        
        for i in range(1, n):
            if time.time() > deadline:
                return None
                
            last = chain[-1]
            options = []
            
            for site in _neighbours(last):
                if site in occupied:
                    continue
                    
                score = 0.0
                # Check if this site creates a contact with a previous H that is a target
                if sequence[i] == 'H':
                    for q in _neighbours(site):
                        if q in occupied:
                            j = occupied[q]
                            if j < i - 1 and sequence[j] == 'H':
                                # Is this a target contact?
                                if j in target_partners[i]:
                                    score += 10.0  # Strong preference for target contacts
                                else:
                                    score += 1.0   # Preference for any H-H contact
                                  
                # Add randomness
                score += rng.random() * 2.0
                options.append((score, site))
                
            if not options:
                return None
                
            # Pick the best option
            options.sort(reverse=True)
            # Sometimes pick the second best to avoid getting stuck?
            if len(options) > 1 and rng.random() < 0.2:
                site = options[1][1]
            else:
                site = options[0][1]
                
            occupied[site] = i
            chain.append(site)
            
        return chain

    # Phase 3: Local Search (Pivot Moves)
    def _pivot_move(chain, occupied, sequence, dimension, rng):
        n = len(chain)
        if n < 3:
            return None, 0
            
        # Pick a pivot point k (1 to n-2 to allow moving both sides? No, standard pivot
        # rotates the tail [k+1, n-1] around chain[k]).
        # Let's pick k from 1 to n-2.
        k = rng.randint(1, n - 2)
        pivot = chain[k]
        
        # The tail is chain[k+1:]
        tail = chain[k+1:]
        
        # We need to rotate the tail around pivot.
        # In 3D, we can rotate around any axis perpendicular to the bond (pivot, chain[k+1]).
        # Or simpler: try all 6 neighbors of pivot that are not chain[k-1] and not in the tail's original position?
        # Actually, a rigid rotation means the relative positions of the tail points remain the same,
        # just the whole tail is rotated.
        
        # Let's implement a simpler "Tail Regeneration" which is more effective for escaping local minima
        # than rigid pivots in Python due to implementation complexity.
        # But the prompt suggests "constraint-satisfaction solver".
        # Let's try a "Local Refold" of a small segment.
        
        # Pick a segment [start, end] to refold.
        seg_len = rng.randint(3, min(10, n - 2))
        start = rng.randint(1, n - seg_len - 1)
        end = start + seg_len
        
        # Fixed parts: [0, start-1] and [end+1, n-1]
        fixed_left = chain[:start]
        fixed_right = chain[end+1:]
        
        # The new segment must connect fixed_left[-1] to fixed_right[0].
        # Length of new segment is seg_len + 1 (nodes from start to end).
        # Wait, we are replacing nodes start..end.
        # The new node at 'start' must be adjacent to fixed_left[-1].
        # The new node at 'end' must be adjacent to fixed_right[0] (if fixed_right exists).
        
        # This is a hard constraint. Let's use a simple greedy growth for the segment.
        
        new_segment = []
        occupied_segment = {}
        
        # Start from fixed_left[-1]
        current = fixed_left[-1]
        
        # We need to generate seg_len+1 nodes.
        # Node 0 of new segment is at a neighbor of current.
        # ...
        # Node seg_len of new segment must be adjacent to fixed_right[0].
        
        # This is a path finding problem with obstacles.
        # Let's just try to grow from the left, and if we can't reach the right end, fail.
        
        # To make it faster, we can bias the growth towards the target end.
        
        target_end = fixed_right[0] if fixed_right else None
        
        # Greedy growth with bias
        path = []
        occupied_path = {}
        
        # The first node of the new segment
        candidates = []
        for site in _neighbours(current):
            if site in occupied and occupied[site] < start: # Collision with fixed left
                continue
            if site in occupied and occupied[site] > end: # Collision with fixed right
                continue
            # Also check if site is in the original segment? No, we are freeing it.
            # But we must not collide with fixed parts.
            
            # Bias: prefer sites closer to target_end
            dist = sum(abs(a-b) for a,b in zip(site, target_end)) if target_end else 0
            candidates.append((-dist, site))
            
        if not candidates:
            return None, 0
            
        candidates.sort()
        first_site = candidates[0][1]
        path.append(first_site)
        occupied_path[first_site] = 0
        
        # Continue growth
        for step in range(1, seg_len + 1):
            last = path[-1]
            options = []
            for site in _neighbours(last):
                # Check collision with fixed parts
                if site in occupied:
                    idx = occupied[site]
                    if idx < start or idx > end:
                        continue
                # Check collision with current path
                if site in occupied_path:
                    continue
                    
                # Bias towards target
                dist = sum(abs(a-b) for a,b in zip(site, target_end)) if target_end else 0
                options.append((-dist, site))
                
            if not options:
                return None, 0
                
            options.sort()
            # Pick best or random from top few
            if len(options) > 1 and rng.random() < 0.3:
                site = options[rng.randint(0, min(2, len(options)-1))][1]
            else:
                site = options[0][1]
                
            path.append(site)
            occupied_path[site] = step
            
        # Check if the last node is adjacent to the fixed right end
        if target_end:
            last_node = path[-1]
            if sum(abs(a-b) for a,b in zip(last_node, target_end)) != 1:
                return None, 0
                
        # Construct new chain
        new_chain = fixed_left + path + fixed_right
        
        # Verify self-avoidance (should be true by construction, but double check)
        if len(set(new_chain)) != len(new_chain):
            return None, 0
            
        # Calculate energy
        old_energy = _energy(sequence, chain)
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - old_energy
        
        return new_chain, delta

    def _energy(sequence, chain):
        where = {p: i for i, p in enumerate(chain)}
        contacts = 0
        for i, p in enumerate(chain):
            if sequence[i] == 'H':
                for q in _neighbours(p):
                    j = where.get(q)
                    if j is not None and j > i + 1 and sequence[j] == 'H':
                        contacts += 1
        return -contacts

    def _neighbours(point):
        for axis in range(len(point)):
            for step in (-1, 1):
                q = list(point)
                q[axis] += step
                yield tuple(q)

    # Main Loop
    best_chain = None
    best_energy = 0
    rng = random.Random(12345 + len(sequence))
    
    # Multiple restarts with different target maps or random seeds
    num_restarts = 20
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Slightly vary the target contacts for diversity
        # Or just use the same ones but different random growth
        chain = _grow_with_constraints(sequence, dimension, rng, selected_contacts, deadline)
        
        if chain is None:
            # Fallback to simple greedy
            chain = _simple_greedy(sequence, dimension, rng)
            if chain is None:
                continue
                
        current_energy = _energy(sequence, chain)
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_chain = chain[:]
            
        # Local Search
        # Run for a fixed time slice
        local_deadline = time.time() + time_per_restart * 0.8
        t = 1.0
        cooling = 0.999
        
        while time.time() < local_deadline:
            new_chain, delta = _pivot_move(chain, {p:i for i,p in enumerate(chain)}, sequence, dimension, rng)
            if new_chain is None:
                t *= cooling
                continue
                
            if delta < 0 or rng.random() < math.exp(-delta / t):
                chain = new_chain
                current_energy += delta
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = chain[:]
            t *= cooling
            
    if best_chain is None:
        # Last resort: straight line
        return [list([i] + [0] * (dimension - 1)) for i in range(n)]
        
    return [list(p) for p in best_chain]

def _simple_greedy(sequence, dimension, rng):
    n = len(sequence)
    start = (0,) * dimension
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, n):
        options = []
        last = chain[-1]
        for site in _neighbours(last):
            if site in occupied:
                continue
            gain = 0
            if sequence[i] == 'H':
                for q in _neighbours(site):
                    if q in occupied:
                        j = occupied[q]
                        if j < i - 1 and sequence[j] == 'H':
                            gain += 1
            score = gain + rng.random()
            options.append((score, site))
            
        if not options:
            return None
            
        options.sort(reverse=True)
        site = options[0][1]
        occupied[site] = i
        chain.append(site)
        
    return chain

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)
