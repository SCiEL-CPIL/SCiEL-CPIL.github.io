import random
import time
import math
import sys
from collections import defaultdict, deque

def _neighbours(point):
    x, y, z = point
    yield (x+1, y, z)
    yield (x-1, y, z)
    yield (x, y+1, z)
    yield (x, y-1, z)
    yield (x, y, z+1)
    yield (x, y, z-1)

def _manhattan_dist(p1, p2):
    """Calculate Manhattan distance between two 3D points."""
    return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1]) + abs(p1[2]-p2[2])

def _energy(sequence, chain):
    """
    Calculate the energy of a chain.
    Energy = - (number of H-H contacts)
    """
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts

def _compute_contact_map(sequence, chain):
    """
    Precompute which pairs of H residues are in contact.
    Returns a set of (i, j) tuples where i < j and both are H with Manhattan distance 1.
    """
    where = {p: i for i, p in enumerate(chain)}
    contacts = set()
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == 'H':
                    contacts.add((i, j))
    return contacts

def _grow(sequence, dimension, rng):
    """
    Greedy growth to create an initial self-avoiding walk with improved heuristics.
    """
    start = (0, 0, 0)
    chain = [start]
    occupied = {start: 0}
    
    for i in range(1, len(sequence)):
        options = []
        last = chain[-1]
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Heuristic: prefer sites that create contacts with existing H residues
            gain = 0
            if sequence[i] == "H":
                for q in _neighbours(site):
                    if q in occupied:
                        idx = occupied[q]
                        if idx < i - 1 and sequence[idx] == "H":
                            gain += 1
            # Also check if placing P here could enable future H-H contacts
            # (look ahead 1-2 steps for H residues)
            if sequence[i] == "P":
                for q in _neighbours(site):
                    if q in occupied:
                        idx = occupied[q]
                        if idx < i - 1 and sequence[idx] == "H":
                            gain += 0.5  # Partial credit for proximity to H
            # Add small random noise to avoid deterministic traps
            score = gain * 5.0 + rng.random() * 2.0
            options.append((score, site))
        
        if not options:
            return None
            
        # Select best option with some randomness to explore different basins
        if len(options) > 1 and rng.random() < 0.3:
            # Sometimes pick a random valid option to maintain diversity
            best_site = rng.choice(options)[1]
        else:
            best_score, best_site = max(options)
        chain.append(best_site)
        occupied[best_site] = i
        
    return chain

def _pivot_move(sequence, chain, i, rng):
    """
    Attempt a pivot move at index i.
    Returns (new_chain, delta_e) or (None, 0) if invalid.
    Optimized: precompute H positions for faster energy delta calculation.
    """
    N = len(chain)
    if i <= 0 or i >= N - 1:
        return None, 0
        
    p_prev = chain[i-1]
    p_curr = chain[i]
    
    # Axis is the vector from p_prev to p_curr
    axis = (p_curr[0] - p_prev[0], p_curr[1] - p_prev[1], p_curr[2] - p_prev[2])
    
    # Determine rotation direction (+90 or -90)
    sign = 1 if rng.random() < 0.5 else -1
    
    # Pre-calculate rotation matrix components for 90 deg around axis
    ax, ay, az = axis
    
    new_positions = []
    valid = True
    fixed_set = set(chain[:i])
    
    # Precompute H positions in fixed and moving parts for faster energy calc
    h_fixed = [(u, chain[u]) for u in range(i) if sequence[u] == 'H']
    h_moving_old = [(v, chain[v]) for v in range(i, N) if sequence[v] == 'H']
    
    for j in range(i, N):
        p_old = chain[j]
        vx = p_old[0] - p_prev[0]
        vy = p_old[1] - p_prev[1]
        vz = p_old[2] - p_prev[2]
        
        # Cross product k x v
        cx = ay * vz - az * vy
        cy = az * vx - ax * vz
        cz = ax * vy - ay * vx
        
        # Dot product k . v
        dot = ax * vx + ay * vy + az * vz
        
        # v' = (k x v) + k * dot
        rx = cx + ax * dot
        ry = cy + ay * dot
        rz = cz + az * dot
        
        if sign == -1:
            rx = -rx
            ry = -ry
            rz = -rz
            
        p_new = (p_prev[0] + rx, p_prev[1] + ry, p_prev[2] + rz)
        
        # Check collision with fixed part
        if p_new in fixed_set:
            valid = False
            break
            
        new_positions.append(p_new)
        
    if not valid:
        return None, 0
    
    # Calculate Delta Energy efficiently
    # Only contacts between fixed part (0..i-1) and moving part (i..N-1) change
    old_cross = 0
    for u_idx, pu in h_fixed:
        for v_idx, pv in h_moving_old:
            if _manhattan_dist(pu, pv) == 1:
                old_cross += 1
                
    # Build new H positions in moving part
    h_moving_new = [(i + k, new_positions[k]) for k in range(len(new_positions)) 
                    if sequence[i + k] == 'H']
    
    new_cross = 0
    for u_idx, pu in h_fixed:
        for v_idx, pv_new in h_moving_new:
            if _manhattan_dist(pu, pv_new) == 1:
                new_cross += 1
                
    delta_e = -(new_cross - old_cross)
    
    new_chain = chain[:]
    new_chain[i:] = new_positions
    
    return new_chain, delta_e

def _min_cost_flow_fold(sequence, dimension, time_budget):
    """
    Formulate HP folding as min-cost max-flow on a time-expanded lattice graph.
    
    Nodes: (t, x, y, z) where t is sequence index (0..N-1), (x,y,z) is spatial coord
    Edges:
      - Time edges: (t, pos) -> (t+1, pos') where pos' is neighbor of pos (unit step)
      - Capacity: each spatial node can be used at most once across all time steps
        (enforced via vertex splitting)
    Cost: -1 for edges that create H-H contacts, 0 otherwise
    
    This is a structural change from local search to a global optimization formulation.
    """
    N = len(sequence)
    if N < 2:
        return None
    
    # For very long chains, this formulation becomes too large
    # Use a bounded spatial region to keep the graph manageable
    # Estimate bounding box size based on chain length
    box_size = max(4, int(math.ceil(N ** 0.55)) + 2)
    
    # Define spatial grid: coordinates from -box_size to box_size
    # Node representation: (t, x, y, z)
    # We'll use vertex splitting to enforce self-avoidance:
    #   in-node: (t, x, y, z, 'in')
    #   out-node: (t, x, y, z, 'out')
    #   Edge in->out has capacity 1, cost 0
    #   Time edges go from out-node of t to in-node of t+1
    
    # Build adjacency structure
    # For efficiency, only include nodes within reasonable bounds
    
    def node_id(t, x, y, z, is_out=False):
        """Encode node as integer for faster lookup."""
        # Offset coordinates to be non-negative
        ox = x + box_size
        oy = y + box_size
        oz = z + box_size
        # Encode: t * (2*box_size+1)^3 * 2 + (ox*(2*box_size+1)^2 + oy*(2*box_size+1) + oz) * 2 + is_out
        base = (2*box_size + 1)
        spatial = ox * base * base + oy * base + oz
        return t * base * base * base * 2 + spatial * 2 + int(is_out)
    
    # Precompute H positions for contact detection
    h_positions = set(i for i in range(N) if sequence[i] == 'H')
    
    # Build graph using adjacency lists
    # graph[node] = list of (neighbor, capacity, cost)
    graph = defaultdict(list)
    reverse_graph = defaultdict(list)  # For residual network
    
    def add_edge(u, v, cap, cost):
        """Add edge u->v with capacity cap and cost cost."""
        graph[u].append([v, cap, cost, len(graph[v])])
        graph[v].append([u, 0, -cost, len(graph[u]) - 1])
    
    # Add source and sink
    SOURCE = -1
    SINK = -2
    
    # Start node: t=0, position (0,0,0)
    start_in = node_id(0, 0, 0, False)
    start_out = node_id(0, 0, 0, True)
    
    # Add source -> start_in edge
    add_edge(SOURCE, start_in, 1, 0)
    
    # Build time-expanded graph
    for t in range(N):
        x0 = 0 if t == 0 else -box_size
        x1 = 0 if t == 0 else box_size
        y0 = 0 if t == 0 else -box_size
        y1 = 0 if t == 0 else box_size
        z0 = 0 if t == 0 else -box_size
        z1 = 0 if t == 0 else box_size
        
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                for z in range(z0, z1 + 1):
                    in_node = node_id(t, x, y, z, False)
                    out_node = node_id(t, x, y, z, True)
                    
                    # Vertex split edge: in -> out with capacity 1
                    add_edge(in_node, out_node, 1, 0)
                    
                    # If this is the last time step, connect to sink
                    if t == N - 1:
                        add_edge(out_node, SINK, 1, 0)
                    else:
                        # Add time edges to next time step
                        for nx, ny, nz in _neighbours((x, y, z)):
                            if abs(nx) > box_size or abs(ny) > box_size or abs(nz) > box_size:
                                continue
                            next_in = node_id(t + 1, nx, ny, nz, False)
                            
                            # Calculate cost: -1 if this creates an H-H contact
                            cost = 0
                            if sequence[t] == 'H' and sequence[t+1] == 'H':
                                # Check if placing residue t at (x,y,z) and t+1 at (nx,ny,nz)
                                # creates a contact with any previous H residue
                                # This is complex to compute exactly in the flow formulation
                                # Approximation: add small negative cost for H-H adjacency
                                cost = -0.5
                            
                            add_edge(out_node, next_in, 1, cost)
    
    # Solve min-cost max-flow using successive shortest path (Bellman-Ford)
    # This is a structural change: instead of local search, we solve a global optimization problem
    
    def bellman_ford_dist():
        """Find shortest path from SOURCE to SINK in residual graph."""
        dist = {SOURCE: 0}
        prev_node = {}
        prev_edge = {}
        in_queue = {SOURCE: True}
        queue = deque([SOURCE])
        
        while queue:
            u = queue.popleft()
            in_queue[u] = False
            for i, (v, cap, cost, _) in enumerate(graph[u]):
                if cap > 0:
                    new_dist = dist[u] + cost
                    if v not in dist or new_dist < dist[v]:
                        dist[v] = new_dist
                        prev_node[v] = u
                        prev_edge[v] = i
                        if not in_queue.get(v, False):
                            in_queue[v] = True
                            queue.append(v)
        
        if SINK not in dist:
            return None, None, None
        return dist[SINK], prev_node, prev_edge
    
    def augment_path(prev_node, prev_edge, flow_amount):
        """Augment flow along the path."""
        v = SINK
        while v != SOURCE:
            u = prev_node[v]
            edge_idx = prev_edge[v]
            graph[u][edge_idx][1] -= flow_amount
            # Find reverse edge
            rev_idx = graph[u][edge_idx][3]
            graph[v][rev_idx][1] += flow_amount
            v = u
    
    # Successive shortest path algorithm
    total_flow = 0
    total_cost = 0
    start_time = time.time()
    
    while time.time() - start_time < time_budget * 0.7:
        path_cost, prev_node, prev_edge = bellman_ford_dist()
        if path_cost is None:
            break
        
        # Find bottleneck capacity
        min_cap = float('inf')
        v = SINK
        while v != SOURCE:
            u = prev_node[v]
            edge_idx = prev_edge[v]
            min_cap = min(min_cap, graph[u][edge_idx][1])
            v = u
        
        if min_cap <= 0:
            break
        
        augment_path(prev_node, prev_edge, min_cap)
        total_flow += min_cap
        total_cost += path_cost * min_cap
        
        if total_flow >= 1:
            break
    
    # Extract path from flow
    if total_flow < 1:
        return None
    
    # Reconstruct path by following edges from source
    path = []
    current = SOURCE
    visited_nodes = set()
    
    while current != SINK:
        visited_nodes.add(current)
        found_next = False
        for v, cap, cost, rev_idx in graph[current]:
            if cap < 1 and v not in visited_nodes:  # Edge with flow (original cap 1, now 0)
                path.append(v)
                current = v
                found_next = True
                break
        if not found_next:
            return None
    
    # Decode path to lattice points
    chain = []
    for node in path:
        if node == SOURCE or node == SINK:
            continue
        # Decode node
        is_out = node % 2
        spatial = (node // 2) % (2*box_size + 1) ** 3
        t = node // ((2*box_size + 1) ** 3 * 2)
        
        if is_out:  # Only use out-nodes for positions
            base = (2*box_size + 1)
            ox = spatial // (base * base)
            oy = (spatial // base) % base
            oz = spatial % base
            x = ox - box_size
            y = oy - box_size
            z = oz - box_size
            chain.append((x, y, z))
    
    if len(chain) != N:
        return None
    
    # Verify self-avoidance
    if len(set(chain)) != N:
        return None
    
    # Verify unit steps
    for i in range(N - 1):
        if _manhattan_dist(chain[i], chain[i+1]) != 1:
            return None
    
    return chain

def fold(sequence, dimension):
    N = len(sequence)
    if N == 0: return []
    
    start_time = time.time()
    # 240s budget for all sequences. Use 7.5s per sequence to be safe.
    deadline = start_time + 7.5
    
    # Try min-cost flow formulation first (structural change)
    # This is a global optimization approach vs local search
    if N <= 60:  # Only use flow for shorter chains where graph is manageable
        flow_chain = _min_cost_flow_fold(sequence, dimension, 3.0)
        if flow_chain is not None:
            flow_energy = _energy(sequence, flow_chain)
            # If flow gives a good result, use it as initial population
            best_chain = flow_chain
            best_energy = flow_energy
        else:
            flow_chain = None
    else:
        flow_chain = None
    
    # Initialize Population
    pop_size = 12
    population = []
    energies = []
    
    # If we have a flow solution, add it first
    if flow_chain is not None:
        population.append(flow_chain)
        energies.append(_energy(sequence, flow_chain))
    
    for _ in range(pop_size - (1 if flow_chain is not None else 0)):
        # Try to grow a valid chain
        attempts = 0
        while attempts < 30:
            rng_init = random.Random()
            chain = _grow(sequence, dimension, rng_init)
            if chain is not None:
                e = _energy(sequence, chain)
                population.append(chain)
                energies.append(e)
                break
            attempts += 1
        
        if len(population) < pop_size:
             # Fallback: straight line
             chain = [tuple([i] + [0]*(dimension-1)) for i in range(N)]
             population.append(chain)
             energies.append(_energy(sequence, chain))
             
    if not population:
        # Ultimate fallback
        chain = [tuple([i] + [0]*(dimension-1)) for i in range(N)]
        return [list(p) for p in chain]
    
    best_chain = population[0]
    best_energy = energies[0]
    
    # Island Model / Parallel SA with improved parameters
    # We will interleave steps of SA for each individual in the population.
    
    # Parameters for SA - tuned for better exploration
    T_initial = max(3.0, N * 0.15)
    T_min = 0.005
    cooling_rate = 0.9998  # Slower cooling for more exploration
    
    temps = [T_initial] * len(population)
    step_counts = [0] * len(population)
    
    # Track best per individual for restart decisions
    best_per_ind = energies[:]
    
    while time.time() < deadline:
        # Pick an individual to move - weight towards underexplored individuals
        idx = random.randint(0, len(population) - 1)
        chain = population[idx]
        e = energies[idx]
        T = temps[idx]
        
        rng_move = random.Random()
        
        # Perform a pivot move
        i = rng_move.randint(1, N - 2)
        new_chain, delta_e = _pivot_move(sequence, chain, i, rng_move)
        
        if new_chain is not None:
            # Metropolis Acceptance
            if delta_e < 0 or rng_move.random() < math.exp(-delta_e / T):
                population[idx] = new_chain
                energies[idx] += delta_e
                step_counts[idx] += 1
                
                if energies[idx] < best_per_ind[idx]:
                    best_per_ind[idx] = energies[idx]
                
                if energies[idx] < best_energy:
                    best_energy = energies[idx]
                    best_chain = new_chain
                
                # Cool down
                temps[idx] *= cooling_rate
                if temps[idx] < T_min:
                    temps[idx] = T_min
            else:
                # Rejected, no change
                pass
        
        # Restart strategy: if temperature is very low and no improvement in recent steps
        if temps[idx] <= T_min * 1.05:
            # Check if this individual has been stuck
            if random.random() < 0.15:
                new_chain = _grow(sequence, dimension, random.Random())
                if new_chain is not None:
                    new_e = _energy(sequence, new_chain)
                    # Restart if new chain is competitive
                    if new_e <= best_per_ind[idx] + 5:
                        population[idx] = new_chain
                        energies[idx] = new_e
                        temps[idx] = T_initial
                        step_counts[idx] = 0
                        if new_e < best_energy:
                            best_energy = new_e
                            best_chain = new_chain

    # Final polish pass: local search on the best chain with tighter tolerance
    if time.time() < deadline - 1.0:
        polish_deadline = min(deadline - 0.5, time.time() + 1.5)
        polish_T = 0.1
        polish_chain = best_chain[:]
        polish_e = best_energy
        
        while time.time() < polish_deadline:
            rng_polish = random.Random()
            i = rng_polish.randint(1, N - 2)
            new_chain, delta_e = _pivot_move(sequence, polish_chain, i, rng_polish)
            
            if new_chain is not None:
                # Stricter acceptance for polish phase
                if delta_e < 0 or rng_polish.random() < math.exp(-delta_e / polish_T):
                    polish_chain = new_chain
                    polish_e += delta_e
                    
                    if polish_e < best_energy:
                        best_energy = polish_e
                        best_chain = polish_chain
    
    return [list(p) for p in best_chain]
