"""Hydrophobic-core decomposition: pack H-blocks as rigid bodies, then thread polar residues."""
# EVOLVE-BLOCK-START
import random
import time
import math

_DEADLINE = 315.0  # safety margin before 330s wall-clock kill
_START_TIME = None

def _get_deadline():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _START_TIME + _DEADLINE

def _time_left():
    return _get_deadline() - time.time()

def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)

def _lattice_adjacent(p1, p2):
    return sum(abs(a - b) for a, b in zip(p1, p2)) == 1

def _identify_h_blocks(sequence):
    """
    Identify contiguous H-blocks in the sequence.
    Returns list of (start_idx, end_idx, length) for each H-block.
    Also returns the positions of P residues between blocks.
    """
    blocks = []
    i = 0
    n = len(sequence)
    while i < n:
        if sequence[i] == 'H':
            start = i
            while i < n and sequence[i] == 'H':
                i += 1
            end = i - 1
            blocks.append((start, end, end - start + 1))
        else:
            i += 1
    return blocks

def _core_contacts(block1, block2, pos1, pos2):
    """
    Count H-H contacts between two rigid H-blocks placed at positions pos1, pos2.
    Each block is a straight line of length L.
    block1: positions pos1[0], pos1[0]+dir1, ..., pos1[0]+(L1-1)*dir1
    block2: positions pos2[0], pos2[0]+dir2, ..., pos2[0]+(L2-1)*dir2
    """
    L1, dir1 = block1
    L2, dir2 = block2
    contacts = 0
    for i in range(L1):
        p1 = tuple(pos1[0][k] + i * dir1[k] for k in range(3))
        for j in range(L2):
            p2 = tuple(pos2[0][k] + j * dir2[k] for k in range(3))
            if _lattice_adjacent(p1, p2):
                contacts += 1
    return contacts

def _pack_h_blocks(sequence, dimension, rng):
    """
    Pack H-blocks as rigid bodies to maximize intra-core contacts.
    Returns a mapping from residue index to position for all H residues,
    or None if packing fails.
    """
    blocks = _identify_h_blocks(sequence)
    if not blocks:
        return None
    
    # For each block, we need to choose a position and direction
    # Start by placing the first block at origin
    n = len(sequence)
    occupied = set()
    block_positions = {}  # block_idx -> (start_pos, direction)
    
    # Place first block
    L0 = blocks[0][2]
    start0 = (0, 0, 0)
    dir0 = (1, 0, 0)
    for i in range(L0):
        pos = tuple(start0[k] + i * dir0[k] for k in range(3))
        occupied.add(pos)
    block_positions[0] = (start0, dir0)
    
    # Directions for each block (try all 6 directions)
    all_dirs = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    
    # For each subsequent block, try to place it adjacent to existing blocks
    # to maximize contacts
    for b_idx in range(1, len(blocks)):
        start, end, L = blocks[b_idx]
        placed = False
        
        # Try to place this block adjacent to each existing block
        # to maximize contacts
        best_score = -1
        best_placement = None
        
        # Candidate positions: adjacent to any occupied site
        candidate_starts = set()
        for pos in list(occupied):
            for q in _neighbours(pos):
                if q not in occupied:
                    candidate_starts.add(q)
        
        # Limit candidates for efficiency
        if len(candidate_starts) > 200:
            candidate_starts = list(candidate_starts)
            rng.shuffle(candidate_starts)
            candidate_starts = set(candidate_starts[:200])
        
        for cand_start in candidate_starts:
            for dir_b in all_dirs:
                # Check if the entire block fits without collision
                valid = True
                block_sites = []
                for i in range(L):
                    pos = tuple(cand_start[k] + i * dir_b[k] for k in range(3))
                    if pos in occupied:
                        valid = False
                        break
                    block_sites.append(pos)
                
                if not valid:
                    continue
                
                # Count contacts with existing blocks
                score = 0
                for eb_idx in range(b_idx):
                    eb_start, eb_dir = block_positions[eb_idx]
                    eb_L = blocks[eb_idx][2]
                    for i in range(L):
                        p_new = block_sites[i]
                        for j in range(eb_L):
                            p_old = tuple(eb_start[k] + j * eb_dir[k] for k in range(3))
                            if _lattice_adjacent(p_new, p_old):
                                score += 1
                
                if score > best_score:
                    best_score = score
                    best_placement = (cand_start, dir_b)
        
        if best_placement is None:
            # Try random placement
            for _ in range(50):
                cand_start = tuple(rng.randint(-20, 20) for _ in range(3))
                dir_b = all_dirs[rng.randint(0, 5)]
                valid = True
                block_sites = []
                for i in range(L):
                    pos = tuple(cand_start[k] + i * dir_b[k] for k in range(3))
                    if pos in occupied:
                        valid = False
                        break
                    block_sites.append(pos)
                if valid:
                    best_placement = (cand_start, dir_b)
                    break
        
        if best_placement is None:
            return None
        
        start_pos, dir_b = best_placement
        for i in range(L):
            pos = tuple(start_pos[k] + i * dir_b[k] for k in range(3))
            occupied.add(pos)
        block_positions[b_idx] = (start_pos, dir_b)
    
    # Build position map for all H residues
    pos_map = {}
    for b_idx, (start, end, L) in enumerate(blocks):
        start_pos, dir_b = block_positions[b_idx]
        for i in range(L):
            residue_idx = start + i
            pos = tuple(start_pos[k] + i * dir_b[k] for k in range(3))
            pos_map[residue_idx] = pos
    
    return pos_map, block_positions, blocks, occupied

def _thread_polar_residues(sequence, dimension, rng, h_pos_map, blocks):
    """
    Thread polar (P) residues between H-blocks to form a valid self-avoiding walk.
    Uses BFS/DFS to find paths between consecutive H-block endpoints through P residues.
    """
    n = len(sequence)
    is_h = [c == 'H' for c in sequence]
    
    # Get the order of H-blocks and their positions
    # We need to connect: start -> block0 -> P's -> block1 -> P's -> block2 -> ... -> end
    
    # Identify segments: alternating H-blocks and P-gaps
    segments = []  # list of ('H', start, end) or ('P', start, end)
    i = 0
    while i < n:
        if is_h[i]:
            start = i
            while i < n and is_h[i]:
                i += 1
            segments.append(('H', start, i - 1))
        else:
            start = i
            while i < n and not is_h[i]:
                i += 1
            segments.append(('P', start, i - 1))
    
    # Build the chain by connecting segments
    chain = [None] * n
    occupied = set()
    
    # Place all H residues first
    for idx, pos in h_pos_map.items():
        chain[idx] = pos
        occupied.add(pos)
    
    # Now fill in P residues by finding paths between consecutive H-blocks
    # Get the sequence of segment endpoints we need to connect
    # For each P-gap between two H-blocks, find a path
    
    # Get the ordered list of "anchor" points: start of each H-block and end of each H-block
    # We need to connect: 
    #   (if starts with P) start of P segment -> first H block start
    #   end of H block k -> start of P gap -> start of H block k+1
    #   end of last H block -> end of P segment (if ends with P)
    
    # Let's identify connection points
    connections = []  # list of (from_idx, to_idx) where we need a path of P residues
    
    # Find all P-gaps
    for seg_idx, seg in enumerate(segments):
        if seg[0] == 'P':
            p_start, p_end = seg[1], seg[2]
            # What comes before this P-gap?
            if seg_idx > 0 and segments[seg_idx-1][0] == 'H':
                from_idx = segments[seg_idx-1][2]  # end of previous H-block
            else:
                from_idx = None  # start of chain
            
            # What comes after this P-gap?
            if seg_idx < len(segments) - 1 and segments[seg_idx+1][0] == 'H':
                to_idx = segments[seg_idx+1][1]  # start of next H-block
            else:
                to_idx = None  # end of chain
            
            connections.append((p_start, p_end, from_idx, to_idx))
    
    # Fill in P residues
    for p_start, p_end, from_idx, to_idx in connections:
        gap_len = p_end - p_start + 1
        
        if from_idx is not None and to_idx is not None:
            # Need to connect chain[from_idx] to chain[to_idx] with gap_len P residues
            from_pos = chain[from_idx]
            to_pos = chain[to_idx]
            
            # BFS to find a path of length gap_len from from_pos to to_pos
            # that doesn't go through occupied sites
            path = _find_path(from_pos, to_pos, gap_len, occupied, dimension, rng)
            if path is None:
                # Try a simpler approach: just try random walks
                path = _random_walk_between(from_pos, to_pos, gap_len, occupied, dimension, rng)
            if path is None:
                return None
            
            for i, pos in enumerate(path):
                chain[p_start + i] = pos
                occupied.add(pos)
        
        elif from_idx is not None and to_idx is None:
            # Chain starts with H-block, then P's at the end
            from_pos = chain[from_idx]
            # Just grow randomly from from_pos
            path = _random_walk(from_pos, gap_len, occupied, dimension, rng)
            if path is None:
                return None
            for i, pos in enumerate(path):
                chain[p_start + i] = pos
                occupied.add(pos)
        
        elif from_idx is None and to_idx is not None:
            # Chain ends with H-block, P's at the beginning
            to_pos = chain[to_idx]
            # Grow backwards from to_pos
            path = _random_walk(to_pos, gap_len, occupied, dimension, rng)
            if path is None:
                return None
            # Reverse the path since we're going backwards
            path = list(reversed(path))
            for i, pos in enumerate(path):
                chain[p_start + i] = pos
                occupied.add(pos)
        
        else:
            # All P's? Just random walk from origin
            path = _random_walk((0,0,0), gap_len, occupied, dimension, rng)
            if path is None:
                return None
            for i, pos in enumerate(path):
                chain[p_start + i] = pos
                occupied.add(pos)
    
    # Verify the chain is valid
    for i in range(n - 1):
        if chain[i] is None or chain[i+1] is None:
            return None
        if not _lattice_adjacent(chain[i], chain[i+1]):
            return None
    
    # Check self-avoidance
    pos_set = set(chain)
    if len(pos_set) != n:
        return None
    
    return chain

def _find_path(start, end, length, occupied, dimension, rng):
    """
    Find a self-avoiding path of exactly 'length' steps from start to end,
    avoiding occupied sites. Uses BFS with depth limit.
    Returns list of 'length' positions (excluding start, including end) or None.
    """
    if length == 0:
        return [] if start == end else None
    
    # For short paths, use direct BFS
    if length <= 10:
        # BFS from start, track path
        from collections import deque
        # State: (position, path)
        queue = deque()
        queue.append((start, []))
        visited = {start}
        
        while queue:
            pos, path = queue.popleft()
            if len(path) == length:
                if pos == end:
                    return path
                continue
            
            for nb in _neighbours(pos):
                if nb in visited or nb in occupied:
                    continue
                if nb == end and len(path) + 1 != length:
                    continue  # Can only reach end at exactly length
                if nb == end and len(path) + 1 == length:
                    return path + [nb]
                visited.add(nb)
                queue.append((nb, path + [nb]))
        
        return None
    
    # For longer paths, use a heuristic random walk
    return _random_walk_between(start, end, length, occupied, dimension, rng)

def _random_walk_between(start, end, length, occupied, dimension, rng):
    """
    Find a path of exactly 'length' steps from start to end using random walk with guidance.
    Returns list of 'length' positions or None.
    """
    # Compute Manhattan distance
    dist = sum(abs(a - b) for a, b in zip(start, end))
    
    if dist > length:
        return None
    
    # We need to add (length - dist) extra steps (detours)
    extra = length - dist
    
    # Try multiple random attempts
    for attempt in range(200):
        path = []
        current = start
        remaining = length
        
        for step in range(length):
            if remaining == 0:
                break
            
            # Determine which directions are useful
            dirs = []
            for axis in range(dimension):
                for s in (-1, 1):
                    d = [0] * dimension
                    d[axis] = s
                    d = tuple(d)
                    next_pos = tuple(current[k] + d[k] for k in range(dimension))
                    if next_pos in occupied or next_pos in path:
                        continue
                    dirs.append((d, next_pos))
            
            if not dirs:
                break
            
            # Bias towards end if we're close
            if remaining <= dist:
                # Must go towards end
                useful_dirs = []
                for d, next_pos in dirs:
                    new_dist = sum(abs(a - b) for a, b in zip(next_pos, end))
                    if new_dist < dist:
                        useful_dirs.append((d, next_pos))
                if useful_dirs:
                    d, next_pos = useful_dirs[rng.randint(0, len(useful_dirs)-1)]
                else:
                    d, next_pos = dirs[rng.randint(0, len(dirs)-1)]
            else:
                # Can go in any direction, but bias towards end
                weights = []
                for d, next_pos in dirs:
                    new_dist = sum(abs(a - b) for a, b in zip(next_pos, end))
                    # Lower distance = higher weight
                    weights.append(1.0 / (new_dist + 1))
                total_w = sum(weights)
                r = rng.random() * total_w
                cum = 0
                for (d, next_pos), w in zip(dirs, weights):
                    cum += w
                    if r <= cum:
                        break
                else:
                    d, next_pos = dirs[-1]
            
            path.append(next_pos)
            current = next_pos
            dist = sum(abs(a - b) for a, b in zip(current, end))
            remaining -= 1
        
        if current == end and len(path) == length:
            return path
    
    return None

def _random_walk(start, length, occupied, dimension, rng):
    """
    Generate a random self-avoiding walk of 'length' steps starting from start.
    Returns list of 'length' positions or None.
    """
    path = []
    current = start
    for _ in range(length):
        dirs = []
        for axis in range(dimension):
            for s in (-1, 1):
                d = [0] * dimension
                d[axis] = s
                d = tuple(d)
                next_pos = tuple(current[k] + d[k] for k in range(dimension))
                if next_pos in occupied or next_pos in path:
                    continue
                dirs.append(next_pos)
        
        if not dirs:
            return None
        
        next_pos = dirs[rng.randint(0, len(dirs)-1)]
        path.append(next_pos)
        current = next_pos
    
    return path

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

def _compute_contacts(sequence, chain):
    """Return set of (i,j) H-H contact pairs (i<j, not chain-adjacent)."""
    n = len(chain)
    where = {p: i for i, p in enumerate(chain)}
    contacts = set()
    for i, p in enumerate(chain):
        if sequence[i] != "H":
            continue
        for q in _neighbours(p):
            j = where.get(q)
            if j is not None and j > i + 1 and sequence[j] == "H":
                contacts.add((i, j))
    return contacts

def _rotation_matrices():
    """Return list of 24 rotation matrices for the cubic group (as 3x3 tuples)."""
    mats = []
    e = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    for x in e:
        for y in e:
            cx = x[1]*y[2] - x[2]*y[1]
            cy = x[2]*y[0] - x[0]*y[2]
            cz = x[0]*y[1] - x[1]*y[0]
            z = (cx, cy, cz)
            if sum(a*b for a,b in zip(x,y)) != 0:
                continue
            if sum(a*b for a,b in zip(x,z)) != 0:
                continue
            if sum(a*b for a,b in zip(y,z)) != 0:
                continue
            m = (x, y, z)
            if m not in mats:
                mats.append(m)
    return mats

_ROT_MATS = _rotation_matrices()

def _apply_rotation(mat, vec):
    r0, r1, r2 = mat
    return (
        r0[0]*vec[0] + r0[1]*vec[1] + r0[2]*vec[2],
        r1[0]*vec[0] + r1[1]*vec[1] + r1[2]*vec[2],
        r2[0]*vec[0] + r2[1]*vec[1] + r2[2]*vec[2],
    )

class _FoldingState:
    """State for incremental local search with optimized contact tracking."""
    def __init__(self, sequence, chain):
        self.seq = sequence
        self.n = len(chain)
        self.chain = list(chain)
        self.occupied = {p: i for i, p in enumerate(chain)}
        self.is_h = [c == 'H' for c in sequence]
        self.contacts = _compute_contacts(sequence, chain)
        
        # contact_map[i] = set of j such that (min(i,j), max(i,j)) is in contacts
        self.contact_map = {i: set() for i in range(self.n)}
        for i, j in self.contacts:
            self.contact_map[i].add(j)
            self.contact_map[j].add(i)
            
        self.energy = -len(self.contacts)
        self.best_chain = list(chain)
        self.best_energy = self.energy

    def _accept_if_better(self):
        if self.energy < self.best_energy:
            self.best_energy = self.energy
            self.best_chain = list(self.chain)

    def _remove_contacts_involving(self, indices):
        """Remove all contacts involving any index in the set. Efficient O(k) where k is degree."""
        indices_set = set(indices)
        removed_count = 0
        for i in indices_set:
            # Copy the set because we are modifying it
            partners = list(self.contact_map[i])
            for j in partners:
                if j in indices_set:
                    # Both i and j are in the moved set.
                    # We must ensure we don't remove the same pair twice.
                    # If i < j, we process it here.
                    if i < j:
                        pair = (i, j)
                        if pair in self.contacts:
                            self.contacts.discard(pair)
                            self.contact_map[i].discard(j)
                            self.contact_map[j].discard(i)
                            removed_count += 1
                else:
                    # j is not in the moved set, so the contact is definitely removed
                    # because i has moved away.
                    pair = (i, j) if i < j else (j, i)
                    if pair in self.contacts:
                        self.contacts.discard(pair)
                        self.contact_map[i].discard(j)
                        self.contact_map[j].discard(i)
                        removed_count += 1
        self.energy += removed_count

    def _add_new_contacts(self, indices, new_positions):
        """Add new contacts for moved indices at their new positions."""
        indices_set = set(indices)
        added_count = 0
        for i in indices_set:
            if not self.is_h[i]:
                continue
            pos = new_positions[i]
            for q in _neighbours(pos):
                j = self.occupied.get(q)
                if j is None or j == i:
                    continue
                if abs(j - i) == 1:
                    continue
                if self.is_h[j]:
                    pair = (i, j) if i < j else (j, i)
                    if pair not in self.contacts:
                        self.contacts.add(pair)
                        self.contact_map[i].add(j)
                        self.contact_map[j].add(i)
                        added_count += 1
        self.energy -= added_count

    def _update_energy_for_move(self, moved_indices, old_positions, new_positions):
        """
        Incrementally update energy after moving residues.
        """
        # 1. Remove all contacts involving moved residues
        self._remove_contacts_involving(moved_indices)
        
        # 2. Add new contacts involving moved residues
        self._add_new_contacts(moved_indices, new_positions)

    def _pivot_move_with_undo(self, rng):
        """True pivot with undo info returned."""
        n = self.n
        if n < 4:
            return None
        pivot_idx = rng.randint(1, n - 3)
        orig_dir = tuple(self.chain[pivot_idx + 1][k] - self.chain[pivot_idx][k] for k in range(3))

        valid_mats = []
        for mat in _ROT_MATS:
            new_dir = _apply_rotation(mat, orig_dir)
            if new_dir != orig_dir and new_dir != tuple(-x for x in orig_dir):
                valid_mats.append((mat, new_dir))

        if not valid_mats:
            return None

        mat, new_dir = valid_mats[rng.randint(0, len(valid_mats) - 1)]
        subchain_len = n - pivot_idx - 1
        subchain_old = list(self.chain[pivot_idx + 1:])

        new_subchain = [None] * subchain_len
        new_subchain[0] = tuple(self.chain[pivot_idx][k] + new_dir[k] for k in range(3))

        if new_subchain[0] in self.occupied:
            return None

        valid = True
        for k in range(1, subchain_len):
            old_step = tuple(subchain_old[k][m] - subchain_old[k-1][m] for m in range(3))
            new_step = _apply_rotation(mat, old_step)
            new_pos = tuple(new_subchain[k-1][m] + new_step[m] for m in range(3))
            if new_pos in self.occupied:
                valid = False
                break
            new_subchain[k] = new_pos

        if not valid:
            return None

        seen = set()
        for pos in new_subchain:
            if pos in seen:
                return None
            seen.add(pos)

        if pivot_idx + subchain_len < n:
            if not _lattice_adjacent(new_subchain[-1], self.chain[pivot_idx + subchain_len]):
                return None

        moved_indices = list(range(pivot_idx + 1, n))
        old_positions = {i: self.chain[i] for i in moved_indices}

        undo_info = {
            'type': 'pivot',
            'pivot_idx': pivot_idx,
            'old_positions': old_positions,
            'new_positions': {i: new_subchain[i - pivot_idx - 1] for i in moved_indices},
            'moved_indices': moved_indices
        }

        for pos in subchain_old:
            del self.occupied[pos]
        for k, pos in enumerate(new_subchain):
            self.chain[pivot_idx + 1 + k] = pos
            self.occupied[pos] = pivot_idx + 1 + k

        new_positions = {i: self.chain[i] for i in moved_indices}
        self._update_energy_for_move(moved_indices, old_positions, new_positions)
        return undo_info

    def _undo_move(self, move_info):
        """Undo a move using the saved info."""
        if move_info is None:
            return
        moved_indices = move_info['moved_indices']
        old_positions = move_info['old_positions']
        new_positions = move_info['new_positions']

        # Remove contacts involving moved residues at new positions
        self._remove_contacts_involving(moved_indices)

        # Restore old positions
        for i in moved_indices:
            del self.occupied[self.chain[i]]
            self.chain[i] = old_positions[i]
            self.occupied[old_positions[i]] = i

        # Add contacts at old positions
        self._add_new_contacts(moved_indices, old_positions)

    def _pivot_move_translate_with_undo(self, rng):
        """Translation-based pivot with undo info."""
        n = self.n
        if n < 4:
            return None
        pivot_idx = rng.randint(1, n - 3)
        orig_dir = tuple(self.chain[pivot_idx + 1][k] - self.chain[pivot_idx][k] for k in range(3))

        possible_dirs = []
        for ax in range(3):
            for s in (-1, 1):
                d = [0, 0, 0]
                d[ax] = s
                d = tuple(d)
                if d != orig_dir and d != tuple(-x for x in orig_dir):
                    possible_dirs.append(d)

        if not possible_dirs:
            return None

        new_dir = possible_dirs[rng.randint(0, len(possible_dirs) - 1)]
        subchain_len = n - pivot_idx - 1
        subchain_old = list(self.chain[pivot_idx + 1:])

        new_subchain = [None] * subchain_len
        new_subchain[0] = tuple(self.chain[pivot_idx][k] + new_dir[k] for k in range(3))

        if new_subchain[0] in self.occupied:
            return None

        valid = True
        for k in range(1, subchain_len):
            old_step = tuple(subchain_old[k][m] - subchain_old[k-1][m] for m in range(3))
            new_pos = tuple(new_subchain[k-1][m] + old_step[m] for m in range(3))
            if new_pos in self.occupied:
                valid = False
                break
            new_subchain[k] = new_pos

        if not valid:
            return None

        seen = set()
        for pos in new_subchain:
            if pos in seen:
                return None
            seen.add(pos)

        if pivot_idx + subchain_len < n:
            if not _lattice_adjacent(new_subchain[-1], self.chain[pivot_idx + subchain_len]):
                return None

        moved_indices = list(range(pivot_idx + 1, n))
        old_positions = {i: self.chain[i] for i in moved_indices}

        undo_info = {
            'type': 'pivot_translate',
            'pivot_idx': pivot_idx,
            'old_positions': old_positions,
            'new_positions': {i: new_subchain[i - pivot_idx - 1] for i in moved_indices},
            'moved_indices': moved_indices
        }

        for pos in subchain_old:
            del self.occupied[pos]
        for k, pos in enumerate(new_subchain):
            self.chain[pivot_idx + 1 + k] = pos
            self.occupied[pos] = pivot_idx + 1 + k

        new_positions = {i: self.chain[i] for i in moved_indices}
        self._update_energy_for_move(moved_indices, old_positions, new_positions)
        return undo_info

    def _crank_move_with_undo(self, rng):
        """Crank with undo info."""
        n = self.n
        if n < 5:
            return None
        start_idx = rng.randint(1, n - 3)
        end_idx = rng.randint(start_idx + 1, n - 1)
        subchain = list(self.chain[start_idx:end_idx + 1])

        ax = rng.randint(0, 2)
        s = rng.choice((-1, 1))
        new_subchain = []
        for p in subchain:
            np = list(p)
            np[ax] += s
            new_subchain.append(tuple(np))

        seen = set()
        for np in new_subchain:
            if np in seen:
                return None
            seen.add(np)

        for p in subchain:
            del self.occupied[p]

        for np in new_subchain:
            if np in self.occupied:
                for i, p in enumerate(subchain):
                    self.occupied[p] = start_idx + i
                return None

        if start_idx > 0:
            if not _lattice_adjacent(self.chain[start_idx - 1], new_subchain[0]):
                for i, p in enumerate(subchain):
                    self.occupied[p] = start_idx + i
                return None

        if end_idx < n - 1:
            if not _lattice_adjacent(new_subchain[-1], self.chain[end_idx + 1]):
                for i, p in enumerate(subchain):
                    self.occupied[p] = start_idx + i
                return None

        moved_indices = list(range(start_idx, end_idx + 1))
        old_positions = {i: self.chain[i] for i in moved_indices}

        undo_info = {
            'type': 'crank',
            'start_idx': start_idx,
            'end_idx': end_idx,
            'old_positions': old_positions,
            'new_positions': {i: new_subchain[i - start_idx] for i in moved_indices},
            'moved_indices': moved_indices
        }

        for i, np in enumerate(new_subchain):
            self.chain[start_idx + i] = np
            self.occupied[np] = start_idx + i

        new_positions = {i: self.chain[i] for i in moved_indices}
        self._update_energy_for_move(moved_indices, old_positions, new_positions)
        return undo_info

def _local_search(sequence, chain, dimension, rng, max_iters=500):
    """Local search with incremental energy updates."""
    if chain is None:
        return None
    n = len(chain)
    if n < 4:
        return chain

    state = _FoldingState(sequence, chain)

    for it in range(max_iters):
        if _time_left() < 1.0:
            break
        r = rng.random()
        if r < 0.35:
            state._pivot_move_with_undo(rng)
        elif r < 0.70:
            state._pivot_move_translate_with_undo(rng)
        else:
            state._crank_move_with_undo(rng)
        
        # Accept if better
        state._accept_if_better()

    return state.best_chain

def _simulated_annealing(sequence, chain, dimension, rng, max_iters=1000):
    """Simulated annealing with efficient incremental undo via move tracking."""
    if chain is None:
        return None
    n = len(chain)
    if n < 4:
        return chain

    state = _FoldingState(sequence, chain)
    T_start = 5.0
    T_end = 0.05
    cooling_rate = (T_end / T_start) ** (1.0 / max_iters)
    T = T_start

    for it in range(max_iters):
        if _time_left() < 1.0:
            break
        T *= cooling_rate

        old_energy = state.energy

        r = rng.random()
        if r < 0.35:
            move_info = state._pivot_move_with_undo(rng)
        elif r < 0.70:
            move_info = state._pivot_move_translate_with_undo(rng)
        else:
            move_info = state._crank_move_with_undo(rng)

        if move_info is None:
            continue

        delta = state.energy - old_energy
        if delta < 0 or (T > 0.01 and rng.random() < math.exp(-delta / T)):
            state._accept_if_better()
        else:
            state._undo_move(move_info)

    return state.best_chain

def _fallback_grow(sequence, dimension, rng):
    """Fallback random greedy growth if core decomposition fails."""
    start = (0,) * dimension
    chain, occupied = [start], {start: 0}
    is_h = [c == 'H' for c in sequence]
    
    for i in range(1, len(sequence)):
        options = []
        for site in _neighbours(chain[-1]):
            if site in occupied:
                continue
            gain = 0
            if is_h[i]:
                for q in _neighbours(site):
                    if q in occupied:
                        idx = occupied[q]
                        if idx < i - 1 and is_h[idx]:
                            gain += 1
            free = sum(1 for q in _neighbours(site) if q not in occupied)
            score = gain * 1.0 + free * 0.1 + rng.random() * 0.5
            options.append((score, site))
        
        if not options:
            return None
        
        site = max(options, key=lambda x: x[0])[1]
        occupied[site] = i
        chain.append(site)
    return chain

def fold(sequence, dimension):
    """
    Fold using hydrophobic-core decomposition:
    1. Identify H-blocks and pack them as rigid bodies
    2. Thread polar residues to connect blocks
    3. Refine with local search
    Falls back to greedy growth + SA if decomposition fails.
    """
    n = len(sequence)
    rng = random.Random(12345 + n)
    best, best_energy = None, 1

    # Strategy 1: Core decomposition
    time_budget = _time_left()
    core_restarts = min(50, max(10, int(time_budget * 0.3)))
    
    for r in range(core_restarts):
        if _time_left() < 5.0:
            break
        
        result = _pack_h_blocks(sequence, dimension, rng)
        if result is None:
            continue
        
        h_pos_map, block_positions, blocks, occupied = result
        chain = _thread_polar_residues(sequence, dimension, rng, h_pos_map, blocks)
        
        if chain is not None:
            # Refine with local search
            if _time_left() > 3.0:
                max_ls = 300 if n <= 50 else (200 if n <= 80 else 100)
                chain = _local_search(sequence, chain, dimension, rng, max_iters=max_ls)
            
            if chain is not None:
                value = _energy(sequence, chain)
                if value < best_energy:
                    best, best_energy = chain, value
    
    # Strategy 2: Fallback greedy + SA if core decomposition didn't find good results
    if best is None or best_energy > -1:
        fallback_restarts = min(100, max(20, int(_time_left() * 0.5)))
        for r in range(fallback_restarts):
            if _time_left() < 3.0:
                break
            
            chain = _fallback_grow(sequence, dimension, rng)
            if chain is None:
                continue
            
            if _time_left() > 2.0:
                if r % 2 == 0:
                    max_sa = 400 if n <= 50 else (250 if n <= 80 else 120)
                    chain = _simulated_annealing(sequence, chain, dimension, rng, max_iters=max_sa)
                else:
                    max_ls = 300 if n <= 50 else (200 if n <= 80 else 100)
                    chain = _local_search(sequence, chain, dimension, rng, max_iters=max_ls)
            
            if chain is not None:
                value = _energy(sequence, chain)
                if value < best_energy:
                    best, best_energy = chain, value

    if best is None:
        best = [tuple([i] + [0] * (dimension - 1)) for i in range(n)]

    return [list(p) for p in best]
# EVOLVE-BLOCK-END
