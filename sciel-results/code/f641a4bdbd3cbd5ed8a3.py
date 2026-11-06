"""HP lattice protein folding using helix initialization + simulated annealing."""
# EVOLVE-BLOCK-START
import random
import math


def _neighbours(point):
    for axis in range(len(point)):
        for step in (-1, 1):
            q = list(point)
            q[axis] += step
            yield tuple(q)


def _helix_init(n, dim):
    """Generate a tight square spiral conformation of n residues."""
    if n <= 0:
        return []
    if n == 1:
        return [(0, 0)]
    
    chain = [(0, 0)]
    dx, dy = 1, 0
    step_len = 1
    segment_count = 0
    
    while len(chain) < n:
        for _ in range(step_len):
            if len(chain) >= n:
                break
            x, y = chain[-1]
            chain.append((x + dx, y + dy))
        if len(chain) >= n:
            break
        dx, dy = -dy, dx
        segment_count += 1
        if segment_count % 2 == 0:
            step_len += 1
    
    return chain[:n]


def _count_contacts(seq, chain):
    """Count H-H contacts (non-consecutive adjacent H pairs)."""
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if seq[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and j > i + 1 and seq[j] == 'H':
                    contacts += 1
    return contacts


def _sa_fold(seq, dim, rng, n_iter=40000, t_start=2.0, t_end=0.01):
    """Simulated annealing to maximize H-H contacts."""
    n = len(seq)
    if n <= 1:
        return _helix_init(n, dim), 0
    
    # Initialize with helix, verify self-avoidance
    chain = _helix_init(n, dim)
    if len(set(chain)) != n:
        chain = [(i, 0) for i in range(n)]
    
    where = {p: i for i, p in enumerate(chain)}
    current_contacts = _count_contacts(seq, chain)
    best_chain = list(chain)
    best_contacts = current_contacts
    
    t = t_start
    cooling = (t_end / t_start) ** (1.0 / max(n_iter, 1))
    
    for it in range(n_iter):
        t *= cooling
        i = rng.randint(1, n - 2)
        
        move_type = rng.random()
        
        if move_type < 0.7:
            # Move residue i to another neighbor of chain[i-1]
            prev = chain[i - 1]
            next_pos = chain[i + 1]
            old_pos = chain[i]
            
            candidates = []
            for q in _neighbours(prev):
                if q == next_pos or q == old_pos:
                    continue
                if q in where and where[q] != i:
                    continue
                candidates.append(q)
            
            if not candidates:
                continue
            
            new_pos = rng.choice(candidates)
            
            # Compute delta in contacts
            delta = 0
            if seq[i] == 'H':
                # Remove old contacts
                for q in _neighbours(old_pos):
                    j = where.get(q)
                    if j is not None and j != i and abs(j - i) > 1 and seq[j] == 'H':
                        delta -= 1
                # Add new contacts
                for q in _neighbours(new_pos):
                    j = where.get(q)
                    if j is not None and j != i and abs(j - i) > 1 and seq[j] == 'H':
                        delta += 1
            
            # Apply
            del where[old_pos]
            where[new_pos] = i
            chain[i] = new_pos
            current_contacts += delta
            
            if current_contacts > best_contacts:
                best_contacts = current_contacts
                best_chain = list(chain)
            
            if delta >= 0 or rng.random() < math.exp(delta / max(t, 1e-8)):
                pass
            else:
                # Revert
                del where[new_pos]
                where[old_pos] = i
                chain[i] = old_pos
                current_contacts -= delta
        
        else:
            # Move residue i+1 to another neighbor of chain[i]
            prev = chain[i]
            next_pos = chain[i + 2] if i + 2 < n else None
            old_pos = chain[i + 1]
            
            candidates = []
            for q in _neighbours(prev):
                if q == old_pos:
                    continue
                if next_pos is not None and q == next_pos:
                    continue
                if q in where and where[q] != i + 1:
                    continue
                candidates.append(q)
            
            if not candidates:
                continue
            
            new_pos = rng.choice(candidates)
            
            delta = 0
            if seq[i + 1] == 'H':
                for q in _neighbours(old_pos):
                    j = where.get(q)
                    if j is not None and j != i + 1 and abs(j - (i + 1)) > 1 and seq[j] == 'H':
                        delta -= 1
                for q in _neighbours(new_pos):
                    j = where.get(q)
                    if j is not None and j != i + 1 and abs(j - (i + 1)) > 1 and seq[j] == 'H':
                        delta += 1
            
            del where[old_pos]
            where[new_pos] = i + 1
            chain[i + 1] = new_pos
            current_contacts += delta
            
            if current_contacts > best_contacts:
                best_contacts = current_contacts
                best_chain = list(chain)
            
            if delta >= 0 or rng.random() < math.exp(delta / max(t, 1e-8)):
                pass
            else:
                del where[new_pos]
                where[old_pos] = i + 1
                chain[i + 1] = old_pos
                current_contacts -= delta
    
    return best_chain, best_contacts


def fold(sequence, dimension):
    """Fold a sequence using helix init + simulated annealing with multiple restarts."""
    n = len(sequence)
    if n <= 1:
        return [[0] * dimension]
    
    rng = random.Random(42 + len(sequence))
    
    best_chain = None
    best_contacts = -1
    
    n_runs = 4 if n <= 25 else (6 if n <= 50 else 8)
    n_iter = 25000 if n <= 25 else (40000 if n <= 50 else 55000)
    
    for run in range(n_runs):
        t_start = 1.5 + run * 0.4
        chain, contacts = _sa_fold(sequence, dimension, rng, n_iter=n_iter, t_start=t_start, t_end=0.01)
        
        if contacts > best_contacts:
            best_contacts = contacts
            best_chain = chain
    
    if best_chain is None:
        best_chain = [(i, 0) for i in range(n)]
    
    return [list(p) for p in best_chain]
# EVOLVE-BLOCK-END
