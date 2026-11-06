import time
import random
from itertools import combinations

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))
N_TRIPLES = len(ALL_TRIPLES)  # 560

# Precompute triple index lookup
_triple_idx = {}
for _i, _t in enumerate(ALL_TRIPLES):
    _triple_idx[_t] = _i

def _tidx(a, b, c):
    """Get index of triple {a,b,c} (all distinct)."""
    if a > b: a, b = b, a
    if b > c: b, c = c, b
    if a > b: a, b = b, a
    return _triple_idx[(a, b, c)]

class Hypergraph:
    def __init__(self):
        self.presence = 0  # 560-bit integer
        self.common = [[0] * V for _ in range(V)]  # 16x16 matrix of 16-bit masks
        self.count = 0

    def add(self, idx):
        a, b, c = ALL_TRIPLES[idx]
        self.presence |= (1 << idx)
        self.count += 1
        # Update common masks
        # {a,b,c} means c is in common[a][b], etc.
        self.common[a][b] |= (1 << c)
        self.common[b][a] |= (1 << c)
        self.common[a][c] |= (1 << b)
        self.common[c][a] |= (1 << b)
        self.common[b][c] |= (1 << a)
        self.common[c][b] |= (1 << a)

    def remove(self, idx):
        a, b, c = ALL_TRIPLES[idx]
        self.presence &= ~(1 << idx)
        self.count -= 1
        # Update common masks
        self.common[a][b] &= ~(1 << c)
        self.common[b][a] &= ~(1 << c)
        self.common[a][c] &= ~(1 << b)
        self.common[c][a] &= ~(1 << b)
        self.common[b][c] &= ~(1 << a)
        self.common[c][b] &= ~(1 << a)

    def is_present(self, idx):
        return (self.presence >> idx) & 1

    def check_conflict(self, a, b, c):
        """
        Check if adding {a,b,c} creates a K(2,2,2).
        We assume {a,b,c} is not currently present.
        """
        # Candidates for a' must be in common[b][c] and not in {a,b,c}
        mask_bc = self.common[b][c] & ~(1 << a) & ~(1 << b) & ~(1 << c)
        
        # Candidates for b' must be in common[a][c] and not in {a,b,c}
        mask_ac = self.common[a][c] & ~(1 << a) & ~(1 << b) & ~(1 << c)
        
        # Candidates for c' must be in common[a][b] and not in {a,b,c}
        mask_ab = self.common[a][b] & ~(1 << a) & ~(1 << b) & ~(1 << c)

        # Iterate over possible a'
        a_p_mask = mask_bc
        while a_p_mask:
            a_p_bit = a_p_mask & (-a_p_mask)
            a_p = a_p_bit.bit_length() - 1
            a_p_mask ^= a_p_bit
            
            # For this a', iterate over possible b'
            # b' must be in mask_ac and not equal to a_p
            b_p_mask = mask_ac & ~(1 << a_p)
            while b_p_mask:
                b_p_bit = b_p_mask & (-b_p_mask)
                b_p = b_p_bit.bit_length() - 1
                b_p_mask ^= b_p_bit
                
                # Check if {a', b', c} is present
                # i.e., c is in common[a_p][b_p]
                if not (self.common[a_p][b_p] & (1 << c)):
                    continue
                
                # If {a', b', c} is present, we need to check if there exists c'
                # such that:
                # {a, b, c'} in H  -> c' in common[a][b]
                # {a, b', c'} in H -> c' in common[a][b_p]
                # {a', b, c'} in H -> c' in common[a_p][b]
                # {a', b', c'} in H -> c' in common[a_p][b_p]
                # and c' not in {a, b, c, a', b'}
                
                c_cand = (self.common[a][b] & 
                          self.common[a][b_p] & 
                          self.common[a_p][b] & 
                          self.common[a_p][b_p])
                
                # Exclude a, b, c, a', b'
                c_cand &= ~(1 << a)
                c_cand &= ~(1 << b)
                c_cand &= ~(1 << c)
                c_cand &= ~(1 << a_p)
                c_cand &= ~(1 << b_p)
                
                if c_cand:
                    return True
        
        return False

def construct():
    start = time.time()
    deadline = start + 115  # Safety margin
    
    best_hg = None
    best_count = 0
    
    while time.time() < deadline:
        # Initialize with a greedy random order
        hg = Hypergraph()
        order = list(range(N_TRIPLES))
        random.shuffle(order)
        
        for idx in order:
            if time.time() > deadline - 10: # Reserve time for local search
                break
            a, b, c = ALL_TRIPLES[idx]
            if not hg.check_conflict(a, b, c):
                hg.add(idx)
        
        # Local Search with proper swap moves
        present_mask = hg.presence
        present_list = [i for i in range(N_TRIPLES) if (present_mask >> i) & 1]
        missing = [i for i in range(N_TRIPLES) if not (present_mask >> i) & 1]
    
        improved = True
        stall_count = 0
        max_stall = 50
    
        while improved and time.time() < deadline - 5:
            improved = False
            if stall_count > max_stall:
                break
        
            random.shuffle(missing)
        
            # Phase 1: Try to add (fast path)
            for idx in missing:
                if time.time() > deadline - 5:
                    break
                a, b, c = ALL_TRIPLES[idx]
                if not hg.check_conflict(a, b, c):
                    hg.add(idx)
                    present_mask |= (1 << idx)
                    missing.remove(idx)
                    present_list.append(idx)
                    improved = True
                    stall_count = 0
                    break
        
            if improved:
                continue
        
            # Phase 2: Try 1-swap moves (remove one present, add one missing)
            if missing:
                # Try several random swap attempts
                swap_attempts = 0
                max_swap_attempts = 20
                while swap_attempts < max_swap_attempts and time.time() < deadline - 5:
                    swap_attempts += 1
                
                    new_idx = random.choice(missing)
                    na, nb, nc = ALL_TRIPLES[new_idx]
                
                    # Find a present triple that might be blocking this addition
                    # A triple blocks if it participates in a K(2,2,2) with the new triple
                    # Heuristic: pick random present triples and try removing them
                    for _ in range(5):
                        if not present_list or time.time() > deadline - 5:
                            break
                        old_idx = random.choice(present_list)
                        if old_idx == new_idx:
                            continue
                    
                        oa, ob, oc = ALL_TRIPLES[old_idx]
                    
                        # Try removing old and adding new
                        hg.remove(old_idx)
                        if not hg.check_conflict(na, nb, nc):
                            hg.add(new_idx)
                            # Update lists
                            present_list.remove(old_idx)
                            missing.remove(new_idx)
                            present_list.append(new_idx)
                            present_mask = hg.presence
                            improved = True
                            stall_count = 0
                            break
                        else:
                            # Restore
                            hg.add(old_idx)
                            present_mask = hg.presence
                
                    if improved:
                        break
        
            if not improved:
                stall_count += 1
        
        if hg.count > best_count:
            best_count = hg.count
            best_hg = hg

    if best_hg is None:
        return []
        
    result = []
    for i in range(N_TRIPLES):
        if (best_hg.presence >> i) & 1:
            result.append(list(ALL_TRIPLES[i]))
            
    return result

random.seed(42)
