import numpy as np
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safe margin before 150s kill

    def get_triple_indices(n):
        """Return list of all triples (a,b,c) with a<b<c."""
        return list(combinations(range(n), 3))

    def build_structures(n):
        """Build K4 and K5 structures with precomputed masks for fast delta computation."""
        triples = get_triple_indices(n)
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}

        # K4: 4 vertices -> 4 triples
        k4s = []
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            k4s.append((
                triple_to_idx[(a, b, c)],
                triple_to_idx[(a, b, d)],
                triple_to_idx[(a, c, d)],
                triple_to_idx[(b, c, d)]
            ))

        # K5: 5 vertices -> 10 triples
        k5s = []
        for quint in combinations(range(n), 5):
            idxs = []
            for triple in combinations(quint, 3):
                idxs.append(triple_to_idx[triple])
            k5s.append(tuple(idxs))

        # Precompute for each triple: list of (k4_idx, other_mask) and (k5_idx, other_mask)
        # other_mask has bits set for the other triples in the clique (excluding this triple)
        triple_to_k4 = [[] for _ in range(num_triples)]
        triple_to_k5 = [[] for _ in range(num_triples)]

        for k4_idx, idxs in enumerate(k4s):
            for pos, t in enumerate(idxs):
                # Build mask of other 3 triples in this K4
                other_mask = 0
                for j, other_t in enumerate(idxs):
                    if j != pos:
                        other_mask |= (1 << other_t)
                triple_to_k4[t].append((k4_idx, other_mask))

        for k5_idx, idxs in enumerate(k5s):
            for pos, t in enumerate(idxs):
                # Build mask of other 9 triples in this K5
                other_mask = 0
                for j, other_t in enumerate(idxs):
                    if j != pos:
                        other_mask |= (1 << other_t)
                triple_to_k5[t].append((k5_idx, other_mask))

        # Precompute full masks for each clique
        k4_masks = []
        for idxs in k4s:
            mask = 0
            for i in idxs:
                mask |= (1 << i)
            k4_masks.append(mask)

        k5_masks = []
        for idxs in k5s:
            mask = 0
            for i in idxs:
                mask |= (1 << i)
            k5_masks.append(mask)

        return {
            'triples': triples,
            'num_triples': num_triples,
            'k4s': k4s,
            'k5s': k5s,
            'triple_to_k4': triple_to_k4,
            'triple_to_k5': triple_to_k5,
            'k4_masks': k4_masks,
            'k5_masks': k5_masks
        }

    def count_violations(colors, struct):
        """Count total violations (red K4s + blue K5s)."""
        k4_masks = struct['k4_masks']
        k5_masks = struct['k5_masks']
        
        red_count = 0
        for mask in k4_masks:
            if (colors & mask) == 0:
                red_count += 1
        
        blue_count = 0
        for mask in k5_masks:
            if (colors & mask) == mask:
                blue_count += 1
                
        return red_count + blue_count

    def delta_cost(colors, t, struct):
        """Compute cost delta when flipping triple t using precomputed other_masks."""
        delta = 0
        bit = 1 << t
        is_blue = (colors & bit) != 0

        # Check K4s containing t
        for k4_idx, other_mask in struct['triple_to_k4'][t]:
            # Count reds (zeros) in other triples
            reds_others = np.count_nonzero(np.array([(colors >> i) & 1 == 0 for i in range(struct['num_triples']) if (other_mask >> i) & 1]))
            
            if is_blue:
                # t was Blue (1). Flipping to Red (0).
                # After: reds_others + 1
                if reds_others == 3:
                    # Becomes 4 reds -> Violation created
                    delta += 1
            else:
                # t was Red (0). Flipping to Blue (1).
                # Before: reds_others + 1
                if reds_others == 3:
                    # Was 4 reds (Violation), becomes 3. Violation removed.
                    delta -= 1

        # Check K5s containing t
        for k5_idx, other_mask in struct['triple_to_k5'][t]:
            # Count blues (ones) in other triples
            blues_others = np.count_nonzero(np.array([(colors >> i) & 1 == 1 for i in range(struct['num_triples']) if (other_mask >> i) & 1]))
            
            if is_blue:
                # t was Blue. Flipping to Red.
                # Before: blues_others + 1
                if blues_others == 9:
                    # Was 10 blues (Violation), becomes 9. Violation removed.
                    delta -= 1
            else:
                # t was Red. Flipping to Blue.
                # After: blues_others + 1
                if blues_others == 9:
                    # Becomes 10 blues -> Violation created
                    delta += 1

        return delta

    def fast_delta_cost(colors, t, struct):
        """Faster delta computation using bit operations on precomputed masks."""
        delta = 0
        bit = 1 << t
        is_blue = (colors & bit) != 0

        # Check K4s containing t
        for k4_idx, other_mask in struct['triple_to_k4'][t]:
            # Count how many of the other bits are 0 (Red)
            # reds_others = popcount(~(colors & other_mask)) within other_mask
            # = popcount(other_mask) - popcount(colors & other_mask)
            other_count = bin(other_mask).count('1')
            set_count = bin(colors & other_mask).count('1')
            reds_others = other_count - set_count
            
            if is_blue:
                if reds_others == 3:
                    delta += 1
            else:
                if reds_others == 3:
                    delta -= 1

        # Check K5s containing t
        for k5_idx, other_mask in struct['triple_to_k5'][t]:
            # Count how many of the other bits are 1 (Blue)
            other_count = bin(other_mask).count('1')
            set_count = bin(colors & other_mask).count('1')
            blues_others = set_count
            
            if is_blue:
                if blues_others == 9:
                    delta -= 1
            else:
                if blues_others == 9:
                    delta += 1

        return delta

    def structured_init(n, variant, seed, struct):
        """Generate structured initial colorings as bitmask."""
        triples = struct['triples']
        num_triples = struct['num_triples']
        mask = 0
        
        if variant == 0:
            for i, (a, b, c) in enumerate(triples):
                if (a + b + c) % 3 != 0:
                    mask |= (1 << i)
        elif variant == 1:
            for i, (a, b, c) in enumerate(triples):
                if (a + b + 2*c) % 3 != 0:
                    mask |= (1 << i)
        elif variant == 2:
            for i, (a, b, c) in enumerate(triples):
                if (a*b + b*c + c*a) % 2 == 1:
                    mask |= (1 << i)
        elif variant == 3:
            for i, (a, b, c) in enumerate(triples):
                if (a + b + c) % 2 == 1:
                    mask |= (1 << i)
        elif variant == 4:
            rng = np.random.RandomState(seed)
            for i in range(num_triples):
                if rng.rand() < 0.5:
                    mask |= (1 << i)
        else:
            for i, (a, b, c) in enumerate(triples):
                if (a + b + c) % 3 != 0:
                    mask |= (1 << i)
        
        return mask

    def sa_lns(n, time_limit, seed, init_variant, struct):
        """Simulated Annealing with LNS."""
        rng = np.random.RandomState(seed)
        num_triples = struct['num_triples']
        
        colors = structured_init(n, init_variant, seed, struct)
        current_cost = count_violations(colors, struct)
        best_cost = current_cost
        best_colors = colors

        T = 5.0  # Initial temperature
        T_min = 0.01
        cooling = 0.995
        max_steps_per_temp = 50

        step = 0
        no_improve = 0
        max_no_improve = 200

        while time.time() < time_limit and T > T_min:
            # Try multiple moves at this temperature
            for _ in range(max_steps_per_temp):
                if time.time() >= time_limit:
                    break
                    
                # Pick a random triple to flip
                t = rng.randint(num_triples)
                dc = fast_delta_cost(colors, t, struct)
                new_cost = current_cost + dc
                
                # Metropolis acceptance
                if dc < 0 or rng.rand() < np.exp(-dc / T):
                    colors ^= (1 << t)
                    current_cost = new_cost
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_colors = colors
                        no_improve = 0
                        if best_cost == 0:
                            return best_colors, True
                    else:
                        no_improve += 1
                else:
                    no_improve += 1

            # LNS: if stuck, do a larger perturbation
            if no_improve > max_no_improve:
                # Flip a random subset of triples involved in violations
                # First, find violated cliques
                violated_triples = set()
                for k4_idx, mask in enumerate(struct['k4_masks']):
                    if (colors & mask) == 0:
                        for t in struct['k4s'][k4_idx]:
                            violated_triples.add(t)
                for k5_idx, mask in enumerate(struct['k5_masks']):
                    if (colors & mask) == mask:
                        for t in struct['k5s'][k5_idx]:
                            violated_triples.add(t)
                
                if violated_triples:
                    # Flip a random subset of violated triples
                    v_list = list(violated_triples)
                    num_flip = max(1, len(v_list) // 3)
                    to_flip = rng.choice(v_list, size=min(num_flip, len(v_list)), replace=False)
                    for t in to_flip:
                        colors ^= (1 << t)
                    
                    current_cost = count_violations(colors, struct)
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_colors = colors
                else:
                    # No violations but stuck? Shouldn't happen if best_cost > 0
                    # Do a random shake
                    num_shake = max(1, num_triples // 20)
                    shake_indices = rng.choice(num_triples, size=num_shake, replace=False)
                    for t in shake_indices:
                        colors ^= (1 << t)
                    current_cost = count_violations(colors, struct)
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_colors = colors
                
                no_improve = 0
                T = max(T * 0.5, T_min)  # Cool down after perturbation

            T *= cooling

        return best_colors, (best_cost == 0)

    def verify(n, colors_bitmask):
        """Verify coloring has no red K4 and no blue K5."""
        triples = get_triple_indices(n)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            i1 = triple_to_idx[(a, b, c)]
            i2 = triple_to_idx[(a, b, d)]
            i3 = triple_to_idx[(a, c, d)]
            i4 = triple_to_idx[(b, c, d)]
            
            if ((colors_bitmask >> i1) & 1) == 0 and \
               ((colors_bitmask >> i2) & 1) == 0 and \
               ((colors_bitmask >> i3) & 1) == 0 and \
               ((colors_bitmask >> i4) & 1) == 0:
                return False

        for quint in combinations(range(n), 5):
            all_blue = True
            for triple in combinations(quint, 3):
                idx = triple_to_idx[triple]
                if ((colors_bitmask >> idx) & 1) != 1:
                    all_blue = False
                    break
            if all_blue:
                return False

        return True

    final_n = 0
    final_colors = []

    target_ns = [35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9]

    for n in target_ns:
        if time.time() > deadline - 5:
            break

        if n <= final_n:
            continue

        remaining = deadline - time.time()
        if remaining < 3:
            break

        # Allocate time based on n
        if n == 35:
            time_budget = min(50.0, remaining * 0.6)
        elif n == 34:
            time_budget = min(40.0, remaining * 0.5)
        elif n >= 30:
            time_budget = min(25.0, remaining * 0.4)
        elif n >= 25:
            time_budget = min(15.0, remaining * 0.3)
        elif n >= 20:
            time_budget = min(10.0, remaining * 0.25)
        else:
            time_budget = min(5.0, remaining * 0.2)

        if time_budget < 1.0:
            break

        time_limit = time.time() + time_budget

        found = False
        # More seeds for larger n
        num_seeds = 25 if n >= 35 else (20 if n >= 30 else 15)

        # Pre-build structures for this n
        struct = build_structures(n)

        for seed in range(num_seeds):
            if time.time() > time_limit:
                break

            variant = seed % 5

            try:
                colors_bitmask, is_valid = sa_lns(n, time_limit, seed=seed, init_variant=variant, struct=struct)

                if is_valid:
                    if verify(n, colors_bitmask):
                        num_triples = struct['num_triples']
                        colors_list = [int((colors_bitmask >> i) & 1) for i in range(num_triples)]
                        final_n = n
                        final_colors = colors_list
                        found = True
                        break
            except Exception:
                continue

        if found:
            break

    if final_n == 0 or not final_colors:
        # Fallback: small valid coloring
        n = 9
        triples = get_triple_indices(n)
        mask = 0
        for i, (a, b, c) in enumerate(triples):
            if (a + b + c) % 3 != 0:
                mask |= (1 << i)
        if verify(9, mask):
            final_n = 9
            final_colors = [int((mask >> i) & 1) for i in range(len(triples))]
        else:
            final_n = 1
            final_colors = [0]

    return {'n': final_n, 'colors': final_colors}
