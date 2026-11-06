"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import math
import time


def fold(sequence, dimension):
    """
    Fold the sequence into a low-energy conformation on the square lattice.
    Uses pre-computed contact tracking for O(1) energy delta updates
    and simulated annealing with kink and loop-reversal moves.
    """
    n = len(sequence)
    if n == 0:
        return []
    if n == 1:
        return [[0] * dimension]
    if n == 2:
        return [[0] * dimension, [1] + [0] * (dimension - 1)]

    is_h = [res == 'H' for res in sequence]
    start_time = time.time()
    time_budget = 11.5  # Leave margin

    def _generate_initial_walk(rng):
        """Generate a random self-avoiding walk."""
        chain = [[0, 0]]
        pos_map = {(0, 0): 0}
        occupied = {(0, 0)}

        for i in range(1, n):
            last = chain[-1]
            options = []
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = last[0] + dx, last[1] + dy
                if (nx, ny) not in occupied:
                    options.append((nx, ny))

            if not options:
                return None

            site = rng.choice(options)
            chain.append([site[0], site[1]])
            pos_map[site] = i
            occupied.add(site)

        return chain, pos_map

    def _compute_initial_contacts(chain, pos_map):
        """Compute the set of H-H contacts (i,j) with i<j."""
        contacts = set()
        for i in range(n):
            if not is_h[i]:
                continue
            px, py = chain[i]
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = px + dx, py + dy
                j = pos_map.get((nx, ny))
                if j is not None and abs(i - j) > 1 and is_h[j]:
                    if i < j:
                        contacts.add((i, j))
        return contacts

    def _sa_search(chain, pos_map, rng, max_iters):
        """Simulated annealing with kink and loop moves."""
        # Build contact set
        contacts = _compute_initial_contacts(chain, pos_map)
        current_energy = -len(contacts)
        best_energy = current_energy
        best_chain = [list(p) for p in chain]

        t_start = 2.5
        t_end = 0.01
        it = 0

        while it < max_iters:
            alpha = it / max_iters
            temp = t_start * math.exp(-8.0 * alpha) + t_end
            it += 1

            move_type = rng.random()

            if move_type < 0.75:
                # Kink move: relocate a single interior residue
                i = rng.randint(1, n - 2)
                if i < 1 or i > n - 2:
                    continue

                px, py = chain[i]
                prev_x, prev_y = chain[i - 1]
                next_x, next_y = chain[i + 1]

                # Candidate positions: adjacent to both prev and next, not occupied
                candidates = []
                # Must be at Manhattan distance 1 from both prev and next
                # The two common neighbors of prev and next (if they are at distance 2)
                # prev and next are at distance 2 from each other (since i is between them)
                # So there are at most 2 common neighbors
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    cx, cy = prev_x + dx, prev_y + dy
                    # Check if adjacent to next
                    if abs(cx - next_x) + abs(cy - next_y) == 1:
                        if (cx, cy) != (px, py) and (cx, cy) not in pos_map:
                            candidates.append((cx, cy))

                if not candidates:
                    continue

                new_pos = rng.choice(candidates)
                new_x, new_y = new_pos

                # Compute energy delta
                delta = 0
                if is_h[i]:
                    # Remove old contacts involving i
                    for (a, b) in list(contacts):
                        if a == i or b == i:
                            delta += 1  # losing a contact increases energy
                            contacts.discard((a, b))

                    # Add new contacts involving i
                    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        nx, ny = new_x + dx, new_y + dy
                        j = pos_map.get((nx, ny))
                        if j is not None and abs(i - j) > 1 and is_h[j]:
                            if i < j:
                                pair = (i, j)
                            else:
                                pair = (j, i)
                            if pair not in contacts:
                                contacts.add(pair)
                                delta -= 1  # gaining a contact decreases energy

                # Accept/reject
                if delta <= 0 or rng.random() < math.exp(-delta / temp):
                    # Apply move
                    del pos_map[(px, py)]
                    chain[i] = [new_x, new_y]
                    pos_map[new_pos] = i
                    current_energy += delta

                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = [list(p) for p in chain]

            else:
                # Loop reversal: reverse a segment [i, j]
                # Pick segment length
                max_seg = min(8, n - 2)
                seg_len = rng.randint(2, max_seg)
                i = rng.randint(1, n - seg_len - 1)
                j = i + seg_len - 1

                prev_x, prev_y = chain[i - 1]
                next_x, next_y = chain[j + 1]
                old_i_x, old_i_y = chain[i]
                old_j_x, old_j_y = chain[j]

                # After reversal:
                #   new_chain[i] = old_chain[j] = (old_j_x, old_j_y)
                #   new_chain[j] = old_chain[i] = (old_i_x, old_i_y)
                # Check connectivity:
                #   (old_j_x, old_j_y) must be adjacent to (prev_x, prev_y)
                #   (old_i_x, old_i_y) must be adjacent to (next_x, next_y)
                if abs(old_j_x - prev_x) + abs(old_j_y - prev_y) != 1:
                    continue
                if abs(old_i_x - next_x) + abs(old_i_y - next_y) != 1:
                    continue

                # The set of positions is unchanged, so self-avoidance is maintained.
                # Only chain-adjacency changes within [i, j].
                # Compute energy delta by checking which contacts change.

                # Contacts that can change:
                # 1. Contacts between residues in [i, j] and residues outside [i, j]
                #    that are lattice-adjacent to positions in the segment
                # 2. Contacts between residues in [i, j] that were chain-adjacent
                #    before but not after, or vice versa

                # For efficiency, let's compute the delta by examining:
                # - Which (a,b) contacts are removed/added due to chain distance changes
                # - The lattice positions are the same, so lattice adjacency is unchanged
                #   Only the chain-index adjacency changes

                # A contact (a,b) exists if: is_h[a], is_h[b], lattice-adjacent, |a-b|>1
                # After reversal, for a,b both in [i,j]:
                #   new index of old residue k (i<=k<=j) is i+j-k
                #   So |new_a - new_b| = |(i+j-a) - (i+j-b)| = |b-a| = |a-b|
                #   So chain distance is preserved for pairs within the segment!
                #   But chain adjacency to outside changes:
                #   - Residue i was adjacent to i-1, now residue j (in new chain) is at position i
                #     Wait, let me think again.
                #
                # After reversal:
                #   new_chain[i] = old_chain[j]  (residue at new index i is the one that was at old index j)
                #   new_chain[i+1] = old_chain[j-1]
                #   ...
                #   new_chain[j] = old_chain[i]
                #
                # So the residue that was at old index k (i<=k<=j) is now at new index i+j-k.
                # The chain neighbors of new index i are i-1 and i+1.
                #   i-1 is outside the segment (unchanged)
                #   i+1 corresponds to old index j-1
                # The chain neighbors of new index j are j-1 and j+1.
                #   j+1 is outside the segment (unchanged)
                #   j-1 corresponds to old index i+1
                #
                # So the chain-adjacency changes are:
                #   Before: (i-1, i) and (j, j+1) were chain-adjacent
                #   After:  (i-1, i) where new i is old j, and (j, j+1) where new j is old i
                #   Also: before (i, i+1) and (j-1, j) were chain-adjacent
                #          after  (i, i+1) where new i+1 is old j-1, and (j-1, j) where new j-1 is old i+1
                #
                # Key insight: For pairs where at least one is outside [i,j], the lattice positions
                # are unchanged, but the chain indices of the segment residues change.
                #
                # Let me compute delta by checking all H-H pairs that are lattice-adjacent
                # and have at least one member in [i,j], and see if the contact status changes.

                delta = 0

                # Build a map: old_index -> new_index for indices in [i, j]
                # new_index = i + j - old_index

                # Check contacts involving residues in [i, j]
                # For each residue a in [i, j], check its 4 lattice neighbors
                # If neighbor b is H and |a-b| > 1 before AND |new_a - new_b| > 1 after, contact preserved
                # If |a-b| > 1 before but |new_a - new_b| <= 1 after, contact lost (delta += 1)
                # If |a-b| <= 1 before but |new_a - new_b| > 1 after, contact gained (delta -= 1)

                # To avoid double counting, only consider a < b in terms of some ordering
                # Let's iterate over residues in [i, j] and their lattice neighbors

                # First, remove all contacts involving any residue in [i, j]
                to_remove = []
                to_add = []

                for a in range(i, j + 1):
                    if not is_h[a]:
                        continue
                    ax, ay = chain[a]
                    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        nx, ny = ax + dx, ay + dy
                        b = pos_map.get((nx, ny))
                        if b is None or not is_h[b]:
                            continue
                        # b is a lattice neighbor of a
                        # Check if this was a contact before
                        was_contact = abs(a - b) > 1
                        # New index of a: new_a = i + j - a
                        new_a = i + j - a
                        # b is outside [i, j] or inside
                        if i <= b <= j:
                            new_b = i + j - b
                        else:
                            new_b = b
                        is_contact_after = abs(new_a - new_b) > 1

                        if was_contact and not is_contact_after:
                            # Contact lost
                            if a < b:
                                to_remove.append((a, b))
                            else:
                                to_remove.append((b, a))
                        elif not was_contact and is_contact_after:
                            # Contact gained
                            if a < b:
                                to_add.append((a, b))
                            else:
                                to_add.append((b, a))

                # Also need to check contacts between two residues both in [i, j]
                # where the chain distance changes
                # Actually the above loop already handles this since b can be in [i, j]
                # But we might double count. Let's use sets.

                remove_set = set()
                for a in range(i, j + 1):
                    if not is_h[a]:
                        continue
                    ax, ay = chain[a]
                    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        nx, ny = ax + dx, ay + dy
                        b = pos_map.get((nx, ny))
                        if b is None or not is_h[b]:
                            continue
                        was_contact = abs(a - b) > 1
                        new_a = i + j - a
                        if i <= b <= j:
                            new_b = i + j - b
                        else:
                            new_b = b
                        is_contact_after = abs(new_a - new_b) > 1

                        if a < b:
                            pair = (a, b)
                        else:
                            pair = (b, a)

                        if was_contact and not is_contact_after:
                            remove_set.add(pair)
                        elif not was_contact and is_contact_after:
                            # Only add if not already in contacts
                            if pair not in contacts:
                                to_add.append(pair)

                # Apply changes
                for pair in remove_set:
                    if pair in contacts:
                        contacts.discard(pair)
                        delta += 1

                for pair in to_add:
                    if pair not in contacts:
                        contacts.add(pair)
                        delta -= 1

                # Accept/reject
                if delta <= 0 or rng.random() < math.exp(-delta / temp):
                    # Apply reversal
                    for k in range(i, j + 1):
                        new_k = i + j - k
                        chain[k], chain[new_k] = chain[new_k], chain[k]
                        pos_map[tuple(chain[k])] = k
                        pos_map[tuple(chain[new_k])] = new_k
                    current_energy += delta

                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_chain = [list(p) for p in chain]

        return best_chain, best_energy

    # Main: multiple restarts
    best_chain_overall = None
    best_energy_overall = float('inf')

    # Adaptive number of trials based on time
    num_trials = 15
    iters_per_trial = 80000

    for trial in range(num_trials):
        if time.time() - start_time > time_budget:
            break

        rng = random.Random(42 + len(sequence) * 7 + trial * 137)

        result = _generate_initial_walk(rng)
        if result is None:
            continue

        chain, pos_map = result
        improved_chain, improved_energy = _sa_search(chain, pos_map, rng, iters_per_trial)

        if improved_energy < best_energy_overall:
            best_energy_overall = improved_energy
            best_chain_overall = improved_chain

    if best_chain_overall is None:
        best_chain_overall = [[i, 0] for i in range(n)]

    return best_chain_overall
# EVOLVE-BLOCK-END
