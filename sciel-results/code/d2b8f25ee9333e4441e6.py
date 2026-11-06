<<<<<<< SEARCH
def _polish(sequence, chain, max_iterations=2000, time_limit=3.0, rng=None):
    """Deterministic hill-climbing polish: accept only improving moves."""
    if chain is None:
        return None
    if rng is None:
        rng = random.Random(42)

    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy

    start_time = time.time()
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 15:
            break

        n = len(current_chain)
        best_move = None
        best_delta = 0

        # Try ALL pivot positions for thorough local search
        for pivot_idx in range(1, n - 1):
            for angle in (90, -90, 180):
                new_chain = _apply_pivot(current_chain, pivot_idx, angle)
                if new_chain is None:
                    continue
                new_energy = _energy(sequence, new_chain)
                delta = new_energy - current_energy
                if delta < best_delta:
                    best_delta = delta
                    best_move = new_chain

        # Try slither
        slither = _slither_move(current_chain, rng)
        if slither is not None:
            new_energy = _energy(sequence, slither)
            delta = new_energy - current_energy
            if delta < best_delta:
                best_delta = delta
                best_move = slither

        if best_move is None:
            break

        current_chain = best_move
        current_energy += best_delta
        if current_energy < best_energy:
            best_energy = current_energy
            best_chain = list(current_chain)

    return best_chain
=======
def _polish(sequence, chain, max_iterations=2000, time_limit=3.0, rng=None):
    """Deterministic hill-climbing polish using ChainState for fast energy."""
    if chain is None:
        return None

    state = _ChainState(sequence, chain)
    best_chain = list(state.chain)
    best_energy = state.energy
    n = state.n

    start_time = time.time()
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 15:
            break

        best_delta = 0
        best_move_type = None  # ('pivot', idx, angle) or ('slither', site)

        # Try ALL pivot positions
        for pivot_idx in range(1, n - 1):
            for angle in (90, -90, 180):
                new_chain, new_energy = state.apply_pivot(pivot_idx, angle)
                if new_chain is None:
                    continue
                delta = new_energy - state.energy
                if delta < best_delta:
                    best_delta = delta
                    best_move_type = ('pivot', pivot_idx, angle)

        # Try slither
        prev = state.chain[-2]
        candidates = [s for s in _neighbours(prev) if state._get_occupied(s[0], s[1]) == -1]
        for site in candidates:
            new_chain, new_energy = state.apply_slither(site)
            if new_chain is None:
                continue
            delta = new_energy - state.energy
            if delta < best_delta:
                best_delta = delta
                best_move_type = ('slither', site)

        if best_move_type is None:
            break

        # Apply the best move
        if best_move_type[0] == 'pivot':
            _, pivot_idx, angle = best_move_type
            state.apply_pivot(pivot_idx, angle)
        else:
            _, site = best_move_type
            state.apply_slither(site)

        if state.energy < best_energy:
            best_energy = state.energy
            best_chain = list(state.chain)

    return best_chain
>>>>>>> REPLACE

Now I need to update the `_grow` function and the main `fold` function to work with the new structure. The `_grow` function can stay mostly the same since it builds chains from scratch, but let me also update the `_apply_pivot` and `_slither_move` helpers to be consistent:

