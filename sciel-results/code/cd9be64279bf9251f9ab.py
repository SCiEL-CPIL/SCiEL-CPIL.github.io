<<<<<<< SEARCH
def construct():
    """Return 12 distinct integer points, |x|,|y| <= 10^9, not all collinear,
    minimizing ordinary lines."""
    best_pts = None
    best_score = float('inf')

    def _consider(pts):
        nonlocal best_pts, best_score
        if not _valid(pts):
            return
        s = _count_ordinary(pts)
        if s < best_score:
            best_score = s
            best_pts = [p[:] for p in pts]

    # Multiple seeds
    seeds = [
        _make_seed_grid(3, 4),
        _make_seed_grid(4, 3),
        _make_seed_lines(),
        _make_seed_two_lines(),
        _make_seed_perturbed(),
    ]

    # Evaluate all seeds
    for s in seeds:
        if len(s) == 12:
            _consider(s)

    # Simulated annealing from each seed
    for s in seeds:
        if len(s) != 12:
            continue
        if time.time() > _DEADLINE - 20:
            break
        pts_sa, score_sa = _simulated_annealing(s, max_iters=100000, seed=random.randint(1, 10000))
        if score_sa < best_score:
            best_score = score_sa
            best_pts = pts_sa

    # Local search from best so far
    if best_pts is not None and time.time() < _DEADLINE - 10:
        pts_ls, score_ls = _local_search(best_pts, max_iters=150000, seed=42)
        if score_ls < best_score:
            best_score = score_ls
            best_pts = pts_ls

    # More simulated annealing from best
    if best_pts is not None and time.time() < _DEADLINE - 15:
        pts_sa2, score_sa2 = _simulated_annealing(best_pts, max_iters=200000, seed=999)
        if score_sa2 < best_score:
            best_score = score_sa2
            best_pts = pts_sa2

    # Final local search polish
    if best_pts is not None and time.time() < _DEADLINE - 8:
        pts_final, score_final = _local_search(best_pts, max_iters=200000, seed=777)
        if score_final < best_score:
            best_score = score_final
            best_pts = pts_final

    if best_pts is None:
        best_pts = _make_seed_grid(3, 4)

    return best_pts
=======
def construct():
    """Return 12 distinct integer points, |x|,|y| <= 10^9, not all collinear,
    minimizing ordinary lines."""
    best_pts = None
    best_score = float('inf')

    def _consider(pts):
        nonlocal best_pts, best_score
        if not _valid(pts):
            return
        s = _count_ordinary(pts)
        if s < best_score:
            best_score = s
            best_pts = [p[:] for p in pts]

    # Multiple seeds
    seeds = [
        _make_seed_grid(3, 4),
        _make_seed_grid(4, 3),
        _make_seed_lines(),
        _make_seed_two_lines(),
        _make_seed_perturbed(),
    ]

    # Evaluate all seeds
    for s in seeds:
        if len(s) == 12:
            _consider(s)

    # Try template-based backtracking
    templates = _generate_templates()
    for lines, _ in templates:
        if time.time() > _DEADLINE - 30:
            break
        sol = _solve_template_backtrack(lines)
        if sol is not None:
            _consider(sol)

    # Simulated annealing from each seed
    for s in seeds:
        if len(s) != 12:
            continue
        if time.time() > _DEADLINE - 20:
            break
        pts_sa, score_sa = _simulated_annealing(s, max_iters=100000, seed=random.randint(1, 10000))
        if score_sa < best_score:
            best_score = score_sa
            best_pts = pts_sa

    # Local search from best so far
    if best_pts is not None and time.time() < _DEADLINE - 10:
        pts_ls, score_ls = _local_search(best_pts, max_iters=150000, seed=42)
        if score_ls < best_score:
            best_score = score_ls
            best_pts = pts_ls

    # More simulated annealing from best
    if best_pts is not None and time.time() < _DEADLINE - 15:
        pts_sa2, score_sa2 = _simulated_annealing(best_pts, max_iters=200000, seed=999)
        if score_sa2 < best_score:
            best_score = score_sa2
            best_pts = pts_sa2

    # Final local search polish
    if best_pts is not None and time.time() < _DEADLINE - 8:
        pts_final, score_final = _local_search(best_pts, max_iters=200000, seed=777)
        if score_final < best_score:
            best_score = score_final
            best_pts = pts_final

    if best_pts is None:
        best_pts = _make_seed_grid(3, 4)

    return best_pts
>>>>>>> REPLACE

Now I need to add the missing `_local_search2` and fix the `_simulated_annealing` function since I removed it. Let me check what's still needed and add back a working SA:

