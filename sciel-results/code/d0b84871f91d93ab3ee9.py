<<<<<<< SEARCH
    initializations = [
        ("grid", make_initial_grid),
        ("bcc", make_bcc_lattice),
        ("fcc", make_fcc_like),
        ("det", make_deterministic_packing),
        ("rand0", lambda: make_random(0)),
        ("rand1", lambda: make_random(42)),
        ("rand2", lambda: make_random(123)),
        ("rand3", lambda: make_random(777)),
        ("rand4", lambda: make_random(2024)),
        ("rand5", lambda: make_random(999)),
        ("rand6", lambda: make_random(3141)),
    ]
=======
    initializations = [
        ("grid", make_initial_grid),
        ("bcc", make_bcc_lattice),
        ("fcc", make_fcc_like),
        ("hcp", make_hcp_like),
        ("det", make_deterministic_packing),
        ("rand0", lambda: make_random(0)),
        ("rand1", lambda: make_random(42)),
        ("rand2", lambda: make_random(123)),
        ("rand3", lambda: make_random(777)),
        ("rand4", lambda: make_random(2024)),
        ("rand5", lambda: make_random(999)),
        ("rand6", lambda: make_random(3141)),
    ]
>>>>>>> REPLACE
