    # Phase 1: Multi-start Powell
    n_starts = 6
    # 0: All 30 deg
    x0_1 = np.array([np.pi/6] + [np.pi/6] * n_hex)
    # 1: All 0 deg
    x0_2 = np.array([0.0] + [0.0] * n_hex)
    # 2: Outer 30, Inner 0
    x0_3 = np.array([np.pi/6] + [0.0] * n_hex)
    # 3: Outer 0, Inner 30
    x0_4 = np.array([0.0] + [np.pi/6] * n_hex)
    # 4: Outer 15, Inner 15
    x0_5 = np.array([np.pi/12] + [np.pi/12] * n_hex)
    # 5: Outer 30, Inner alternating 0/30
    inner_alts = [0.0, np.pi/6, 0.0, np.pi/6, 0.0, np.pi/6, 0.0, np.pi/6, 0.0, np.pi/6, 0.0]
    x0_6 = np.array([np.pi/6] + inner_alts)
    
    starts = [x0_1, x0_2, x0_3, x0_4, x0_5, x0_6]
