def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    def run_hill_climbing(initial_set, time_limit):
        # ... hill climbing logic ...
        return best_set, best_size

    # Initial candidates
    candidates = [
        list(range(12)),
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15],
        [0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32],
        [0, 1, 3, 7, 12, 18, 25, 33, 42, 52, 63, 75],
    ]
    
    best_set = None
    best_size = 13
    
    random.seed(42)
    
    # Try each candidate with a short hill climb
    for c in candidates:
        if time.time() > deadline - 5:
            break
        if len(set(c)) != 12 or any(x < 0 or x > 128 for x in c):
            continue
        
        sz = get_sidon_size(c, 12)
        if sz < best_size:
            best_size = sz
            best_set = list(c)
        
        # Run hill climb from this candidate
        # Limit time per candidate
        local_deadline = time.time() + 5.0
        # ... run hill climb ...
        
    # Then do global random restarts with hill climb
    while time.time() < deadline - 5:
        s = random.sample(range(129), 12)
        sz = get_sidon_size(s, 12)
        if sz < best_size:
            best_size = sz
            best_set = list(s)
            
        # Run hill climb from s
        local_deadline = time.time() + 2.0
        # ... run hill climb ...
        
    return best_set
