<<<<<<< SEARCH
    def walk_one(assignment, rng, max_steps, p_flip):
        """Run WalkSAT with incremental tracking."""
        clause_sat_count, unsat_set = init_state(assignment)

        for step in range(max_steps):
            if time.time() > deadline:
                return assignment, False

            if not unsat_set:
                return assignment, True

            # Pick a random unsatisfied clause
            ci = rng.choice(list(unset_set))
=======
    def walk_one(assignment, rng, max_steps, p_flip):
        """Run WalkSAT with incremental tracking."""
        clause_sat_count, unsat_set = init_state(assignment)

        for step in range(max_steps):
            if time.time() > deadline:
                return assignment, False

            if not unsat_set:
                return assignment, True

            # Pick a random unsatisfied clause
            ci = rng.choice(list(unset_set))
>>>>>>> REPLACE
