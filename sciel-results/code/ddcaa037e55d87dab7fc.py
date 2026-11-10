<<<<<<< SEARCH
            # Local improvement on this candidate
            if time.time() < _DEADLINE - 25:
                improved_chain, improved_energy = _local_improve_fast(sequence, chain, rng, max_moves=5000)
                if improved_energy < energy:
                    energy = improved_energy
                    chain = improved_chain
            
            # Min-conflicts search
            if time.time() < _DEADLINE - 22:
                mc_chain, mc_energy = _min_conflicts_search(sequence, chain, rng, max_iters=2000)
                if mc_energy < energy:
                    energy = mc_energy
                    chain = mc_chain
            
            # SA on the improved candidate
            if time.time() < _DEADLINE - 18:
                sa_chain, sa_energy = _regrow_sa_fast(sequence, chain, rng, max_iters=5000, start_temp=6.0, cool_rate=0.995)
                if sa_energy < energy:
                    energy = sa_energy
                    chain = sa_chain
            
            # Second local improvement pass
            if time.time() < _DEADLINE - 12:
                improved_chain2, improved_energy2 = _local_improve_fast(sequence, chain, rng, max_moves=3000)
                if improved_energy2 < energy:
                    energy = improved_energy2
                    chain = improved_chain2
=======
            # Local improvement on this candidate
            if time.time() < _DEADLINE - 25:
                improved_chain, improved_energy = _local_improve_fast(sequence, chain, rng, max_moves=6000)
                if improved_energy < energy:
                    energy = improved_energy
                    chain = improved_chain
            
            # Min-conflicts search
            if time.time() < _DEADLINE - 22:
                mc_chain, mc_energy = _min_conflicts_search(sequence, chain, rng, max_iters=2500)
                if mc_energy < energy:
                    energy = mc_energy
                    chain = mc_chain
            
            # SA on the improved candidate
            if time.time() < _DEADLINE - 18:
                sa_chain, sa_energy = _regrow_sa_fast(sequence, chain, rng, max_iters=6000, start_temp=7.0, cool_rate=0.994)
                if sa_energy < energy:
                    energy = sa_energy
                    chain = sa_chain
            
            # Second local improvement pass
            if time.time() < _DEADLINE - 12:
                improved_chain2, improved_energy2 = _local_improve_fast(sequence, chain, rng, max_moves=4000)
                if improved_energy2 < energy:
                    energy = improved_energy2
                    chain = improved_chain2
>>>>>>> REPLACE

