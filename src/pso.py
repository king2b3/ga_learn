import pyswarms as ps
import random

swarm_size = 
dim = 
epsilon = 
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

# Call instance of PSO
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options=options)

# Perform optimization
best_cost, best_pos = optimizer.optimize(f, iters=100)

self.repr = {
	"lr": random.uniform(0.001, 0.2),
	"num_layers": random.randrange(2,10),
	"layers": [],
	"loss": random.choice(LOSS)
}