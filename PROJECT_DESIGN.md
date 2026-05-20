# Dynamic Routing Project Design

## 1. Overview
We study a **Dynamic TSP** on the Madrid road network (OSM). The tour starts and ends at a fixed depot. Travel times change over time and some nodes can become unavailable. We compare GA, PSO, and ACO on the same dynamic environment.

---

## 2. Goals
- Build a reproducible experimental setup using a real road network.
- Implement and compare **GA**, **PSO**, and **ACO** under dynamic conditions.
- Report performance under identical budgets and update frequency.

---

## 3. Data & Network
- Source: OpenStreetMap via `osmnx`
- Network type: `drive` (road network)
- Place: Madrid, Spain
- Nodes: uniformly sampled from road graph
- Edge weights: `travel_time` (seconds)

---

## 4. Dynamic Model
- Update interval: every `N` iterations (default: 10)
- Dynamics:
  - **Travel time noise**: multiplicative factor in range `[0.9, 1.1]`
  - **Node unavailability**: drop fraction of nodes (e.g., 5%)
- Evaluation:
  - Tour cost = sum of pairwise shortest‑time matrix entries
  - Tour returns to depot

---

## 5. Experiment Setup
- Build road graph once
- Sample `k` nodes
- Compute base shortest‑time matrix
- Dynamic matrix = base matrix * noise at update steps
- Algorithms use same `DynamicRoadTSP` environment

---

## 6. Algorithms (to implement)
### 6.1 GA
- Encoding: permutation of node indices
- Operators: selection, crossover, mutation
- Dynamic handling: remove unavailable nodes + optional reinsertion

### 6.2 PSO
- Encoding: random keys → permutation
- Update via velocity/position
- Dynamic handling: use current matrix on fitness

### 6.3 ACO
- Pheromone matrix over nodes
- Probabilistic construction
- Dynamic handling: evaluate using current matrix each iteration

---

## 7. Metrics
- Best cost over iterations
- Convergence speed
- Stability after dynamic updates
- (Optional) runtime per iteration

---

## 8. Reproducibility
- Fixed random seeds for:
  - Node sampling
  - Dynamic updates
  - Algorithm initialization
- Log configuration + seeds

---

## 9. Open Questions
- How large should `node_count` be for final experiments?
- How aggressive should node unavailability be?
- Should we allow reinsertion of unavailable nodes when they return?

---

## 10. TODO
- [ ] Implement GA
- [ ] Implement PSO
- [ ] Implement ACO
- [ ] Create comparison runner & plots
- [ ] Add experiment logging
