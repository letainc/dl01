# Knowledge Graphs & Large Graph Models — Mathematical Dependency Map

> **Purpose:** mathematical foundations are mandatory in the Knowledge Graph/Graph AI track, not optional enrichment.
> **State (2026-10-09):** curriculum mapped; no mathematical competencies in the new track assessed.
> **Authority:** topical syllabus: `curriculum/syllabus/module-knowledge-graphs-graph-ai.md`; evidenced learner progress: `tracker/learning-state.md`.
> **Pedagogy:** comprehensive mathematics, taught progressively and just in time. Never presume readiness merely because a symbol was used in another ML lesson. Apply `feature-engineering/retention/mathematical-literacy-formula-reading-protocol.md` and `CS229_LEARNING_INSTRUCTIONS.md`.

## Required mathematical learning and explanation protocol

Every **important equation** in each study lesson must address:

1. **Question and intuition:** what problem motivates this equation, and why do we need it?
2. **Visual interpretation:** graph, geometry, distribution, gradient or computational flow.
3. **Reading aloud:** complete natural-language translation of the notation.
4. **Full symbol dictionary:** every variable, index, set, operator, superscript, summation limit, transpose, norm, inverse and assumption; include dimensions and units where meaningful.
5. **Expanded operations:** show evaluation order, intermediate steps and what the computation does.
6. **Derivation:** reason from definitions or established results; no opaque “it follows”; distinguish definition from identity, theorem, objective, model design and approximation.
7. **Manual worked example:** use 2–4 nodes, 2D vectors, a tiny probability table or a scalar derivative, and calculate intermediate numeric values.
8. **ML and Feature Engineering bridge:** connect to vectors, matrix multiplication, probability, statistics, gradients, backpropagation, loss functions, SQL relations and feature correctness.
9. **Implementation:** learner predicts and attempts Python/NumPy or query code; verify the manual result later. Do not immediately supply full answers to learner-authored exercises.
10. **Sanity check and a failure mode:** verify units/shapes, edge cases, assumptions, numerical stability, misleading interpretations and counterexamples.
11. **Transfer check:** derive or compute the same mechanism on a different tiny graph/task.
12. **Durable notes:** record misconceptions, corrections, derivations and evidence of understanding in the lesson; update only KG tracking after actual assessment.

A formula being remembered is **not mastery**. The readiness bar is correct derivation at appropriate rigor, independent manual calculation, explanation of assumptions and at least one transfer/application. A definition needs motivation and consequences, not an invented proof; a theorem needs its conditions.

## Dependency registry

Use states `PLANNED`, `OPEN`, `IN_STUDY`, `READY`, `REINFORCE`. Every module below begins `PLANNED`. Do not mark `READY` without evidence from actual learner work.

| ID | Math area | Skills and representative equations to derive or calculate manually | Why it matters | Bridge |
|---|---|---|---|---|
| KG-M01 | Mathematical language and discrete foundations | Sets, membership, ordered pairs, cardinality, Cartesian product, relations, functions, finite counting, sums/products, logic, quantifiers, proofs and counterexamples | Semantic triples, graph definitions, constraints and relation algebra | Existing mathematical-literacy protocol |
| KG-M02 | Graph theory | `G=(V,E)`, degree, direction, paths vs walks, cycles, connectivity, multigraphs, bipartite graphs; undirected handshaking identity `Σᵥ degree(v)=2|E|` under its assumptions | Precise graph representation and graph algorithms | PK/FK relationships; FE lesson 13 |
| KG-M03 | Discrete algorithms and complexity | BFS, DFS, trees, shortest paths, connected components, topological order, adjacency-list vs dense complexity, `O(|V|+|E|)` traversal bound under adjacency-list assumptions | Traversal, multi-hop queries and scale | SQL joins, data structures |
| KG-M04 | Linear algebra of graphs | Vectors, dot products, norms, matrix multiplication, transposes, permutations, adjacency `A`, degree matrix `D`, incidence; `(A²)ᵢⱼ` and walk-count interpretation; sparse multiplication | GNN tensors, graph paths, numerical representations | Linear Algebra retention, CS229 |
| KG-M05 | Spectral and numerical graph math | Eigenvalues, eigenvectors, Laplacian `L=D−A`, `xᵀLx=Σ_{undirected edges {i,j}}(xᵢ−xⱼ)²` for unweighted simple undirected graphs; normalized operators and degree caveats | Spectral graph analysis, GCN, partitioning | PCA, numerical conditioning |
| KG-M06 | Probability and stochastic graphs | Conditional probability, Bayes, expectation, variance, independent events, random walks, transition matrices `Pᵢⱼ=Aᵢⱼ/dᵢ` for nonzero out-degree; tiny Markov chain calculations | Random walks, graph inference, uncertainty | CS229 probability / Bayes |
| KG-M07 | Statistics and evaluation | Bernoulli/categorical models, likelihood/log likelihood, sampling, estimators, confidence/uncertainty, calibration, edge-split design, ranking metrics and selection bias | Link prediction, negative sampling and valid benchmarking | FE lesson 21, leakage protocols |
| KG-M08 | Logic and KG reasoning | Propositional/predicate logic, quantifiers, implication, constraints, open vs closed world, entailment, RDF/OWL rule interpretation and careful proof conditions | Ontology, schema semantics and symbolic inference | Data contracts, categorical semantics |
| KG-M09 | Geometry and graph embeddings | Euclidean distance `||u−v||₂`, inner products `uᵀv`, cosine similarity `(uᵀv)/||u||||v||`, normalization and relational scoring | Entity and relation embeddings, retrieval | FE lessons 09, 18; vectors |
| KG-M10 | Optimization, calculus, losses | Partial derivatives, chain rule, Jacobian intuition, sigmoid/softmax, cross-entropy, margin objectives, gradients, SGD update `θₜ₊₁=θₜ−η∇θL(θₜ)`, one-step manual optimization | Graph representation learning, link prediction and parameter updates | CS229, gradient descent, backprop |
| KG-M11 | GNN message passing | Neighbor sets, aggregation, degree normalization, matrix multiplication, permutation invariance/equivariance, graph-layer computation and gradients; illustrative `hᵥ⁽ᵏ⁺¹⁾=σ(W·AGG({hᵤ⁽ᵏ⁾:u∈N(v)}))` | GCN, GraphSAGE, relational GNNs | Neural nets, aggregation features, chain rule |
| KG-M12 | Large-model attention math | Query/key/value projection, softmax `eᶻⁱ/Σⱼeᶻʲ`, scaled dot-product attention, structural/positional encodings, computational and memory complexity | Graph transformers, graph foundation models | Attention and numerical stability |
| KG-M13 | Information theory and uncertain KGs | Entropy `H(p)=−Σᵢpᵢlog pᵢ`, cross-entropy, KL divergence, mutual information, contrastive objectives and probabilistic edge confidence | Graph objectives, uncertain knowledge and evaluation | FE entropy, probability, statistics |
| KG-M14 | Large-scale graph systems math | Asymptotic complexity, sparse memory, power-law degree distributions with caveats, sampling variance, partitioning, neighborhood expansion, approximation error | Billion-edge data, serving, training, operational trade-offs | Distributed DE, Ray, sparse FE |

**Advanced branches as demanded by models/papers:** graph signal processing, semirings/query algebra, Weisfeiler–Lehman expressivity, matrix concentration and random graph models, Bayesian graphical models (distinct from KGs), information geometry, dynamic and temporal graphs, hypergraphs, causal inference on relational data, attention complexity, and graph privacy.

## Lesson-by-lesson mathematical gates

| Topical lesson | Required math areas | What must be independently demonstrated |
|---|---|---|
| KG-01 fundamentals | M01, early M02, adjacency definition from M04 | Parse `G=(V,E)`; enumerate entities/relations; correctly write a directed 4x4 adjacency matrix |
| KG-02 graph algorithms | M02–M04 | Degree calculation, trace BFS/DFS, paths vs walks, multiply a small adjacency matrix |
| KG-03 data models and semantics | M01–M02, M08 | Write logical relations, state inference assumptions and translate graph patterns to SQL/RDF/property graph |
| KG-04 construction and inference | M06–M08 | Reason about uncertainty, source claims, identity and rule entailment with simple examples |
| KG-05 graph features and embeddings | M04, M06–M07, M09–M10, optional M13 | Calculate structural features and vector scores; derive/evaluate one loss and optimizer step |
| KG-06 neural graph models | M04–M05, M09–M11 | Manually compute a GNN aggregation layer, trace gradients and identify invariant/variant operations |
| KG-07 large graph AI | M03, M05–M07, M10–M14 as actually needed | Manually compute small attention; justify memory/complexity and sampling/evaluation trade-offs |

Sessions may require several sub-lessons. A topic milestone is not a deadline and **a mathematical prerequisite cannot be bypassed to meet a calendar goal**. When a gap blocks progress, name the exact math ID, open a focused micro-session, demonstrate readiness, then return to the original graph lesson instead of restarting the entire math curriculum.

## Academic math anchors and in-repository support

- MIT Mathematics for Computer Science (6.042J): https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-fall-2010/
- Stanford CS520 Knowledge Graphs: https://web.stanford.edu/class/cs520/
- Stanford CS224W Machine Learning with Graphs: https://web.stanford.edu/class/cs224w/
- MIT 6.7960 Deep Learning: https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/
- Hamilton, Graph Representation Learning: https://www.cs.mcgill.ca/~wlh/grl_book/
- Hogan et al., Knowledge Graphs: https://kgbook.org/
- W3C RDF and OWL: https://www.w3.org/TR/rdf11-concepts/ and https://www.w3.org/TR/owl2-overview/
- Existing notes: `feature-engineering/retention/linear-algebra-for-feature-engineering.md`, `feature-engineering/retention/calculus-for-feature-engineering.md`, `feature-engineering/retention/CS229_LEARNING_INSTRUCTIONS.md`, `feature-engineering/math-dependencies.md`.

Primary courses, standards, original research and carefully checked additional explanations may all be used; no single course is sufficient coverage of the full math stack.
