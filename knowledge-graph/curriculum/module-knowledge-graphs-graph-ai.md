# Knowledge Graphs, Graph Machine Learning, and Enterprise Graph AI — Mastery Track

> **Status:** active parallel learning priority, started 2026-10-08. Course materials prepared; learner mastery **not yet assessed**.
> **Prerequisite:** none for graph foundations. Reuse the existing Linear Algebra, Probability, CS229, and Feature Engineering lessons as just-in-time connections.
> **Primary intent:** deep competence discussing SAP-team large graph model architecture, graph AI, and production data/ML systems. Specific team implementation is **not** assumed or documented here.
> **Existing-track preservation:** this syllabus supplements—not replaces—`curriculum/roadmap.md` and `feature-engineering/curriculum.md`. Do not alter unassessed/unfinished states on those tracks.
> **Coordinator-only conversation:** this chat manages planning and repo state; deep teaching occurs in fresh separate sessions. See `curriculum/knowledge-graphs-session-coordinator.md`.
> **Mathematics is a first-class mandatory curriculum:** see the 14-part dependency map and derivation gates in `curriculum/knowledge-graphs-mathematics.md`. Each required topic must be independently understood, not just covered.

## 1. Learning contract

Teach one concept at a time in this progression:

**Intuition → Visualization → Practical example → Mathematics (manual) → ML/Feature Engineering connections → Engineering implementation → Limitations/failure modes → Advanced concepts.**

For every equation: read it aloud, define each symbol, use tiny numbers, translate to a diagram and Python/SQL, and connect it to the learning objective. Do not front-load notation before the learner can describe the relationship intuitively. Follow `feature-engineering/retention/CS229_LEARNING_INSTRUCTIONS.md` and `feature-engineering/retention/mathematical-literacy-formula-reading-protocol.md`.

**Mandatory mathematics depth:** discrete mathematics, graph theory/algorithms, graph linear algebra and spectral foundations, probability and random walks, statistics and sampling, logic and inference, embedding geometry, calculus/matrix derivatives, optimization and losses, GNN message passing, attention/transformer mathematics, information theory, and large-scale complexity. For each important equation: explain the need, define *every* symbol/operator and assumption, derive intermediate steps, calculate a small numerical example by hand, visualize the result, connect to existing ML/Feature Engineering, implement and test failure cases. Check mathematical readiness by independent calculation, derivation, reasoning and transfer. A source reading is not a mastery certificate; a definition does not demand a fabricated proof. Detailed math IDs KG-M01–KG-M14 are in `curriculum/knowledge-graphs-mathematics.md`.

**Separate-session policy:** this coordinator conversation handles syllabus, academic sources, prerequisites, assignments, durable notes and concise kickoffs; it does not conduct lessons. Each new study chat uses the kickoff in `curriculum/knowledge-graphs-session-coordinator.md` and updates verifiable KG-only progress.

Practice contract:
1. Predict the behavior of a small graph before calculating/querying.
2. Draw it, enumerate nodes and typed/directed edges, and trace a path by hand.
3. Calculate small examples manually, then check with code.
4. Learner writes core exercise code; tutor reviews, hints, and diagnoses before showing reference solutions.
5. Transfer a concept into a different domain (finance, enterprise master data, supply chain, documents).
6. Break the example intentionally: duplicate identities, missing edges, reversed directions, invalid timestamps, high-degree hub, or leakage.
7. Evaluate Apply and Transfer/Design against `assessments/rubric.md`, not against memorized vocabulary.

Do not conflate:
- graph theory: mathematical structures and algorithms;
- knowledge graph: modeled entities and semantically meaningful relations;
- graph database: a persistence/query engine (may hold a KG, but not required);
- RDF/SPARQL: a semantic-graph data model and query language;
- property graph/Cypher: another modeling/query approach;
- graph analytics: deterministic computations over structure;
- KG embeddings: learned numerical representations;
- GNNs: message-passing/graph-structured learned models (do not require a KG);
- graph foundation models: pretrained graph models aiming to transfer across tasks/domains, with varying research definitions;
- GraphRAG: a retrieval-generation design using graphs; not automatically a GNN or graph foundation model;
- relational foundation models / RPT: potentially related enterprise representation learning, but **not automatically identical** to a KG or graph neural network. Require architecture evidence before making a classification.

## 2. Source hierarchy

**University anchors**
- Stanford CS520 (Spring 2021): https://web.stanford.edu/class/cs520/ — KG models, schema, construction, inference, interaction, applications.
- MIT 6.042J (Fall 2010) readings, Ch. 5 Graph Theory and Ch. 6 Directed Graphs: https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-fall-2010/pages/readings/ — discrete mathematics and graph algorithms.
- Stanford CS224W (Fall 2026 as available; use archived public materials otherwise): https://web.stanford.edu/class/cs224w/ — node embeddings, GNNs, knowledge graphs, heterogeneous graphs and graph foundation models.
- MIT 6.7960 Deep Learning (Fall 2024), Lec. 5 Graph Neural Networks: https://ocw.mit.edu/courses/6-7960-deep-learning-fall-2024/resources/mit6_7960f24_lec05_mp4/ — message passing, MLP/CNN connections, limits.

**Primary texts and standards**
- Hogan et al. (2021), `Knowledge Graphs`, free HTML book: https://kgbook.org/ — schema, semantics, construction, inference, access and governance.
- W3C RDF 1.1 Concepts (current W3C Recommendation): https://www.w3.org/TR/rdf11-concepts/
- W3C SPARQL 1.1 Query Language: https://www.w3.org/TR/sparql11-query/
- W3C RDF 1.2 Concepts (candidate standard as of 2026-10-08): https://www.w3.org/TR/rdf12-concepts/ — label its maturity accurately when discussed.
- Hamilton, `Graph Representation Learning`: https://www.cs.mcgill.ca/~wlh/grl_book/
- Sanchez-Lengeling et al., `A Gentle Introduction to Graph Neural Networks` (Distill 2021, interactive): https://distill.pub/2021/gnn-intro/

**Hands-on and extension**
- GraphAcademy graph data modeling: https://graphacademy.neo4j.com/courses/modeling-fundamentals
- GraphAcademy Neo4j fundamentals: https://graphacademy.neo4j.com/courses/neo4j-fundamentals
- Wikidata SPARQL sandbox: https://query.wikidata.org/
- PyTorch Geometric documentation: https://pytorch-geometric.readthedocs.io/
- Edge et al., `From Local to Global: A Graph RAG Approach to Query-Focused Summarization`, 2024: https://arxiv.org/abs/2404.16130

Additional articles must be checked against original papers, official documentation, or standards; distinguish research proposals from production recommendations. See `references/knowledge-graphs-academic-sources.md`.

## 3. Initial sessions (flexible pacing, mastery-gated)

| Session | Topic and core questions | Manual artifact / lab | Unlock / bridge |
|---|---|---|---|
| KG-01 | What is a graph? Entities, edges, types, directions, attributes, paths. When is a graph also a knowledge graph? | Draw 4–6 entities, write triples, count edges/paths, compute 4×4 adjacency matrix | SQL keys and relationships; ML feature identity |
| KG-02 | Graph mathematics, reachability, degree, walks, cycles, BFS/DFS, sparse adjacency, complexity | Trace traversals and compute adjacency powers by hand; verify with Python | matrices, sparse vectors, relational join paths |
| KG-03 | KG semantics, schema/ontology, identity, RDF vs property graphs, SPARQL vs Cypher, SQL mapping | Model one enterprise domain twice; query equivalent relationship paths | relational modeling; feature-engineering lesson 13 |
| KG-04 | Construction, entity extraction/linking, deduplication, provenance, uncertainty, rules and inference | Build extraction→identity→edge-quality workflow and diagnose a bad merge | probabilistic evidence, data quality, leakage |
| KG-05 | Graph statistics, features, node/KG embeddings, relation types, negative sampling, link prediction | Degree/common-neighbor baselines; manual vector arithmetic; train/test edge splits | feature engineering lessons 09, 13, 18; CS229 loss/optimization |
| KG-06 | GNN message passing and aggregation; GCN, GraphSAGE, relational GNNs; expressivity | One-layer neighbor aggregation with tiny matrices; implement manually first | linear algebra, gradients, backprop, permutation invariance |
| KG-07 | Heterogeneous/large graph models, retrieval, graph transformers/foundation models, graph-native AI and production | Evidence-grounded design review; scale, data lineage, query budget and security checklist | enterprise systems, GraphRAG, SAP architecture discussion |

**Beyond initial seven sessions:** deliberately revisit symbolic inference (RDFS/OWL, rule systems), probabilistic KGs, graph algorithms, embeddings with relation-aware loss functions (e.g., TransE and R-GCN), distribution/partitioning, temporal and multimodal graphs, dynamic representations, robust evaluation, and responsible use.

The session number is a learning order, **not** proof of mastery. Some topics may require multiple days.

## SAP Enterprise LGM onboarding companion (public research only)

A second **applied onboarding/research reading queue** complements (but does not replace) Stanford CS520/CS224W, MIT 6.042J/6.7960 and our foundational KG mathematics. See the curated [public-paper and restricted-source register](../../references/sap-lgm-public-research-reading-plan.md). It separates five priority onboarding reading slots, technical background, optional research and restricted collaboration collections. **All remain pending**. The recommended first public applied papers are RelBench (2024) then the graph-centric relational Griffin (2025); the similarly named 2024 recurrent language-model Griffin is an unrelated work.

Never infer the SAP team's deployed architecture from a public paper or copy internal documents/links into a LetA-Tech repository without explicit organizational authorization. Internal sources must be handled only in a storage environment approved by SAP. Study maths and graph foundations at the same time as occasional contextual readings; do not skip mastery gates to accelerate an onboarding checklist.

## 4. Milestone gates and outcomes

**Gate A — graph literacy:** Given a paragraph, independently identify entities vs literals; types; edge direction; properties; one-to-many cardinality; and a multi-hop query. Reproduce an adjacency matrix and a path manually.

**Gate B — data modeling:** Explain RDF vs property graphs without false equivalence, sketch a minimally useful schema/ontology, discuss identity resolution, and demonstrate equivalent query intent in SQL and graph languages.

**Gate C — graph ML:** Formulate graph learning inputs/targets/evaluation; identify train/test leakage from neighborhood, shared entities or future edges; compute simple embeddings/aggregations; and justify a baseline against GNNs.

**Gate D — senior production architecture:** Defend choices in storage, indexing, adjacency representation, distribution/partitioning, update semantics, consistency, hot-spot hubs, monitoring, authorization, provenance, cost and rollback. Include when a traditional relational solution is better.

**Gate E — SAP-team technical conversation:** Ask informed architecture questions without presuming internal implementation:
- What constitutes a node and edge in this system: records, business entities, relations, events, documents?
- Is the graph physical storage, a logical semantic layer, a feature graph, a GNN computation graph, or some combination?
- What data models and query languages are in play: RDF/OWL, LPG/GQL/Cypher, relational foreign keys, bespoke tensors?
- How are identity resolution, cardinality, provenance, temporal validity and authorization handled?
- What does “large graph model” mean for this team: graph database, representation learner, graph foundation model, or distributed enterprise graph?
- Which tasks actually require learned representations versus deterministic graph queries?
- How are scale, updates, evaluation and explainability measured?

## 5. Explicit prior-curriculum bridges

| Knowledge graph topic | Existing glitch-ml source | Shared mechanism and essential caution |
|---|---|---|
| Nodes/entities versus row IDs | `feature-engineering/chapters/13-relational-feature-engineering-pk-fk-context-and-fan-out.md` | business identity is not just PK; join/path multiplicity and fan-out |
| Enterprise relation representations | `feature-engineering/chapters/14-relational-foundation-model-rpt-era-representation.md` | preserving relations versus flattening; cannot assume RPT is a conventional KG |
| Graph embedding vectors | `feature-engineering/chapters/09-categorical-embeddings-and-learned-representations.md` | embeddings carry learned geometry; avoid arbitrary ID numeric order |
| Text→entities/graph | `feature-engineering/chapters/17-text-metadata-and-lexical-features.md` and `18-bag-of-words-n-grams-tf-idf-hashing-and-embeddings.md` | information extraction, lexical/dense signals, provenance and temporal leakage |
| Matrix and graph representations | `feature-engineering/retention/linear-algebra-for-feature-engineering.md` | adjacency, sparse operators and neighborhood aggregation |
| Training and evaluation | `feature-engineering/retention/CS229_LEARNING_INSTRUCTIONS.md` and `feature-engineering/chapters/21-evaluating-whether-a-feature-actually-helps.md` | loss functions, valid splits, baselines, meaningful ablations |
| Time/version governance | `feature-engineering/chapters/12-stateful-features-late-events-snapshots-and-backfills.md` | timestamp-aware edges, known-at-time graph snapshots, late changes |

## 6. Learning artifacts and durable workflow

- Syllabus (this file): stable map, not a session log.
- Lesson note per concept: `notebooks/Knowledge-Graphs-01-What-Is-a-Graph.md`; subsequent notes increment numerically.
- Small deliberate exercises: `exercises/knowledge-graphs-01-foundations.md`; answers reviewed only after learner attempts.
- Sources and literature: `references/knowledge-graphs-academic-sources.md`.
- Mandatory mathematical foundations: `curriculum/knowledge-graphs-mathematics.md` (14 dependency areas and worked-calculation gates).
- Coordinator role, new-session kickoff and post-session reporting: `curriculum/knowledge-graphs-session-coordinator.md`.
- Stable learning state: curated section in `tracker/learning-state.md`.
- Module statuses: `tracker/progress.md`.
- Next action: a scoped section in `tracker/backlog.md`.
- Session history: append-only dated `sessions/` file, per `CLAUDE.md`.

**At the end of every session:** only update KG-specific tracking fields and append the dated history. Do not silently claim cross-track mastery, re-grade other curricula or replace their order. Keep the learner's misconceptions and corrected calculations as durable material after working through them.

**Current handoff:** Start KG-01 with the diagram and four-node example. Have the learner answer Exercise 1 (entities, edges, direction) in their own words **before** scoring or moving forward to KG-02.
