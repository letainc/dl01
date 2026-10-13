# Approved professional KG mastery expansion — coverage overlay

**Approved teaching depth and scope:** 2026-10-12. **Status:** expansion plan only, not learner mastery or a new K-series numbering.  
**Authority:** [original KG-01–KG-07 syllabus](module-knowledge-graphs-graph-ai.md) and [KG-M01–KG-M14 math dependency registry](knowledge-graphs-mathematics.md).

The original teaching sequence and previously demonstrated work take precedence. This overlay identifies comprehensive senior data engineering/data science competencies, to be integrated at appropriate existing gates and only then assigned further stable lesson IDs.

| Coverage cluster | Current course bridge | Mathematics and reasoning | Practical evidence required |
|---|---|---|---|
| KG identity and representation | KG-01, KG-03 | sets, relations, directed paths, semantics, RDF and LPG assumptions | diagram, triples, model comparison, queries |
| Graph theory and search | KG-02 | degrees, walks/trails/paths/cycles, reachability, sparse matrices, BFS/DFS, complexity | manual matrix arithmetic, original traversal code |
| Formal semantics and ontologies | KG-03, KG-04 | predicate logic, entailment, schemas, OWL, SHACL, open world | model constraints, detect invalid inference |
| Query processing | KG-03, KG-07 | SPARQL/Cypher, graph matching, join equivalence, query cost | correct queries, plans and latency comparisons |
| Construction and integration | KG-04 | ID resolution, quality, confidence, provenance, timestamps, R2RML | reproducible relational/text-to-KG ingestion |
| Feature engineering | KG-05 | structural/path/semantic/temporal features, aggregation, leakage | predictive baselines and feature ablations |
| Statistical ML and evaluation | KG-05 | losses, likelihood, sampling, splits, calibration, MRR, Hits@K | clean held-out benchmarks and error analysis |
| Node and KG embeddings | KG-05 | vector geometry, matrix factorization, skip-gram, TransE, DistMult, ComplEx, RotatE | implement and train link predictors, compare limitations |
| Deep learning / GNN | KG-06 | gradients, message passing, graph Laplacian, normalization, invariance | manual layer calculation; PyTorch experiments |
| Heterogeneous relational GNN | KG-06, KG-07 | relation-specific aggregation, R-GCN, GraphSAGE, graph attention | train with baselines and ablations |
| Advanced GNN theory | KG-07 and advanced extensions | Weisfeiler–Lehman, oversmoothing, oversquashing, graph transformers | investigate failures and model complexity |
| Reasoning and uncertainty | KG-04, KG-07 and advanced | symbolic rules, probabilistic reasoning, neuro-symbolic trade-offs | compare inferred, observed and predicted claims |
| Temporal and dynamic KG | KG-04, KG-07 and advanced | event-time vs knowledge-time, temporal sampling, chronological splits | build temporal no-leakage evaluation |
| Graph retrieval / LLM integration | KG-07 and advanced | graph grounding, semantic retrieval, citation quality | evidence-based retrieval experiment |
| Scale and platform engineering | KG-07 and advanced | partitioning, sparse cost, high-degree skew, query planning, reliability | reproducible benchmarking and review |
| SAP-relevant architecture | KG-07 and advanced | semantic vs property graphs, integration, authorization, provenance | public/synthetic enterprise design review |
| Capstone / ML operations | advanced after gates | full pipeline correctness, experiment tracking, drift, costs, controls | executable, tested and documented end-to-end study |

**Stanford primary:** CS520 for KG engineering and semantics; CS224W for graph ML/embeddings/GNNs; CS229 for prerequisite mathematics/optimization. Use complete relevant official lectures, notes, recommended textbooks and research papers, assignments/Colabs where accessible—not slide-only summaries. Historic supplementary resources in the original syllabus remain reference material, but do not displace Stanford.

**Mastery gate:** explicit evidence at Explain, Apply, Transfer/Design, mathematical calculation, implementation and failure analysis. The original repository rubric requires robust Apply and Design/Debug ratings 4–5/5 to claim senior mastery. Do not convert prior 'prepared' material into a score.
