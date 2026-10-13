# Feature Engineering - Senior DE + ML Knowledge Vault

**Status:** approved curriculum; full knowledge-base construction is in progress. Lessons 00-33 now have standalone knowledge chapters. The first-pass knowledge-vault construction is complete; learner mastery remains unassessed.

## Construction sequence

```text
BUILD THE FULL KNOWLEDGE BASE
-> AUDIT IT FOR GAPS AND QUALITY
-> BEGIN DEEP STUDY
-> PRACTICE
-> IMPLEMENT
-> REVIEW AND REFINE
```

The approved syllabus remains authoritative. Lessons are built in its existing order. A prepared chapter is material readiness, not evidence of learner mastery.

## Knowledge chapters

| Lesson | Standalone chapter | Status |
|---|---|---|
| 00 | [What a Feature Actually Is](chapters/00-representation-and-prediction-contracts.md) | prepared |
| 01 | [Prediction Contract, Data-Generating Process, and Leakage](chapters/01-prediction-contract-data-generating-process-and-leakage.md) | prepared |
| 02 | [Numeric Scale and Geometry](chapters/02-numeric-scale-and-geometry.md) | prepared |
| 03 | [Skew, Heavy Tails, Monotonic Transforms, Clipping, and Binning](chapters/03-skew-heavy-tails-monotonic-transforms-clipping-and-binning.md) | prepared |
| 04 | [Missingness Is Information, Not Merely NaN Removal](chapters/04-missingness-is-information-not-merely-nan-removal.md) | prepared |
| 05 | [Outliers, Anomalies, and Extreme-Value Policy](chapters/05-outliers-anomalies-and-extreme-value-policy.md) | prepared |
| 06 | [Categorical Semantics Before Encoding](chapters/06-categorical-semantics-before-encoding.md) | prepared |
| 07 | [One-Hot, Ordinal, Count/Frequency, and Hashing](chapters/07-one-hot-ordinal-count-frequency-and-hashing.md) | prepared |
| 08 | [Target Encoding and Target-Derived Categorical Statistics](chapters/08-target-encoding-and-target-derived-categorical-statistics.md) | prepared |
| 09 | [Categorical Embeddings and Learned Representations](chapters/09-categorical-embeddings-and-learned-representations.md) | prepared |
| 10 | [Datetime and Cyclic Representations](chapters/10-datetime-and-cyclic-representations.md) | prepared |
| 11 | [Lags, Windows, Recency, Frequency, and Trend](chapters/11-lags-windows-recency-frequency-and-trend.md) | prepared |
| 12 | [Stateful Features, Late Events, Snapshots, and Backfills](chapters/12-stateful-features-late-events-snapshots-and-backfills.md) | prepared |
| 13 | [Relational Feature Engineering: PK/FK Context and Fan-Out](chapters/13-relational-feature-engineering-pk-fk-context-and-fan-out.md) | prepared |
| 14 | [Relational Foundation-Model / RPT-Era Representation](chapters/14-relational-foundation-model-rpt-era-representation.md) | prepared |
| 15 | [Ratios, Differences, Crosses, and Conditional Features](chapters/15-ratios-differences-crosses-and-conditional-features.md) | prepared |
| 16 | [Polynomial Features, Splines, Monotonic Bases, and Automatic Interactions](chapters/16-polynomial-features-splines-monotonic-bases-and-automatic-interactions.md) | prepared |
| 17 | [Text Metadata and Lexical Features](chapters/17-text-metadata-and-lexical-features.md) | prepared |
| 18 | [Bag-of-Words, N-Grams, TF-IDF, Hashing, and Embeddings](chapters/18-bag-of-words-n-grams-tf-idf-hashing-and-embeddings.md) | prepared |
| 19 | [Sequence Features](chapters/19-sequence-features.md) | prepared |
| 20 | [Feature Selection: Relevance, Redundancy, Stability, and Cost](chapters/20-feature-selection-relevance-redundancy-stability-and-cost.md) | prepared |
| 21 | [Evaluating Whether a Feature Actually Helps](chapters/21-evaluating-whether-a-feature-actually-helps.md) | prepared |
| 22 | [Linear and Logistic Models](chapters/22-linear-and-logistic-models.md) | prepared |
| 23 | [Trees, Random Forests, XGBoost, LightGBM, and CatBoost](chapters/23-trees-random-forests-xgboost-lightgbm-catboost.md) | prepared |
| 24 | [Neural Networks, Recommendation, and Ranking](chapters/24-neural-networks-recommendation-and-ranking.md) | prepared |
| 25 | [Forecasting Models](chapters/25-forecasting-models.md) | prepared |
| 26 | [Tabular Foundation Models / RPT, LLMs, and Vision](chapters/26-tabular-foundation-models-rpt-llms-and-vision.md) | prepared |
| 27 | [Feature Contracts and Lineage](chapters/27-feature-contracts-and-lineage.md) | prepared |
| 28 | [Offline/Online Parity and Feature Stores](chapters/28-offline-online-parity-and-feature-stores.md) | prepared |
| 29 | [Reproducibility, Backfills, CDC, and Late Data](chapters/29-reproducibility-backfills-cdc-and-late-data.md) | prepared |
| 30 | [Monitoring and Drift](chapters/30-monitoring-and-drift.md) | prepared |
| 31 | [SQL/DuckDB/Polars/PyArrow Feature Computation](chapters/31-sql-duckdb-polars-pyarrow-feature-computation.md) | prepared |
| 32 | [Spark and Ray Data Feature Computation](chapters/32-spark-and-ray-data-feature-computation.md) | prepared |
| 33 | [Approximate and Streaming Features](chapters/33-approximate-and-streaming-features.md) | prepared |

The [detailed syllabus](../curriculum/syllabus/module-feature-engineering.md) remains authoritative. [Curriculum navigation](curriculum.md), [competency map](competency-map.md), [source map](sources.md), and [math registry](math-dependencies.md) retain their roles.

## Quality contract

Each chapter must teach the knowledge directly and stand on its own. Sources are synthesis inputs; provenance belongs in `Extended Reading and References`. Where relevant, chapters should include ontology, mechanism, mathematical/statistical intuition, decision procedures, examples and counterexamples, failure modes, temporal and relational implications, Data Engineering concerns, model-family differences, production contracts, monitoring, and reasoning questions.

`chapters/` holds integrated reading. `notebooks/` holds executable investigation. `src/`, `sql/`, `labs/`, and `tests/` hold implementations and contracts when useful. The progressive playbook is populated only after later study and demonstrated transfer.

## Current next construction action

Run the **full-vault audit** now that Lessons 00-33 all have first-class chapters. Audit the vault against the approved syllabus and quality contract for missing concepts, duplicated or contradictory semantics, stale library/platform behavior, broken cross-lesson references, weak production transfer, terminology drift, and gaps in exercises/failure attacks. Do not populate the progressive playbook yet; learner mastery remains unassessed. After the audit/refinement pass, begin deep study and later Phase K integrated transfer projects.
