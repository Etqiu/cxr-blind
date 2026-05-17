# Combined CXR Retrieval Pipeline

**Models:** BioMedCLIP (Image), BioMed-RoBERTa (Text)
**Dataset:** IU X-Ray (Full Dataset Evaluation)

## Setup & Execution

### Option A: Running on Modal Serverless GPU (Recommended)
This pipeline is engineered to run end-to-end on a Modal serverless backend with a dynamically allocated GPU.

1. Ensure you have a Modal account and authenticate your CLI:
   ```bash
   pip install modal
   modal setup
   ```
2. Your radiology dataset (containing `indiana_reports.csv`, `indiana_projections.csv`, and an `images/` directory) must be uploaded to a Modal Volume named `radiology-data`.
3. Dispatch the pipeline to the cloud directly from your local terminal:
   ```bash
   modal run combined_pipeline/run_modal.py
   ```
   For the MIMIC subset on the same volume, the entrypoint now supports automatic preparation and concept explanations:
   ```bash
   modal run combined_pipeline/run_modal.py --dataset mimic --prepare auto --interpretability concepts
   ```
4. Analysis artifacts, CSV tables, and visualization plots will be saved to your remote Modal Volume. You can pull them down to your local machine using:
   ```bash
   
   modal volume get radiology-data /archive/modal_artifacts/blind_pair_types.png .
   modal volume get radiology-data /archive/modal_artifacts/missing_entities_grouped_bar.png .
   modal volume get radiology-data /archive/modal_artifacts/missing_entities_heatmap.png .
   modal volume get radiology-data /archive/modal_artifacts/top_missing_entities_by_type.csv .
   ```
   The concept explanation export is written to:
   ```bash
   modal volume get radiology-data /modal_artifacts/blind_pair_concepts.csv .
   ```

### Option B: Running Locally (Without Modal)
If executing natively on your own machine or a dedicated environment (bypassing Modal), ensure you have a CUDA-capable GPU with sufficient VRAM.

1. Install all required dependencies:
   ```bash
   pip install -r combined_pipeline/requirements.txt
   ```
2. Configure your dataset paths. The pipeline defaults to looking at `/mnt/radiology-data/archive`. To override this to point to your local dataset folder, set the environment variable:
   ```bash
   export RADIOLOGY_BASE_PATH="/path/to/your/local/dataset/directory"
   ```
3. Execute the pipeline using the local entrypoint script:
   ```bash
   python -m combined_pipeline.src.run_pipeline
   ```

---

## Theory & Method

**Visual Consensus Definition (Scenario B)**

For each concept $c$ (defined as a **RadGraph entity**, formatted as `"token|label"`), such as `cardiomegaly|observation::definitely present`

Following the idea of visual expectation without a true reference ground truth, we define:

$$consistency(c) = |\{j : c \in E_{neighbor}^{(j)}\}|$$

Where $E_{neighbor}^{(j)}$ represents the set of RadGraph entities extracted from the $j$-th closest visual neighbor report ($k=3$):
- **Consensus Valid:** The concept appears in $\ge 2$ out of 3 visual neighbors **(Visual Consensus)**
- **Consensus Invalid:** The concept appears in only $1$ or $0$ visual neighbors **(Visually Unsupported)**

Let $C$ denote the set of all concepts where $consistency(c) \ge 2$.

**Concept-Level Deviation Definition**

For each sample:

**Extraction:**
- Extract entity set $P_{target}$ from the **target report** using **RadGraph-XL**
- Extract entity set $C$ from the **visual consensus** (top-3 visual neighbors) using **RadGraph-XL**

**Labeling concepts in $P_{target}$:**
- **Supported (TP equivalent):** $concept \in P_{target} \cap C$ (visually expected finding)
- **Unsupported (FP equivalent):** $concept \in P_{target} \setminus C$ (hallucination or over-calling)

- **Labeling concepts in $C$:**
- **Missing (FN equivalent):** $concept \in C \setminus P_{target}$ (omission or missed finding)

**Total RadGraph Deviation:**
$$Deviation = |P_{target} \setminus C| + |C \setminus P_{target}|$$

| Absolute Weirdness ($|Z|$) | Expected RadGraph Deviation Rate |
| :--- | :--- |
| $\sim 0.0$ | Low (High text-image match aligns with visual consensus) |
| $> 1.0$ | Elevated (Potential hallucination or omission presence) |
| $> 2.0$ | High (Severe clinical and semantic mismatch from norm) |

*(Method Validation: By calculating the Pearson/Spearman correlation between the absolute Z-score of the RAG Image-Text similarity difference against the total RadGraph Deviation, we can prove the un-supervised weirdness metric functions as a valid proxy for detecting errors).*

---

## Interpretability Findings

### CLIP-Space Concept Explainer

- We added a blind-pair concept explainer built on the 14 CheXpert pathologies.
- For each pathology, BioMedCLIP text prompts define a concept direction in the shared image-text space, and each blind pair is explained by the query-neighbor concept score deltas along those axes.
- In practice, raw dot-product deltas were hard to read directly, so we standardized each concept delta by that concept's global score standard deviation before summarizing results.
- On IU X-Ray, the strongest standardized blind-pair drivers were typically **Support Devices**, **Lung Lesion**, **No Finding**, **Edema**, and **Enlarged Cardiomediastinum**.

### Limits Of The CLIP-Only View

- The CLIP concept explainer is useful as a **representation-space diagnostic**, but it is still partly circular: CLIP retrieval is being explained using CLIP-derived concept directions.
- Pair-level explanations are often clinically plausible, but they should be interpreted as “what CLIP appears to separate on” rather than a fully model-independent explanation of the retrieval failure.

### CLIP vs RAD-DINO Prototype Comparison

- To reduce that circularity, we also built image-prototype concept directions independently in both CLIP and RAD-DINO spaces using the same CheXpert supervision.
- For each concept, the direction is formed as the mean positive image embedding minus the mean negative image embedding inside the relevant space.
- We then score the same blind pairs in both spaces and compare their top concept drivers directly.
- In this prototype-based comparison, CLIP and RAD-DINO agreed on the same top concept for only about **8.5-9.0%** of blind pairs across Types 1, 2, and 3.
- This suggests that blind-pair explanations are strongly representation-dependent: the same query-neighbor failure can look clinically different depending on whether it is viewed through CLIP or RAD-DINO.

### Useful Artifacts

- `blind_pair_concepts.csv`: pair-level CLIP concept explanations.
- `concept_standardized_dashboard.png`: standardized aggregate concept summary across blind types.
- `representative_blind_pairs_raw_and_standardized.png`: representative blind pairs with thumbnails plus raw and standardized CLIP concept views.
- `representative_blind_pairs_clip_vs_dino_prototypes.png`: the same representative pairs compared across CLIP and RAD-DINO prototype concept spaces.
