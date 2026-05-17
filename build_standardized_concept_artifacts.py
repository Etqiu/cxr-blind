"""
Build standardized concept-interpretability summaries and figures from the
downloaded IU blind-pair artifacts.

This script keeps the existing raw-score outputs intact and adds standardized
views where each concept delta is divided by the global standard deviation of
that concept's raw score distribution.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent / "modal_downloads" / "indiana_artifacts"
IMAGE_BASE = BASE / "representative_images"
TYPES = ["Type 1", "Type 2", "Type 3"]


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    concept_df = pd.read_csv(BASE / "blind_pair_concepts.csv")
    blind_pairs_df = pd.read_csv(BASE / "blind_pairs_analysis.csv")
    with (BASE / "concept_scores.pkl").open("rb") as file_obj:
        score_cache = pickle.load(file_obj)
    return concept_df, blind_pairs_df, score_cache


def concept_scale_map(score_cache: dict[str, object]) -> dict[str, float]:
    labels = list(score_cache["pathology_cols"])
    raw_scores = np.asarray(score_cache["raw_scores"], dtype=np.float32)
    stds = raw_scores.std(axis=0)
    stds = np.where(stds < 1e-8, 1.0, stds)
    return {label: float(std) for label, std in zip(labels, stds)}


def add_standardized_columns(concept_df: pd.DataFrame, scale_map: dict[str, float]) -> pd.DataFrame:
    df = concept_df.copy()

    standardized_delta_dicts = []
    standardized_top_lists = []
    max_abs_standardized = []

    for row in df.itertuples(index=False):
        raw_deltas = json.loads(row.concept_score_deltas)
        standardized = {
            concept: round(float(delta) / scale_map[concept], 6)
            for concept, delta in raw_deltas.items()
        }
        ranked = sorted(standardized.items(), key=lambda item: abs(item[1]), reverse=True)

        query_scores = json.loads(row.query_concept_scores)
        neighbor_scores = json.loads(row.neighbor_concept_scores)
        query_probs = json.loads(row.query_concept_probabilities)
        neighbor_probs = json.loads(row.neighbor_concept_probabilities)

        top_items = []
        for concept, std_delta in ranked[:5]:
            top_items.append(
                {
                    "concept": concept,
                    "standardized_delta": round(float(std_delta), 6),
                    "raw_score_delta": round(float(raw_deltas[concept]), 6),
                    "query_score": round(float(query_scores[concept]), 6),
                    "neighbor_score": round(float(neighbor_scores[concept]), 6),
                    "query_standardized_score": round(float(query_scores[concept]) / scale_map[concept], 6),
                    "neighbor_standardized_score": round(float(neighbor_scores[concept]) / scale_map[concept], 6),
                    "query_probability": round(float(query_probs[concept]), 6),
                    "neighbor_probability": round(float(neighbor_probs[concept]), 6),
                }
            )

        standardized_delta_dicts.append(json.dumps(standardized))
        standardized_top_lists.append(json.dumps(top_items))
        max_abs_standardized.append(abs(top_items[0]["standardized_delta"]) if top_items else 0.0)

    df["concept_standardized_deltas"] = standardized_delta_dicts
    df["top_standardized_deltas"] = standardized_top_lists
    df["max_abs_standardized_delta"] = max_abs_standardized
    return df


def summarize_standardized(
    concept_df: pd.DataFrame,
    blind_pairs_df: pd.DataFrame,
    labels: list[str],
) -> pd.DataFrame:
    pair_types = blind_pairs_df[["pair_id", "blind_type"]].drop_duplicates()
    merged = concept_df[["pair_id", "concept_standardized_deltas", "top_standardized_deltas"]].merge(
        pair_types,
        on="pair_id",
        how="left",
    )
    merged = merged.dropna(subset=["blind_type"]).reset_index(drop=True)

    rows = []
    for blind_type, group in merged.groupby("blind_type"):
        pair_count = len(group)
        delta_matrix = np.zeros((pair_count, len(labels)), dtype=np.float32)
        top1_counts = {label: 0 for label in labels}
        top5_counts = {label: 0 for label in labels}

        for row_index, row in enumerate(group.itertuples(index=False)):
            deltas = json.loads(row.concept_standardized_deltas)
            delta_matrix[row_index] = [float(deltas.get(label, 0.0)) for label in labels]

            top_items = json.loads(row.top_standardized_deltas)
            if top_items:
                top1_counts[top_items[0]["concept"]] += 1
                for item in top_items:
                    top5_counts[item["concept"]] += 1

        mean_abs = np.abs(delta_matrix).mean(axis=0)
        mean_signed = delta_matrix.mean(axis=0)

        for label_index, label in enumerate(labels):
            rows.append(
                {
                    "blind_type": blind_type,
                    "concept": label,
                    "mean_abs_standardized_delta": round(float(mean_abs[label_index]), 6),
                    "mean_signed_standardized_delta": round(float(mean_signed[label_index]), 6),
                    "top1_frequency_pct": round(float(top1_counts[label] / pair_count * 100.0), 3),
                    "top5_frequency_pct": round(float(top5_counts[label] / pair_count * 100.0), 3),
                    "pair_count": pair_count,
                }
            )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(BASE / "concept_standardized_summary.csv", index=False)
    return summary_df


def build_global_leaderboard(concept_df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    top1_counts = {label: 0 for label in labels}
    top5_counts = {label: 0 for label in labels}
    values = {label: [] for label in labels}

    for row in concept_df.itertuples(index=False):
        deltas = json.loads(row.concept_standardized_deltas)
        top_items = json.loads(row.top_standardized_deltas)
        for label in labels:
            values[label].append(float(deltas.get(label, 0.0)))
        if top_items:
            top1_counts[top_items[0]["concept"]] += 1
            for item in top_items:
                top5_counts[item["concept"]] += 1

    pair_count = len(concept_df)
    rows = []
    for label in labels:
        arr = np.asarray(values[label], dtype=np.float32)
        rows.append(
            {
                "concept": label,
                "overall_mean_abs_standardized_delta": round(float(np.abs(arr).mean()), 6),
                "overall_mean_signed_standardized_delta": round(float(arr.mean()), 6),
                "overall_top1_frequency_pct": round(float(top1_counts[label] / pair_count * 100.0), 3),
                "overall_top5_frequency_pct": round(float(top5_counts[label] / pair_count * 100.0), 3),
                "pair_count": pair_count,
            }
        )

    leaderboard_df = pd.DataFrame(rows).sort_values(
        ["overall_mean_abs_standardized_delta", "overall_top1_frequency_pct"],
        ascending=False,
    )
    leaderboard_df.to_csv(BASE / "concept_standardized_global_leaderboard.csv", index=False)
    return leaderboard_df


def make_dashboard(
    summary_df: pd.DataFrame,
    leaderboard_df: pd.DataFrame,
    blind_pairs_df: pd.DataFrame,
    labels: list[str],
) -> None:
    plt.rcParams.update({"font.size": 10})
    fig = plt.figure(figsize=(16, 12), facecolor="#f8fafc")
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.35], width_ratios=[0.95, 1.25], hspace=0.28, wspace=0.26)

    counts = blind_pairs_df["blind_type"].value_counts()

    ax1 = fig.add_subplot(grid[0, 0])
    values = [int(counts.get(kind, 0)) for kind in TYPES]
    colors = ["#4f46e5", "#f59e0b", "#dc2626"]
    bars = ax1.bar(TYPES, values, color=colors, width=0.6)
    for bar, value in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}", ha="center", va="bottom", fontweight="bold")
    ax1.set_title("Blind Pair Counts", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Number of pairs")
    ax1.spines[["top", "right"]].set_visible(False)

    ax2 = fig.add_subplot(grid[0, 1])
    heatmap_df = summary_df.pivot(index="concept", columns="blind_type", values="mean_abs_standardized_delta").fillna(0.0)
    heatmap_df = heatmap_df.reindex(index=labels, columns=TYPES, fill_value=0.0)
    image = ax2.imshow(heatmap_df.values, cmap="YlOrRd", aspect="auto")
    ax2.set_xticks(np.arange(len(TYPES)))
    ax2.set_xticklabels(TYPES)
    ax2.set_yticks(np.arange(len(labels)))
    ax2.set_yticklabels(labels)
    ax2.set_title("Mean |Standardized Concept Delta| by Blind Type", fontsize=14, fontweight="bold")
    max_val = float(heatmap_df.values.max()) if heatmap_df.size else 0.0
    threshold = max_val / 2.0 if max_val > 0 else 0.0
    for row_index in range(len(labels)):
        for col_index in range(len(TYPES)):
            value = float(heatmap_df.values[row_index, col_index])
            color = "white" if value > threshold else "black"
            ax2.text(col_index, row_index, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8)
    colorbar = fig.colorbar(image, ax=ax2, fraction=0.03, pad=0.02)
    colorbar.set_label("Mean |delta / sigma|")

    ax3 = fig.add_subplot(grid[1, 0])
    plot_df = leaderboard_df.sort_values("overall_mean_abs_standardized_delta", ascending=True)
    ax3.barh(plot_df["concept"], plot_df["overall_mean_abs_standardized_delta"], color="#0f766e")
    ax3.set_title("Overall Standardized Concept Leaderboard", fontsize=14, fontweight="bold")
    ax3.set_xlabel("Mean |standardized delta| across all blind pairs")
    ax3.spines[["top", "right"]].set_visible(False)
    ax3b = ax3.twiny()
    ax3b.scatter(plot_df["overall_top1_frequency_pct"], plot_df["concept"], color="#7c3aed", s=38, zorder=3)
    ax3b.set_xlabel("Top-1 driver frequency (%)")

    ax4 = fig.add_subplot(grid[1, 1])
    ax4.axis("off")
    ax4.set_title("Type-Specific Standardized Signatures", fontsize=14, fontweight="bold", loc="left")
    text_lines = []
    for blind_type in TYPES:
        text_lines.append(blind_type)
        subset = summary_df[summary_df["blind_type"] == blind_type].sort_values("mean_abs_standardized_delta", ascending=False).head(3)
        for rank, row in enumerate(subset.itertuples(index=False), start=1):
            text_lines.append(
                f"  #{rank}: {row.concept}  (|d/s|={row.mean_abs_standardized_delta:.2f}, top1={row.top1_frequency_pct:.1f}%)"
            )
        text_lines.append("")
    ax4.text(0.0, 0.98, "\n".join(text_lines), va="top", ha="left", fontsize=11, family="DejaVu Sans Mono", color="#111827")

    fig.suptitle("IU Blind-Pair Interpretability Summary (Standardized)", fontsize=18, fontweight="bold", x=0.47, y=0.98)
    fig.text(0.5, 0.012, "Standardized delta = raw concept-score delta divided by the global standard deviation of that concept.", ha="center", fontsize=11, color="#374151")
    fig.savefig(BASE / "concept_standardized_dashboard.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_representative_pair_figure(
    concept_df: pd.DataFrame,
    blind_pairs_df: pd.DataFrame,
    scale_map: dict[str, float],
) -> pd.DataFrame:
    pair_meta = blind_pairs_df[["pair_id", "blind_type", "primary_pathology"]].drop_duplicates()
    merged = concept_df.merge(pair_meta, on="pair_id", how="left")

    selected_rows = []
    for blind_type in TYPES:
        subset = merged[merged["blind_type"] == blind_type].copy()
        if subset.empty:
            continue
        subset = subset.sort_values("max_abs_standardized_delta", ascending=False)
        selected_rows.append(subset.iloc[0])

    selected_df = pd.DataFrame(selected_rows)
    selected_df.to_csv(BASE / "representative_blind_pairs_standardized.csv", index=False)

    fig, axes = plt.subplots(
        len(selected_df),
        4,
        figsize=(18, 4.8 * len(selected_df)),
        facecolor="#f8fafc",
        gridspec_kw={"width_ratios": [0.85, 0.85, 1.5, 1.5]},
    )
    if len(selected_df) == 1:
        axes = np.asarray([axes])

    for row_index, row in enumerate(selected_df.itertuples(index=False)):
        top_items = json.loads(row.top_standardized_deltas)[:5]
        concepts = [item["concept"] for item in top_items][::-1]
        query_raw = [item["query_score"] for item in top_items][::-1]
        neighbor_raw = [item["neighbor_score"] for item in top_items][::-1]
        query_std = [item["query_standardized_score"] for item in top_items][::-1]
        neighbor_std = [item["neighbor_standardized_score"] for item in top_items][::-1]

        y = np.arange(len(concepts))
        height = 0.36

        query_image_path = IMAGE_BASE / row.query_filename
        neighbor_image_path = IMAGE_BASE / row.neighbor_filename

        ax_query_img = axes[row_index, 0]
        _plot_image_panel(
            ax_query_img,
            query_image_path,
            title=f"{row.blind_type}: query\n{row.query_filename}",
        )

        ax_neighbor_img = axes[row_index, 1]
        _plot_image_panel(
            ax_neighbor_img,
            neighbor_image_path,
            title=f"{row.blind_type}: neighbor\n{row.neighbor_filename}",
        )

        ax_raw = axes[row_index, 2]
        ax_raw.barh(y - height / 2, query_raw, height, label="Query", color="#2563eb")
        ax_raw.barh(y + height / 2, neighbor_raw, height, label="Neighbor", color="#ea580c")
        ax_raw.axvline(0.0, color="#6b7280", linewidth=1)
        ax_raw.set_yticks(y)
        ax_raw.set_yticklabels(concepts)
        ax_raw.set_title(
            "Raw concept scores",
            fontsize=11,
            fontweight="bold",
        )
        ax_raw.set_xlabel("Raw concept score")
        ax_raw.spines[["top", "right"]].set_visible(False)
        if row_index == 0:
            ax_raw.legend(frameon=False, loc="lower right")

        ax_std = axes[row_index, 3]
        ax_std.barh(y - height / 2, query_std, height, label="Query", color="#2563eb")
        ax_std.barh(y + height / 2, neighbor_std, height, label="Neighbor", color="#ea580c")
        ax_std.axvline(0.0, color="#6b7280", linewidth=1)
        ax_std.set_yticks(y)
        ax_std.set_yticklabels(concepts)
        top_concept = top_items[0]["concept"] if top_items else "NA"
        top_std = top_items[0]["standardized_delta"] if top_items else 0.0
        ax_std.set_title(
            f"{row.blind_type}: standardized scores\nTop driver: {top_concept} (delta/sigma={top_std:.2f})",
            fontsize=11,
            fontweight="bold",
        )
        ax_std.set_xlabel("Standardized concept score")
        ax_std.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Representative Blind Pairs: Images with Raw and Standardized Concept Views", fontsize=16, fontweight="bold", y=0.995)
    fig.text(0.5, 0.005, "Columns: query image, neighbor image, raw CLIP concept scores, and concept scores standardized by each concept's global standard deviation.", ha="center", fontsize=10, color="#374151")
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.98))
    fig.savefig(BASE / "representative_blind_pairs_raw_and_standardized.png", dpi=180)
    plt.close(fig)
    return selected_df


def _plot_image_panel(ax, image_path: Path, title: str) -> None:
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    if image_path.exists():
        image = mpimg.imread(image_path)
        ax.imshow(image, cmap="gray")
    else:
        ax.text(0.5, 0.5, "Image\nnot found", ha="center", va="center", fontsize=11, color="#6b7280")
    for spine in ax.spines.values():
        spine.set_visible(False)


def main() -> None:
    concept_df, blind_pairs_df, score_cache = load_inputs()
    labels = list(score_cache["pathology_cols"])
    scale_map = concept_scale_map(score_cache)
    enriched_df = add_standardized_columns(concept_df, scale_map)
    enriched_df.to_csv(BASE / "blind_pair_concepts_standardized.csv", index=False)

    summary_df = summarize_standardized(enriched_df, blind_pairs_df, labels)
    leaderboard_df = build_global_leaderboard(enriched_df, labels)
    make_dashboard(summary_df, leaderboard_df, blind_pairs_df, labels)
    make_representative_pair_figure(enriched_df, blind_pairs_df, scale_map)

    print("Wrote standardized concept artifacts to", BASE)
    print("\nTop standardized concepts:")
    print(leaderboard_df.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
