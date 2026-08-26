# HistoSeg Contour Transcript Serve App

This directory contains a standalone SciLifeLab Serve Gradio app for transcript analysis on top of contours that HistoSeg has already generated.

It does not run HistoSeg itself. Instead, it expects:

1. a HistoSeg output bundle that already contains contour files such as `structure_1_contour_0.npy`
2. a fresh Xenium `transcript.parquet`

The app then:

1. lists the structures found in the contour bundle
2. lets the user choose one or more structures as the reference contour
3. expands inward and outward from that contour inside the tissue region implied by all uploaded structures
4. computes signed-distance transcript curves for all genes
5. ranks the most spatially variant genes

## Accepted contour input

Upload or mount either:

- a `.zip` exported from a HistoSeg run directory
- a mounted folder that contains the HistoSeg contour files directly

Expected files include at least:

- `structure_*_contour_*.npy`

Optional files that improve labels or provenance:

- `structure_contour_metrics.json`
- `cells_with_structure_partition.parquet`

## Accepted transcript input

The app expects a Xenium-style `transcript.parquet` with columns similar to:

- `feature_name`
- `x_location`
- `y_location`
- optional `qv`
- optional `is_gene`

It auto-detects common alternate column names.

## Output files

Each run writes:

- `selected_structure_context.png`
- `top_spatially_variant_gene_curves.png`
- `top_spatially_variant_gene_heatmap.png`
- `top_gene_<GENE>_overlay.png`
- `gene_spatial_variation_ranking.csv`
- `gene_distance_density_curves_wide.csv`
- `gene_distance_curves_long.csv`
- `distance_bin_summary.csv`
- `params.json`
- a ZIP archive when possible

## Ranking rule

The app ranks genes by an entropy-based distance-profile concentration score:

- flat profiles score low
- genes concentrated in specific inward or outward distance bins score high

This score is saved as `distance_profile_variation_score`.

## Local Docker test

```bash
docker build --platform linux/amd64 -t histoseg-contour-transcript-serve:local .
docker run --rm -it -p 7860:7860 histoseg-contour-transcript-serve:local
```

Then open `http://localhost:7860`.

## SciLifeLab Serve setup

Create a Gradio app and use:

- `Port`: `7860`
- `Image`: `ghcr.io/<your-github-owner>/histoseg-contour-transcript-serve:sha-<commit>`

Mounted-storage mode is recommended for large `transcript.parquet` files.
