# Paper Draft (LaTeX)

## Compile

```bash
cd paper
latexmk -pdf main.tex
```

Output PDF:

- `paper/main.pdf`

## Data sources used in this draft

- `../results/paper/summary.csv`
- `../results/paper/reduction_vs_baseline.csv`
- `../results/*metrics_paper*.csv`
- `../results/paper/summary_bar.pdf`

## Generate 4 paper tables from logs

```bash
cd /path/to/RoPINN-main-3
python scripts/paper_make_tables.py
```

Generated LaTeX tables:

- `paper/tables/table_reaction_main.tex`
- `paper/tables/table_reaction_mseed.tex`
- `paper/tables/table_wave_mseed.tex`
- `paper/tables/table_convection_mseed.tex`

Generated statistics files:

- `results/paper/tables/seed_rows.csv`
- `results/paper/tables/seed_summary.csv`

If your multi-seed logs are missing locally, the multi-seed tables will be generated with `NA`. Re-run the script on the machine that has `results/mseed_*.log`.

## Generate statistical tests (p-value + 95% CI)

```bash
cd /path/to/RoPINN-main-3
python scripts/paper_significance_tests.py
```

Generated files:

- `results/paper/tables/stat_tests.csv`
- `paper/tables/table_stat_tests.tex`

## Notes

- Current draft reports both positive and negative cases.
- Keep the 4 tables synchronized with logs by re-running `scripts/paper_make_tables.py` before final submission.
