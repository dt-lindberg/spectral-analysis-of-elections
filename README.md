# Spectral Clustering Election Stats

## Quick start
1. Clone the DAP repo and ensure the `dap/` directory exists in the repo root.
> https://github.com/Project-PRAGMA/diversity-agreement-polarization-IJCAI23.git

2. Install Python dependencies:
   ```bash
   pip install -r spectral-clustering/requirements.txt
   ```

3. Enable the scree plot workflow in `spectral-clustering/spectral_analysis.py` by uncommenting the `__main__` block near the bottom and setting `ELECTIONS_TO_COMPARE`.

4. Run the script:
   ```bash
   python spectral-clustering/spectral_analysis.py
   ```
   This writes the scree plot (for example `spectral-clustering/scree_plot_3.png`).

## Additional plot (uniform scree)
- Re-enable the top-level `__main__` block and run the script to generate the uniform-election scree grid at `spectral-clustering/uniform_scree_grid_m3_m6.png`.
