MHED-TOE FIGURES
================

This directory contains all 13 figures for the MHED-TOE arXiv submission.

MAIN PAPER FIGURES (8):
----------------------
1. fig1_utlor_convergence.png    - Figure 1: U_TL-OR bridge convergence
   Shows convergence of toroidal Langlands-OR bridge to 2.87
   
2. fig2_fermion_spectrum.png     - Figure 2: Fermion mass spectrum
   MHED-TOE vs Standard Model fermion masses (7.3% average error)
   
3. fig3_coherence_cascade.png    - Figure 3: Coherence cascade
   From single tubulin (12.4 fs) to axon (143 fs) to Orch-OR (9.2 ms)
   
4. fig4_holographic_phi.png      - Figure 4: Holographic Φ
   Φ_holo from Ryu-Takayanagi surfaces on hex lattice (max = 2.847)
   
5. fig5_causal_sets.png          - Figure 5: Causal set sprinkling
   Poisson sprinkling on monadic lattice, U_causal=2.84 (target: 2.87)
   
6. fig6_revelation_tensor.png    - Figure 6: Revelation tensor
   rev[5×5×3] tensor with 27 bridges >0.03 strength
   
7. fig7_lhc_prediction.png       - Figure 7: LHC prediction
   0.83 TeV dark matter singlet cross-section σ=10⁻⁴ pb
   
8. fig8_eeg_prediction.png       - Figure 8: EEG prediction
   Gamma-band modulation 36→39 Hz from Yukawa defects

SUPPLEMENTARY FIGURES (5):
-------------------------
9. supp_utlor_convergence.png    - Supplementary Figure S1
   Detailed U_TL-OR convergence with error bars
   
10. supp_spectrum_details.png    - Supplementary Figure S2
   Full 3-generation spectrum with error analysis
   
11. supp_orch_or_timing.png      - Supplementary Figure S3
   Extended Orch-OR timing curves for different N
   
12. supp_causal_sets_detailed.png - Supplementary Figure S4
   Detailed causal set statistics and chain distributions
   
13. supp_validation_summary.png  - Supplementary Figure S5
   Complete validation summary across all tests

GENERATION:
All figures were generated using Python 3.9+ with:
- NumPy 1.21.0+
- SciPy 1.7.0+
- Matplotlib 3.5.0+
- QuTiP 4.7.1+
- NetworkX 3.1+

Regenerate all figures by running:
  python examples/generate_figures.py

COLOR SCHEMES:
- Blue: MHED-TOE predictions
- Orange: Standard Model/Experimental values
- Red: Strong connections/errors
- Green: Thresholds/targets
- Viridis colormap: Revelation tensor

RESOLUTION:
All figures: 300 DPI, suitable for publication
Format: PNG with transparency where appropriate

LICENSE:
CC BY 4.0 - Attribution required
