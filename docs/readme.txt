MHED-TOE: DOCUMENTATION PACKAGE
===============================

This directory contains the complete documentation for:
"Microtubule Yukawa Dynamics Generate E8→SM Fermion Spectrum and Orchestrated Objective Reduction Consciousness"

FILES:
- MHED_TOE_PAPER.tex     - Main LaTeX paper (47 pages)
- MHED_TOE_PAPER.pdf     - Compiled PDF of main paper
- MHED_TOE_PAPER.bbl     - Bibliography for main paper
- MHED_TOE_PAPER.bib     - BibTeX database
- supplementary.tex      - Supplementary material (38 pages)
- supplementary.pdf      - Compiled supplementary PDF
- Makefile              - Compilation automation
- README.txt           - This file

COMPILATION INSTRUCTIONS:

1. To compile everything:
   $ make all

2. To compile just the main paper:
   $ make paper

3. To compile supplementary material:
   $ make supplementary

4. To generate figures (requires Python):
   $ make figures

5. To create arXiv submission package:
   $ make arxiv

6. To clean generated files:
   $ make clean

DEPENDENCIES:
- LaTeX distribution (TeX Live 2020+ or MikTeX)
- Python 3.9+ with:
  - NumPy, SciPy, Matplotlib
  - QuTiP 4.7.1+
  - NetworkX 3.1+
- GNU Make

FIGURE GENERATION:
All figures can be regenerated using the Python scripts:
- python examples/generate_figures.py
- python notebooks/05_validation_figures.ipynb

The figures directory contains:
- 8 main figures for the paper
- 5 supplementary figures
- Validation summary figures

PAPER STRUCTURE:
Main Paper (47 pages):
  1. Introduction
  2. Theoretical Framework
  3. Simulation Results
  4. MHED Master Equation
  5. Falsifiable Predictions (2026-2027)
  6. Discussion
  7. Conclusion
  8. Appendix: Supplementary Information

Supplementary Material (38 pages):
  1. Mathematical Details
  2. Implementation Details
  3. Extended Results
  4. Mathematical Proofs
  5. Experimental Protocols
  6. Code Examples
  7. Data Availability

CITATION:
Please cite as:
  Colvin, T. R. IV (2026). Microtubule Yukawa Dynamics Generate E8→SM Fermion
  Spectrum and Orchestrated Objective Reduction Consciousness. arXiv:2602.00001.

CODE REPOSITORY:
https://github.com/atomadic/MHED-TOE

CONTACT:
Thomas Ralph Colvin IV
atomadic@proton.me

VERSION:
1.0.0 (February 9, 2026)
