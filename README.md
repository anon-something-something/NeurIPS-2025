# Pharmacophore-Guided Generative Design of Novel Drug-Like Molecules

This repository contains the code, data, and models for our NeurIPS 2025 submission:
**"Pharmacophore-Guided Generative Design of Novel Drug-Like Molecules"**.

## Abstract
The integration of artificial intelligence (AI) in early-stage drug discovery offers unprecedented opportunities for exploring chemical space and accelerating hit-to lead optimization. However, using docking as a reward function during generative model training is computationally expensive and may yield inaccurate results. Here, we present a novel generative framework that balances pharmacophore similarity to reference compounds with structural diversity from active molecules. The framework allows users to provide custom reference sets, including FDA approved drugs or clinical candidates, and guides the de novo generation of potential therapeutics. We demonstrate its applicability through a case study targeting alpha estrogen receptor modulators and antagonists for breast cancer. The generated compounds maintain high pharmacophoric fidelity to known active molecules while introducing substantial structural novelty, suggesting strong potential for functional innovation and patentability. Comprehensive evaluation of the generated molecules against common drug-like properties confirms the robustness and pharmaceutical relevance of the approach.


<img width="5657" height="1573" alt="image" src="https://github.com/user-attachments/assets/2d09db0c-3439-4f03-8c40-8d3284c81456" />


Baseline comparisons are performed with QVina docking on the α-estrogen receptor (PDB: 8AWG).

## Links

- [ChemDiv — Complete list of compound libraries](https://www.chemdiv.com/catalog/complete-list-of-compounds-libraries/)  
- [PDB entry 8AWG (RCSB)](https://www.rcsb.org/structure/8AWG)  
- [ChEMBL](https://www.ebi.ac.uk/chembl/)  
- [ZINC database](https://zinc.docking.org/)  
- [PubChem](https://pubchem.ncbi.nlm.nih.gov/)

## Citation
Under review for NeurIPS 2025 AI4Mat
