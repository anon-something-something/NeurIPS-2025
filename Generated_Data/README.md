# Generated Molecules

This folder contains molecules generated during the experiments described in the NeurIPS-2025 paper.  
Each CSV file corresponds to a specific configuration of the reward function.

## Files

- **baseline_dock_qed.csv**  
  Molecules generated with **baseline reward** (QED + docking).  
  Used as reference to compare against pharmacophore-guided setups.

- **cats_cosine_MAP4.csv**  
  Molecules generated with reward: **QED + MAP4 (structural dissimilarity) + CATS Cosine (pharmacophoric similarity)**.  
  Balances drug-likeness, pharmacophore fidelity, and scaffold novelty.

- **cats_cosine_tanimoto.csv**  
  Molecules generated with reward: **QED + Tanimoto (MACCS-based structural similarity) + CATS Cosine**.  
  Tends to produce compounds with high pharmacophoric alignment but closer structural analogs.

- **cats_euclid_MAP4.csv**  
  Molecules generated with reward: **QED + MAP4 + CATS Euclidean distance**.  
  Encourages pharmacophore similarity (Euclidean) while maximizing structural novelty (MAP4).

- **cats_euclid_tanimoto.csv**  
  Molecules generated with reward: **QED + Tanimoto + CATS Euclidean distance**.  
  Optimizes for pharmacophoric overlap with more diverse scaffolds compared to cosine setups.

## Notes
- All molecules (except those in the baseline set) were generated using the FREED++ RL framework with pharmacophore-guided rewards.  
- Docking scores and any properties not explicitly included in the rewards were computed post-hoc using FREED++ functions.
- We recommend using a reward weighting scheme of **QED: 1, all other rewards: 2** for balanced optimization between drug-likeness and pharmacophore-guided objectives.
