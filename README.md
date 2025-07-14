## Machine Learning Workflow for Detecting Chronic Drug Effects in In Vitro Neuronal Networks on Microelectrode Arrays

This repository contains the codebase for the analysis workflow which has been presented at the 13th international Meeting on Neural and Electrogenic Cell Interfacing, Vienna, (2025). This workflow is a modification from Ciba et al., 2025. If you use this code, please cite the publication Ciba et al. (2025) and this poster:

Ciba, Manuel; Mayer, Margot; L. Flores, Ismael; Alves, Caroline; Namyslo, Jannick; Thielemann, Christiane (2025). Machine Learning Workflow for Detecting Longterm Drug Effects in In Vitro Neuronal Networks on Microelectrode Arrays (presented at 13th international Meeting on Neural and Electrogenic Cell Interfacing, Vienna, 2025). figshare. Poster. https://doi.org/10.6084/m9.figshare.29559020.v2

## 🧠 Abstract

Assessing the chronic (long-term) effects of drugs on in vitro neuronal networks can be challenging due to the subtle nature of these effects, which may not reach statistical significance. Moreover, such experiments often involve repeated measurements over time, necessitating p-value adjustments in traditional inferential statistical tests (e.g., t-tests) to account for multiple testing. These adjustments exacerbate the difficulty of detecting subtle drug effects, particularly when analyzing multiple features simultaneously. Typically, datasets from microelectrode array (MEA) chips include a broad range of features quantifying spikes, bursts, network bursts, synchrony, functional connectivity, and complex network properties.

We recently proposed a machine learning-based workflow for detecting drug effects without the need for multiple testing adjustments (Ciba et al., 2025). This workflow, initially developed for acute drug experiments, integrates various machine learning models and preprocessing steps with explainable machine learning methods such as SHapley Additive exPlanations (SHAP). In this study, we extend the workflow to assess chronic drug effects by incorporating repeated measurements into the analysis.

As a case study, we applied the extended workflow to data from experiments investigating the chronic effects of the brain-derived neurotrophic factor
(BDNF) which is known to be increased by psychedlic substances such as LSD. In these experiments, the active components were applied for several hours and then washed out, resulting in subtle post-washout effects. Traditional statistical tests failed to detect these chronic effects, while the machine learning-based workflow successfully identified them. This demonstrates the potential of machine learning as a powerful alternative
for analyzing subtle and complex drug effects in in vitro neuronal network studies.

---

## 📌 Acknowledgements

This work was supported by the **Bundesministerium für Bildung und Forschung (BMBF)** under the ESTRANGE project (grant 02NUK081B), and by the **Deutsche Forschungsgemeinschaft (DFG)** under the project *\"Untersuchung der Wirkung von psychedelischen Substanzen auf neuronale dreidimensionale Zellkulturen\"* (grant TH 1448/5-1).

---

## 📚 Reference

Ciba, M., Petzold, M., Alves, C. L., Rodrigues, F. A., Jimbo, Y., & Thielemann, C. (2025).  
*Machine learning and complex network analysis of drug effects on neuronal microelectrode biosensor data*.  
**Scientific Reports, 15**(1), 15128. https://doi.org/10.1038/s41598-025-99479-7

---

**Corresponding Author:**  
Manuel Ciba — [Manuel.Ciba@th-ab.de](mailto:Manuel.Ciba@th-ab.de)

