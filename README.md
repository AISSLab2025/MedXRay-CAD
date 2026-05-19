# 🩻 MEDXRAY-CAD: Multimodal Agentic Chest X-Ray Diagnosis

> **News:** This paper has been published in the **IEEE Journal of Biomedical and Health Informatics (JBHI)**.  
> 📄 **[Read the Paper here](https://ieeexplore.ieee.org/document/11297148)**

> **A novel agentic AI framework for chest X-ray diagnosis combining multimodal retrieval-augmented generation (RAG), classification, and human-in-the-loop interpretability.**

---

![Demo](MedXray-CAD.gif)

<div align="center">
  <img src="docs/MedXray-CAD-Architecture.png" width="100%" alt="MedXray CAD Architecture">
</div>

<div align="center">
  <img src="docs/MedXray-CAD-Retrieval-Architecture.png" width="100%" alt="Retrieval Architecture">
</div>

## 🚀 Overview

**MEDXRAY-CAD** is an intelligent, multimodal agentic system designed for automated chest X-ray (CXR) diagnosis.  
It integrates **vision encoders**, **binary and multi-label classifiers**, and a **vector database (Chroma)** for context-aware medical retrieval.  
The framework also features a **Gradio-based dashboard** for real-time interaction, visualization, and report generation.

---

## 🎯 Purpose

To automate and enhance diagnostic decision-making in CXR analysis by:
- Combining **deep visual encoders** and **retrieval-based reasoning**.
- Leveraging **medical report embeddings** for contextual retrieval.
- Enabling **interactive exploration** of predictions and Grad-CAM visualizations.

---

## 🧩 Key Components

- **Binary & Multi-label Classifiers** → Detect conditions such as *Pleural Effusion*, *Pneumonia*, etc.  
- **Vector Database (Chroma)** → Efficient storage and querying of medical report embeddings.  
- **Agentic Workflow** → Incorporates localisation, visualisation tools etc.  
- **Interactive Dashboard (Gradio)** → Real-time predictions, retrievals, and clinician-style queries.  

---

## 📚 Citation

```bibtex
@article{11297148,
  author={Raza, Mukhlis and Salem, Saied and Kwon, Hyunwook and Hussain, Jamil and Gu, Yeong Hyeon and Al-Antari, Mugahed A.},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={Multimodal Knowledge-Infused VLM for Respiratory Disease Prediction and Clinical Report Generation}, 
  year={2025},
  pages={1-14},
  doi={10.1109/JBHI.2025.3631264},
  url={https://ieeexplore.ieee.org/document/11297148}
}
```

## 📚 Related Work

```bibtex
@inproceedings{Salem2025AutoSpineAI,
  title={AutoSpineAI: Lightweight Multimodal CAD Framework for Lumbar Spine MRI Assessments},
  author={Salem, S. and Habib, A. and Raza, M. and Al-Huda, Z. and Al-maqtari, O. and Ertuğrul, B. and others},
  booktitle={IEEE-EMBS International Conference on Biomedical and Health Informatics},
  year={2025},
  url={https://doi.org/10.1109/BHI67747.2025.11269503}
}

@article{AlAntari2025Evaluating,
  title={Evaluating AI-powered predictive solutions for MRI in lumbar spinal stenosis: a systematic review},
  author={Al-Antari, M. A. and Salem, S. and Raza, M. and Elbadawy, A. S. and Bütün, E. and Aydin, A. A. and others},
  journal={Artificial Intelligence Review},
  volume={58},
  number={8},
  pages={221},
  year={2025},
  publisher={Springer},
  doi={10.1007/s10462-025-11185-y},
  url={https://doi.org/10.1007/s10462-025-11185-y}
}

@inproceedings{Raza2025MedXpert,
  title={MedXpert-CAD: A Multimodal Multi-agentic System for Clinical Imaging Analysis via Model Context Protocol LLM-Driven Agentic Workflows},
  author={Raza, M. and Salem, S. and Habib, A. and Abdulmahmod, O. and Kwon, H. and Hussain, J. and Al-antari, M. A.},
  booktitle={MICCAI2025: Agentic AI in Medicine Workshop},
  pages={55--64},
  year={2025},
  month={September},
  publisher={Springer Nature Switzerland},
  doi={10.1007/978-3-032-06004-4_6},
  url={https://doi.org/10.1007/978-3-032-06004-4_6}
}
```
