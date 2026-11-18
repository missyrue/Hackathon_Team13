# PathSnap: Distributed Map-Matching & Route Reconstruction

> **Course:** UE24CS352A — Machine Learning | PES University  
> **Author:** A S Vincent (`vincent2maran@gmail.com`)  
> **Project:** `vincent-ml-hackathon`

---

## 🌟 Overview
PathSnap is a distributed machine learning system that reconstructs realistic vehicle trajectories by matching noisy GPS observations (~1 Hz) to an underlying physical road network graph. It combines a spatial candidate extraction engine with a 5-model supervised classification zoo and an HMM / Viterbi sequential decoder, deployed across a two-laptop FastAPI + Streamlit architecture.

---

## 📁 Project Structure

```
vincent-ml-hackathon/
├── notebooks/                              # Google Colab & Jupyter Notebooks (.ipynb)
│   ├── 01_EDA_and_Preprocessing.ipynb      # Coordinate projection, 1 Hz audit, KD-Tree, Folium map
│   ├── 02_Model_Zoo_and_Comparison.ipynb   # 5 Models (RF, LightGBM, MLP, KNN, LogReg) + Table 3.1
│   ├── 03_Bayesian_Optimization.ipynb      # Optuna hyperparameter tuning, convergence plots
│   ├── 04_HMM_Viterbi_Reconstruction.ipynb # Dijkstra road distances, HMM trellis, continuity
│   └── 05_Master_Full_Pipeline_Colab.ipynb # Turnkey Master Notebook for Google Colab
├── src/                                    # Modular production source code
│   ├── data/
│   │   ├── spatial_projection.py           # WGS84 degrees -> UTM Zone 32N (metres)
│   │   ├── road_network.py                 # Graph parser, segment lengths, bearings, DiGraph
│   │   └── trajectory_loader.py            # .track reader, speeds, gap audit
│   ├── pipeline/
│   │   └── features.py                     # Sampled KD-Tree, monotonic route alignment, features
│   ├── models/
│   │   ├── cv_harness.py                   # GroupKFold by trip_id, Top-1 accuracy, Macro F1
│   │   ├── model_zoo.py                    # 5 distinct model implementations & Table 3.1 generator
│   │   └── bayesian_opt.py                 # Optuna Bayesian hyperparameter search
│   ├── hmm/
│   │   ├── transitions.py                  # Shortest road network distance vs Euclidean penalty
│   │   ├── emissions.py                    # Gaussian spatial error + ML confidence likelihood
│   │   └── viterbi.py                      # Log-space Viterbi dynamic programming trellis decoder
│   ├── api/
│   │   └── main.py                         # FastAPI backend server (Laptop 1)
│   └── ui/
│       └── app.py                          # Streamlit evaluator UI + Folium map (Laptop 2)
├── artifacts/                              # Exported model weights & metadata
├── requirements.txt                        # Dependencies
└── README.md
```

---

## 🚀 Running in Google Colab

1. Open [Google Colab](https://colab.research.google.com).
2. Upload any notebook from the `notebooks/` folder:
   - `01_EDA_and_Preprocessing.ipynb`
   - `02_Model_Zoo_and_Comparison.ipynb`
   - `03_Bayesian_Optimization.ipynb`
   - `04_HMM_Viterbi_Route_Reconstruction.ipynb`
   - Or upload `05_Master_Full_Pipeline_Colab.ipynb` for the full all-in-one run.
3. In Colab, mount Google Drive or upload your `Student folder.zip` into the files pane:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. Run all cells (`Runtime -> Run all`).

---

## 💻 Two-Laptop Distributed Deployment

The hackathon requires deploying across two separate physical machines connected over the local network (Wi-Fi or mobile hotspot):

### Laptop 1 (Model Server & FastAPI Backend):
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the FastAPI server on port 8000:
   ```bash
   uvicorn src.api.main:app --host 0.0.0.0 --port 8000
   ```
3. Find your local IP address in terminal (`ipconfig` on Windows $\rightarrow$ e.g., `192.168.1.45`).

### Laptop 2 (Evaluator UI & Folium Visualizer):
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit web interface:
   ```bash
   streamlit run src/ui/app.py
   ```
3. In the sidebar, enter Laptop 1's IP: `http://192.168.1.45:8000`.
4. Select any of the 10 evaluation trips and click **"Run Map-Matching & Route Reconstruction"**.
5. The UI dynamically receives the reconstructed route, displays KPI metric cards, and renders the interactive 4-layer Folium map!
