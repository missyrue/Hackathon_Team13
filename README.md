# PathSnap: Distributed Map-Matching & Route Reconstruction

> **UE24CS352A — Machine Learning Hackathon | PES University**
> Team 13: A S Vincent, Aaruni, Ananya, Akshata

## Overview

PathSnap is an end-to-end distributed ML system that reconstructs realistic vehicle
trajectories by matching noisy GPS observations to an underlying physical road network
(Map-Matching). The system uses a multi-model classification pipeline combined with an
HMM / Viterbi sequential decoder, deployed across a two-laptop FastAPI + Streamlit
architecture.

## Repository Structure

```
Hackathon_Team13/
├── src/
│   ├── data/           # Road network graph parser, trajectory loader, UTM projection
│   ├── pipeline/       # Candidate segment generator (KD-Tree), spatial feature extractor
│   ├── models/         # Model zoo: RF, LightGBM, MLP, KNN, Logistic Regression
│   ├── tuning/         # Bayesian hyperparameter optimisation (Optuna)
│   ├── hmm/            # HMM transition/emission models, Viterbi decoder
│   ├── api/            # FastAPI backend server (Laptop 1)
│   └── ui/             # Streamlit evaluator UI + Folium maps (Laptop 2)
├── tests/              # Unit & integration tests
├── notebooks/          # EDA and experiment notebooks
├── artifacts/          # Serialised model weights & scalers
├── docs/
│   ├── eda/trajectory_audit/ # Training-track inventory and anomaly summary
│   └── figures/             # Report figures, Folium HTML exports
├── report/sections/         # Technical report source files
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# 1. Clone
git clone https://github.com/missyrue/Hackathon_Team13.git
cd Hackathon_Team13

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run backend (Laptop 1)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# 5. Run frontend (Laptop 2)
streamlit run src/ui/app.py
```

## Dataset

- **80 training trips** — used for model training (GroupKFold by trip ID)
- **10 student-testing trips** — used for student-side validation
- **10 evaluator-testing trips** — reserved for live demo (ground-truth stays on Laptop 1)

Each trip contains `.nodes` (road vertices), `.arcs` (road segments), `.track` (noisy GPS),
and `.route` (ground-truth arc sequence).

## Team

| Member | Contributions |
|--------|--------------|
| A S Vincent | Road network parser, RF/Extra-Trees, Bayesian tuning objective, HMM transitions, FastAPI backend |
| Aaruni | Trajectory loader, LightGBM/XGBoost, Bayesian trials, HMM emissions, API logging |
| Ananya | KD-Tree candidates, MLP neural net, Viterbi decoder, Streamlit UI |
| Akshata | Feature engineering, KNN/LogReg, artifact export, gap handling, Folium maps |

## License

Academic use only — PES University UE24CS352A.
