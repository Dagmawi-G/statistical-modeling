# 10 Academy AIM Week 3: Insurance Risk Analytics

## Overview
This repository contains the code and analysis for Week 3 of the 10 Academy Artificial Intelligence Mastery program, focusing on end-to-end insurance risk analytics and predictive modeling for AlphaCare Insurance Solutions (ACIS).

## Objectives
- Analyze historical insurance claim data to identify low-risk segments.
- Optimize marketing strategies and premium pricing.
- Perform EDA, hypothesis testing, and predictive modeling.

## Repository Structure
- `data/`: Stores the dataset (not committed due to size; tracked with DVC).
- `scripts/`: Python scripts for EDA and utilities.
- `notebooks/`: Jupyter notebooks for exploratory analysis.
- `.github/workflows/`: CI/CD configuration.
- `requirements.txt`: Python dependencies.
- `README.md`: Project documentation.

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/10Academy-AIM-Week3.git
## Data Version Control Setup
- DVC is used to version-control the dataset.
- Remote storage: `C:/Users/hp/Desktop/dvc_storage`
- Dataset: `insurance.csv` is tracked with DVC.
- To reproduce:
  1. Clone the repository.
  2. Run `dvc pull` to retrieve the dataset from the remote storage.