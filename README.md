# AIDEOM-VN Dashboard

## AI-Driven Economic Decision Support System for Vietnam

AIDEOM-VN (Artificial Intelligence Driven Economic Optimization Model for Vietnam) is an integrated decision support system designed to assist economic planning and policy analysis in the context of digital transformation and artificial intelligence.

The project combines optimization models, multi-criteria decision analysis, dynamic simulation, stochastic programming, and reinforcement learning to evaluate policy alternatives and support evidence-based decision-making for Vietnam's long-term development.

---

## Research Background

This dashboard was developed as the final project for the course **Decision Support Systems**, with the research topic:

**"Decision Support System for Vietnam's Economic Development in the AI Era"**

The system integrates eleven analytical modules and one comprehensive policy dashboard.

---

## Main Features

### Bài 1 – Cobb-Douglas Growth Model with AI

* Estimate economic output under digital and AI investments.
* Analyze marginal productivity of capital, digitalization, AI, and human capital.

### Bài 2 – Public Investment Optimization

* Linear Programming (LP) model for budget allocation.
* Maximize economic returns under fiscal constraints.

### Bài 3 – Industry Prioritization

* Rank strategic industries based on economic contribution and digital readiness.

### Bài 4 – Regional Investment Allocation

* Optimize resource distribution across economic regions.

### Bài 5 – Project Selection

* Mixed Integer Programming (MIP) model for selecting public investment projects.

### Bài 6 – Regional AI Readiness Assessment

* TOPSIS-based evaluation of six Vietnamese economic regions.
* Compare expert-weight and entropy-weight scenarios.

### Bài 7 – Multi-Objective Optimization

* NSGA-II algorithm.
* Generate Pareto-optimal solutions balancing:

  * Economic growth
  * Social inclusion
  * Environmental sustainability
  * Data security

### Bài 8 – Dynamic Economic Simulation (2026–2035)

* Optimal investment trajectory.
* Shock analysis and long-term development scenarios.

### Bài 9 – Labor Market and AI

* Evaluate job creation, displacement, and retraining strategies.
* Workforce transition analysis.

### Bài 10 – Stochastic Programming

* Two-stage decision model under uncertainty.
* VSS and EVPI analysis.

### Bài 11 – Reinforcement Learning

* Q-Learning policy optimization.
* Adaptive policy responses under different economic states.

### Bài 12 – Integrated AIDEOM-VN Dashboard

* Comprehensive decision support platform.
* Combines all previous analytical modules into a unified system.

---

## Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Plotly
* Matplotlib
* Seaborn
* PuLP
* Pyomo
* CVXPY
* SciPy
* Pymoo (NSGA-II)
* Reinforcement Learning (Q-Learning)

---

## Dashboard Deployment

Live application:

https://aideom-vn-dashboard.streamlit.app/

---

## Repository Structure

```text
AIDEOM-VN-dashboard
│
├── app.py
├── requirements.txt
├── package.txt
├── README.md
│
├── data/
│   ├── vietnam_macro_2020_2025.csv
│   ├── vietnam_regions_*.csv
│   ├── vietnam_sectors_*.csv
│   └── ...
│
├── pages/
│   ├── Bài 1 – Cobb-Douglas + AI
│   ├── Bài 2 – LP Budget Allocation
│   ├── ...
│   └── Bài 12 – Integrated Dashboard
│
└── assets/
```

---

## Research Contributions

This project demonstrates how optimization, AI, and decision analytics can be integrated into a unified framework for:

* Strategic economic planning
* Digital transformation policy design
* AI development strategy evaluation
* Workforce transition management
* Regional development planning
* Decision-making under uncertainty

The framework can serve as a foundation for future extensions involving:

* CGE and DSGE models
* Real-time economic data integration
* Deep Reinforcement Learning
* Multi-Agent Reinforcement Learning

---

## Author

**Vũ Thế Mạnh**

Faculty of Economics

Final Project – Decision Support Systems

Vietnam
