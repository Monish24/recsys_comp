Hybrid Recommender System - Group 24
A tier-specific hybrid recommender system for generating top-10 personalized item recommendations.
Overview
This project implements a modular hybrid recommender system that adapts its strategy based on user interaction history. The system integrates multiple recommendation paradigms including sequential pattern mining, collaborative filtering, content-based retrieval, and bought-together pattern extraction.
Key Features

Tier-specific Architecture: Adapts recommendations based on user interaction count (Cold/Warm/Hot users)
Multi-signal Fusion: Combines sequential, content-based, collaborative, and popularity signals
Bought-together Enhancement: Extracts co-purchase patterns from transaction data
Wilson Score Quality: Enhanced item quality scoring with statistical confidence
Hyperparameter Optimization: Bayesian optimization with tier-specific parameters

Performance

Overall Recall@10: 0.0426
Cold Users (94.6%): 0.0422 recall
Warm Users (5.3%): 0.0323 recall
Hot Users (0.1%): 0.0294 recall

Files Structure
├── code/
│   └── hybrid.py                    # Main recommender system implementation
├── report/
│   └── Recommender_Systems_Report.pdf  # Detailed technical report
├── data/
│   ├── train.csv                    # Training interactions
│   ├── test.csv                     # Test interactions
│   ├── item_meta.csv               # Item metadata
│   └── sample_submission.csv       # Submission template
└── requirements.txt                # Python dependencies
Installation
bash# Clone repository
git clone https://github.com/YOUR_USERNAME/hybrid-recommender-system.git
cd hybrid-recommender-system

# Install dependencies
pip install -r requirements.txt
Usage
python# Load and run the recommender system
python code/hybrid.py

# This will:
# 1. Load and preprocess data
# 2. Extract bought-together patterns
# 3. Build the hybrid recommender
# 4. Validate performance
# 5. Generate submission file
Key Innovations

Enhanced Metadata: Automatic extraction of bought-together patterns from transaction data
Adaptive Weighting: Dynamic module weights based on user interaction tier
Combo Bonuses: 25% score boost for items appearing in multiple recommendation sources
Regularization: Tier-specific strategies to prevent overfitting

Technical Details
User Segmentation

Cold (1 interaction): Content + popularity emphasis
Warm (2-4 interactions): Sequential + content + collaborative
Hot (5+ interactions): Full sequential modeling with regularization

Module Weights

Content similarity: 2.0 (cold) → 1.8 (warm) → 1.8 (hot)
Sequential patterns: 0.8 (cold) → 2.5 (warm) → 2.5 (hot)
Collaborative filtering: 0.5 (cold) → 1.3 (warm) → 1.5 (hot)

Dependencies
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=1.0.0
scipy>=1.7.0
sentence-transformers>=2.0.0
Team

Joren van den Berg
Sepehr Moghiseh
Monish Shah
SeyedehSheida Nehzati

License
This project is for academic purposes as part of the Recommender Systems course.