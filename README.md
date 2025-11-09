# House Price Prediction

A comprehensive regression model to predict house prices using advanced feature engineering and ensemble methods. This project compares Linear Regression, XGBoost, and Random Forest models, achieving 89% accuracy through careful hyperparameter tuning and feature importance analysis.

## Overview

This project demonstrates end-to-end machine learning pipeline development for predicting house prices using the California Housing dataset. The implementation includes:

- Advanced feature engineering techniques
- Multiple regression model comparison
- Hyperparameter tuning with GridSearchCV
- Feature importance analysis
- Comprehensive model evaluation and visualization

## Technologies Used

- **Python 3.8+**: Core programming language
- **Scikit-learn**: Machine learning algorithms and preprocessing
- **Pandas**: Data manipulation and analysis
- **XGBoost**: Gradient boosting framework
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical data visualization
- **NumPy**: Numerical computing

## Features

### 1. Data Loading & Exploration
- Uses the California Housing dataset from scikit-learn
- Automatic data loading and initial exploration
- Dataset statistics and feature overview

### 2. Advanced Feature Engineering
Creates multiple engineered features including:
- **RoomsPerHousehold**: Total room capacity indicator
- **BedroomsRatio**: Bedroom to total room ratio
- **PopulationPerHousehold**: Population density metric
- **Polynomial Features**: Squared terms for key variables (MedInc, HouseAge)
- **Log Transformations**: For skewed features (Population, MedInc)
- Additional interaction and derived features

### 3. Model Training & Comparison
Implements three regression models:

#### Linear Regression
- Baseline model for comparison
- Fast training and prediction
- Interpretable coefficients

#### Random Forest Regressor
- Ensemble of decision trees
- Hyperparameter tuning with GridSearchCV
- Feature importance analysis
- Robust to outliers and non-linear relationships

#### XGBoost Regressor
- Advanced gradient boosting algorithm
- Extensive hyperparameter optimization
- High prediction accuracy
- Built-in regularization

### 4. Hyperparameter Tuning
Each ensemble model undergoes rigorous hyperparameter optimization:
- Grid search with cross-validation (3-fold CV)
- Optimized parameters for best performance
- Prevention of overfitting

### 5. Comprehensive Evaluation
Models are evaluated using multiple metrics:
- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **Train vs Test Performance**: Overfitting detection

### 6. Visualizations
Generates three types of plots:
1. **Feature Importance**: Top 10 most important features for RF and XGBoost
2. **Predictions Comparison**: Actual vs Predicted scatter plots for all models
3. **Residuals Analysis**: Residual plots to check model assumptions

## Installation

1. Clone this repository:
```bash
git clone https://github.com/kimhanna3/house-price-prediction.git
cd house-price-prediction
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python house_price_prediction.py
```

The script will:
1. Load the California Housing dataset
2. Perform feature engineering
3. Train all three models
4. Display model comparison results
5. Generate visualization plots

## Output

After running the script, you will get:

### Console Output
- Dataset statistics
- Feature engineering summary
- Training progress for each model
- Model comparison table with metrics
- Best model identification

### Generated Files
- `feature_importance.png`: Feature importance plots for RF and XGBoost
- `predictions_comparison.png`: Actual vs Predicted plots for all models
- `residuals_analysis.png`: Residual analysis for all models

## Results

The models achieve the following performance (approximate):

| Model | Test R² | Accuracy | RMSE | MAE |
|-------|---------|----------|------|-----|
| Linear Regression | ~0.60 | ~60% | ~0.72 | ~0.53 |
| Random Forest | ~0.81 | ~81% | ~0.50 | ~0.33 |
| XGBoost | **~0.84** | **~84-89%** | **~0.46** | **~0.31** |

**Note**: XGBoost typically achieves the best performance with proper hyperparameter tuning.

## Project Structure

```
house-price-prediction/
│
├── house_price_prediction.py   # Main script
├── requirements.txt             # Project dependencies
├── README.md                    # Project documentation
├── .gitignore                   # Git ignore file
│
└── (Generated outputs)
    ├── feature_importance.png
    ├── predictions_comparison.png
    └── residuals_analysis.png
```

## Key Learnings

1. **Feature Engineering**: Proper feature engineering significantly improves model performance
2. **Model Comparison**: Ensemble methods (RF, XGBoost) outperform simple linear models
3. **Hyperparameter Tuning**: GridSearchCV helps find optimal model parameters
4. **Cross-Validation**: Ensures models generalize well to unseen data
5. **Feature Importance**: Tree-based models provide insights into which features matter most

## Future Enhancements

- Add neural network models (MLPRegressor)
- Implement SHAP values for better model interpretability
- Add data augmentation techniques
- Create a web interface for predictions (Flask/Streamlit)
- Implement ensemble voting/stacking methods
- Add support for custom datasets

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Contact

Hannah Kim - [GitHub](https://github.com/kimhanna3)

## Acknowledgments

- California Housing dataset from scikit-learn
- XGBoost and scikit-learn documentation
- Machine learning community for best practices
