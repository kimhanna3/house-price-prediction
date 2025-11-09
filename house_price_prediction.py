"""
House Price Prediction using Advanced Feature Engineering and Ensemble Methods
================================================================================

This script implements a comprehensive regression model to predict house prices,
comparing Linear Regression, XGBoost, and Random Forest models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import os
warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Output directory setup
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


class HousePricePrediction:
    """
    A comprehensive house price prediction system with feature engineering
    and model comparison capabilities.
    """

    def __init__(self):
        """Initialize the prediction system."""
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}

    def load_data(self):
        """Load the California Housing dataset."""
        print("Loading California Housing dataset...")
        housing = fetch_california_housing()
        self.data = pd.DataFrame(housing.data, columns=housing.feature_names)
        self.data['target'] = housing.target

        print(f"Dataset shape: {self.data.shape}")
        print(f"\nFeatures: {list(self.data.columns[:-1])}")
        print(f"\nFirst few rows:")
        print(self.data.head())
        return self.data

    def feature_engineering(self):
        """
        Perform advanced feature engineering to create new meaningful features.
        """
        print("\n" + "="*70)
        print("Performing Feature Engineering...")
        print("="*70)

        # Create a copy for feature engineering
        df = self.data.copy()

        # 1. Rooms per household
        df['RoomsPerHousehold'] = df['AveRooms'] * df['AveOccup']

        # 2. Bedrooms ratio
        df['BedroomsRatio'] = df['AveBedrms'] / df['AveRooms']

        # 3. Population per household
        df['PopulationPerHousehold'] = df['Population'] / df['HouseAge']

        # 4. Price per room (interaction with target for feature creation)
        df['PricePerRoom'] = df['target'] / (df['AveRooms'] + 1)

        # 5. Total rooms
        df['TotalRooms'] = df['AveRooms'] * df['Population']

        # 6. Location wealth indicator (based on median income and house value)
        df['WealthIndicator'] = df['MedInc'] * df['target']

        # 7. Polynomial features for important variables
        df['MedInc_squared'] = df['MedInc'] ** 2
        df['HouseAge_squared'] = df['HouseAge'] ** 2

        # 8. Log transformations for skewed features
        df['Log_Population'] = np.log1p(df['Population'])
        df['Log_MedInc'] = np.log1p(df['MedInc'])

        print(f"Original features: {len(self.data.columns) - 1}")
        print(f"Features after engineering: {len(df.columns) - 1}")
        print(f"New features created: {len(df.columns) - len(self.data.columns)}")

        self.data = df
        return df

    def prepare_data(self, test_size=0.2, random_state=42):
        """
        Prepare data for model training.

        Args:
            test_size (float): Proportion of dataset to include in test split
            random_state (int): Random state for reproducibility
        """
        print("\n" + "="*70)
        print("Preparing Data for Training...")
        print("="*70)

        # Separate features and target
        X = self.data.drop('target', axis=1)

        # Remove target-dependent features for proper train/test split
        cols_to_remove = ['PricePerRoom', 'WealthIndicator']
        X = X.drop(cols_to_remove, axis=1)

        y = self.data['target']

        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Scale the features
        self.X_train = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.X_train.columns
        )
        self.X_test = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.X_test.columns
        )

        print(f"Training set size: {self.X_train.shape}")
        print(f"Test set size: {self.X_test.shape}")

    def train_linear_regression(self):
        """Train Linear Regression model."""
        print("\n" + "="*70)
        print("Training Linear Regression Model...")
        print("="*70)

        lr = LinearRegression()
        lr.fit(self.X_train, self.y_train)

        # Predictions
        y_pred_train = lr.predict(self.X_train)
        y_pred_test = lr.predict(self.X_test)

        # Evaluation
        train_r2 = r2_score(self.y_train, y_pred_train)
        test_r2 = r2_score(self.y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        mae = mean_absolute_error(self.y_test, y_pred_test)

        self.models['Linear Regression'] = lr
        self.results['Linear Regression'] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'rmse': rmse,
            'mae': mae,
            'predictions': y_pred_test
        }

        print(f"Train R² Score: {train_r2:.4f}")
        print(f"Test R² Score: {test_r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")

    def train_random_forest(self):
        """Train Random Forest model with hyperparameter tuning."""
        print("\n" + "="*70)
        print("Training Random Forest Model with Hyperparameter Tuning...")
        print("="*70)

        # Initial model
        rf = RandomForestRegressor(random_state=42)

        # Hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }

        # Grid search with cross-validation
        print("Performing Grid Search (this may take a few minutes)...")
        grid_search = GridSearchCV(
            rf, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1
        )
        grid_search.fit(self.X_train, self.y_train)

        # Best model
        best_rf = grid_search.best_estimator_

        print(f"\nBest parameters: {grid_search.best_params_}")

        # Predictions
        y_pred_train = best_rf.predict(self.X_train)
        y_pred_test = best_rf.predict(self.X_test)

        # Evaluation
        train_r2 = r2_score(self.y_train, y_pred_train)
        test_r2 = r2_score(self.y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        mae = mean_absolute_error(self.y_test, y_pred_test)

        self.models['Random Forest'] = best_rf
        self.results['Random Forest'] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'rmse': rmse,
            'mae': mae,
            'predictions': y_pred_test,
            'best_params': grid_search.best_params_
        }

        print(f"\nTrain R² Score: {train_r2:.4f}")
        print(f"Test R² Score: {test_r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")

    def train_xgboost(self):
        """Train XGBoost model with hyperparameter tuning."""
        print("\n" + "="*70)
        print("Training XGBoost Model with Hyperparameter Tuning...")
        print("="*70)

        # Initial model
        xgb = XGBRegressor(random_state=42, objective='reg:squarederror')

        # Hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0]
        }

        # Grid search with cross-validation
        print("Performing Grid Search (this may take a few minutes)...")
        grid_search = GridSearchCV(
            xgb, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1
        )
        grid_search.fit(self.X_train, self.y_train)

        # Best model
        best_xgb = grid_search.best_estimator_

        print(f"\nBest parameters: {grid_search.best_params_}")

        # Predictions
        y_pred_train = best_xgb.predict(self.X_train)
        y_pred_test = best_xgb.predict(self.X_test)

        # Evaluation
        train_r2 = r2_score(self.y_train, y_pred_train)
        test_r2 = r2_score(self.y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        mae = mean_absolute_error(self.y_test, y_pred_test)

        self.models['XGBoost'] = best_xgb
        self.results['XGBoost'] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'rmse': rmse,
            'mae': mae,
            'predictions': y_pred_test,
            'best_params': grid_search.best_params_
        }

        print(f"\nTrain R² Score: {train_r2:.4f}")
        print(f"Test R² Score: {test_r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")

    def compare_models(self):
        """Compare all trained models."""
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)

        comparison_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Train R²': [self.results[m]['train_r2'] for m in self.results],
            'Test R²': [self.results[m]['test_r2'] for m in self.results],
            'RMSE': [self.results[m]['rmse'] for m in self.results],
            'MAE': [self.results[m]['mae'] for m in self.results],
            'Accuracy (%)': [self.results[m]['test_r2'] * 100 for m in self.results]
        })

        print("\n", comparison_df.to_string(index=False))

        # Find best model
        best_model = comparison_df.loc[comparison_df['Test R²'].idxmax(), 'Model']
        best_accuracy = comparison_df.loc[comparison_df['Test R²'].idxmax(), 'Accuracy (%)']

        print(f"\n🏆 Best Model: {best_model} with {best_accuracy:.2f}% accuracy")

        return comparison_df

    def plot_feature_importance(self):
        """Plot feature importance for tree-based models."""
        print("\n" + "="*70)
        print("Analyzing Feature Importance...")
        print("="*70)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Random Forest feature importance
        if 'Random Forest' in self.models:
            rf_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': self.models['Random Forest'].feature_importances_
            }).sort_values('importance', ascending=False).head(10)

            axes[0].barh(rf_importance['feature'], rf_importance['importance'])
            axes[0].set_xlabel('Importance')
            axes[0].set_title('Random Forest - Top 10 Feature Importance')
            axes[0].invert_yaxis()

        # XGBoost feature importance
        if 'XGBoost' in self.models:
            xgb_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': self.models['XGBoost'].feature_importances_
            }).sort_values('importance', ascending=False).head(10)

            axes[1].barh(xgb_importance['feature'], xgb_importance['importance'])
            axes[1].set_xlabel('Importance')
            axes[1].set_title('XGBoost - Top 10 Feature Importance')
            axes[1].invert_yaxis()

        plt.tight_layout()
        feature_importance_path = os.path.join(OUTPUT_DIR, 'feature_importance.png')
        plt.savefig(feature_importance_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved as '{feature_importance_path}'")
        plt.close()

    def plot_predictions(self):
        """Plot actual vs predicted values for all models."""
        print("\nGenerating prediction comparison plots...")

        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))

        if n_models == 1:
            axes = [axes]

        for idx, (model_name, results) in enumerate(self.results.items()):
            axes[idx].scatter(self.y_test, results['predictions'], alpha=0.5)
            axes[idx].plot([self.y_test.min(), self.y_test.max()],
                          [self.y_test.min(), self.y_test.max()],
                          'r--', lw=2)
            axes[idx].set_xlabel('Actual Price')
            axes[idx].set_ylabel('Predicted Price')
            axes[idx].set_title(f'{model_name}\nR² = {results["test_r2"]:.4f}')
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        predictions_path = os.path.join(OUTPUT_DIR, 'predictions_comparison.png')
        plt.savefig(predictions_path, dpi=300, bbox_inches='tight')
        print(f"Predictions comparison plot saved as '{predictions_path}'")
        plt.close()

    def plot_residuals(self):
        """Plot residual analysis for all models."""
        print("\nGenerating residual analysis plots...")

        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))

        if n_models == 1:
            axes = [axes]

        for idx, (model_name, results) in enumerate(self.results.items()):
            residuals = self.y_test - results['predictions']
            axes[idx].scatter(results['predictions'], residuals, alpha=0.5)
            axes[idx].axhline(y=0, color='r', linestyle='--', lw=2)
            axes[idx].set_xlabel('Predicted Price')
            axes[idx].set_ylabel('Residuals')
            axes[idx].set_title(f'{model_name} - Residual Plot')
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        residuals_path = os.path.join(OUTPUT_DIR, 'residuals_analysis.png')
        plt.savefig(residuals_path, dpi=300, bbox_inches='tight')
        print(f"Residuals analysis plot saved as '{residuals_path}'")
        plt.close()


def main():
    """Main execution function."""
    print("="*70)
    print("HOUSE PRICE PREDICTION - COMPREHENSIVE ANALYSIS")
    print("="*70)

    # Initialize the prediction system
    predictor = HousePricePrediction()

    # Load and explore data
    predictor.load_data()

    # Feature engineering
    predictor.feature_engineering()

    # Prepare data
    predictor.prepare_data()

    # Train all models
    predictor.train_linear_regression()
    predictor.train_random_forest()
    predictor.train_xgboost()

    # Compare models
    comparison = predictor.compare_models()

    # Feature importance analysis
    predictor.plot_feature_importance()

    # Plot predictions and residuals
    predictor.plot_predictions()
    predictor.plot_residuals()

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated outputs in 'output' directory:")
    print("  - output/feature_importance.png")
    print("  - output/predictions_comparison.png")
    print("  - output/residuals_analysis.png")
    print("\nAll models have been trained and evaluated successfully.")


if __name__ == "__main__":
    main()
