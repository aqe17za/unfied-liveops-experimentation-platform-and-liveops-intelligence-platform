"""Quit predictor: XGBoost model predicting match abandonment with SHAP explanations."""

import logging
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
import shap
import yaml
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "config.yaml"


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class QuitPredictor:
    """Predicts the probability a player abandons a match early
    (ragequit) using XGBoost, with SHAP-based per-prediction
    explanations of the top contributing drivers."""

    def __init__(self, config_path=CONFIG_PATH):
        self.logger = logging.getLogger(__name__)
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['liveops']

        self.model_config = self.config['model']['quit_predictor']
        self.db_path = str(ROOT / self.config['data']['db_path'])
        self.processed_path = ROOT / self.config['data']['processed_path']
        self.models_path = ROOT / 'models'
        self.random_state = 42
        self.high_risk_threshold = self.model_config['high_risk_threshold']

        self.models_path.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.label_encoders = {}

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def load_and_prepare(self) -> pd.DataFrame:
        """
        Load PXI scores (player features + a partial MQI merge) and
        MQI scores (full match-level features), and merge on match_id
        to reconstruct the complete feature set needed for modeling.
        """
        pxi_df = pd.read_csv(self.processed_path / 'pxi_scores.csv')
        mqi_df = pd.read_csv(self.processed_path / 'mqi_scores.csv')

        mqi_extra_cols = [
            'match_id', 'skill_gap', 'score_diff_abs',
            'match_competitiveness', 'comeback_flag',
            'time_bucket', 'day_of_week', 'is_weekend'
        ]
        df = pxi_df.merge(mqi_df[mqi_extra_cols], on='match_id', how='left')

        self.logger.info(
            f"Loaded and merged data: shape={df.shape}, "
            f"ragequit rate={df['ragequit_flag'].mean():.1%}"
        )
        return df

    def build_feature_matrix(self, df: pd.DataFrame) -> tuple:
        """
        Select and engineer features for the model.
        Returns (X, y, feature_names).
        """
        df = df.reset_index(drop=True)
        y = df['ragequit_flag'].reset_index(drop=True)

        skill_balance_col = 'comp_skill_balance' if 'comp_skill_balance' in df.columns else 'normalized_skill_gap'

        feature_cols = [
            'mqi_score', skill_balance_col, 'normalized_skill_gap', 'score_diff_abs',
            'match_competitiveness', 'comeback_flag',
            'pxi_score', 'pxi_avg_mqi', 'pxi_session_consistency',
            'pxi_engagement_trend', 'pxi_streak_factor',
            'losing_streak', 'winning_streak', 'ragequit_rate_historical',
            'win_rate_last10', 'session_match_number', 'matches_this_week',
            'recency_days', 'won', 'is_weekend'
        ]
        feature_cols = list(dict.fromkeys(feature_cols))

        X = df[feature_cols].copy()

        for col in ['time_bucket', 'day_of_week']:
            le = LabelEncoder()
            X[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le

        null_counts = X.isna().sum()
        if null_counts.sum() > 0:
            self.logger.info(f"Nulls before fill: {null_counts[null_counts > 0].to_dict()}")
            X = X.fillna(X.median(numeric_only=True))

        self.feature_columns = X.columns.tolist()
        self.logger.info(f"Feature matrix: {X.shape}, Target: ragequit rate = {y.mean():.1%}")

        return X, y, self.feature_columns

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Train XGBoost with stratified K-fold CV.
        Handle class imbalance with scale_pos_weight.
        """
        n_negative = int((y == 0).sum())
        n_positive = int((y == 1).sum())
        scale_pos_weight = n_negative / n_positive
        self.logger.info(
            f"Class distribution: negative={n_negative}, positive={n_positive}, "
            f"scale_pos_weight={scale_pos_weight:.4f}"
        )

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        params = dict(
            n_estimators=self.model_config['n_estimators'],
            learning_rate=self.model_config['learning_rate'],
            max_depth=self.model_config['max_depth'],
            min_child_weight=self.model_config['min_child_weight'],
            subsample=self.model_config['subsample'],
            colsample_bytree=self.model_config['colsample_bytree'],
            random_state=self.random_state,
            scale_pos_weight=scale_pos_weight,
            eval_metric='auc',
            use_label_encoder=False,
        )

        skf = StratifiedKFold(
            n_splits=self.model_config['cv_folds'],
            shuffle=True,
            random_state=self.random_state
        )

        fold_aucs = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y), 1):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            fold_model = XGBClassifier(**params)
            fold_model.fit(X_train, y_train)
            val_proba = fold_model.predict_proba(X_val)[:, 1]
            fold_auc = roc_auc_score(y_val, val_proba)
            fold_aucs.append(fold_auc)
            self.logger.info(f"Fold {fold_idx} AUC: {fold_auc:.4f}")

        cv_auc_mean = float(np.mean(fold_aucs))
        cv_auc_std = float(np.std(fold_aucs))
        self.logger.info(f"CV AUC scores: {fold_aucs}")
        self.logger.info(f"Mean CV AUC: {cv_auc_mean:.4f} ± {cv_auc_std:.4f}")

        self.model = XGBClassifier(**params)
        self.model.fit(X_scaled, y)

        y_pred_proba = self.model.predict_proba(X_scaled)[:, 1]
        y_pred = (y_pred_proba >= self.high_risk_threshold).astype(int)

        metrics = {
            'cv_auc_mean': cv_auc_mean,
            'cv_auc_std': cv_auc_std,
            'cv_auc_scores': fold_aucs,
            'train_auc': roc_auc_score(y, y_pred_proba),
            'avg_precision': average_precision_score(y, y_pred_proba),
            'classification_report': classification_report(y, y_pred),
            'confusion_matrix': confusion_matrix(y, y_pred),
            'n_positive': n_positive,
            'n_negative': n_negative,
            'ragequit_rate': float(y.mean()),
            'threshold': self.high_risk_threshold,
        }

        self.logger.info(f"Train AUC: {metrics['train_auc']:.4f}")
        self.logger.info(f"Average Precision: {metrics['avg_precision']:.4f}")
        self.logger.info(f"Classification Report:\n{metrics['classification_report']}")
        self.logger.info(f"Confusion Matrix:\n{metrics['confusion_matrix']}")

        return metrics

    # ------------------------------------------------------------------
    # Prediction and explanation
    # ------------------------------------------------------------------

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict ragequit probability for new data."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def explain(self, X: pd.DataFrame, player_ids=None) -> pd.DataFrame:
        """
        Compute SHAP values and return top 3 drivers per player
        prediction.
        """
        X_scaled = self.scaler.transform(X)
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_scaled)

        quit_probs = self.model.predict_proba(X_scaled)[:, 1]
        feature_names = self.feature_columns

        rows = []
        for i in range(len(X)):
            abs_vals = np.abs(shap_values[i])
            top_3_idx = np.argsort(abs_vals)[-3:][::-1]

            row = {
                'player_id': player_ids[i] if player_ids is not None else i,
                'quit_probability': quit_probs[i],
            }
            for rank, idx in enumerate(top_3_idx, 1):
                shap_val = shap_values[i][idx]
                row[f'driver_{rank}_feature'] = feature_names[idx]
                row[f'driver_{rank}_shap'] = shap_val
                row[f'driver_{rank}_direction'] = (
                    'increases quit risk' if shap_val > 0 else 'decreases quit risk'
                )
            rows.append(row)

        explanations_df = pd.DataFrame(rows)
        self.logger.info(f"SHAP explanations computed for {len(explanations_df)} players")
        return explanations_df

    def flag_high_risk(self, df: pd.DataFrame, X: pd.DataFrame, player_ids=None) -> pd.DataFrame:
        """Identify and return high-risk player-match records."""
        quit_probs = self.predict_proba(X)
        high_risk_mask = quit_probs >= self.high_risk_threshold
        high_risk_indices = np.where(high_risk_mask)[0]

        X_high_risk = X.iloc[high_risk_indices].reset_index(drop=True)
        player_ids_high_risk = (
            np.asarray(player_ids)[high_risk_indices] if player_ids is not None else None
        )

        explanations_high_risk = self.explain(X_high_risk, player_ids_high_risk)

        context_cols = [
            'player_id', 'match_id', 'pxi_score', 'pxi_tier',
            'mqi_score', 'mqi_tier', 'losing_streak',
            'skill_gap', 'ragequit_flag'
        ]
        context_df = df.iloc[high_risk_indices][context_cols].reset_index(drop=True)

        high_risk_df = pd.concat(
            [context_df, explanations_high_risk.drop(columns=['player_id', 'quit_probability'])],
            axis=1
        )
        high_risk_df['quit_probability'] = quit_probs[high_risk_indices]
        high_risk_df = high_risk_df.sort_values('quit_probability', ascending=False).reset_index(drop=True)

        self.logger.info(
            f"Flagged {len(high_risk_df)} high-risk matches "
            f"({len(high_risk_df) / len(df):.1%} of total)"
        )

        return high_risk_df

    # ------------------------------------------------------------------
    # Save and load
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Save model artifacts to models/."""
        joblib.dump(self.model, self.models_path / 'quit_predictor.pkl')
        joblib.dump(self.scaler, self.models_path / 'scaler.pkl')
        joblib.dump(self.feature_columns, self.models_path / 'feature_columns.pkl')
        joblib.dump(self.label_encoders, self.models_path / 'label_encoders.pkl')
        self.logger.info(f"Model artifacts saved to {self.models_path}")

    def load(self) -> None:
        """Load model artifacts from models/."""
        self.model = joblib.load(self.models_path / 'quit_predictor.pkl')
        self.scaler = joblib.load(self.models_path / 'scaler.pkl')
        self.feature_columns = joblib.load(self.models_path / 'feature_columns.pkl')
        self.label_encoders = joblib.load(self.models_path / 'label_encoders.pkl')
        self.logger.info(f"Model artifacts loaded from {self.models_path}")

    def save_predictions(self, df: pd.DataFrame, predictions: np.ndarray, explanations: pd.DataFrame) -> None:
        """Save full predictions to processed/ and DuckDB."""
        out_df = df.reset_index(drop=True).copy()
        out_df['quit_probability'] = predictions
        out_df['is_high_risk'] = out_df['quit_probability'] >= self.high_risk_threshold

        explanation_cols = explanations.drop(columns=['player_id', 'quit_probability']).reset_index(drop=True)
        out_df = pd.concat([out_df, explanation_cols], axis=1)

        out_path = self.processed_path / 'quit_predictions.csv'
        out_df.to_csv(out_path, index=False)

        con = duckdb.connect(self.db_path)
        con.execute("CREATE OR REPLACE TABLE quit_predictions AS SELECT * FROM out_df")
        con.close()

        self.logger.info(f"Saved {len(out_df)} rows to {out_path}")
        self.logger.info(
            f"High-risk count: {out_df['is_high_risk'].sum()} "
            f"({out_df['is_high_risk'].mean():.1%})"
        )

    def run(self) -> tuple:
        """
        Full quit predictor pipeline.
        Returns (metrics, high_risk_df, explanations_df)
        """
        self.logger.info("Starting quit predictor pipeline...")

        df = self.load_and_prepare()
        X, y, feature_names = self.build_feature_matrix(df)
        metrics = self.train(X, y)

        self.logger.info("Generating predictions and SHAP...")
        predictions = self.predict_proba(X)

        player_ids = df['player_id'].values if 'player_id' in df.columns else None
        explanations = self.explain(X, player_ids)
        high_risk_df = self.flag_high_risk(df, X, player_ids)

        self.save()
        self.save_predictions(df, predictions, explanations)

        self.logger.info("Quit predictor pipeline complete.")
        return metrics, high_risk_df, explanations


if __name__ == "__main__":
    predictor = QuitPredictor()
    metrics, high_risk_df, explanations = predictor.run()

    print("\n=== QUIT PREDICTOR RESULTS ===")
    print(f"Ragequit rate in data: {metrics['ragequit_rate']:.1%}")
    print(f"CV AUC: {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")
    print(f"Train AUC: {metrics['train_auc']:.4f}")
    print("\nClassification Report:")
    print(metrics['classification_report'])
    print(f"\nHigh-risk matches flagged: {len(high_risk_df)}")
    print("\nSample SHAP explanations:")
    print(explanations.head(5)[[
        'player_id', 'quit_probability',
        'driver_1_feature', 'driver_1_shap',
        'driver_2_feature', 'driver_2_shap'
    ]])
