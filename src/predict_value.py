"""
Commercial Property Value Prediction Module

This module provides functionality to predict property values and assess
value decline risk for commercial properties in Philadelphia.

Author: Jamiu Olamilekan Badmus
Email: jamiubadmus001@gmail.com
GitHub: https://github.com/jamiubadmusng
LinkedIn: https://www.linkedin.com/in/jamiu-olamilekan-badmus-9276a8192/
Website: https://sites.google.com/view/jamiu-olamilekan-badmus/
"""

import pandas as pd
import numpy as np
import joblib
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import warnings

warnings.filterwarnings('ignore')


class PropertyValuePredictor:
    """
    A class for predicting commercial property values and assessing decline risk.
    
    This predictor uses pre-trained models to:
    1. Predict whether a property is at risk of significant value decline
    2. Estimate the market value of a property
    
    Attributes:
        classifier: Trained classification model for decline prediction
        regressor: Trained regression model for value prediction
        scaler_class: Scaler for classification features
        scaler_reg: Scaler for regression features
        model_loaded: Boolean indicating if models are loaded successfully
    """
    
    def __init__(self, models_path: str = '../models'):
        """
        Initialize the predictor with trained models.
        
        Args:
            models_path: Path to directory containing saved models
        """
        self.models_path = Path(models_path)
        self.model_loaded = False
        self._load_models()
    
    def _load_models(self) -> None:
        """Load pre-trained models and scalers."""
        try:
            self.classifier = joblib.load(
                self.models_path / 'value_decline_classifier.joblib'
            )
            self.regressor = joblib.load(
                self.models_path / 'property_value_regressor.joblib'
            )
            self.scaler_class = joblib.load(
                self.models_path / 'scaler_classification.joblib'
            )
            self.scaler_reg = joblib.load(
                self.models_path / 'scaler_regression.joblib'
            )
            self.model_loaded = True
            print("✓ Models loaded successfully")
        except FileNotFoundError as e:
            print(f"⚠ Error loading models: {e}")
            print("  Please run the analysis notebook first to train and save models.")
            self.model_loaded = False
    
    def predict_decline_risk(
        self,
        property_data: Dict[str, Union[float, str]],
        return_probability: bool = True
    ) -> Dict[str, Union[int, float, str]]:
        """
        Predict the risk of significant value decline for a property.
        
        Args:
            property_data: Dictionary containing property features:
                - total_area: Total property area (sq ft)
                - total_livable_area: Livable area (sq ft)
                - building_age: Age of building in years
                - number_stories: Number of stories
                - market_value: Current market value ($)
                - value_per_sqft: Value per square foot ($)
                - years_count: Years of assessment history
                - property_type: Type (Office, Retail, Industrial, etc.)
                - size_category: Size category
                - age_category: Age category
            return_probability: Whether to return decline probability
        
        Returns:
            Dictionary with prediction results:
                - risk_prediction: 0 (no decline) or 1 (decline expected)
                - risk_probability: Probability of decline (if requested)
                - risk_level: Categorical risk level
        """
        if not self.model_loaded:
            return {'error': 'Models not loaded'}
        
        # Prepare features
        features = self._prepare_classification_features(property_data)
        
        # Scale numerical features
        num_cols = ['total_area', 'total_livable_area', 'building_age', 
                    'number_stories', 'market_value', 'value_per_sqft', 'years_count']
        features[num_cols] = self.scaler_class.transform(features[num_cols])
        
        # Make prediction
        prediction = self.classifier.predict(features)[0]
        
        result = {
            'risk_prediction': int(prediction),
            'risk_level': self._get_risk_level(prediction)
        }
        
        if return_probability:
            proba = self.classifier.predict_proba(features)[0][1]
            result['risk_probability'] = float(proba)
            result['risk_level'] = self._get_risk_level_from_proba(proba)
        
        return result
    
    def predict_value(
        self,
        property_data: Dict[str, Union[float, str]]
    ) -> Dict[str, Union[float, str]]:
        """
        Predict the market value of a property.
        
        Args:
            property_data: Dictionary containing property features:
                - total_area: Total property area (sq ft)
                - total_livable_area: Livable area (sq ft)
                - building_age: Age of building in years
                - number_stories: Number of stories
                - property_type: Type (Office, Retail, Industrial, etc.)
                - size_category: Size category
                - age_category: Age category
        
        Returns:
            Dictionary with prediction results:
                - predicted_value: Estimated market value ($)
                - value_range_low: Lower bound of estimate
                - value_range_high: Upper bound of estimate
        """
        if not self.model_loaded:
            return {'error': 'Models not loaded'}
        
        # Prepare features
        features = self._prepare_regression_features(property_data)
        
        # Scale numerical features
        num_cols = ['total_area', 'total_livable_area', 'building_age', 
                    'number_stories', 'value_per_sqft', 'years_count']
        # Filter to only columns that exist
        num_cols = [c for c in num_cols if c in features.columns]
        features[num_cols] = self.scaler_reg.transform(features[num_cols])
        
        # Make prediction (model predicts log value)
        log_prediction = self.regressor.predict(features)[0]
        predicted_value = np.expm1(log_prediction)
        
        # Estimate range (±20% as rough confidence interval)
        value_range_low = predicted_value * 0.8
        value_range_high = predicted_value * 1.2
        
        return {
            'predicted_value': float(predicted_value),
            'value_range_low': float(value_range_low),
            'value_range_high': float(value_range_high),
            'formatted_value': f"${predicted_value:,.0f}"
        }
    
    def _prepare_classification_features(
        self,
        data: Dict[str, Union[float, str]]
    ) -> pd.DataFrame:
        """Prepare features for classification model."""
        # Create base dataframe
        df = pd.DataFrame([data])
        
        # One-hot encode categorical variables
        categorical_cols = ['property_type', 'size_category', 'age_category']
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        # Ensure all expected columns exist (with 0 for missing)
        expected_cols = self._get_expected_classification_columns()
        for col in expected_cols:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        
        return df_encoded[expected_cols]
    
    def _prepare_regression_features(
        self,
        data: Dict[str, Union[float, str]]
    ) -> pd.DataFrame:
        """Prepare features for regression model."""
        # Create base dataframe
        df = pd.DataFrame([data])
        
        # One-hot encode categorical variables
        categorical_cols = ['property_type', 'size_category', 'age_category']
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        # Ensure all expected columns exist
        expected_cols = self._get_expected_regression_columns()
        for col in expected_cols:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        
        return df_encoded[expected_cols]
    
    def _get_expected_classification_columns(self) -> list:
        """Get expected columns for classification model."""
        # Base numerical columns
        cols = ['total_area', 'total_livable_area', 'building_age', 
                'number_stories', 'market_value', 'value_per_sqft', 'years_count']
        
        # Add categorical columns (assuming these were used in training)
        property_types = ['Industrial', 'Mixed Use', 'Office', 'Other Commercial', 
                         'Parking', 'Restaurant', 'Retail']
        size_cats = ['Large', 'Mega', 'Medium', 'Very Large']
        age_cats = ['Historic', 'Mature', 'Modern', 'Very Old']
        
        cols.extend([f'property_type_{t}' for t in property_types])
        cols.extend([f'size_category_{s}' for s in size_cats])
        cols.extend([f'age_category_{a}' for a in age_cats])
        
        return cols
    
    def _get_expected_regression_columns(self) -> list:
        """Get expected columns for regression model."""
        # Base numerical columns (excluding market_value which is the target)
        cols = ['total_area', 'total_livable_area', 'building_age', 
                'number_stories', 'value_per_sqft', 'years_count']
        
        # Add categorical columns
        property_types = ['Industrial', 'Mixed Use', 'Office', 'Other Commercial', 
                         'Parking', 'Restaurant', 'Retail']
        size_cats = ['Large', 'Mega', 'Medium', 'Very Large']
        age_cats = ['Historic', 'Mature', 'Modern', 'Very Old']
        
        cols.extend([f'property_type_{t}' for t in property_types])
        cols.extend([f'size_category_{s}' for s in size_cats])
        cols.extend([f'age_category_{a}' for a in age_cats])
        
        return cols
    
    @staticmethod
    def _get_risk_level(prediction: int) -> str:
        """Convert binary prediction to risk level."""
        return 'HIGH RISK' if prediction == 1 else 'LOW RISK'
    
    @staticmethod
    def _get_risk_level_from_proba(probability: float) -> str:
        """Convert probability to categorical risk level."""
        if probability >= 0.7:
            return 'CRITICAL'
        elif probability >= 0.5:
            return 'HIGH'
        elif probability >= 0.3:
            return 'ELEVATED'
        elif probability >= 0.15:
            return 'MODERATE'
        else:
            return 'LOW'


def assess_property(
    parcel_number: str,
    data_path: str = '../data/processed/commercial_properties_processed.csv',
    models_path: str = '../models'
) -> Dict:
    """
    Assess a specific property by parcel number.
    
    Args:
        parcel_number: The property's parcel number
        data_path: Path to processed property data
        models_path: Path to saved models
    
    Returns:
        Dictionary with property details and predictions
    """
    # Load processed data
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        return {'error': f'Data file not found: {data_path}'}
    
    # Find property
    property_data = data[data['parcel_number'] == parcel_number]
    
    if len(property_data) == 0:
        return {'error': f'Property not found: {parcel_number}'}
    
    # Get latest record
    latest = property_data.sort_values('year', ascending=False).iloc[0]
    
    # Initialize predictor
    predictor = PropertyValuePredictor(models_path)
    
    if not predictor.model_loaded:
        return {'error': 'Models not loaded'}
    
    # Prepare property data
    property_dict = {
        'total_area': latest.get('total_area', 0),
        'total_livable_area': latest.get('total_livable_area', 0),
        'building_age': latest.get('building_age', 50),
        'number_stories': latest.get('number_stories', 1),
        'market_value': latest.get('market_value', 0),
        'value_per_sqft': latest.get('market_value', 0) / max(latest.get('total_area', 1), 1),
        'years_count': len(property_data),
        'property_type': latest.get('property_type', 'Other Commercial'),
        'size_category': _get_size_category(latest.get('total_area', 0)),
        'age_category': _get_age_category(latest.get('building_age', 50))
    }
    
    # Get predictions
    decline_result = predictor.predict_decline_risk(property_dict)
    value_result = predictor.predict_value(property_dict)
    
    return {
        'parcel_number': parcel_number,
        'property_type': latest.get('property_type'),
        'current_value': float(latest.get('market_value', 0)),
        'building_age': int(latest.get('building_age', 0)),
        'total_area': float(latest.get('total_area', 0)),
        'decline_risk': decline_result,
        'value_prediction': value_result
    }


def _get_size_category(area: float) -> str:
    """Categorize property by size."""
    if area < 2000:
        return 'Small'
    elif area < 5000:
        return 'Medium'
    elif area < 10000:
        return 'Large'
    elif area < 50000:
        return 'Very Large'
    else:
        return 'Mega'


def _get_age_category(age: float) -> str:
    """Categorize property by age."""
    if age < 20:
        return 'New'
    elif age < 50:
        return 'Modern'
    elif age < 80:
        return 'Mature'
    elif age < 100:
        return 'Historic'
    else:
        return 'Very Old'


def print_assessment_report(assessment: Dict) -> None:
    """Print a formatted assessment report."""
    if 'error' in assessment:
        print(f"Error: {assessment['error']}")
        return
    
    print("\n" + "="*60)
    print("COMMERCIAL PROPERTY ASSESSMENT REPORT")
    print("="*60)
    
    print(f"\n📍 Parcel Number: {assessment['parcel_number']}")
    print(f"🏢 Property Type: {assessment['property_type']}")
    print(f"💰 Current Value: ${assessment['current_value']:,.0f}")
    print(f"📏 Total Area: {assessment['total_area']:,.0f} sq ft")
    print(f"📅 Building Age: {assessment['building_age']} years")
    
    print("\n" + "-"*60)
    print("RISK ASSESSMENT")
    print("-"*60)
    
    decline = assessment['decline_risk']
    if 'error' not in decline:
        print(f"⚠️  Risk Level: {decline.get('risk_level', 'N/A')}")
        if 'risk_probability' in decline:
            print(f"📊 Decline Probability: {decline['risk_probability']*100:.1f}%")
    
    print("\n" + "-"*60)
    print("VALUE ESTIMATE")
    print("-"*60)
    
    value = assessment['value_prediction']
    if 'error' not in value:
        print(f"💵 Predicted Value: {value.get('formatted_value', 'N/A')}")
        print(f"📈 Range: ${value.get('value_range_low', 0):,.0f} - ${value.get('value_range_high', 0):,.0f}")
    
    print("\n" + "="*60)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Commercial Property Value Prediction Tool'
    )
    parser.add_argument(
        '--parcel_number', '-p',
        type=str,
        help='Parcel number of the property to assess'
    )
    parser.add_argument(
        '--data_path', '-d',
        type=str,
        default='../data/processed/commercial_properties_processed.csv',
        help='Path to processed property data'
    )
    parser.add_argument(
        '--models_path', '-m',
        type=str,
        default='../models',
        help='Path to saved models directory'
    )
    
    args = parser.parse_args()
    
    if args.parcel_number:
        assessment = assess_property(
            args.parcel_number,
            args.data_path,
            args.models_path
        )
        print_assessment_report(assessment)
    else:
        print("Please provide a parcel number with --parcel_number or -p")
        print("Example: python predict_value.py -p 123456789")


if __name__ == '__main__':
    main()
