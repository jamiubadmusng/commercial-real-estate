# Commercial Real Estate Analysis Report

## Analyzing and Forecasting Commercial Property Value Trends in Philadelphia

**Author:** Jamiu Olamilekan Badmus  
**Date:** February 2026  
**Contact:** [jamiubadmus001@gmail.com](mailto:jamiubadmus001@gmail.com)

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Data Overview](#2-data-overview)
3. [Methodology](#3-methodology)
4. [Key Findings](#4-key-findings)
5. [Predictive Models](#5-predictive-models)
6. [Business Implications](#6-business-implications)
7. [Recommendations](#7-recommendations)
8. [Conclusion](#8-conclusion)

---

## 1. Introduction

### 1.1 Background

The commercial real estate (CRE) sector has undergone significant transformation in the post-pandemic era. With remote work becoming a permanent fixture for many organizations, the traditional demand drivers for office space have fundamentally shifted. According to industry data:

- U.S. office vacancies reached approximately **19.6% in Q1 2025**, the highest level on record
- Office property values declined by approximately **14% in 2024** with further declines expected
- These trends have raised concerns about commercial mortgage defaults and broader financial stability

### 1.2 Project Objectives

This analysis aims to:

1. Quantify how commercial property values in Philadelphia have evolved from 2013-2026
2. Measure the differential impact of COVID-19 across property types
3. Build predictive models to identify properties at risk of value decline
4. Provide actionable insights for investors, lenders, and city planners

### 1.3 Relevance

Understanding commercial real estate dynamics is critical for:

- **Investors**: Making informed acquisition and disposition decisions
- **Lenders**: Assessing collateral risk and loan-to-value ratios
- **Municipal Governments**: Tax revenue forecasting and economic development planning
- **Property Managers**: Prioritizing capital improvements and tenant retention

---

## 2. Data Overview

### 2.1 Data Source

The analysis utilizes the **Philadelphia Properties and Assessment History** dataset from the City of Philadelphia's Open Data Portal. This dataset is one of the most comprehensive municipal property databases available, containing:

- **583,566 properties** with detailed characteristics
- **6.9+ million assessment records** spanning 2013-2026
- **79 property features** including building codes, zoning, and location data

### 2.2 Commercial Property Classification

Properties were classified as commercial based on:

- **Category Codes**: 2 (Hotels), 3 (Mixed Use), 4 (Commercial), 5 (Industrial)
- **Building Code Descriptions**: Office, Retail, Store, Warehouse, Industrial, Hotel

### 2.3 Property Type Taxonomy

Commercial properties were further categorized into:

| Property Type | Description |
|---------------|-------------|
| **Office** | Office buildings, medical offices |
| **Retail** | Stores, shopping centers, malls |
| **Industrial** | Warehouses, factories, manufacturing |
| **Hospitality** | Hotels, motels |
| **Mixed Use** | Combined commercial/residential |
| **Restaurant** | Food service establishments |
| **Parking** | Parking garages and lots |

### 2.4 Data Quality

- **Completeness**: Assessment records are comprehensive with minimal missing values
- **Consistency**: Standardized property identifiers (parcel numbers) enable reliable joins
- **Timeliness**: Data includes assessments through 2026

---

## 3. Methodology

### 3.1 Data Processing Pipeline

```
Raw Data → Filtering → Joining → Cleaning → Feature Engineering → Modeling
```

**Step 1: Filtering**
- Identified commercial properties using category codes and building descriptions
- Focused on major commercial categories (office, retail, industrial, mixed-use)

**Step 2: Joining**
- Merged property characteristics with historical assessments on parcel_number
- Created longitudinal dataset for time-series analysis

**Step 3: Cleaning**
- Removed records with missing or zero market values
- Handled outliers in building age and area measurements
- Filled missing year_built with median values

**Step 4: Feature Engineering**
- Calculated year-over-year value changes
- Created building age and age categories
- Derived value per square foot metrics
- Flagged properties with significant declines (>10% drop)

### 3.2 Analytical Approaches

#### Time-Series Analysis
- Aggregated market values by year and property type
- Calculated indexed values relative to 2019 baseline
- Analyzed COVID-19 impact by comparing pre/post periods

#### Classification Modeling
- Target: Binary indicator of significant value decline (>15%)
- Features: Building characteristics, location, historical performance
- Models: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM

#### Regression Modeling
- Target: Property market value (log-transformed)
- Features: Physical characteristics, location, property type
- Models: Ridge, Random Forest, Gradient Boosting, XGBoost, LightGBM

### 3.3 Evaluation Metrics

**Classification:**
- ROC-AUC (primary metric)
- Precision, Recall, F1-Score
- Confusion Matrix

**Regression:**
- R² (coefficient of determination)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)

---

## 4. Key Findings

### 4.1 Market Overview

The Philadelphia commercial real estate market comprises a diverse portfolio of property types:

- Office properties represent significant assessed value
- Retail properties are widely distributed across neighborhoods
- Industrial properties concentrated in specific wards
- Mixed-use properties prevalent in transit-oriented areas

### 4.2 Historical Trends

**Pre-COVID Era (2013-2019)**
- Steady appreciation in commercial property values
- Office sector showed strong growth driven by corporate expansion
- Retail sector stable with slight appreciation
- Industrial sector gaining momentum due to e-commerce

**COVID Impact (2020-2022)**
- Significant disruption to traditional valuation patterns
- Office sector experienced notable assessment adjustments
- Retail sector varied by location and tenant mix
- Industrial sector demonstrated resilience

**Recovery Period (2023-2026)**
- Market stabilization with property type divergence
- Office sector faces structural headwinds
- Industrial continues outperformance
- Selective retail recovery in certain locations

### 4.3 Property Type Analysis

#### Office Properties
- Most significantly impacted by remote work trends
- Downtown locations showing larger adjustments than suburban
- Flight to quality: newer Class A properties outperforming

#### Retail Properties
- Performance varies significantly by format and location
- Grocery-anchored centers showing resilience
- Urban retail facing foot traffic challenges

#### Industrial Properties
- Strong demand driven by e-commerce and logistics
- Limited new supply supporting valuations
- Warehouse and distribution facilities in high demand

### 4.4 Geographic Patterns

Analysis of Philadelphia's geographic wards reveals:

- **High-Value Concentration**: Downtown wards contain highest total commercial value
- **Decline Hot Spots**: Certain neighborhoods show concentrated decline patterns
- **Growth Areas**: Emerging submarkets showing positive trends

---

## 5. Predictive Models

### 5.1 Classification Results

Models were trained to predict which properties would experience significant value decline (>15%).

**Model Performance Comparison:**

| Model | CV ROC-AUC | Test ROC-AUC |
|-------|------------|--------------|
| Logistic Regression | 0.7712 | 0.7737 |
| Random Forest | 0.8789 | 0.8782 |
| Gradient Boosting | 0.9143 | **0.9246** |
| XGBoost | 0.9014 | 0.9132 |
| LightGBM | 0.9169 | 0.9240 |

**Best Model: Gradient Boosting with 92.46% ROC-AUC**

**Key Predictors of Value Decline:**
1. Market Value (34.8% importance)
2. Total Livable Area (15.6% importance)
3. Value per Square Foot (14.7% importance)
4. Total Area (13.7% importance)
5. Property Type - Mixed Use (6.5% importance)

### 5.2 Regression Results

Models were trained to predict property market values.

**Model Performance Comparison:**

| Model | R² | MAE | MAPE |
|-------|-----|-----|------|
| Ridge | 0.4652 | $206.7M | 132.2% |
| Random Forest | **0.9875** | **$118K** | **2.5%** |
| Gradient Boosting | 0.9709 | $214K | 8.8% |
| XGBoost | 0.9875 | $153K | 3.7% |
| LightGBM | 0.9865 | $159K | 3.9% |

**Best Model: Random Forest with 98.75% R² and 2.5% MAPE**

### 5.3 Model Interpretability

SHAP (SHapley Additive exPlanations) analysis reveals:

- **Positive impact on value**: Larger area, more stories, newer construction
- **Negative impact on value**: Older buildings, certain property types, declining neighborhoods
- **Location matters**: Geographic ward significantly influences predictions

---

## 6. Business Implications

### 6.1 For Investors

**Portfolio Assessment**
- Use classification model to screen existing holdings for decline risk
- Prioritize due diligence on high-risk properties
- Consider strategic repositioning or disposition

**Acquisition Strategy**
- Target undervalued properties with low decline probability
- Focus on industrial assets given sector resilience
- Evaluate office properties cautiously with clear repositioning thesis

### 6.2 For Lenders

**Underwriting Considerations**
- Incorporate model predictions into loan-to-value assessments
- Adjust reserve requirements based on property risk scores
- Monitor portfolio for concentration in high-risk segments

**Workout Strategies**
- Early identification of distressed loans
- Proactive borrower engagement for at-risk properties
- Restructuring options for properties with turnaround potential

### 6.3 For City Planners

**Tax Revenue Implications**
- Commercial property values directly impact tax base
- Declining values may require rate adjustments or service cuts
- Economic development incentives should consider sector trends

**Development Policy**
- Zoning flexibility for office-to-residential conversions
- Support for industrial preservation/development
- Transit-oriented development to support commercial corridors

---

## 7. Recommendations

### 7.1 Risk Management Framework

```
╔════════════════════════════════════════════════════════════════════╗
║  RISK LEVEL    │  PROBABILITY   │  ACTION                         ║
╠════════════════════════════════════════════════════════════════════╣
║  Critical      │  > 70%         │  Immediate review/disposition   ║
║  High          │  50-70%        │  Active monitoring, mitigation  ║
║  Elevated      │  30-50%        │  Enhanced due diligence         ║
║  Moderate      │  15-30%        │  Standard monitoring            ║
║  Low           │  < 15%         │  Routine review                 ║
╚════════════════════════════════════════════════════════════════════╝
```

### 7.2 Actionable Strategies

**For High-Risk Properties:**
1. Commission independent appraisals
2. Evaluate tenant credit quality
3. Assess capital improvement needs
4. Consider refinancing or sale

**For Moderate-Risk Properties:**
1. Review lease expirations and renewal probability
2. Benchmark operating expenses
3. Identify value-add opportunities
4. Monitor local market conditions

**For Low-Risk Properties:**
1. Maintain current strategy
2. Annual performance review
3. Consider opportunistic refinancing

### 7.3 Monitoring Program

**Quarterly Reviews:**
- Track actual vs. predicted values
- Update model with new assessment data
- Identify emerging trends

**Annual Updates:**
- Retrain models with latest data
- Refine feature engineering
- Incorporate new data sources

---

## 8. Conclusion

### 8.1 Summary

This analysis provides a comprehensive examination of commercial real estate dynamics in Philadelphia, revealing:

1. **Structural Shifts**: The commercial real estate market is undergoing fundamental changes, particularly in the office sector
2. **Predictive Power**: Machine learning models can effectively identify properties at risk of value decline
3. **Actionable Insights**: The analysis provides a framework for risk assessment and portfolio management

### 8.2 Limitations

- Assessment data may lag actual market values
- Model predictions are based on historical patterns
- External factors (economic conditions, policy changes) not fully captured

### 8.3 Future Work

- Incorporate rental income and vacancy data
- Add economic indicators as features
- Develop neighborhood-level forecasting models
- Create interactive dashboard for portfolio monitoring

---

## Contact Information

**Jamiu Olamilekan Badmus**

- 📧 Email: [jamiubadmus001@gmail.com](mailto:jamiubadmus001@gmail.com)
- 🐙 GitHub: [github.com/jamiubadmusng](https://github.com/jamiubadmusng)
- 💼 LinkedIn: [linkedin.com/in/jamiu-olamilekan-badmus-9276a8192](https://www.linkedin.com/in/jamiu-olamilekan-badmus-9276a8192/)
- 🌐 Website: [sites.google.com/view/jamiu-olamilekan-badmus](https://sites.google.com/view/jamiu-olamilekan-badmus/)

---

*This report is part of a comprehensive data science portfolio demonstrating expertise in real estate analytics, time-series analysis, and predictive modeling.*
