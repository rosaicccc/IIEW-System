<div align="center">

# IIEW-System

### Intelligent Identification and Early Warning System for Adolescent Stress

*A multimodal time-series data fusion system for real-time physiological monitoring, intelligent stress identification, and early warning.*

</div>

---

## Overview

In response to the rising trend of hidden and early-onset psychological stress among adolescents, as well as the limitations of traditional subjective and delayed assessment methods, this study proposes an intelligent adolescent stress identification and early warning system based on multimodal time-series data fusion. Grounded in the international WESAD dataset standard, the research systematically integrates multidimensional physiological signals such as heart rate variability (HRV) and electrodermal activity (EDA), establishing a data preprocessing pipeline featuring feature engineering and dimensionality reduction. Building upon this, an innovative "precise-portable" dual-track Random Forest (RF) model for current stress state classification and a Long Short-Term Memory (LSTM) network for short-term trend prediction were developed. Experimental results demonstrate that the precise RF model achieves a classification accuracy of 85.5% on an independent test set. By extracting core features compatible with consumer-grade wearables (Xiaomi Smart Band 7), the portable RF model maintains an accuracy of 79.2%. Furthermore, the lightweight LSTM model achieves a global accuracy of 0.8583 and a high recall rate of 0.956 in predicting high-stress periods 48 hours in advance. This system successfully breaks through the deployment limitations of professional medical equipment, facilitating a paradigm shift in mental health management from "reactive treatment" to "proactive prevention," thereby providing objective and efficient technological support for early adolescent stress warning in school and family settings.

---

## Results

### 1. Physiological Monitoring at `09:09:40`

<p align="center">
  <img
    src="./results/S16_090940.png"
    width="85%"
    alt="Real-time physiological monitoring results for Subject S16 at 09:09:40"
  />
</p>

<p align="center">
  <em>Figure 1. Real-time physiological monitoring of Subject S16 at 09:09:40.</em>
</p>

---

### 2. Physiological Monitoring at `09:12:10`

<p align="center">
  <img
    src="./results/S16_091210.png"
    width="85%"
    alt="Real-time physiological monitoring results for Subject S16 at 09:12:10"
  />
</p>

<p align="center">
  <em>Figure 2. Real-time physiological monitoring of Subject S16 at 09:12:10.</em>
</p>

---

### 3. Physiological Monitoring at `09:13:20`

<p align="center">
  <img
    src="./results/S16_091320.png"
    width="85%"
    alt="Real-time physiological monitoring results for Subject S16 at 09:13:20"
  />
</p>

<p align="center">
  <em>Figure 3. Real-time physiological monitoring of Subject S16 at 09:13:20.</em>
</p>

---

<div align="center">

**IIEW-System — Intelligent Stress Identification and Early Warning through Multimodal Data Fusion**

</div>
