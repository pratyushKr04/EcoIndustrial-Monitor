# EcoIndustrial-Monitor 🛰️🌱

**EcoIndustrial-Monitor** is an AI-powered environmental monitoring system designed to track industrial compliance using multi-temporal satellite imagery. By leveraging **Sentinel-2** data and deep learning, the system detects land-use changes and monitors vegetation health around industrial zones globally.

## 🚀 Key Features
- **Global ROI Geocoding**: Monitor any industrial zone worldwide by simply entering a city name.
- **Dual Deep Learning Models**:
  - **Vegetation U-Net**: Segment and quantify green cover with high precision.
  - **Siamese Change U-Net**: Detect structural or environmental changes between two time periods.
- **Water Body Filtering**: Integrated NDWI (Normalized Difference Water Index) to filter out ports and coastal waters for cleaner data.
- **Multi-City Training**: Pre-trained on a diverse dataset of 11 cities across 6 continents for robust generalization.
- **Interactive Dashboard**: Premium dark-themed web interface with real-time statistics, compliance reports, and map visualizations.
- **Local Caching**: Optimized data pipeline that caches satellite imagery and OSM data to save bandwidth and time.

## 🛠️ Technology Stack
- **Deep Learning**: TensorFlow / Keras 3
- **Satellite Data**: Google Earth Engine (GEE API)
- **GIS**: OpenStreetMap (OSMNX), GeoPandas, Rasterio
- **Frontend**: Vanilla JS, Chart.js, CSS3 (Glassmorphism)
- **Backend**: Python HTTP Server

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/EcoIndustrial-Monitor.git
   cd EcoIndustrial-Monitor
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Authenticate Google Earth Engine**:
   ```bash
   earthengine authenticate
   ```

## 🚦 Usage

### 1. Model Training
Train the models using the multi-city dataset defined in `config.py`:
```bash
python main.py --mode train
```

### 2. Run Inference
Analyze a specific region (e.g., Chennai, India):
```bash
python main.py --mode infer --roi "Chennai, India"
```

### 3. Launch Dashboard
Visualize the results and compliance metrics:
```bash
python server.py
```
Visit `http://localhost:8000` in your browser.

## 📊 Evaluation Metrics
The system is validated using:
- **IoU (Intersection over Union)**
- **Pixel-wise Precision & Recall**
- **Binary Crossentropy + Dice Loss**

## 📂 Project Structure
```text
├── main.py              # Main entry point
├── server.py            # Dashboard server
├── config.py            # System configuration
├── models/              # U-Net & Siamese architectures
├── training/            # Training pipelines
├── inference/           # Inference & map generation logic
├── utils/               # Satellite, OSM, and NDVI utilities
└── frontend/            # Web dashboard source
```

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
