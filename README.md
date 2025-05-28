# Brain Tumor Analysis
BrainTumorAnalysis is an open-source mobile application powered by expo, designed for the automated detection and classification of brain tumors using MRI scans. Leveraging advanced convolutional neural networks (CNNs), the app precisely identifies tumor types, including glioma, meningioma, pituitary, and normal (no tumor), directly from medical images.

The system provides an intuitive and responsive mobile interface built with react Native (expo), allowing medical professionals and users to quickly upload MRI images and instantly receive accurate predictions. Additionally, the backend is supported by a robust Flask-based REST API integrated with state-of-the-art HuggingFace language models (Llama-2/Zephyr), generating concise, real-time summaries to facilitate clinical assessments and enhance diagnostic workflows.

---

##  Features

* Automatic brain tumor detection from MRI images
* Tumor type classification: glioma, meningioma, pituitary, no\_tumor
* REST API for fast predictions (`/predict`)
* Real-time AI summary of recent analysis results (`/gemma_summary`)
* Easy integration with other systems
* Result pool for collecting and summarizing past results
* HuggingFace LLM (Llama-2, Zephyr) for natural language summary

---

##  Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/BrainTumorAnalysis.git
cd BrainTumorAnalysis
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up HuggingFace API key

* Get your token from [HuggingFace Tokens](https://huggingface.co/settings/tokens)
* Add to a `.env` file in the project root:

  ```
  HF_TOKEN=your_huggingface_token_here
  ```
* *(Optional)* Place your trained model file in the `models/` directory.

---

##  Frontend Installation (React or Next.js)

> It's recommended to keep your frontend code inside a `/frontend` folder.



### Start the frontend app:

```bash
cd frontend
npm install
npx expo start
# or
npm start     # For React
```

---

##  API Usage

### Prediction Endpoint

* **POST** `/predict`
* **Form-data:** `"image"` (your MRI image file)

**Example using curl:**

```bash
curl -X POST -F "image=@/path/to/image.jpg" http://localhost:500/predict
```

**Sample response:**

```json
{
  "predicted_class": "glioma",
  "confidence": 0.94,
  "probabilities": [0.94, 0.02, 0.03, 0.01],
  "timestamp": 1717000000.123
}
```

---

### Summary Endpoint

* **GET** `/gemma_summary`
* Returns AI-generated summary of the latest predictions.

**Sample response:**

```json
{
  "summary": "5 MR görüntüsü incelendi. 2 tanesinde tümör tespit edildi. Glioma ve meningioma saptandı. Hastaların ileri tetkiki önerilir.",
  "result_count": 5
}
```

---

##  Example Usage

**Add a new result to pool:**

```python
from ai.add_result import add_result
import time

add_result({
    "predicted_class": "glioma",
    "confidence": 0.93,
    "timestamp": time.time()
})
```

**Generate a summary:**

```python
from ai.gemma_summary import generate_gemma_summary

print(generate_gemma_summary([
    {"predicted_class": "glioma"},
    {"predicted_class": "no_tumor"},
    {"predicted_class": "pituitary"}
]))
```

---

##  Technologies Used

* Python 3.9+ 
* Flask / React Native
* TensorFlow / Keras
* HuggingFace Hub (LLM for summary)
* Pillow, NumPy, threading

---

##  License

MIT License.
All Rights Reserved.

---

##  Acknowledgements

* HuggingFace LLM API
* Brain Tumor MRI Dataset (Kaggle)
* Open source contributors

---

##  Author

Developed by **Veysel SAYAR**
Manisa Celal Bayar University – Software Engineering

---
