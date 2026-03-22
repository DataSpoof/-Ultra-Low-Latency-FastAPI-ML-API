# -Ultra-Low-Latency-FastAPI-ML-API

# 🚀 Ultra-Low Latency FastAPI ML API

---

## 🎯 Goal

- Achieve **sub-millisecond model inference**
- OR achieve **~1–2 ms end-to-end API latency**

> ⚠️ Note:  
> **< 1 ms total API latency is nearly unrealistic** due to unavoidable overhead (network + serialization + framework).

---

## ⚠️ Latency Breakdown (Where time goes)

| Component        | Approx Time |
|------------------|------------|
| JSON parsing     | 0.2–1 ms   |
| FastAPI overhead | 0.5–2 ms   |
| Model inference  | 0.1–1 ms   |
| Preprocessing    | 0.1–0.5 ms |

👉 Typical total latency: **2–5 ms**

---

## 🔥 Core Optimization Strategy

---

### 1️⃣ Remove Pandas (Critical Optimization)

❌ Avoid:
```python
df = pd.DataFrame([features])
```

✅ Use NumPy:
```python
features = np.array(features).reshape(1, -1)
```

✔ Benefits:
- Saves **1–2 ms**
- Reduces memory overhead

---

### 2️⃣ Load Model at Startup

❌ Avoid loading inside API:
```python
def predict():
    model = load_model()
```

✅ Correct approach:
```python
model = load_model("model.h5")
scaler = joblib.load("scaler.pkl")
```

✔ Benefits:
- Eliminates disk I/O latency

---

### 3️⃣ Switch TensorFlow → ONNX ⚡

#### Why?
- TensorFlow runtime is heavy
- ONNX is lightweight and optimized

#### Setup:
```bash
pip install tf2onnx onnxruntime
```

#### Inference:
```python
session = ort.InferenceSession("model.onnx")
session.run(None, {"input": data})
```

✔ Benefits:
- **2–5x speed improvement**

---

### 4️⃣ Optimize Server Runtime

#### Install:
```bash
pip install uvloop httptools
```

#### Run:
```bash
uvicorn app:app --workers 4 --loop uvloop --http httptools
```

✔ Benefits:
- Faster event loop
- Better concurrency

---

### 5️⃣ Reduce Validation Overhead

❌ Using Pydantic:
```python
def predict(data: InputSchema):
```

✅ Use raw dict:
```python
def predict(data: dict):
```

✔ Benefits:
- Saves **~0.3–0.8 ms**

---

### 6️⃣ Avoid JSON (Advanced Optimization)

#### Problem:
- JSON parsing is slow

#### Alternatives:
- MessagePack (`msgpack`)
- Raw binary formats

✔ Benefits:
- Reduces serialization cost

---

### 7️⃣ Use Batching (High Throughput Systems)

Instead of:
- 1 request → 1 prediction

Use:
- N requests → batch inference

✔ Benefits:
- Better CPU utilization
- Higher throughput

---

### 8️⃣ CPU Optimization

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

✔ Benefits:
- Prevents thread overhead
- Improves latency consistency

---

## ⚡ Optimized FastAPI Example

```python
from fastapi import FastAPI
import numpy as np
import joblib
import onnxruntime as ort

app = FastAPI()

scaler = joblib.load("model/scaler.pkl")
session = ort.InferenceSession("model.onnx")

@app.post("/predict")
def predict(data: dict):
    features = np.array(data["features"], dtype=np.float32).reshape(1, -1)

    features = scaler.transform(features)

    pred = session.run(None, {"input": features})[0][0][0]

    return {
        "prediction": "malware" if pred > 0.5 else "benign",
        "confidence": float(pred)
    }
```

---

## 🧪 Latency Measurement

```python
import time

start = time.perf_counter()
# API call
end = time.perf_counter()

latency_ms = (end - start) * 1000
print(latency_ms)
```

---

## 📊 Realistic Latency Expectations

| Setup                           | Latency    |
|--------------------------------|-----------|
| Default FastAPI                | 5–20 ms   |
| Optimized FastAPI              | 1–3 ms    |
| Extreme tuning (ONNX + no JSON)| 0.5–1.5 ms|
| Pure model inference           | <0.5 ms   |

---

## 🚨 When <1 ms is Required

FastAPI may NOT be suitable.

### Alternatives:
- C++ inference service
- Rust + ONNX Runtime
- TensorRT (GPU optimized)
- gRPC (faster than HTTP)

---

## 🎯 Best Practical Target

👉 Aim for:
- **~2 ms latency**
- Stable and scalable system

✔ This is:
- Production-ready
- Realistic
- Cost-efficient

---

## 🧠 Key Takeaways

- Biggest performance gains:
  - ❌ Remove Pandas
  - ⚡ Use ONNX
  - 🚀 Optimize server runtime

- Avoid unnecessary over-optimization
- Always benchmark before and after changes

---

## 🤝 Next Steps

You can extend this project into:
- 📄 IEEE-style research paper
- 🎥 Instagram reel content
- 📊 Benchmark experiment design

---
