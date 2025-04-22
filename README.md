
# MongoDB vs PostgreSQL Benchmark: 1 Million Record Insertion 🚀

This project benchmarks the insertion performance of **MongoDB** and **PostgreSQL** using a realistic e-commerce dataset. It evaluates how each database handles **single**, **batch**, and **concurrent** insert operations.

> ⚡️ Designed for developers, database enthusiasts, and teams exploring performance under heavy writes.

---

## 📦 Features

- 🛠️ Generates 1M+ realistic records (customers, products, orders)
- 🚀 Benchmarks single, batch, and multi-threaded inserts
- 📊 Measures throughput (records/sec), CPU usage, and memory usage
- 📈 Generates performance charts and a Markdown case study report

---

## 🧰 Requirements

- Python 3.8+
- Docker (for running MongoDB and PostgreSQL)
- Python packages:

```bash
pip install -r requirements.txt
```

---

## 🔧 Setup & Usage

### 1. Clone the Repo

### 2. Start MongoDB and PostgreSQL with Docker

```bash
# MongoDB
docker run -d --name mongo -p 27017:27017 mongo

# PostgreSQL
docker run -d --name postgres -e POSTGRES_PASSWORD=postgres -p 5432:5432 postgres
```

### 3. Generate the Dataset

```bash
python data_generator.py --customers 50000 --products 50000 --orders 900000
```

This will output JSON files in the `./data/` folder.

### 4. Run Benchmarks

#### PostgreSQL

```bash
python benchmark.py --data-dir data --output-dir results \
  --postgres-host localhost --postgres-port 5432 \
  --postgres-user postgres --postgres-password postgres --postgres-db performance_test
```

#### MongoDB (optional – uncomment in code)

```bash
python benchmark.py --data-dir data --output-dir results \
  --mongo-host localhost --mongo-port 27017
```

### 5. Generate Charts and Markdown Report

```bash
python report_generator.py --results-dir results --output-file results/case_study_report.md --charts-dir results
```

This will create visual PNG charts and a full markdown report at `case_study_report.md`.

---

## 📊 Output Example

- Overall throughput bar charts
- Line plots for batch and concurrency scaling
- CPU/memory usage bar charts
- Tables for single/batch/concurrent insert speeds

See sample images in the `/charts/` folder.

---

## 🧠 What’s Next

Future enhancements could include:

- Query performance benchmarks (joins, aggregations)
- Mixed workload simulation (read/write)
- Larger datasets (10M+ records)
- Indexing impact on inserts
- Resource usage over time

---

## 📝 License

MIT License

Copyright (c) 2025 LearnWithHasan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

**THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
