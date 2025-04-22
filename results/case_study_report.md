# MongoDB vs PostgreSQL: 1 Million Record Insertion Performance Case Study

*Generated on: 2025-04-22 07:22:08*

## 1. Introduction

This case study compares the performance of MongoDB and PostgreSQL when inserting a large number of records. We tested various insertion methods, batch sizes, and concurrency levels to provide a comprehensive analysis of each database's performance characteristics.

![Overall Performance](./charts/overall_performance.png)

## 2. Setup and Environment

### Database Setup

Both databases were deployed as Docker containers on the same host machine to ensure fair comparison. Default configurations were used for both databases with minimal tuning to represent real-world usage.

### Data Model

We used a realistic e-commerce data model with three main entities:

- **Customers**: User profiles with personal information, preferences, and summary data
- **Products**: Product catalog with descriptions, pricing, inventory, and attributes
- **Orders**: Customer orders with line items, shipping details, and payment information

The data includes a variety of data types including strings, numbers, dates, arrays, and nested objects/JSON.

## 3. Testing Methodology

We performed the following types of tests:

1. **Single-record insertions**: Each record inserted with an individual query
2. **Batch insertions**: Multiple records inserted in batches of varying sizes
3. **Concurrent insertions**: Multiple threads/workers inserting data simultaneously

For each test, we measured:

- Total insertion time
- Records inserted per second
- CPU usage
- Memory consumption

## 4. Results

### 4.1 Overall Performance Summary

Average insertion rate (records per second):

- MongoDB: 46012.45
- PostgreSQL: 8900.00

On average, MongoDB was 5.17x faster than PostgreSQL for all insertion operations.

### 4.2 Single Insertion Performance

![Single Insertion Performance](./charts/single_insertion_performance.png)

Single record insertion rates (records per second):

| Database | Customers | Products | Orders |
|----------|-----------|----------|--------|
| Mongodb | 1642.83 | 1590.09 | 1634.56 |
| Postgresql | 2078.72 | 2113.37 | 1888.95 |

### 4.3 Batch Insertion Performance

![Batch Performance - Customers](./charts/batch_performance_customers.png)

![Batch Performance - Products](./charts/batch_performance_products.png)

![Batch Performance - Orders](./charts/batch_performance_orders.png)

Batch insertion performance by batch size (records per second):

**Customers:**

| Database | Batch 1 | Batch 100 | Batch 1000 | Batch 10000 |
|----------|---------|-----------|------------|-------------|
| Mongodb | 1587.59 | 46721.64 | 69319.11 | 78300.73 |
| Postgresql | 1972.53 | 1970.68 | 16466.28 | 29736.00 |

**Products:**

| Database | Batch 1 | Batch 100 | Batch 1000 | Batch 10000 |
|----------|---------|-----------|------------|-------------|
| Mongodb | 1585.99 | 47142.34 | 53167.74 | 58882.23 |
| Postgresql | 1889.19 | 1965.98 | 15549.23 | 26809.28 |

**Orders:**

| Database | Batch 1 | Batch 100 | Batch 1000 | Batch 10000 |
|----------|---------|-----------|------------|-------------|
| Mongodb | 1361.96 | 35111.36 | 40729.57 | 42412.36 |
| Postgresql | 1845.20 | 1947.76 | 17124.67 | 19490.82 |

### 4.4 Concurrent Insertion Performance

![Concurrency Performance - Customers](./charts/concurrency_performance_customers.png)

![Concurrency Performance - Products](./charts/concurrency_performance_products.png)

![Concurrency Performance - Orders](./charts/concurrency_performance_orders.png)

Concurrent insertion performance with batch size 1000 (records per second):

**Customers:**

| Database | 1 Worker | 2 Workers | 4 Workers | 8 Workers |
|----------|----------|-----------|-----------|-----------|
| Mongodb | 69589.87 | 91616.54 | 179212.51 | 181658.42 |
| Postgresql | 16062.99 | 15795.33 | 16589.26 | 11268.10 |

**Products:**

| Database | 1 Worker | 2 Workers | 4 Workers | 8 Workers |
|----------|----------|-----------|-----------|-----------|
| Mongodb | 51901.02 | 83145.62 | 141249.47 | 157419.50 |
| Postgresql | 15645.95 | 16546.31 | 15163.47 | 16349.75 |

**Orders:**

| Database | 1 Worker | 2 Workers | 4 Workers | 8 Workers |
|----------|----------|-----------|-----------|-----------|
| Mongodb | 38285.12 | 64477.26 | 108450.70 | 110510.80 |
| Postgresql | 14631.44 | 15484.67 | 14959.00 | 14644.39 |

### 4.5 Resource Usage

![CPU Usage](./charts/cpu_usage.png)

![Memory Usage](./charts/memory_usage.png)

**Average CPU Usage by Method (%):**

| Database | Single | Batch | Concurrent |
|----------|--------|-------|------------|
| Mongodb | 24.27 | 24.60 | 78.01 |
| Postgresql | 12.30 | 18.77 | 23.72 |

**Average Memory Usage by Method (MB):**

| Database | Single | Batch | Concurrent |
|----------|--------|-------|------------|
| Mongodb | 30.70 | 0.22 | -0.44 |
| Postgresql | 0.08 | 1.73 | 0.59 |

## 5. Analysis

![Improvement Factors](./charts/improvement_factors.png)

### 5.1 Single Insertion Analysis

In single-record insertion tests:

- PostgreSQL was 1.25x faster overall for single insertions
- The performance difference is likely due to MongoDB's simpler write path, as it doesn't need to validate schemas, check constraints, or manage ACID transactions by default
- For PostgreSQL, the overhead of transaction management affects performance even for single-record inserts

### 5.2 Batch Insertion Analysis

In batch insertion tests:

- MongoDB showed a 36.90x performance improvement with large batch sizes (10,000) compared to single inserts
- PostgreSQL showed a 12.50x performance improvement with large batch sizes
- MongoDB benefited more from batch processing than PostgreSQL
- Both databases showed significant improvements with batch sizes of 1,000 or more
- The optimal batch size appears to be around 10,000 records for both databases

### 5.3 Concurrency Analysis

In concurrent insertion tests:

- MongoDB showed a 2.81x performance improvement with 8 workers compared to 1 worker
- PostgreSQL showed a 0.91x performance improvement with 8 workers
- MongoDB scaled better with concurrent operations than PostgreSQL
- Both databases showed diminishing returns beyond 4 workers, likely due to contention
- Combining batching with concurrency provided the best overall performance for both databases

### 5.4 Resource Usage Analysis

- MongoDB consumed more CPU resources across all test types (57.44% vs 21.32%)
- MongoDB used more memory during the insertion tests (2.16 MB vs 0.90 MB)
- Concurrent operations increased resource usage for both databases, as expected
- Batch operations tended to be more resource-efficient for the same throughput

## 6. Conclusions and Recommendations

### 6.1 Summary of Findings

- **Overall Performance Winner**: MongoDB
- **Best for Single Inserts**: PostgreSQL
- **Best for Batch Processing**: MongoDB
- **Best for Concurrent Operations**: MongoDB
- **Most Resource Efficient**: PostgreSQL

### 6.2 Recommendations

Based on the results of our benchmark, we can make the following recommendations:

- **For write-heavy applications with high throughput requirements**: Choose MongoDB and utilize batch insertions with concurrency
- **For applications with strict transactional requirements**: Choose PostgreSQL, but ensure batch operations are used
- **For optimal MongoDB performance**: Use batch sizes of 1,000-10,000 and 4-8 concurrent workers
- **For optimal PostgreSQL performance**: Focus on larger batch sizes (1,000-10,000) rather than increased concurrency
- **For resource-constrained environments**: PostgreSQL may be preferable due to lower resource utilization

### 6.3 Future Work

This benchmark focused solely on insertion performance. For a more comprehensive comparison, future work could include:

- Query performance benchmarks (simple queries, complex joins/aggregations)
- Update and delete performance
- Performance under mixed workloads (reads and writes)
- Impact of indexing on write performance
- Performance with larger datasets (10M+ records)
- Testing with different hardware configurations
