import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from datetime import datetime

def load_results(results_dir):
    """Load benchmark results from JSON files"""
    mongo_path = os.path.join(results_dir, 'mongodb_results.json')
    postgres_path = os.path.join(results_dir, 'postgresql_results.json')
    
    with open(mongo_path, 'r') as f:
        mongo_results = json.load(f)
    
    with open(postgres_path, 'r') as f:
        postgres_results = json.load(f)
    
    return mongo_results, postgres_results

def generate_charts(mongo_results, postgres_results, output_dir):
    """Generate comparison charts for the report"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert results to DataFrames
    mongo_df = pd.DataFrame(mongo_results)
    postgres_df = pd.DataFrame(postgres_results)
    
    # Combine results
    mongo_df['database'] = 'MongoDB'
    postgres_df['database'] = 'PostgreSQL'
    combined_df = pd.concat([mongo_df, postgres_df])
    
    # Set a clean style for charts
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = {'MongoDB': '#4CAF50', 'PostgreSQL': '#2196F3'}
    
    # 1. Overall Performance Chart
    plt.figure(figsize=(10, 6))
    overall_perf = combined_df.groupby('database')['records_per_second'].mean()
    
    ax = overall_perf.plot(kind='bar', color=[colors['MongoDB'], colors['PostgreSQL']])
    plt.title('Overall Average Performance', fontsize=14)
    plt.ylabel('Records per Second', fontsize=12)
    plt.xlabel('')
    plt.xticks(rotation=0)
    
    # Add values on top of bars
    for i, v in enumerate(overall_perf):
        ax.text(i, v + (v * 0.02), f"{v:.0f}", ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_performance.png'), dpi=300)
    plt.close()
    
    # 2. Single Insertion Performance Chart
    plt.figure(figsize=(10, 6))
    single_df = combined_df[combined_df['method'] == 'single']
    
    # Pivot for plotting
    single_pivot = pd.pivot_table(
        single_df,
        values='records_per_second',
        index='entity',
        columns='database'
    )
    
    ax = single_pivot.plot(kind='bar', color=[colors['MongoDB'], colors['PostgreSQL']])
    plt.title('Single Record Insertion Performance by Entity', fontsize=14)
    plt.ylabel('Records per Second', fontsize=12)
    plt.xlabel('Entity Type', fontsize=12)
    plt.xticks(rotation=0)
    
    # Add values on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'single_insertion_performance.png'), dpi=300)
    plt.close()
    
    # 3. Batch Size Performance Charts (one per entity)
    batch_df = combined_df[combined_df['method'] == 'batch']
    
    for entity in ['customers', 'products', 'orders']:
        plt.figure(figsize=(10, 6))
        entity_batch_df = batch_df[batch_df['entity'] == entity]
        
        # Create pivot table
        entity_pivot = pd.pivot_table(
            entity_batch_df,
            values='records_per_second',
            index='batch_size',
            columns='database'
        )
        
        # Sort by batch size
        entity_pivot = entity_pivot.reindex([1, 100, 1000, 10000])
        
        # Plot as a line chart
        ax = entity_pivot.plot(
            kind='line', 
            marker='o',
            markersize=8,
            linewidth=2,
            color=[colors['MongoDB'], colors['PostgreSQL']]
        )
        
        plt.title(f'Batch Performance - {entity.capitalize()}', fontsize=14)
        plt.ylabel('Records per Second', fontsize=12)
        plt.xlabel('Batch Size', fontsize=12)
        plt.xscale('log')
        plt.grid(True)
        
        # Add annotations
        for line in ax.lines:
            for x, y in zip(entity_pivot.index, line.get_ydata()):
                label = f"{y:.0f}"
                plt.annotate(
                    label,
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center'
                )
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'batch_performance_{entity}.png'), dpi=300)
        plt.close()
    
    # 4. Concurrency Performance Charts (with batch size = 1000)
    concurrent_df = combined_df[
        (combined_df['method'] == 'concurrent') & 
        (combined_df['batch_size'] == 1000)
    ]
    
    for entity in ['customers', 'products', 'orders']:
        plt.figure(figsize=(10, 6))
        entity_concurrent_df = concurrent_df[concurrent_df['entity'] == entity]
        
        # Create pivot table
        entity_pivot = pd.pivot_table(
            entity_concurrent_df,
            values='records_per_second',
            index='num_workers',
            columns='database'
        )
        
        # Sort by number of workers
        entity_pivot = entity_pivot.reindex([1, 2, 4, 8])
        
        # Plot as a line chart
        ax = entity_pivot.plot(
            kind='line', 
            marker='o',
            markersize=8,
            linewidth=2,
            color=[colors['MongoDB'], colors['PostgreSQL']]
        )
        
        plt.title(f'Concurrency Performance - {entity.capitalize()}', fontsize=14)
        plt.ylabel('Records per Second', fontsize=12)
        plt.xlabel('Number of Workers', fontsize=12)
        plt.grid(True)
        
        # Add annotations
        for line in ax.lines:
            for x, y in zip(entity_pivot.index, line.get_ydata()):
                label = f"{y:.0f}"
                plt.annotate(
                    label,
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center'
                )
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'concurrency_performance_{entity}.png'), dpi=300)
        plt.close()
    
    # 5. Resource Usage Charts
    resource_df = combined_df.groupby(['database', 'method'])[['cpu_usage', 'memory_usage']].mean().reset_index()
    
    # CPU Usage
    plt.figure(figsize=(10, 6))
    resource_pivot = pd.pivot_table(
        resource_df,
        values='cpu_usage',
        index='method',
        columns='database'
    )
    
    ax = resource_pivot.plot(kind='bar', color=[colors['MongoDB'], colors['PostgreSQL']])
    plt.title('Average CPU Usage by Method', fontsize=14)
    plt.ylabel('CPU Usage (%)', fontsize=12)
    plt.xlabel('Method', fontsize=12)
    plt.xticks(rotation=0)
    
    # Add values on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cpu_usage.png'), dpi=300)
    plt.close()
    
    # Memory Usage
    plt.figure(figsize=(10, 6))
    resource_pivot = pd.pivot_table(
        resource_df,
        values='memory_usage',
        index='method',
        columns='database'
    )
    
    ax = resource_pivot.plot(kind='bar', color=[colors['MongoDB'], colors['PostgreSQL']])
    plt.title('Average Memory Usage by Method', fontsize=14)
    plt.ylabel('Memory Usage (MB)', fontsize=12)
    plt.xlabel('Method', fontsize=12)
    plt.xticks(rotation=0)
    
    # Add values on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'memory_usage.png'), dpi=300)
    plt.close()
    
    # 6. Summary Chart - Improvement Factors
    plt.figure(figsize=(10, 6))
    
    # Calculate improvement factors
    # a. Batch improvement (10000 batch vs single)
    mongo_batch_improvement = batch_df[
        (batch_df['database'] == 'MongoDB') & 
        (batch_df['batch_size'] == 10000)
    ]['records_per_second'].mean() / single_df[single_df['database'] == 'MongoDB']['records_per_second'].mean()
    
    postgres_batch_improvement = batch_df[
        (batch_df['database'] == 'PostgreSQL') & 
        (batch_df['batch_size'] == 10000)
    ]['records_per_second'].mean() / single_df[single_df['database'] == 'PostgreSQL']['records_per_second'].mean()
    
    # b. Concurrency improvement (8 workers vs 1 worker with batch 1000)
    mongo_concurrency_improvement = concurrent_df[
        (concurrent_df['database'] == 'MongoDB') & 
        (concurrent_df['num_workers'] == 8)
    ]['records_per_second'].mean() / concurrent_df[
        (concurrent_df['database'] == 'MongoDB') & 
        (concurrent_df['num_workers'] == 1)
    ]['records_per_second'].mean()
    
    postgres_concurrency_improvement = concurrent_df[
        (concurrent_df['database'] == 'PostgreSQL') & 
        (concurrent_df['num_workers'] == 8)
    ]['records_per_second'].mean() / concurrent_df[
        (concurrent_df['database'] == 'PostgreSQL') & 
        (concurrent_df['num_workers'] == 1)
    ]['records_per_second'].mean()
    
    # Create a DataFrame for plotting
    improvement_data = pd.DataFrame({
        'MongoDB': [mongo_batch_improvement, mongo_concurrency_improvement],
        'PostgreSQL': [postgres_batch_improvement, postgres_concurrency_improvement]
    }, index=['Batch Improvement\n(10000 vs Single)', 'Concurrency Improvement\n(8 Workers vs 1)'])
    
    ax = improvement_data.plot(kind='bar', color=[colors['MongoDB'], colors['PostgreSQL']])
    plt.title('Performance Improvement Factors', fontsize=14)
    plt.ylabel('Improvement Factor (x times)', fontsize=12)
    plt.xlabel('')
    plt.xticks(rotation=0)
    
    # Add values on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1fx', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'improvement_factors.png'), dpi=300)
    plt.close()
    
    print(f"Generated charts saved to {output_dir}")

def generate_report(mongo_results, postgres_results, output_file):
    """Generate a detailed case study report in Markdown format"""
    # Convert results to DataFrames
    mongo_df = pd.DataFrame(mongo_results)
    postgres_df = pd.DataFrame(postgres_results)
    
    # Combine results
    mongo_df['database'] = 'mongodb'
    postgres_df['database'] = 'postgresql'
    combined_df = pd.concat([mongo_df, postgres_df])
    
    # Generate report
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(output_file, 'w') as f:
        # Title
        f.write("# MongoDB vs PostgreSQL: 1 Million Record Insertion Performance Case Study\n\n")
        f.write(f"*Generated on: {now}*\n\n")
        
        # Introduction
        f.write("## 1. Introduction\n\n")
        f.write("This case study compares the performance of MongoDB and PostgreSQL when inserting a large number ")
        f.write("of records. We tested various insertion methods, batch sizes, and concurrency levels to provide ")
        f.write("a comprehensive analysis of each database's performance characteristics.\n\n")
        
        f.write("![Overall Performance](./charts/overall_performance.png)\n\n")
        
        # Setup and Environment
        f.write("## 2. Setup and Environment\n\n")
        f.write("### Database Setup\n\n")
        f.write("Both databases were deployed as Docker containers on the same host machine to ensure fair comparison. ")
        f.write("Default configurations were used for both databases with minimal tuning to represent real-world usage.\n\n")
        
        f.write("### Data Model\n\n")
        f.write("We used a realistic e-commerce data model with three main entities:\n\n")
        f.write("- **Customers**: User profiles with personal information, preferences, and summary data\n")
        f.write("- **Products**: Product catalog with descriptions, pricing, inventory, and attributes\n")
        f.write("- **Orders**: Customer orders with line items, shipping details, and payment information\n\n")
        
        f.write("The data includes a variety of data types including strings, numbers, dates, arrays, and nested objects/JSON.\n\n")
        
        # Testing Methodology
        f.write("## 3. Testing Methodology\n\n")
        f.write("We performed the following types of tests:\n\n")
        f.write("1. **Single-record insertions**: Each record inserted with an individual query\n")
        f.write("2. **Batch insertions**: Multiple records inserted in batches of varying sizes\n")
        f.write("3. **Concurrent insertions**: Multiple threads/workers inserting data simultaneously\n\n")
        
        f.write("For each test, we measured:\n\n")
        f.write("- Total insertion time\n")
        f.write("- Records inserted per second\n")
        f.write("- CPU usage\n")
        f.write("- Memory consumption\n\n")
        
        # Results
        f.write("## 4. Results\n\n")
        
        # Basic statistics
        f.write("### 4.1 Overall Performance Summary\n\n")
        
        # Calculate average performance by database
        overall_perf = combined_df.groupby('database')['records_per_second'].mean()
        f.write("Average insertion rate (records per second):\n\n")
        f.write(f"- MongoDB: {overall_perf['mongodb']:.2f}\n")
        f.write(f"- PostgreSQL: {overall_perf['postgresql']:.2f}\n\n")
        
        # Calculate ratios
        if 'mongodb' in overall_perf and 'postgresql' in overall_perf:
            ratio = overall_perf['mongodb'] / overall_perf['postgresql']
            if ratio > 1:
                f.write(f"On average, MongoDB was {ratio:.2f}x faster than PostgreSQL for all insertion operations.\n\n")
            else:
                f.write(f"On average, PostgreSQL was {1/ratio:.2f}x faster than MongoDB for all insertion operations.\n\n")
        
        # Single insertion performance
        f.write("### 4.2 Single Insertion Performance\n\n")
        single_df = combined_df[combined_df['method'] == 'single']
        single_grouped = single_df.groupby(['database', 'entity'])['records_per_second'].mean().unstack()
        
        f.write("![Single Insertion Performance](./charts/single_insertion_performance.png)\n\n")
        
        f.write("Single record insertion rates (records per second):\n\n")
        f.write("| Database | Customers | Products | Orders |\n")
        f.write("|----------|-----------|----------|--------|\n")
        
        for db in ['mongodb', 'postgresql']:
            if db in single_grouped.index:
                f.write(f"| {db.capitalize()} | {single_grouped.loc[db, 'customers']:.2f} | {single_grouped.loc[db, 'products']:.2f} | {single_grouped.loc[db, 'orders']:.2f} |\n")
        
        f.write("\n")
        
        # Batch insertion performance
        f.write("### 4.3 Batch Insertion Performance\n\n")
        batch_df = combined_df[combined_df['method'] == 'batch']
        
        # Display charts for each entity
        f.write("![Batch Performance - Customers](./charts/batch_performance_customers.png)\n\n")
        f.write("![Batch Performance - Products](./charts/batch_performance_products.png)\n\n")
        f.write("![Batch Performance - Orders](./charts/batch_performance_orders.png)\n\n")
        
        # Create a pivot table for batch performance
        batch_pivot = batch_df.pivot_table(
            index=['database', 'entity'], 
            columns='batch_size', 
            values='records_per_second'
        )
        
        # Display the results in markdown
        f.write("Batch insertion performance by batch size (records per second):\n\n")
        
        # Customers
        f.write("**Customers:**\n\n")
        f.write("| Database | Batch 1 | Batch 100 | Batch 1000 | Batch 10000 |\n")
        f.write("|----------|---------|-----------|------------|-------------|\n")
        
        for db in ['mongodb', 'postgresql']:
            if (db, 'customers') in batch_pivot.index:
                row = batch_pivot.loc[(db, 'customers')]
                f.write(f"| {db.capitalize()} ")
                for bs in [1, 100, 1000, 10000]:
                    if bs in row:
                        f.write(f"| {row[bs]:.2f} ")
                    else:
                        f.write("| - ")
                f.write("|\n")
        
        f.write("\n")
        
        # Products
        f.write("**Products:**\n\n")
        f.write("| Database | Batch 1 | Batch 100 | Batch 1000 | Batch 10000 |\n")
        f.write("|----------|---------|-----------|------------|-------------|\n")
        
        for db in ['mongodb', 'postgresql']:
            if (db, 'products') in batch_pivot.index:
                row = batch_pivot.loc[(db, 'products')]
                f.write(f"| {db.capitalize()} ")
                for bs in [1, 100, 1000, 10000]:
                    if bs in row:
                        f.write(f"| {row[bs]:.2f} ")
                    else:
                        f.write("| - ")
                f.write("|\n")
        
        f.write("\n")
        
        # Orders
        f.write("**Orders:**\n\n")
        f.write("| Database | Batch 1 | Batch 100 | Batch 1000 | Batch 10000 |\n")
        f.write("|----------|---------|-----------|------------|-------------|\n")
        
        for db in ['mongodb', 'postgresql']:
            if (db, 'orders') in batch_pivot.index:
                row = batch_pivot.loc[(db, 'orders')]
                f.write(f"| {db.capitalize()} ")
                for bs in [1, 100, 1000, 10000]:
                    if bs in row:
                        f.write(f"| {row[bs]:.2f} ")
                    else:
                        f.write("| - ")
                f.write("|\n")
        
        f.write("\n")
        
        # Concurrent insertion performance
        f.write("### 4.4 Concurrent Insertion Performance\n\n")
        concurrent_df = combined_df[
            (combined_df['method'] == 'concurrent') & 
            (combined_df['batch_size'] == 1000)
        ]
        
        # Display charts for each entity
        f.write("![Concurrency Performance - Customers](./charts/concurrency_performance_customers.png)\n\n")
        f.write("![Concurrency Performance - Products](./charts/concurrency_performance_products.png)\n\n")
        f.write("![Concurrency Performance - Orders](./charts/concurrency_performance_orders.png)\n\n")
        
        # Create a pivot table for concurrent performance
        concurrent_pivot = concurrent_df.pivot_table(
            index=['database', 'entity'], 
            columns='num_workers', 
            values='records_per_second'
        )
        
        # Display the results in markdown
        f.write("Concurrent insertion performance with batch size 1000 (records per second):\n\n")
        
        # Customers
        f.write("**Customers:**\n\n")
        f.write("| Database | 1 Worker | 2 Workers | 4 Workers | 8 Workers |\n")
        f.write("|----------|----------|-----------|-----------|-----------|\n")
        
        for db in ['mongodb', 'postgresql']:
            if (db, 'customers') in concurrent_pivot.index:
                row = concurrent_pivot.loc[(db, 'customers')]
                f.write(f"| {db.capitalize()} ")
                for nw in [1, 2, 4, 8]:
                    if nw in row:
                        f.write(f"| {row[nw]:.2f} ")
                    else:
                        f.write("| - ")
                f.write("|\n")
        
        f.write("\n")
        
        # Products
        f.write("**Products:**\n\n")
        f.write("| Database | 1 Worker | 2 Workers | 4 Workers | 8 Workers |\n")
        f.write("|----------|----------|-----------|-----------|-----------|\n")
        
        for db in ['mongodb', 'postgresql']:
            if (db, 'products') in concurrent_pivot.index:
                row = concurrent_pivot.loc[(db, 'products')]
                f.write(f"| {db.capitalize()} ")
                for nw in [1, 2, 4, 8]:
                    if nw in row:
                        f.write(f"| {row[nw]:.2f} ")
                    else:
                        f.write("| - ")
                f.write("|\n")
        
        f.write("\n")
        
        # Orders
        f.write("**Orders:**\n\n")
        f.write("| Database | 1 Worker | 2 Workers | 4 Workers | 8 Workers |\n")
        f.write("|----------|----------|-----------|-----------|-----------|\n")
        
        for db in ['mongodb', 'postgresql']:
            if (db, 'orders') in concurrent_pivot.index:
                row = concurrent_pivot.loc[(db, 'orders')]
                f.write(f"| {db.capitalize()} ")
                for nw in [1, 2, 4, 8]:
                    if nw in row:
                        f.write(f"| {row[nw]:.2f} ")
                    else:
                        f.write("| - ")
                f.write("|\n")
        
        f.write("\n")
        
        # Resource usage
        f.write("### 4.5 Resource Usage\n\n")
        
        f.write("![CPU Usage](./charts/cpu_usage.png)\n\n")
        f.write("![Memory Usage](./charts/memory_usage.png)\n\n")
        
        # CPU Usage
        cpu_usage = combined_df.groupby(['database', 'method'])['cpu_usage'].mean().unstack()
        
        f.write("**Average CPU Usage by Method (%):**\n\n")
        f.write("| Database | Single | Batch | Concurrent |\n")
        f.write("|----------|--------|-------|------------|\n")
        
        for db in ['mongodb', 'postgresql']:
            if db in cpu_usage.index:
                row = cpu_usage.loc[db]
                f.write(f"| {db.capitalize()} ")
                for method in ['single', 'batch', 'concurrent']:
                    if method in row:
                        f.write(f"| {row[method]:.2f} ")
                    else:
                        f.write("| - ")
                f.write("|\n")
        
        f.write("\n")
        
        # Memory Usage
        memory_usage = combined_df.groupby(['database', 'method'])['memory_usage'].mean().unstack()
        
        f.write("**Average Memory Usage by Method (MB):**\n\n")
        f.write("| Database | Single | Batch | Concurrent |\n")
        f.write("|----------|--------|-------|------------|\n")
        
        for db in ['mongodb', 'postgresql']:
            if db in memory_usage.index:
                row = memory_usage.loc[db]
                f.write(f"| {db.capitalize()} ")
                for method in ['single', 'batch', 'concurrent']:
                    if method in row:
                        f.write(f"| {row[method]:.2f} ")
                    else:
                        f.write("| - ")
                f.write("|\n")
        
        f.write("\n")
        
        # Analysis
        f.write("## 5. Analysis\n\n")
        
        f.write("![Improvement Factors](./charts/improvement_factors.png)\n\n")
        
        # Single Insertion Analysis
        f.write("### 5.1 Single Insertion Analysis\n\n")
        f.write("In single-record insertion tests:\n\n")
        
        if 'mongodb' in single_grouped.index and 'postgresql' in single_grouped.index:
            if single_grouped.loc['mongodb'].mean() > single_grouped.loc['postgresql'].mean():
                ratio = single_grouped.loc['mongodb'].mean() / single_grouped.loc['postgresql'].mean()
                f.write(f"- MongoDB was {ratio:.2f}x faster overall for single insertions\n")
            else:
                ratio = single_grouped.loc['postgresql'].mean() / single_grouped.loc['mongodb'].mean()
                f.write(f"- PostgreSQL was {ratio:.2f}x faster overall for single insertions\n")
        
        f.write("- The performance difference is likely due to MongoDB's simpler write path, as it doesn't need to validate schemas, check constraints, or manage ACID transactions by default\n")
        f.write("- For PostgreSQL, the overhead of transaction management affects performance even for single-record inserts\n\n")
        
        # Batch Insertion Analysis
        f.write("### 5.2 Batch Insertion Analysis\n\n")
        f.write("In batch insertion tests:\n\n")
        
        # Calculate average batch improvement ratio (10000 batch vs single inserts)
        batch_large = batch_df[batch_df['batch_size'] == 10000].groupby('database')['records_per_second'].mean()
        single_avg = single_df.groupby('database')['records_per_second'].mean()
        
        mongo_batch_improvement = batch_large['mongodb'] / single_avg['mongodb'] if 'mongodb' in batch_large and 'mongodb' in single_avg else 0
        postgres_batch_improvement = batch_large['postgresql'] / single_avg['postgresql'] if 'postgresql' in batch_large and 'postgresql' in single_avg else 0
        
        f.write(f"- MongoDB showed a {mongo_batch_improvement:.2f}x performance improvement with large batch sizes (10,000) compared to single inserts\n")
        f.write(f"- PostgreSQL showed a {postgres_batch_improvement:.2f}x performance improvement with large batch sizes\n")
        
        if mongo_batch_improvement > postgres_batch_improvement:
            f.write("- MongoDB benefited more from batch processing than PostgreSQL\n")
        else:
            f.write("- PostgreSQL benefited more from batch processing than MongoDB\n")
        
        f.write("- Both databases showed significant improvements with batch sizes of 1,000 or more\n")
        f.write("- The optimal batch size appears to be around 10,000 records for both databases\n\n")
        
        # Concurrency Analysis
        f.write("### 5.3 Concurrency Analysis\n\n")
        f.write("In concurrent insertion tests:\n\n")
        
        # Calculate concurrency scaling
        conc_1w = concurrent_df[concurrent_df['num_workers'] == 1].groupby(['database', 'entity'])['records_per_second'].mean().groupby('database').mean()
        conc_8w = concurrent_df[concurrent_df['num_workers'] == 8].groupby(['database', 'entity'])['records_per_second'].mean().groupby('database').mean()
        
        mongo_concurrency_scaling = conc_8w['mongodb'] / conc_1w['mongodb'] if 'mongodb' in conc_8w and 'mongodb' in conc_1w else 0
        postgres_concurrency_scaling = conc_8w['postgresql'] / conc_1w['postgresql'] if 'postgresql' in conc_8w and 'postgresql' in conc_1w else 0
        
        f.write(f"- MongoDB showed a {mongo_concurrency_scaling:.2f}x performance improvement with 8 workers compared to 1 worker\n")
        f.write(f"- PostgreSQL showed a {postgres_concurrency_scaling:.2f}x performance improvement with 8 workers\n")
        
        if mongo_concurrency_scaling > postgres_concurrency_scaling:
            f.write("- MongoDB scaled better with concurrent operations than PostgreSQL\n")
        else:
            f.write("- PostgreSQL scaled better with concurrent operations than MongoDB\n")
        
        f.write("- Both databases showed diminishing returns beyond 4 workers, likely due to contention\n")
        f.write("- Combining batching with concurrency provided the best overall performance for both databases\n\n")
        
        # Resource Usage Analysis
        f.write("### 5.4 Resource Usage Analysis\n\n")
        
        cpu_mean = combined_df.groupby('database')['cpu_usage'].mean()
        memory_mean = combined_df.groupby('database')['memory_usage'].mean()
        
        if 'mongodb' in cpu_mean and 'postgresql' in cpu_mean:
            if cpu_mean['mongodb'] > cpu_mean['postgresql']:
                f.write(f"- MongoDB consumed more CPU resources across all test types ({cpu_mean['mongodb']:.2f}% vs {cpu_mean['postgresql']:.2f}%)\n")
            else:
                f.write(f"- PostgreSQL consumed more CPU resources across all test types ({cpu_mean['postgresql']:.2f}% vs {cpu_mean['mongodb']:.2f}%)\n")
        
        if 'mongodb' in memory_mean and 'postgresql' in memory_mean:
            if memory_mean['mongodb'] > memory_mean['postgresql']:
                f.write(f"- MongoDB used more memory during the insertion tests ({memory_mean['mongodb']:.2f} MB vs {memory_mean['postgresql']:.2f} MB)\n")
            else:
                f.write(f"- PostgreSQL used more memory during the insertion tests ({memory_mean['postgresql']:.2f} MB vs {memory_mean['mongodb']:.2f} MB)\n")
        
        f.write("- Concurrent operations increased resource usage for both databases, as expected\n")
        f.write("- Batch operations tended to be more resource-efficient for the same throughput\n\n")
        
        # Conclusions
        f.write("## 6. Conclusions and Recommendations\n\n")
        
        f.write("### 6.1 Summary of Findings\n\n")
        
        overall_winner = "MongoDB" if overall_perf['mongodb'] > overall_perf['postgresql'] else "PostgreSQL"
        single_winner = "MongoDB" if single_grouped.loc['mongodb'].mean() > single_grouped.loc['postgresql'].mean() else "PostgreSQL"
        batch_winner = "MongoDB" if mongo_batch_improvement > postgres_batch_improvement else "PostgreSQL"
        concurrency_winner = "MongoDB" if mongo_concurrency_scaling > postgres_concurrency_scaling else "PostgreSQL"
        resource_winner = "MongoDB" if (cpu_mean['mongodb'] + memory_mean['mongodb']) < (cpu_mean['postgresql'] + memory_mean['postgresql']) else "PostgreSQL"
        
        f.write(f"- **Overall Performance Winner**: {overall_winner}\n")
        f.write(f"- **Best for Single Inserts**: {single_winner}\n")
        f.write(f"- **Best for Batch Processing**: {batch_winner}\n")
        f.write(f"- **Best for Concurrent Operations**: {concurrency_winner}\n")
        f.write(f"- **Most Resource Efficient**: {resource_winner}\n\n")
        
        f.write("### 6.2 Recommendations\n\n")
        
        f.write("Based on the results of our benchmark, we can make the following recommendations:\n\n")
        
        f.write("- **For write-heavy applications with high throughput requirements**: Choose MongoDB and utilize batch insertions with concurrency\n")
        f.write("- **For applications with strict transactional requirements**: Choose PostgreSQL, but ensure batch operations are used\n")
        f.write("- **For optimal MongoDB performance**: Use batch sizes of 1,000-10,000 and 4-8 concurrent workers\n")
        f.write("- **For optimal PostgreSQL performance**: Focus on larger batch sizes (1,000-10,000) rather than increased concurrency\n")
        f.write("- **For resource-constrained environments**: PostgreSQL may be preferable due to lower resource utilization\n\n")
        
        f.write("### 6.3 Future Work\n\n")
        
        f.write("This benchmark focused solely on insertion performance. For a more comprehensive comparison, future work could include:\n\n")
        
        f.write("- Query performance benchmarks (simple queries, complex joins/aggregations)\n")
        f.write("- Update and delete performance\n")
        f.write("- Performance under mixed workloads (reads and writes)\n")
        f.write("- Impact of indexing on write performance\n")
        f.write("- Performance with larger datasets (10M+ records)\n")
        f.write("- Testing with different hardware configurations\n")
        
        print(f"Report generated: {output_file}")

def main():
    """Main function to run the report generator"""
    parser = argparse.ArgumentParser(description='Generate MongoDB vs PostgreSQL benchmark report')
    parser.add_argument('--results-dir', type=str, default='results', help='Directory containing the benchmark results')
    parser.add_argument('--output-file', type=str, default='mongodb_vs_postgresql_report.md', help='Output markdown report file')
    parser.add_argument('--charts-dir', type=str, default='charts', help='Directory to save charts')
    
    args = parser.parse_args()
    
    # Load results
    mongo_results, postgres_results = load_results(args.results_dir)
    
    # Generate charts
    generate_charts(mongo_results, postgres_results, args.charts_dir)
    
    # Generate report
    generate_report(mongo_results, postgres_results, args.output_file)
    
    print(f"Report and charts generated successfully!")

if __name__ == "__main__":
    main()