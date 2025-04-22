import json
import time
import psutil
import argparse
import concurrent.futures
import pymongo
import psycopg2
import psycopg2.extras
from typing import Dict, List, Any, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("benchmark.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class DatabaseBenchmark:
    def __init__(self, db_type: str, connection_params: Dict[str, Any], data_dir: str):
        """
        Initialize the benchmark for a specific database
        
        Args:
            db_type: Type of database ('mongodb' or 'postgresql')
            connection_params: Connection parameters for the database
            data_dir: Directory containing the data files
        """
        self.db_type = db_type
        self.connection_params = connection_params
        self.data_dir = data_dir
        self.results = {}
        
        # Load data
        self.customers = self._load_data('customers')
        self.products = self._load_data('products')
        self.orders = self._load_data('orders')
        
        logger.info(f"Loaded {len(self.customers)} customers, {len(self.products)} products, and {len(self.orders)} orders")
        
    def _load_data(self, entity_name: str) -> List[Dict[str, Any]]:
        """Load data from JSON file"""
        file_path = os.path.join(self.data_dir, f"{entity_name}.json")
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def connect(self):
        """Connect to the database"""
        if self.db_type == 'mongodb':
            client = pymongo.MongoClient(**self.connection_params)
            self.db = client['performance_test']
            self.connection = client
        elif self.db_type == 'postgresql':
            self.connection = psycopg2.connect(**self.connection_params)
            self.connection.autocommit = False  # We'll handle transactions manually
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")
        
        logger.info(f"Connected to {self.db_type}")
    
    def disconnect(self):
        """Disconnect from the database"""
        if hasattr(self, 'connection') and self.connection:
            self.connection.close()
            logger.info(f"Disconnected from {self.db_type}")
    
    def clear_data(self):
        """Clear all data from the database"""
        if self.db_type == 'mongodb':
            self.db.customers.delete_many({})
            self.db.products.delete_many({})
            self.db.orders.delete_many({})
        elif self.db_type == 'postgresql':
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT clear_all_tables();")
                self.connection.commit()
        
        logger.info(f"Cleared all data from {self.db_type}")
    
    def _insert_mongodb_single(self, collection_name: str, data: List[Dict[str, Any]]) -> Tuple[float, int]:
        """Insert data into MongoDB one by one"""
        collection = self.db[collection_name]
        start_time = time.time()
        
        for item in data:
            collection.insert_one(item)
            
        elapsed_time = time.time() - start_time
        return elapsed_time, len(data)
    
    def _insert_mongodb_batch(self, collection_name: str, data: List[Dict[str, Any]], batch_size: int) -> Tuple[float, int]:
        """Insert data into MongoDB in batches"""
        collection = self.db[collection_name]
        start_time = time.time()
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            collection.insert_many(batch)
            
        elapsed_time = time.time() - start_time
        return elapsed_time, len(data)
    
    def _insert_postgresql_single(self, table_name: str, data: List[Dict[str, Any]]) -> Tuple[float, int]:
        """Insert data into PostgreSQL one by one"""
        start_time = time.time()
        
        with self.connection.cursor() as cursor:
            for item in data:
                if table_name == 'customers':
                    cursor.execute(
                        "INSERT INTO customers (customer_id, email, name, address, phone, created_at, last_login, "
                        "preferences, account_status, total_orders, total_spent) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                        (
                            item['customer_id'], 
                            item['email'], 
                            item['name'], 
                            json.dumps(item['address']), 
                            item['phone'],
                            item['created_at'], 
                            item['last_login'], 
                            item['preferences'], 
                            item['account_status'],
                            item['total_orders'], 
                            item['total_spent']
                        )
                    )
                elif table_name == 'products':
                    cursor.execute(
                        "INSERT INTO products (product_id, name, description, price, category, tags, inventory, "
                        "supplier_id, created_at, updated_at, dimensions, is_available) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                        (
                            item['product_id'], 
                            item['name'], 
                            item['description'], 
                            item['price'], 
                            item['category'],
                            item['tags'], 
                            item['inventory'], 
                            item['supplier_id'], 
                            item['created_at'], 
                            item['updated_at'],
                            json.dumps(item['dimensions']), 
                            item['is_available']
                        )
                    )
                elif table_name == 'orders':
                    cursor.execute(
                        "INSERT INTO orders (order_id, customer_id, order_date, status, items, shipping_address, "
                        "payment_info, total_amount, discount_applied, shipping_cost, tax) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                        (
                            item['order_id'], 
                            item['customer_id'], 
                            item['order_date'], 
                            item['status'], 
                            json.dumps(item['items']),
                            json.dumps(item['shipping_address']), 
                            json.dumps(item['payment_info']), 
                            item['total_amount'],
                            item['discount_applied'], 
                            item['shipping_cost'], 
                            item['tax']
                        )
                    )
            
            self.connection.commit()
            
        elapsed_time = time.time() - start_time
        return elapsed_time, len(data)
    
    def _insert_postgresql_batch(self, table_name: str, data: List[Dict[str, Any]], batch_size: int) -> Tuple[float, int]:
        """Insert data into PostgreSQL in batches using execute_values"""
        start_time = time.time()
        
        with self.connection.cursor() as cursor:
            if table_name == 'customers':
                values = [(
                    item['customer_id'], 
                    item['email'], 
                    item['name'], 
                    json.dumps(item['address']), 
                    item['phone'],
                    item['created_at'], 
                    item['last_login'], 
                    item['preferences'], 
                    item['account_status'],
                    item['total_orders'], 
                    item['total_spent']
                ) for item in data]
                
                psycopg2.extras.execute_values(
                    cursor,
                    "INSERT INTO customers (customer_id, email, name, address, phone, created_at, last_login, "
                    "preferences, account_status, total_orders, total_spent) VALUES %s",
                    values,
                    page_size=batch_size
                )
                
            elif table_name == 'products':
                values = [(
                    item['product_id'], 
                    item['name'], 
                    item['description'], 
                    item['price'], 
                    item['category'],
                    item['tags'], 
                    item['inventory'], 
                    item['supplier_id'], 
                    item['created_at'], 
                    item['updated_at'],
                    json.dumps(item['dimensions']), 
                    item['is_available']
                ) for item in data]
                
                psycopg2.extras.execute_values(
                    cursor,
                    "INSERT INTO products (product_id, name, description, price, category, tags, inventory, "
                    "supplier_id, created_at, updated_at, dimensions, is_available) VALUES %s",
                    values,
                    page_size=batch_size
                )
                
            elif table_name == 'orders':
                values = [(
                    item['order_id'], 
                    item['customer_id'], 
                    item['order_date'], 
                    item['status'], 
                    json.dumps(item['items']),
                    json.dumps(item['shipping_address']), 
                    json.dumps(item['payment_info']), 
                    item['total_amount'],
                    item['discount_applied'], 
                    item['shipping_cost'], 
                    item['tax']
                ) for item in data]
                
                psycopg2.extras.execute_values(
                    cursor,
                    "INSERT INTO orders (order_id, customer_id, order_date, status, items, shipping_address, "
                    "payment_info, total_amount, discount_applied, shipping_cost, tax) VALUES %s",
                    values,
                    page_size=batch_size
                )
            
            self.connection.commit()
            
        elapsed_time = time.time() - start_time
        return elapsed_time, len(data)
    
    def _insert_concurrent(self, entity_name: str, data: List[Dict[str, Any]], num_workers: int, batch_size: int) -> Tuple[float, int]:
        """Insert data using multiple concurrent workers"""
        # Split data into chunks for each worker
        chunk_size = len(data) // num_workers
        chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            if self.db_type == 'mongodb':
                futures = [executor.submit(self._insert_mongodb_batch, entity_name, chunk, batch_size) for chunk in chunks]
            else:
                # For PostgreSQL, we need to create a new connection for each worker
                futures = []
                for chunk in chunks:
                    # Create a new connection for this worker
                    conn = psycopg2.connect(**self.connection_params)
                    cursor = conn.cursor()
                    
                    # Submit the task
                    if batch_size == 1:
                        futures.append(executor.submit(self._insert_postgresql_single, entity_name, chunk))
                    else:
                        futures.append(executor.submit(self._insert_postgresql_batch, entity_name, chunk, batch_size))
            
            # Wait for all futures to complete
            concurrent.futures.wait(futures)
        
        elapsed_time = time.time() - start_time
        return elapsed_time, len(data)
    
    def run_benchmark(self, test_name: str, entity_name: str, method: str, batch_size: int = 1, num_workers: int = 1) -> Dict[str, Any]:
        """Run a benchmark test"""
        logger.info(f"Running {test_name} test for {self.db_type} with {entity_name}")
        
        # Get the data for this entity
        if entity_name == 'customers':
            data = self.customers
        elif entity_name == 'products':
            data = self.products
        elif entity_name == 'orders':
            data = self.orders
        else:
            raise ValueError(f"Unknown entity: {entity_name}")
        
        # Start monitoring resources
        process = psutil.Process(os.getpid())
        start_cpu_percent = process.cpu_percent()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run the appropriate insertion method
        if method == 'single':
            if self.db_type == 'mongodb':
                elapsed_time, count = self._insert_mongodb_single(entity_name, data)
            else:
                elapsed_time, count = self._insert_postgresql_single(entity_name, data)
        elif method == 'batch':
            if self.db_type == 'mongodb':
                elapsed_time, count = self._insert_mongodb_batch(entity_name, data, batch_size)
            else:
                elapsed_time, count = self._insert_postgresql_batch(entity_name, data, batch_size)
        elif method == 'concurrent':
            elapsed_time, count = self._insert_concurrent(entity_name, data, num_workers, batch_size)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # End monitoring resources
        end_cpu_percent = process.cpu_percent()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Calculate metrics
        records_per_second = count / elapsed_time
        cpu_usage = end_cpu_percent - start_cpu_percent
        memory_usage = end_memory - start_memory
        
        logger.info(f"Completed {test_name} in {elapsed_time:.2f}s, {records_per_second:.2f} records/second")
        
        # Store results
        result = {
            'test_name': test_name,
            'database': self.db_type,
            'entity': entity_name,
            'method': method,
            'batch_size': batch_size,
            'num_workers': num_workers,
            'count': count,
            'elapsed_time': elapsed_time,
            'records_per_second': records_per_second,
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage
        }
        
        self.results[test_name] = result
        return result
        
    def run_all_benchmarks(self, batch_sizes=[1, 100, 1000, 10000], num_workers_list=[1, 2, 4, 8]):
        """Run all benchmark tests and return the results"""
        all_results = []
        
        # Single inserts
        self.clear_data()
        all_results.append(self.run_benchmark('customer_single', 'customers', 'single'))

        all_results.append(self.run_benchmark('product_single', 'products', 'single'))
        
       
        all_results.append(self.run_benchmark('order_single', 'orders', 'single'))

        self.clear_data()
        
        # Batch inserts with different batch sizes
        for batch_size in batch_sizes:
            
            all_results.append(self.run_benchmark(f'customer_batch_{batch_size}', 'customers', 'batch', batch_size))
            
            
            all_results.append(self.run_benchmark(f'product_batch_{batch_size}', 'products', 'batch', batch_size))
            
            
            all_results.append(self.run_benchmark(f'order_batch_{batch_size}', 'orders', 'batch', batch_size))
            self.clear_data()
        
        # Concurrent inserts with different number of workers
        for num_workers in num_workers_list:
            for batch_size in [1, 1000]:  # Test with single inserts and batches
                
                all_results.append(self.run_benchmark(
                    f'customer_concurrent_{num_workers}w_{batch_size}b', 
                    'customers', 'concurrent', batch_size, num_workers
                ))
                
                
                all_results.append(self.run_benchmark(
                    f'product_concurrent_{num_workers}w_{batch_size}b', 
                    'products', 'concurrent', batch_size, num_workers
                ))
                
                
                all_results.append(self.run_benchmark(
                    f'order_concurrent_{num_workers}w_{batch_size}b', 
                    'orders', 'concurrent', batch_size, num_workers
                ))
                self.clear_data()
        
        return all_results


def generate_charts(mongo_results, postgres_results, output_dir='results'):
    """Generate charts comparing MongoDB and PostgreSQL performance"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert results to DataFrames
    mongo_df = pd.DataFrame(mongo_results)
    postgres_df = pd.DataFrame(postgres_results)
    
    # Combine results
    combined_df = pd.concat([mongo_df, postgres_df])
    
    # Chart 1: Single insertion performance by entity type
    plt.figure(figsize=(12, 6))
    single_df = combined_df[combined_df['method'] == 'single']
    
    # Group by database and entity
    single_grouped = single_df.groupby(['database', 'entity'])['records_per_second'].mean().unstack()
    single_grouped.plot(kind='bar')
    
    plt.title('Single Record Insertion Performance')
    plt.ylabel('Records per Second')
    plt.xlabel('Database')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/single_insertion_performance.png')
    
    # Chart 2: Batch insertion performance by batch size
    plt.figure(figsize=(12, 8))
    batch_df = combined_df[combined_df['method'] == 'batch']
    
    # Plot each entity type separately
    for entity in ['customers', 'products', 'orders']:
        entity_df = batch_df[batch_df['entity'] == entity]
        
        plt.subplot(3, 1, ['customers', 'products', 'orders'].index(entity) + 1)
        
        # Group by database and batch size
        entity_grouped = entity_df.pivot_table(
            index='batch_size', 
            columns='database', 
            values='records_per_second'
        )
        
        entity_grouped.plot(kind='line', marker='o')
        plt.title(f'Batch Insertion Performance - {entity.capitalize()}')
        plt.ylabel('Records per Second')
        plt.xlabel('Batch Size')
        plt.xscale('log')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/batch_insertion_performance.png')
    
    # Chart 3: Concurrent insertion performance
    plt.figure(figsize=(12, 8))
    concurrent_df = combined_df[combined_df['method'] == 'concurrent']
    
    # Plot each entity type separately
    for entity in ['customers', 'products', 'orders']:
        entity_df = concurrent_df[concurrent_df['entity'] == entity]
        
        # Filter for batch size = 1000
        batch_entity_df = entity_df[entity_df['batch_size'] == 1000]
        
        plt.subplot(3, 1, ['customers', 'products', 'orders'].index(entity) + 1)
        
        # Group by database and num_workers
        entity_grouped = batch_entity_df.pivot_table(
            index='num_workers', 
            columns='database', 
            values='records_per_second'
        )
        
        entity_grouped.plot(kind='line', marker='o')
        plt.title(f'Concurrent Insertion Performance (Batch=1000) - {entity.capitalize()}')
        plt.ylabel('Records per Second')
        plt.xlabel('Number of Workers')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/concurrent_insertion_performance.png')
    
    # Chart 4: CPU usage comparison
    plt.figure(figsize=(12, 6))
    
    # Average CPU usage by database and method
    cpu_usage = combined_df.groupby(['database', 'method'])['cpu_usage'].mean().unstack()
    cpu_usage.plot(kind='bar')
    
    plt.title('Average CPU Usage by Method')
    plt.ylabel('CPU Usage (%)')
    plt.xlabel('Database')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cpu_usage_comparison.png')
    
    # Chart 5: Memory usage comparison
    plt.figure(figsize=(12, 6))
    
    # Average memory usage by database and method
    memory_usage = combined_df.groupby(['database', 'method'])['memory_usage'].mean().unstack()
    memory_usage.plot(kind='bar')
    
    plt.title('Average Memory Usage by Method')
    plt.ylabel('Memory Usage (MB)')
    plt.xlabel('Database')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/memory_usage_comparison.png')
    
    logger.info(f"Charts saved to {output_dir}")


def main():
    """Main function to run the benchmark"""
    parser = argparse.ArgumentParser(description='Run MongoDB vs PostgreSQL insertion performance benchmark')
    parser.add_argument('--data-dir', type=str, default='data', help='Directory containing the test data')
    parser.add_argument('--output-dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--mongo-host', type=str, default='localhost', help='MongoDB host')
    parser.add_argument('--mongo-port', type=int, default=27017, help='MongoDB port')
    parser.add_argument('--postgres-host', type=str, default='localhost', help='PostgreSQL host')
    parser.add_argument('--postgres-port', type=int, default=5432, help='PostgreSQL port')
    parser.add_argument('--postgres-user', type=str, default='postgres', help='PostgreSQL username')
    parser.add_argument('--postgres-password', type=str, default='postgres', help='PostgreSQL password')
    parser.add_argument('--postgres-db', type=str, default='performance_test', help='PostgreSQL database name')
    
    args = parser.parse_args()
    
    # Create results directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # MongoDB connection parameters
    mongo_params = {
        'host': args.mongo_host,
        'port': args.mongo_port
    }
    
    # PostgreSQL connection parameters
    postgres_params = {
        'host': args.postgres_host,
        'port': args.postgres_port,
        'user': args.postgres_user,
        'password': args.postgres_password,
        'dbname': args.postgres_db
    }
    
    # Run MongoDB benchmarks
    # logger.info("Starting MongoDB benchmarks")
    # mongo_benchmark = DatabaseBenchmark('mongodb', mongo_params, args.data_dir)
    # mongo_benchmark.connect()
    # mongo_results = mongo_benchmark.run_all_benchmarks()
    # mongo_benchmark.disconnect()
    
    # # Save MongoDB results
    # with open(f'{args.output_dir}/mongodb_results.json', 'w') as f:
    #     json.dump(mongo_results, f, indent=2)
    
    # Run PostgreSQL benchmarks
    logger.info("Starting PostgreSQL benchmarks")
    postgres_benchmark = DatabaseBenchmark('postgresql', postgres_params, args.data_dir)
    postgres_benchmark.connect()
    postgres_results = postgres_benchmark.run_all_benchmarks()
    postgres_benchmark.disconnect()
    
    # Save PostgreSQL results
    with open(f'{args.output_dir}/postgresql_results.json', 'w') as f:
        json.dump(postgres_results, f, indent=2)
    
    # Generate comparison charts
    #generate_charts(mongo_results, postgres_results, args.output_dir)
    
    logger.info("Benchmark completed successfully!")


if __name__ == "__main__":
    main()