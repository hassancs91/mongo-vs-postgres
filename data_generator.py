import json
import random
import uuid
import datetime
import faker
import argparse
from typing import Dict, List, Any

# Initialize faker
fake = faker.Faker()

def generate_customer() -> Dict[str, Any]:
    """Generate a random customer record."""
    return {
        "customer_id": str(uuid.uuid4()),
        "email": fake.unique.email(),
        "name": fake.name(),
        "address": {
            "street": fake.street_address(),
            "city": fake.city(),
            "state": fake.state(),
            "zip": fake.zipcode(),
            "country": fake.country()
        },
        "phone": fake.phone_number(),
        "created_at": fake.date_time_between(start_date="-3y", end_date="now").isoformat(),
        "last_login": fake.date_time_between(start_date="-1m", end_date="now").isoformat(),
        "preferences": random.sample(["electronics", "clothing", "books", "home", "sports", "food", "toys", "beauty", "health"], random.randint(1, 5)),
        "account_status": random.choice(["active", "inactive", "suspended", "pending"]),
        "total_orders": random.randint(0, 50),
        "total_spent": round(random.uniform(0, 10000), 2)
    }

def generate_product() -> Dict[str, Any]:
    """Generate a random product record."""
    return {
        "product_id": str(uuid.uuid4()),
        "name": fake.catch_phrase(),
        "description": fake.paragraph(nb_sentences=5),
        "price": round(random.uniform(1, 1000), 2),
        "category": random.choice(["electronics", "clothing", "books", "home", "sports", "food", "toys", "beauty", "health"]),
        "tags": random.sample(["sale", "new", "featured", "clearance", "limited", "exclusive", "trending", "popular"], random.randint(1, 4)),
        "inventory": random.randint(0, 1000),
        "supplier_id": str(uuid.uuid4()),
        "created_at": fake.date_time_between(start_date="-2y", end_date="now").isoformat(),
        "updated_at": fake.date_time_between(start_date="-1m", end_date="now").isoformat(),
        "dimensions": {
            "weight": round(random.uniform(0.1, 100), 2),
            "width": round(random.uniform(1, 100), 2),
            "height": round(random.uniform(1, 100), 2),
            "depth": round(random.uniform(1, 100), 2)
        },
        "is_available": random.random() > 0.1  # 90% of products are available
    }

def generate_order(customer_ids: List[str], product_ids: List[str]) -> Dict[str, Any]:
    """Generate a random order record with references to existing customers and products."""
    items_count = random.randint(1, 5)
    items = []
    total_amount = 0
    
    # Select random products for this order
    order_products = random.sample(product_ids, items_count)
    
    for product_id in order_products:
        quantity = random.randint(1, 5)
        price = round(random.uniform(10, 500), 2)
        item_total = quantity * price
        total_amount += item_total
        
        items.append({
            "product_id": product_id,
            "quantity": quantity,
            "price_at_purchase": price
        })
    
    shipping_cost = round(random.uniform(5, 25), 2)
    tax = round(total_amount * 0.08, 2)  # 8% tax
    discount = round(total_amount * random.uniform(0, 0.2), 2) if random.random() > 0.7 else 0  # 30% chance of discount
    
    final_total = total_amount + shipping_cost + tax - discount
    
    return {
        "order_id": str(uuid.uuid4()),
        "customer_id": random.choice(customer_ids),
        "order_date": fake.date_time_between(start_date="-1y", end_date="now").isoformat(),
        "status": random.choice(["pending", "processing", "shipped", "delivered", "cancelled"]),
        "items": items,
        "shipping_address": {
            "street": fake.street_address(),
            "city": fake.city(),
            "state": fake.state(),
            "zip": fake.zipcode(),
            "country": fake.country()
        },
        "payment_info": {
            "method": random.choice(["credit_card", "paypal", "bank_transfer", "crypto"]),
            "transaction_id": str(uuid.uuid4()),
            "status": random.choice(["pending", "completed", "failed", "refunded"])
        },
        "total_amount": round(final_total, 2),
        "discount_applied": round(discount, 2),
        "shipping_cost": shipping_cost,
        "tax": tax
    }

def generate_dataset(num_customers: int, num_products: int, num_orders: int) -> Dict[str, List[Dict[str, Any]]]:
    """Generate a complete dataset with the specified number of records."""
    print(f"Generating {num_customers} customers...")
    customers = [generate_customer() for _ in range(num_customers)]
    
    print(f"Generating {num_products} products...")
    products = [generate_product() for _ in range(num_products)]
    
    # Extract IDs for reference
    customer_ids = [customer["customer_id"] for customer in customers]
    product_ids = [product["product_id"] for product in products]
    
    print(f"Generating {num_orders} orders...")
    orders = [generate_order(customer_ids, product_ids) for _ in range(num_orders)]
    
    return {
        "customers": customers,
        "products": products,
        "orders": orders
    }

def save_dataset(dataset: Dict[str, List[Dict[str, Any]]], output_dir: str):
    """Save the dataset to JSON files."""
    import os
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for entity_name, entities in dataset.items():
        filename = f"{output_dir}/{entity_name}.json"
        with open(filename, 'w') as f:
            json.dump(entities, f)
        print(f"Saved {len(entities)} {entity_name} to {filename}")

def main():
    parser = argparse.ArgumentParser(description='Generate test data for MongoDB vs PostgreSQL comparison')
    parser.add_argument('--customers', type=int, default=50000, help='Number of customers to generate')
    parser.add_argument('--products', type=int, default=50000, help='Number of products to generate')
    parser.add_argument('--orders', type=int, default=900000, help='Number of orders to generate')
    parser.add_argument('--output', type=str, default='data', help='Output directory for the generated data')
    
    args = parser.parse_args()
    
    total_records = args.customers + args.products + args.orders
    print(f"Generating a total of {total_records} records...")
    
    dataset = generate_dataset(args.customers, args.products, args.orders)
    save_dataset(dataset, args.output)
    
    print("Data generation complete!")

if __name__ == "__main__":
    main()