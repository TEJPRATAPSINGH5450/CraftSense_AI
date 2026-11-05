from pathlib import Path
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

products = {
    "Pottery": [
        "Blue pottery vase", "Terracotta clay pot",
        "Handmade ceramic bowl", "Decorative clay lamp"
    ],
    "Textiles": [
        "Phulkari embroidered dupatta", "Handwoven cotton shawl",
        "Traditional handloom saree", "Embroidered cushion cover"
    ],
    "Wooden Crafts": [
        "Carved wooden elephant", "Handmade wooden box",
        "Wooden wall decor", "Traditional wooden toy"
    ],
    "Jewellery": [
        "Terracotta necklace", "Handmade beaded earrings",
        "Traditional pendant", "Handcrafted bracelet"
    ],
    "Paintings": [
        "Madhubani folk painting", "Warli art canvas",
        "Miniature traditional painting", "Handmade folk art"
    ],
    "Home Decor": [
        "Handmade wall hanging", "Decorative bamboo basket",
        "Macrame home decor", "Handcrafted table decoration"
    ],
}

price_ranges = {
    "Pottery": (150, 1200),
    "Textiles": (300, 2500),
    "Wooden Crafts": (200, 1800),
    "Jewellery": (100, 1000),
    "Paintings": (250, 3000),
    "Home Decor": (150, 2000),
}

# These are intentionally simulated co-occurrence tendencies.
pairings = {
    "Pottery": ["Home Decor", "Wooden Crafts", "Paintings"],
    "Textiles": ["Jewellery", "Home Decor"],
    "Wooden Crafts": ["Pottery", "Home Decor"],
    "Jewellery": ["Textiles", "Paintings"],
    "Paintings": ["Home Decor", "Pottery"],
    "Home Decor": ["Pottery", "Wooden Crafts"],
}

rows = []
start = datetime(2025, 1, 1)
for invoice_num in range(1, 1801):
    customer = f"C{random.randint(1001, 1250)}"
    main = random.choice(list(products))
    selected = [main]
    for category in pairings[main]:
        if random.random() < 0.45:
            selected.append(category)

    invoice_date = start + timedelta(days=random.randint(0, 364))
    for category in set(selected):
        product = random.choice(products[category])
        low, high = price_ranges[category]
        rows.append({
            "InvoiceNo": f"INV{invoice_num:05d}",
            "CustomerID": customer,
            "Product": product,
            "Category": category,
            "Quantity": random.randint(1, 3),
            "UnitPrice": round(random.uniform(low, high), 2),
            "InvoiceDate": invoice_date.strftime("%Y-%m-%d"),
        })

df = pd.DataFrame(rows)

# Add a small amount of imperfect source data to demonstrate cleaning.
missing_idx = np.random.choice(df.index, size=max(1, int(len(df) * 0.005)), replace=False)
df.loc[missing_idx, "Product"] = np.nan
df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # one duplicate row

output = Path("data")
output.mkdir(exist_ok=True)
df.to_csv(output / "artisan_demo_transactions.csv", index=False)
print(f"Created {output / 'artisan_demo_transactions.csv'} with {len(df)} rows.")
print("NOTE: This is simulated demonstration data, not real artisan sales.")
