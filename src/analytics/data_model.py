import os
import pandas as pd
from pathlib import Path
from typing import Optional

# Use project root to find Data/Processed
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "Data" / "Processed"

class AnalyticalDataModel:
    """
    Central analytical data model for Phase 3A.
    Loads and caches canonical processed datasets to avoid repetitive file I/O.
    Provides standard approved joins.
    """
    _cache = {}

    def __init__(self, processed_dir: str = str(PROCESSED_DIR)):
        self.processed_dir = Path(processed_dir)

    def _load_csv(self, filename: str) -> pd.DataFrame:
        if filename not in self.__class__._cache:
            filepath = self.processed_dir / filename
            if not filepath.exists():
                # fallback to data/Processed if Data/Processed doesn't exist
                filepath = Path(str(self.processed_dir).replace("Data", "data", 1)) / filename
                if not filepath.exists():
                    raise FileNotFoundError(f"Missing canonical dataset: {filepath}")
            df = pd.read_csv(filepath)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            self.__class__._cache[filename] = df
        return self.__class__._cache[filename]

    def get_sales(self) -> pd.DataFrame:
        return self._load_csv("fact_sales_monthly.csv")

    def get_products(self) -> pd.DataFrame:
        return self._load_csv("dim_product.csv")

    def get_customers(self) -> pd.DataFrame:
        return self._load_csv("dim_customer.csv")

    def get_markets(self) -> pd.DataFrame:
        return self._load_csv("dim_market.csv")

    def get_inventory(self) -> pd.DataFrame:
        return self._load_csv("fact_inventory_monthly.csv")

    def get_marketing(self) -> pd.DataFrame:
        return self._load_csv("fact_marketing_monthly.csv")

    def get_pricing(self) -> pd.DataFrame:
        return self._load_csv("fact_competitor_pricing_monthly.csv")

    def get_support(self) -> pd.DataFrame:
        return self._load_csv("fact_support_tickets.csv")

    def get_crm(self) -> pd.DataFrame:
        return self._load_csv("fact_crm_notes.csv")

    def get_sales_calls(self) -> pd.DataFrame:
        return self._load_csv("fact_sales_calls.csv")

    def get_joined_sales(self) -> pd.DataFrame:
        """Returns fact_sales_monthly joined with product, customer, and market dimensions."""
        if "joined_sales" not in self.__class__._cache:
            sales = self.get_sales()
            products = self.get_products()
            customers = self.get_customers()
            markets = self.get_markets()

            joined = sales.merge(products, on="product_code", how="left")
            joined = joined.merge(customers, on="customer_code", how="left")
            joined = joined.merge(markets, on="market", how="left")
            self.__class__._cache["joined_sales"] = joined
        return self.__class__._cache["joined_sales"]

    def apply_scope(self, df: pd.DataFrame, request: dict) -> pd.DataFrame:
        """
        Applies request scope filters (market, product_code, category, channel) to df.
        If a column (like category or channel) is not in df but can be resolved via
        merging with products/customers, does so before filtering.
        """
        df_filtered = df.copy()
        
        # 1. Handle market
        if 'market' in request:
            if 'market' not in df_filtered.columns and 'customer_code' in df_filtered.columns:
                customers = self.get_customers()[['customer_code', 'market']]
                df_filtered = df_filtered.merge(customers, on='customer_code', how='left')
            if 'market' in df_filtered.columns:
                df_filtered = df_filtered[df_filtered['market'] == request['market']]
                
        # 2. Handle product_code
        if 'product_code' in request:
            if 'product_code' in df_filtered.columns:
                df_filtered = df_filtered[df_filtered['product_code'] == request['product_code']]
                
        # 3. Handle category
        if 'category' in request:
            if 'category' not in df_filtered.columns and 'product_code' in df_filtered.columns:
                products = self.get_products()[['product_code', 'category']]
                df_filtered = df_filtered.merge(products, on='product_code', how='left')
            if 'category' in df_filtered.columns:
                df_filtered = df_filtered[df_filtered['category'] == request['category']]
                
        # 4. Handle channel
        if 'channel' in request:
            if 'channel' not in df_filtered.columns and 'customer_code' in df_filtered.columns:
                customers = self.get_customers()[['customer_code', 'channel']]
                df_filtered = df_filtered.merge(customers, on='customer_code', how='left')
            if 'channel' in df_filtered.columns:
                df_filtered = df_filtered[df_filtered['channel'] == request['channel']]
                
        return df_filtered

