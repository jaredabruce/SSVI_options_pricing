# data_cleaning.py

import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

def load_and_clean_options(csv_path: str, chunksize=100_000) -> pd.DataFrame:
    """
    Reads the raw CSV in chunks (with a progress bar), merges calls & puts if present,
    and returns a DataFrame with columns:
      ['quote_date', 'expire_date', 'option_type', 'strike', 'underlying_last',
       'iv', 'mid_price', 'time_to_expiry_years'].

    :param csv_path: Path to the raw CSV file.
    :param chunksize: Number of rows to read per chunk. Adjust as needed.
    """
    # 1) We'll read the file in chunks to show a progress bar if it's large
    chunks = []
    with pd.read_csv(csv_path, low_memory=True, chunksize=chunksize) as reader:
        for chunk in tqdm(reader, desc="Reading CSV in chunks"):
            chunks.append(chunk)
    
    # 2) Concatenate all chunks into one DataFrame
    df = pd.concat(chunks, ignore_index=True)
    
    # 3) Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # 4) Example rename (adjust as needed)
    rename_map = {
        "[QUOTE_DATE]": "quote_date",
        "[EXPIRE_DATE]": "expire_date",
        "[UNDERLYING_LAST]": "underlying_last",
        "[STRIKE]": "strike",
        "[C_IV]": "c_iv",
        "[P_IV]": "p_iv",
        "[C_BID]": "c_bid",
        "[C_ASK]": "c_ask",
        "[P_BID]": "p_bid",
        "[P_ASK]": "p_ask",
        "[DTE]": "dte"
    }
    for old_col, new_col in rename_map.items():
        if old_col in df.columns:
            df.rename(columns={old_col: new_col}, inplace=True)

    # 5) Convert date cols to datetime
    for col in ["quote_date", "expire_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # 6) Convert numeric cols
    for col in ["strike", "underlying_last", "c_iv", "p_iv", "c_bid", "c_ask", "p_bid", "p_ask", "dte"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 7) Construct calls DataFrame
    calls = pd.DataFrame(columns=["quote_date", "expire_date", "underlying_last",
                                  "strike","iv","bid","ask","dte","option_type"])
    if "c_iv" in df.columns:
        calls = df[[
            "quote_date", "expire_date", "underlying_last", "strike",
            "c_iv", "c_bid", "c_ask", "dte"
        ]].copy()
        calls["option_type"] = "C"
        calls.rename(columns={
            "c_iv": "iv",
            "c_bid": "bid",
            "c_ask": "ask"
        }, inplace=True)

    # 8) Construct puts DataFrame
    puts = pd.DataFrame(columns=["quote_date", "expire_date", "underlying_last",
                                 "strike","iv","bid","ask","dte","option_type"])
    if "p_iv" in df.columns:
        puts = df[[
            "quote_date", "expire_date", "underlying_last", "strike",
            "p_iv", "p_bid", "p_ask", "dte"
        ]].copy()
        puts["option_type"] = "P"
        puts.rename(columns={
            "p_iv": "iv",
            "p_bid": "bid",
            "p_ask": "ask"
        }, inplace=True)

    # 9) Combine
    long_df = pd.concat([calls, puts], ignore_index=True)
    long_df.sort_values(by=["quote_date", "expire_date", "strike", "option_type"], inplace=True)
    long_df.reset_index(drop=True, inplace=True)

    # 10) Compute mid-price
    long_df["mid_price"] = 0.5 * (long_df["bid"] + long_df["ask"])

    # 11) Convert dte to time_to_expiry_years
    long_df["time_to_expiry_years"] = long_df["dte"] / 365.0

    # 12) Filter out nonsense
    long_df.dropna(subset=["quote_date", "expire_date", "strike", "underlying_last", 
                           "time_to_expiry_years"], inplace=True)
    long_df = long_df[long_df["time_to_expiry_years"] > 0]

    return long_df

def main():
    if len(sys.argv) < 2:
        print("Usage: python data_cleaning.py <raw_options.csv>")
        sys.exit(1)

    in_csv = sys.argv[1]
    out_csv = "options_cleaned.csv"

    df_clean = load_and_clean_options(in_csv, chunksize=100_000)
    df_clean.to_csv(out_csv, index=False)
    print(f"\nSaved cleaned data to {out_csv}, shape={df_clean.shape}")

if __name__ == "__main__":
    main()
