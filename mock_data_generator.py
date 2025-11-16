import pandas as pd
import numpy as np # <<< ADDED THIS LINE
from datetime import timedelta

# --- Configuration ---
INPUT_FILE = 'news_sentiment_report.csv'
OUTPUT_FILE = 'news_sentiment_report_7day_mock.csv'
DAYS_OF_HISTORY_NEEDED = 7

def generate_mock_history(input_file, output_file, days):
    """
    Loads data and synthetically creates a history spanning the required number of days.
    """
    try:
        df_original = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found. Please ensure the file is in the current directory.")
        return

    # Convert the publishedAt column to datetime
    df_original['publishedAt'] = pd.to_datetime(df_original['publishedAt'])

    # Determine the earliest date in your current dataset
    earliest_date = df_original['publishedAt'].dt.normalize().min()

    # List to hold all the synthetic DataFrames
    all_dfs = []

    # Loop to create data for each day needed
    for i in range(days):
        # Create a new date, counting backwards from the earliest original date
        new_date = earliest_date - timedelta(days=i)
        
        # Create a copy of the original data
        df_new = df_original.copy()
        
        # Shift the publishedAt column to the new date, keeping the original time component
        df_new['publishedAt'] = df_new['publishedAt'].dt.time.apply(
            lambda t: pd.to_datetime(f"{new_date.date()} {t}")
        )
        
        # Apply a small random shift to the sentiment score to make the trend non-linear
        # *** CORRECTED ERROR HERE: CHANGED pd.np.random.rand TO np.random.rand ***
        df_new['sentiment_score'] = df_new['sentiment_score'] + pd.Series(
            (0.15 * (1 - 2 * np.random.rand(len(df_new)))), index=df_new.index
        )
        
        all_dfs.append(df_new)

    # Concatenate all synthetic data into a single DataFrame
    df_mock = pd.concat(all_dfs, ignore_index=True).sort_values(by='publishedAt', ascending=True)

    # Save the new mock data file
    df_mock.to_csv(output_file, index=False)
    
    # Check the actual number of unique dates created
    unique_days = df_mock['publishedAt'].dt.normalize().nunique()
    
    print(f"\n✅ Mock 7-day history created successfully!")
    print(f"File saved as '{output_file}' with {len(df_mock)} total records.")
    print(f"It spans {unique_days} unique days of data.")

if __name__ == "__main__":
    # Suppress the UserWarning about non-integer labels for indexing that occurs with pd.np.random.rand
    with pd.option_context('mode.chained_assignment', None):
        generate_mock_history(INPUT_FILE, OUTPUT_FILE, DAYS_OF_HISTORY_NEEDED)