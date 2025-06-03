import pandas as pd
import statistics

def analyze_row_word_counts(csv_file_path):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Initialize lists to store word counts
    row_word_counts = []
    
    # Process each row
    for _, row in df.iterrows():
        word_count = 0
        # Count words in each cell of the row
        for cell in row:
            if isinstance(cell, str):  # Check if the cell is a string
                word_count += len(cell.split())
        row_word_counts.append(word_count)
    
    if not row_word_counts:
        return "No valid text found in the CSV."
    
    # Calculate average and maximum word count
    avg_word_count = statistics.mean(row_word_counts)
    max_word_count = max(row_word_counts)
    
    return {
        'average_row_word_count': round(avg_word_count, 2),
        'max_row_word_count': max_word_count
    }

# Example usage
if __name__ == "__main__":
    # Replace with your CSV file path
    file_path = 'E:/SVMpredictionmodel/IMDB Dataset.csv/IMDB Dataset.csv'
    result = analyze_row_word_counts(file_path)
    print(f"Average words per row: {result['average_row_word_count']}")
    print(f"Maximum words in a row: {result['max_row_word_count']}")